import argparse
import os
import subprocess
import sys

import modal
import modal.gpu

#==============================================================================
# Runtime Container Configurations
#==============================================================================
CUDA_VERSION = "12.4.0"
CONTAINER_OS_VERSION = "ubuntu22.04"
CONTAINER_TAG = f"{CUDA_VERSION}-devel-{CONTAINER_OS_VERSION}"
CONTAINER_PYTHON_VERSION = "3.10"

#==============================================================================
# Global Settings
#==============================================================================
GPU = "L40S"
TIMEOUT = 60 * 60 * 3 # 3 hours

VOLUME_NAME_MODEL = "models"
VOLUME_NAME_OUTPUT = "outputs"

#==============================================================================
app = modal.App(f"sdxl-lora-gen")

model_volume = modal.Volume.from_name(VOLUME_NAME_MODEL)
output_volume = modal.Volume.from_name(VOLUME_NAME_OUTPUT, create_if_missing=True)

#==============================================================================
# Runtime Image Definition
#==============================================================================
image = (
  modal.Image.from_registry(f"nvidia/cuda:{CONTAINER_TAG}", add_python=CONTAINER_PYTHON_VERSION)
  .apt_install(
    "git",
    "libglib2.0-0",
    "libsm6",
    "libxrender1",
    "libxext6",
    "ffmpeg",
    "libgl1")
  .run_commands(
    # 1. PyTorch を CUDA 12.4 ビルドでインストール（公式手順 Step 1）
    "pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124",
    # 2. sd-scripts をクローンして依存をインストール（公式手順 Step 2）
    #    --extra-index-url で cu124 を参照させ、torch/torchvision が CPU 版で上書きされるのを防ぐ
    "git clone https://github.com/kohya-ss/sd-scripts.git",
    "cd sd-scripts && pip install --upgrade -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124",
    # 3. xformers を最後にインストール（公式手順 Step 3）
    "pip install xformers --index-url https://download.pytorch.org/whl/cu124")
)

#==============================================================================
@app.function(image=image, gpu=GPU, volumes={'/model': model_volume, '/output': output_volume}, timeout=TIMEOUT)
def generate(name: str):
  print(f"Remote function called. target: {name}")

  local_input = '/tmp/input'
  os.makedirs(local_input, exist_ok=True)

  # input-{name} volume からローカル /tmp/input にファイルをコピー
  # （with_options が不要なため、volume API で直接読み込む）
  input_vol = modal.Volume.from_name(f"input-{name}")
  print(f"Syncing input-{name} → {local_input}")
  for entry in input_vol.listdir('/'):
    if getattr(entry, 'is_dir', False):
      continue
    data = b''.join(input_vol.read_file(entry.path))
    with open(os.path.join(local_input, os.path.basename(entry.path)), 'wb') as f:
      f.write(data)
  print(f"Sync complete.")

  # config.toml / dataset.toml の /input パスを /tmp/input に書き換え
  # （/output は Volume マウント先のまま維持）
  for fname in ('config.toml', 'dataset.toml'):
    fpath = os.path.join(local_input, fname)
    if not os.path.exists(fpath):
      continue
    text = open(fpath).read()
    # '/input/foo' や '/input' 形式の両方を置換
    text = text.replace("'/input/", f"'{local_input}/")
    text = text.replace('"/input/', f'"{local_input}/')
    text = text.replace("'/input'", f"'{local_input}'")
    text = text.replace('"/input"', f'"{local_input}"')
    open(fpath, 'w').write(text)

  os.chdir('/sd-scripts')
  subprocess.run("accelerate config default --mixed_precision fp16", shell=True, check=True)
  result = subprocess.run(
    "accelerate launch --num_cpu_threads_per_process=4 "
    "sdxl_train_network.py "
    f"--config_file {local_input}/config.toml "
    f"--output_name {name} ",
    shell=True)
  if result.returncode != 0:
    raise RuntimeError(f"Training failed with exit code {result.returncode}")
  output_volume.commit()

@app.local_entrypoint()
def main(name: str):
  print(f"Target name: {name}")
  if not os.path.exists(name):
    print(f"Target path does not exist: {name}")
    sys.exit(1)

  input_volume_name = f"input-{name}"

  # 既存の入力 Volume を削除してクリーンな状態にする
  try:
    modal.Volume.objects.delete(input_volume_name)
  except Exception:
    pass

  input_vol = modal.Volume.from_name(input_volume_name, create_if_missing=True)
  with input_vol.batch_upload(force=True) as batch:
    batch.put_directory(name, '/')
  print("Uploaded")

  # 入力 Volume は generate 内で volume API 経由でコピーする（複数同時実行対応）
  generate.remote(name)

  out_vol = modal.Volume.from_name(VOLUME_NAME_OUTPUT)
  with open(f"{name}.safetensors", "wb") as f:
    for chunk in out_vol.read_file(f"{name}.safetensors"):
      f.write(chunk)

  # 入力 Volume を削除してクリーンアップ
  modal.Volume.objects.delete(input_volume_name)
  print(f"Done: {name}.safetensors")
