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
TIMEOUT = 60 * 60 * 2 # 2 hours

VOLUME_NAME_MODEL = "models"
VOLUME_NAME_INPUT = "inputs"
VOLUME_NAME_OUTPUT = "outputs"

#==============================================================================
app = modal.App(f"sdxl-lora-gen")

model_volume = modal.Volume.from_name(VOLUME_NAME_MODEL)

try:
  print("Deleting input volume")
  modal.Volume.delete(VOLUME_NAME_INPUT)
except:
  print("Input volume is not created yet.")

try:
  print("Deleting output volume")
  modal.Volume.delete(VOLUME_NAME_OUTPUT)
except:
  print("Output volume is not created yet.")

print("Creating input and output volumes")
input_volume = modal.Volume.from_name(VOLUME_NAME_INPUT, create_if_missing=True)
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
  .pip_install(
    [
      "torch==2.1.2",
      "torchvision==0.16.2"
    ],
    index_url="https://download.pytorch.org/whl/cu118")
  .pip_install(
    "xformers==0.0.23.post1",
    index_url="https://download.pytorch.org/whl/cu118")
  .run_commands(
    "git clone https://github.com/kohya-ss/sd-scripts.git",
    "cd sd-scripts && pip install --upgrade -r requirements.txt")
)

#==============================================================================
volumes= {
  '/model': model_volume,
  '/input': input_volume,
  '/output': output_volume
}

@app.function(image=image, gpu=GPU, volumes=volumes, timeout=TIMEOUT)
def generate(name: str):
  print(f"Remote function called. target: {name}")
  os.chdir('/sd-scripts')
  subprocess.run("accelerate config default --mixed_precision fp16", shell=True)
  subprocess.run(
    "accelerate launch --num_cpu_threads_per_process=4 "
    "sdxl_train_network.py "
    f"--config_file /input/config.toml "
    f"--output_name {name} ",
    shell=True)
  global output_volume
  output_volume.commit()

@app.local_entrypoint()
def main(name: str):
  print(f"Target name: {name}")
  if not os.path.exists(name):
    print(f"Target path does not exist: {name}")
    sys.exit(1)

  global input_volume, output_volume

  with input_volume.batch_upload(force=True) as batch:
    batch.put_directory(name, '/')
  print("Uploaded")

  generate.remote(name)

  v = modal.Volume.from_name(VOLUME_NAME_OUTPUT)
  with open(f"{name}.safetensors", "wb") as f:
    for chunk in v.read_file(f"{name}.safetensors"):
      f.write(chunk)
   