# SDXL 用 LoRA を Modal で作るスクリプト
SDXL 用の LoRA を [Modal](https://modal.com/) で作成するためのサンプルスクリプトです。

## Modal とは
[Modal](https://modal.com/) は任意の Python のコードをクラウド上で実行するための環境です。AI や ML に親和性が高く、GPU を使用した環境が安価に使用できます。
無料プランでも毎月30ドル分のクレジットがもらえます。

料金は CPU、メモリ、GPU の使用時間に応じて課金され、A10G を一時間動かしても1ドルちょっとなので、比較的気軽に LoRA 生成ができます。

## 準備
Python 3.10 以降が必要です。

```
git clone sdxl-lora-gen-with-modal
cd sdxl-lora-gen-with-modal
python -m venv venv
venv/Script/activate
pip install -r requirements.txt
```

Modal のサイトに登録しておきます。
`python -m modal setup` を実行して初期設定をしておきます。

## 使い方
例：東北キリタンの LoRA を作る

1. 素材をダウンロードする  
[AI画像モデル用学習データ](https://zunko.jp/con_illust.html)が配布されているので、「02_LoRA学習用データ_B氏提供版_背景透過」から kiritan をダウンロード。

2. 素材の加工  
SDXL の学習に 1024 x 1024 のサイズが良いらしいので素材ファイルの画像を加工します。
`python resizer.py kiritan` でそれぞれのファイルサイズを 1024 x 1024 に修正します。

3. ベースモデルのアップロード  
LoRA 作成の元となるベースモデルを Modal にアップロードしておきます。
```
modal volume create models
modal volume put models <モデルファイル名> /
```

ここまでで基本となる準備は完了です。
4と5を満足するまで繰り返します。

4. 設定ファイルを素材フォルダに入れる  
`config.toml` と `dataset.toml` を編集し、素材フォルダ（kiritan）にコピーします。  
基本的には `pretrained_model_name_or_path = '/model/<モデルファイル名>'` のファイル名を修正します。  
他のパラメータは各種サイトを参考に設定して下さい。  
例：https://hoshikat.hatenablog.com/entry/2023/05/26/223229  
作成結果を見ながら調整してみてください。

5. トレーニングを開始する  
`modal run generate_lora.py --name kiritan` と実行すると素材ファイルのアップロードと LoRA のトレーニングが開始されます。  
状況はコンソールと、Modal のサイトから確認できます。  
問題がありそうだったら Modal の Logs から Stop Now で止めてください。
トレーニングは A10G で 30 分ぐらいかかります。  
完了すると kiritan.safetensors がローカルに作成されます。

## 後始末
結果に満足したら Modal のストレージを消しておきましょう。
```
modal volume delete inputs
modal volume delete outputs
modal volume delete models
```

## Tips
- 他のデータセットで試したい場合、オプションの `--name` の名前と素材のディレクトリ名をあわせてください
- GPU を変更したい場合は `generate_lora.py` の `GPU = "L40S"` のところを変更してください
- 設定ファイルの書き方は [sd-script](https://github.com/kohya-ss/sd-scripts) 等を参照ください

## Modal で使用できる GPU 一覧

`generate_lora.py` の `GPU = "..."` に指定できる GPU の一覧です。  
料金は [Modal Pricing](https://modal.com/pricing) より（2026年3月時点、税抜）。

| GPU 文字列 | アーキテクチャ | VRAM | 料金（目安） | 概要 |
|---|---|---|---|---|
| `"T4"` | Turing | 16 GB | 約 $0.59 / hr | 低コスト。小規模な実験向き。A10G 比でML学習性能は約 1/3。 |
| `"L4"` | Ada Lovelace | 24 GB | 約 $0.80 / hr | コスパ重視のミドルクラス。RTX（レイトレーシング）サポート。 |
| `"A10G"` | Ampere | 24 GB | 約 $1.10 / hr | T4 比で ML 学習性能最大 3.3 倍、ML 推論性能最大 3 倍。LoRA 学習に十分なコスパ。 |
| `"L40S"` | Ada Lovelace | 48 GB | 約 $1.95 / hr | A10G の約 2 倍の VRAM。FP8 精度をサポート。大きい `batch_size` や高 `network_dim` を狙う場合に有効。**本スクリプトのデフォルト。** |
| `"A100-40GB"` | Ampere | 40 GB | 約 $2.10 / hr | Ampere 世代のフラッグシップ（40GB版）。 |
| `"A100-80GB"` | Ampere | 80 GB | 約 $2.50 / hr | Ampere 世代のフラッグシップ（80GB版）。複数 GPU 使用も可能。 |
| `"RTX_PRO_6000"` | Ada Lovelace | 48 GB | 約 $3.03 / hr | プロフェッショナルワークステーション GPU。FP8 / RTX サポート。 |
| `"H100"` | Hopper | 80 GB | 約 $3.95 / hr | Hopper 世代のフラッグシップ。FP8 サポートと Transformer Engine を搭載。GPT-3 規模モデルで前世代比最大 4 倍高速。 |
| `"H200"` | Hopper | 141 GB | 約 $4.54 / hr | H100 の大容量 HBM3e 版。超大規模モデルや複数 GPU 構成向け。 |
| `"B200"` | Blackwell | 192 GB | 約 $6.25 / hr | 最新 Blackwell アーキテクチャ。最高性能・最高価格。 |
| `"any"` | — | — | — | 空き状況に応じて自動で最適な GPU を選択。 |

### 複数 GPU を使う場合

文字列で `"T4:4"` のように `:<台数>` を付けると複数台を同時に使用できます。

```python
GPU = "A100-80GB:2"  # A100 80GB × 2 台
```

> **LoRA 学習向けの選択指針**  
> - コストを抑えたい → `"L4"` または `"A10G"`  
> - バランス重視（デフォルト） → `"L40S"`  
> - 大きな `batch_size` や解像度 1024 で余裕を持ちたい → `"A100-80GB"` または `"H100"`
