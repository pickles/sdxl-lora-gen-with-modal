# SDXL LoRA 学習設定ガイド

このドキュメントは `config.toml` および `dataset.toml` の各プロパティの意味と推奨値を解説します。  
学習エンジンは [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) の `sdxl_train_network.py` を使用しています。

> **出典**: 本ドキュメントの説明・推奨値は主に以下の公式ドキュメントを参照しています。
> - [train_network_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md) — LoRA 学習固有のオプション
> - [train_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_README-ja.md) — 共通の学習オプション

---

## config.toml

### パス設定

| プロパティ | 説明 |
|---|---|
| `pretrained_model_name_or_path` | ベースモデルのパス。Stable Diffusion の `.ckpt` / `.safetensors`、Diffusers 形式のローカルディレクトリ、または HuggingFace のモデル ID（`"stabilityai/stable-diffusion-xl-base-1.0"` など）が指定できる。Modal コンテナ内では `/model/` 以下に配置する。 |
| `dataset_config` | `dataset.toml` のパス。Modal コンテナ内では `/input/dataset.toml` に固定。 |
| `output_dir` | 学習済み LoRA ファイルの出力先ディレクトリ。Modal コンテナ内では `/output` に固定。 |

---

### ハイパーパラメータ

#### 学習率スケジューラ

| プロパティ | 説明 |
|---|---|
| `lr_scheduler` | 学習率のスケジューリング方式。下表を参照。デフォルトは `"constant"`。 |
| `lr_scheduler_num_cycles` | `"cosine_with_restarts"` 使用時のリスタート回数。`"polynomial"` 使用時の polynomial power にも使われる（`lr_scheduler_power` で別途指定も可）。 |
| `lr_warmup_steps` | 学習率をゼロから目標値まで線形に上昇させるウォームアップのステップ数。急激な勾配更新による発散防止。`constant_with_warmup` や `cosine` と組み合わせて使う。AdaFactor で `relative_step=False` の場合は `constant_with_warmup` が推奨。 |

**`lr_scheduler` の選択肢**

| 値 | 挙動 |
|---|---|
| `"constant"` | 学習率を一定に保つ（デフォルト）。 |
| `"constant_with_warmup"` | ウォームアップ後に一定。安定した LoRA 学習で最も汎用的。 |
| `"linear"` | ウォームアップ後、線形に学習率を 0 まで下げる。 |
| `"cosine"` | ウォームアップ後、コサイン曲線で 0 まで下げる。 |
| `"cosine_with_restarts"` | コサイン減衰を `lr_scheduler_num_cycles` 回リスタート。 |
| `"polynomial"` | 多項式カーブで減衰。`lr_scheduler_power` で次数を指定。 |
| 任意のスケジューラ | `lr_scheduler_args` でオプション引数を渡すことも可能。 |

#### トークン・エポック設定

| プロパティ | 説明 |
|---|---|
| `max_token_length` | キャプションの最大トークン長。デフォルトは `75`。`150` または `225` を指定するとトークン長を拡張して学習できる。SDXL では `225` が上限で、長いキャプションを使う場合に指定する。ただし Web UI（Automatic1111）とは分割仕様が微妙に異なるため、必要がなければ `75` のままでも可。 |
| `max_train_epochs` | 学習エポック数（データセット全体を何周するか）。`max_train_steps` と同時に指定するとエポック数が優先される。過学習と学習不足のバランスで調整する。 |

#### オプティマイザ

| プロパティ | 説明 |
|---|---|
| `optimizer_type` | 使用するオプティマイザ。下表を参照。 |
| `optimizer_args` | オプティマイザへの追加引数をリスト形式で指定。`key=value` の形式で複数指定可能。value はカンマ区切りで複数の値が指定できる（例: `["betas=.9,.999", "weight_decay=0.01"]`）。各オプティマイザの仕様に合わせて指定する。 |
| `max_grad_norm` | 勾配クリッピングの閾値。勾配の爆発を防ぐ。`1.0` が標準。`0` で無効化。AdaFactor + `relative_step=False` 使用時はクリッピングしないことが推奨されているため `0.0` にする。 |
| `seed` | 乱数シード。再現性のある学習を行うために固定する。`3407` は論文 "Bag of Tricks" 由来の値。 |

**`optimizer_type` の選択肢**（出典: [train_README-ja.md オプティマイザ関係](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_README-ja.md)）

| 値 | 特徴 |
|---|---|
| `"AdamW"` | 標準的な Adam + weight decay。デフォルト。 |
| `"AdamW8bit"` | 8bit 量子化 AdamW（bitsandbytes）。VRAM 使用量を削減。 |
| `"PagedAdamW8bit"` | Paged メモリを使う 8bit AdamW。VRAM がさらに限られる場合に有効。 |
| `"Lion"` | Google DeepMind 開発。Adam より少ないメモリで同等性能とされる。学習率は Adam の 1/3〜1/10 程度に下げる必要あり。 |
| `"Lion8bit"` | 8bit 量子化 Lion。 |
| `"SGDNesterov"` | SGD + Nesterov momentum。`momentum` が必須引数として自動追加される。 |
| `"DAdaptation"` / `"DAdaptAdam"` | 学習率を自動調整（Facebook Research）。`learning_rate=1.0` を指定し、Text Encoder のみ学習率を変えたい場合は `--text_encoder_lr=0.5 --unet_lr=1.0` のように比率で指定する。 |
| `"Prodigy"` | 学習率自動調整オプティマイザ。DAdaptation の後継的位置付け。 |
| `"AdaFactor"` | メモリ効率が高い。`relative_step=True`（デフォルト）で学習率自動調整が可能。自動調整しない場合は `["relative_step=False", "scale_parameter=False", "warmup_init=False"]` を指定し、`lr_scheduler="constant_with_warmup"` かつ `max_grad_norm=0.0` にすることが推奨。 |

---

### メモリ・速度最適化

| プロパティ | 説明 |
|---|---|
| `cache_latents` | 事前に画像を VAE でエンコードしてメモリキャッシュに保存。毎ステップの VAE エンコード処理を省略でき学習が高速化される。`color_aug` を使う場合は指定不可。`true` 推奨。 |
| `cache_latents_to_disk` | Latent キャッシュをメインメモリではなくディスクに保存。スクリプトを再起動してもキャッシュが有効なままになる。大規模データセットや長期学習に有効。`true` 推奨。 |
| `gradient_checkpointing` | 中間活性化を保持せず都度再計算することで VRAM 使用量を削減するテクニック。一般的には速度が低下するが、バッチサイズを大きくできるためトータルの学習時間はむしろ短縮できる場合もある。オンオフは精度に直接影響しない。 |
| `lowram` | RAM が少ない環境向けの最適化フラグ。モデルの一部をできるだけ GPU 側に保持することでメインメモリ使用を抑える。通常は `false` でよい。 |
| `mixed_precision` | 混合精度学習の型。`"fp16"` または `"bf16"`。`accelerate config` で設定した値と一致させる必要がある。SDXL では数値安定性の高い `"bf16"` を推奨（RTX 30 シリーズ以降で利用可能）。`"no"` で fp32（最も安定だが遅くメモリも多い）。 |
| `xformers` | xFormers ライブラリによる効率的な CrossAttention 計算を有効化。VRAM 削減と高速化に有効。xFormers をインストール済みの場合は `true` 推奨。インストールされていないか `mixed_precision="no"` の場合はエラーになる場合があり、その際は `mem_eff_attn` を代わりに使う（`xformers` より低速）。 |
| `min_snr_gamma` | Min-SNR Weighting Strategy（[arxiv:2303.09556](https://arxiv.org/abs/2303.09556)）の係数。ノイズレベル（SNR）に応じた損失の重みを均一化することで学習安定性と収束速度を改善する。**公式ドキュメントおよび論文では `5` が推奨値**。コメントアウトで無効。出典: [sd-scripts PR#308](https://github.com/kohya-ss/sd-scripts/pull/308) |
| `noise_offset` | 拡散モデルのノイズにオフセットを加えることで、純黒・純白など極端な明暗の画像生成能力を改善するテクニック。`0.1` 程度の値が推奨。LoRA 学習でも有効とされる。出典: [Diffusion with Offset Noise](https://www.crosslabs.org/blog/diffusion-with-offset-noise) |

---

### チェックポイント保存

| プロパティ | 説明 |
|---|---|
| `save_every_n_epochs` | 何エポックごとに中間モデルを保存するか。`1` で毎エポック保存。`save_every_n_steps` と同時に使用可能。 |
| `save_every_n_steps` | 何ステップごとに中間モデルを保存するか。エポック単位ではなくステップ単位で細かく管理したい場合に使用。 |
| `save_last_n_epochs` | 直近 N エポック分のみ保存し、それより古いファイルを自動削除。ディスク容量節約に有効。`5` なら最新 5 エポック分を保持。 |
| `save_last_n_steps` | 直近 N ステップ分のチェックポイントのみ保存する（ステップ単位管理時）。 |

---

### LoRA ネットワーク設定

#### 学習率

| プロパティ | 説明 |
|---|---|
| `learning_rate` | LoRA 全体の学習率。`unet_lr` / `text_encoder_lr` を個別指定しない場合のデフォルト値。**通常の DreamBooth や fine tuning より高めの `1e-4`〜`1e-3` 程度を指定するとよい**（出典: [train_network_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md)）。 |
| `unet_lr` | U-Net 部分の LoRA モジュールのみ異なる学習率を使う場合に指定。`learning_rate` を上書きする。 |
| `text_encoder_lr` | Text Encoder 部分の LoRA モジュールのみ異なる学習率を使う場合に指定。**Text Encoder の学習率は U-Net より若干低め（`5e-5` など）にしたほうが良いという意見もある**（出典: [train_network_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md)）。 |

#### 学習対象の制御

| プロパティ | 説明 |
|---|---|
| `network_train_unet_only` | `true` にすると Text Encoder を学習せず U-Net のみを学習する。fine tuning 的な学習に有効。**キャプション方式での LoRA 学習では一般的に `true` が使われる**。`cache_text_encoder_outputs = true` との組み合わせで大幅に高速化できる。 |
| `network_train_text_encoder_only` | `true` にすると Text Encoder のみを学習。Textual Inversion 的な効果が期待できる。上記と同時には指定しない。 |

#### ネットワーク構造

| プロパティ | 説明 |
|---|---|
| `network_module` | LoRA の実装モジュール。標準は `"networks.lora"` (LoRA-LierLa)。DyLoRA を使う場合は `"networks.dylora"`。LyCORIS 等のサードパーティ実装も指定可能。 |
| `network_dim` | LoRA のランク (rank)。値が大きいほど表現力は増すが、学習に必要なメモリと時間が増加し、過学習リスクも高まる。**闇雲に増やしても良くはない**（出典: [train_network_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md)）。省略時は `4`。 |
| `network_alpha` | アンダーフローを防ぎ安定して学習するための alpha 値。実効スケールは `network_alpha / network_dim` で決まる。`network_dim` と同じ値を指定すると旧バージョンと同じ動作になる。デフォルトは `1`。 |
| `network_args` | LoRA ネットワークへの追加引数。用途別に下記を参照。 |
| `network_weights` | 学習を継続（追加学習）する場合に既存 LoRA の重みを読み込む。新規学習の場合はコメントアウトしておく。 |

#### `network_args` の詳細オプション

**① LoRA-C3Lier（Conv2d 3x3 への拡張）**

通常の LoRA-LierLa は Linear 層と 1x1 Conv2d にのみ適用されるが、`conv_dim` を指定することで 3x3 Conv2d にも拡張できる（LoRA-C3Lier）。適用層が増えるため精度が向上する可能性がある。

```toml
network_args = ["conv_dim=4", "conv_alpha=1"]
```

| 引数 | 説明 |
|---|---|
| `conv_dim` | Conv2d (3x3) に適用する LoRA の rank。 |
| `conv_alpha` | Conv2d (3x3) LoRA の alpha 値。省略時は `1`。 |

**② 階層別学習率（Block-level LR）**（出典: [sd-scripts PR#355](https://github.com/kohya-ss/sd-scripts/pull/355)）

U-Net の各ブロックごとに学習率の重みを設定できる。SDXL では `down` 9 個、`mid` 3 個、`up` 9 個の値を指定する。

```toml
# 例: down を抑えめ、mid を強め、up を中程度にする場合
network_args = [
  "down_lr_weight=0.5,0.5,0.5,0.5,0.5,0.5,1.0,1.0,1.0",
  "mid_lr_weight=2.0,2.0,2.0",
  "up_lr_weight=1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.5,0.5"
]
```

| 引数 | 説明 |
|---|---|
| `down_lr_weight` | U-Net down blocks の学習率重み（SDXL: 9 個）。プリセット `sine`, `cosine`, `linear`, `reverse_linear`, `zeros` も使用可能。`+数値` で加算できる（例: `cosine+.25`）。 |
| `mid_lr_weight` | U-Net mid block の学習率重み（SDXL: 3 個）。 |
| `up_lr_weight` | U-Net up blocks の学習率重み（SDXL: 9 個）。 |
| `block_lr_zero_threshold` | この値以下の重みが指定されたブロックの LoRA モジュールを省略する。デフォルトは `0`。 |

**③ 階層別 rank（Block-level dim）**

ブロックごとに異なる rank を指定できる。SDXL では 23 個の値を指定する。

```toml
network_args = [
  "block_dims=2,4,4,4,8,8,8,8,12,12,12,16,8,8,8,8,8,8,8,4,4,4,2",
  "block_alphas=2,2,2,2,4,4,4,4,6,6,6,8,4,4,4,4,4,4,4,2,2,2,2"
]
```

| 引数 | 説明 |
|---|---|
| `block_dims` | 各ブロックの rank（SDXL: 23 個）。 |
| `block_alphas` | 各ブロックの alpha（SDXL: 23 個）。省略時は `network_alpha` の値が使われる。 |
| `conv_block_dims` | Conv2d 3x3 拡張時の各ブロックの rank。 |
| `conv_block_alphas` | Conv2d 3x3 拡張時の各ブロックの alpha。省略時は `conv_alpha` が使われる。 |

**④ DyLoRA（Dynamic LoRA）**（出典: [arxiv:2210.07558](https://arxiv.org/abs/2210.07558)）

`network_module = "networks.dylora"` と組み合わせて使用。指定した dim 以下のさまざまな rank を同時に学習することで、最適な rank の探索コストを削減できる。

```toml
network_module = "networks.dylora"
network_dim = 16
network_args = ["unit=4"]  # unit は network_dim を割り切れる値
```

#### その他

| プロパティ | 説明 |
|---|---|
| `persistent_data_loader_workers` | DataLoader のワーカープロセスを永続化し、エポック間の待ち時間を短縮する。**Windows 環境で指定すると特に効果が大きい**（出典: [train_network_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md)）。`true` 推奨。 |
| `max_data_loader_n_workers` | データ読み込みワーカーのプロセス数。デフォルトは `min(8, CPU スレッド数-1)`。GPU 使用率が 90% 以上ならメモリと相談しながら `1`〜`2` まで下げると良い。 |
| `prior_loss_weight` | DreamBooth 学習時の事前保全損失（正則化画像の損失）の重み。通常の caption 方式 LoRA 学習では `1.0` で問題ない。 |
| `alpha_mask` | 画像のアルファチャンネルをマスクとして学習に使用する。透過 PNG などを学習する場合に有効。出典: [sd-scripts PR#1223](https://github.com/kohya-ss/sd-scripts/pull/1223) |

---

### SDXL 固有設定

| プロパティ | 説明 |
|---|---|
| `cache_text_encoder_outputs` | Text Encoder の出力を事前キャッシュして学習を高速化する。**`network_train_unet_only = true` の場合のみ有効**。U-Net のみ学習する場合は `true` を推奨。 |

---

## dataset.toml

### [general] セクション

| プロパティ | 説明 |
|---|---|
| `caption_extension` | キャプションファイルの拡張子。画像ファイルと同名の指定拡張子のファイルをキャプションとして読み込む。`.txt` の場合は `'.txt'` と指定する。`.caption` との使い分けも可。 |
| `enable_bucket` | アスペクト比バケット学習（Aspect Ratio Bucketing）を有効化。異なる縦横比の画像を 64 ピクセル単位で最適なバケット解像度に分類・処理する。画像のトリミング量が減り、キャプションと画像の対応関係をより正確に学習できる。`true` 推奨。 |
| `shuffle_caption` | カンマ区切りのキャプションタグをランダムな順序でシャッフルして学習する。特定タグの位置依存性をなくし過学習防止と汎化性能向上に有効。`true` 推奨。 |

### [[datasets]] セクション

| プロパティ | 説明 |
|---|---|
| `resolution` | 学習画像の解像度（ピクセル）。整数 1 つで正方形（例: `1024` → 1024×1024）、`[幅, 高さ]` で非正方形（例: `[1024, 768]`）を指定できる。SDXL は `1024` が標準。`enable_bucket = true` の場合はこの解像度と同じ面積を上限としてバケットが作成される。 |
| `batch_size` | 1 ステップで処理する画像枚数。バッチサイズを増やすと精度が向上しやすいが VRAM を消費する。`バッチサイズ × ステップ数` が実際に使用するデータ量であるため、バッチサイズを増やした分ステップ数を減らすと良い。その場合は学習率もやや大きめにすること。 |

### [[datasets.subsets]] セクション

| プロパティ | 説明 |
|---|---|
| `image_dir` | 学習画像とキャプション `.txt` ファイルが格納されたディレクトリ。Modal コンテナ内では `/input` 以下に自動アップロードされる。 |
| `num_repeats` | このサブセットの画像を 1 エポック内で何回繰り返すか。画像枚数が少ない場合に増やすことで学習データを水増しできる。繰り返し後のデータが 1 周すると 1 epoch となる。 |
| `class_tokens` | DreamBooth class+identifier 方式で使う `"identifier class"` のトークン文字列（例: `"shs 1girl"`）。キャプション方式（`.txt` ファイル）を使う場合は不要。 |
| `is_reg` | `true` にするとこのサブセットが正則化画像（Prior Preservation 用）として扱われる。通常の学習データとは別に DreamBooth 学習で使う。 |

---

## 設定のポイントと推奨事項

### 品質重視の設定例

```toml
# より高品質だが学習に時間がかかる設定
network_dim = 32
network_alpha = 16
learning_rate = 5e-5
max_train_epochs = 20
min_snr_gamma = 5          # 論文・公式ドキュメントともに 5 を推奨
cache_text_encoder_outputs = true  # network_train_unet_only = true の場合のみ有効
noise_offset = 0.1         # 暗い・明るい画像の生成品質を改善
```

### 軽量・高速な設定例

```toml
# ファイルサイズを抑えて高速に学習する設定
network_dim = 8
network_alpha = 1
learning_rate = 1e-4
max_train_epochs = 10
network_train_unet_only = true
cache_text_encoder_outputs = true
```

### VRAM 節約の設定例

```toml
# 限られた VRAM で動かすための設定
gradient_checkpointing = true
mixed_precision = "bf16"
xformers = true
cache_latents = true
cache_latents_to_disk = true
optimizer_type = "AdamW8bit"   # 8bit Adam に切り替え
batch_size = 1                  # dataset.toml 側で設定
```

### `network_dim` と `network_alpha` の関係

実効スケール = `network_alpha / network_dim`

| dim | alpha | 実効スケール | 用途 |
|-----|-------|------------|------|
| 8 | 1 | 0.125 | 小さめ・安定重視 |
| 8 | 8 | 1.0 | 旧バージョンと同等の挙動 |
| 16 | 8 | 0.5 | バランス型（軽量） |
| 32 | 16 | 0.5 | バランス型（標準） |
| 64 | 32 | 0.5 | 高表現力 |
| 128 | 64 | 0.5 | 最高表現力（過学習に注意） |

> **推奨**: 実効スケールを `0.5` 前後に揃えると、異なる dim 間での挙動比較がしやすい。論文（DyLoRA）によると rank は高ければよいわけではなく、データセット・タスクに応じた最適値を探す必要がある。

---

## Modal での実行方法

```bash
# name は学習対象キャラクター・スタイル名（フォルダ名と一致させる）
modal run generate_lora.py --name <name>
```

- ローカルの `<name>/` ディレクトリ内の画像・キャプションが `/input/` に自動アップロードされます。
- `config.toml` と `dataset.toml` も同ディレクトリに配置してください。
- 出力された `<name>.safetensors` がローカルにダウンロードされます。
