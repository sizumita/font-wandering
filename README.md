# font-wandering

`font-wandering` は、入力した文字列を macOS 上のシステムフォントごとに固定サイズ glyph 画像へ描画し、特徴抽出・クラスタリングした結果を `GPUI` 上で 2D map として眺めるためのデスクトップアプリです。

現在の runtime は次の流れで動きます。

1. システムフォントを走査し、全文字列を単独 face で描けるフォントだけを採用
2. `224x224` grayscale へ glyph をレンダリング
3. `glyph encoder + morphology` で hybrid feature を生成
4. 高次元 feature に対して `HDBSCAN` を実行
5. 同じ高次元 feature を `UMAP` で 2D に圧縮して map 表示

`UMAP` は表示専用で、クラスタラベル自体は高次元 feature から決まります。

## 動作環境

- macOS
- Apple Silicon 推奨
- Rust toolchain
- Xcode Command Line Tools
- Metal backend を使う場合は Metal Toolchain

`xcrun metal` が通らない場合は、次を実行してください。

```bash
xcodebuild -downloadComponent MetalToolchain
```

## 起動

```bash
cargo run
```

起動後は文字列を入力して `Analyze` を押します。

UI でできること:

- 2D map の pan / zoom
- glyph thumbnail の hover / click
- cluster / weight / stretch / style の詳細確認
- 除外されたフォントの理由表示

## テスト

```bash
cargo check
cargo test
```

## モデル asset

runtime は次の asset を読みます。

- `assets/models/glyph_encoder_bootstrap_f37072fd.pth`
- `assets/models/glyph_feature_stats_bootstrap.json`

現状の `glyph_encoder_bootstrap_f37072fd.pth` は bootstrap 用の重み参照で、学習済み checkpoint を export すると差し替えられます。

## 学習パイプライン

`training/` には glyph encoder を offline 学習して runtime bundle を更新するためのスクリプトがあります。

`uv` を使う場合の最短手順:

```bash
uv venv training/.venv --python 3.12
uv pip install --python training/.venv/bin/python -r training/requirements.txt
uv run --python training/.venv/bin/python training/prepare_dataset.py --text 'Aaあ@' --output training/data/aa_ja_at
uv run --python training/.venv/bin/python training/train_glyph_encoder.py --manifest training/data/aa_ja_at/manifest.jsonl --output training/runs/aa_ja_at.pt --init-backbone assets/models/resnet18-f37072fd.pth
uv run --python training/.venv/bin/python training/export_runtime_bundle.py --checkpoint training/runs/aa_ja_at.pt --manifest training/data/aa_ja_at/manifest.jsonl
uv run --python training/.venv/bin/python training/evaluate_model.py --checkpoint training/runs/aa_ja_at.pt --manifest training/data/aa_ja_at/manifest.jsonl
```

MPS/Metal が利用できる環境では、training 側は `mps -> cuda -> cpu` の順で device を選びます。

詳細は [training/README.md](training/README.md) を参照してください。

## 実装メモ

- UI: `src/ui.rs`
- フォント走査・レンダリング: `src/font_pipeline.rs`
- morphology 特徴: `src/morphology.rs`
- hybrid feature / UMAP / HDBSCAN: `src/ml_pipeline.rs`
- データ契約: `src/models.rs`

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照してください。
