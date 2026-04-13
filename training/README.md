# Training Pipeline

`font-wandering` の runtime は `assets/models/glyph_encoder_bootstrap_f37072fd.pth` と
`assets/models/glyph_feature_stats_bootstrap.json` を読み込みます。

この `training/` ディレクトリは、その 2 つを glyph 向けに更新するための offline weak supervision パイプラインです。

## 使い方

1. 仮想環境を作成して依存を入れる
   - `python3 -m venv training/.venv`
   - `source training/.venv/bin/activate`
   - `pip install -r training/requirements.txt`
2. ローカルフォントから dataset を生成する
   - `python training/prepare_dataset.py --text あ --output training/data/ja-weight`
3. encoder を学習する
   - `python training/train_glyph_encoder.py --manifest training/data/ja-weight/manifest.jsonl --output training/runs/ja-weight.pt`
4. runtime 用 bundle を書き出す
   - `python training/export_runtime_bundle.py --checkpoint training/runs/ja-weight.pt`
5. 評価する
   - `python training/evaluate_model.py --checkpoint training/runs/ja-weight.pt --manifest training/data/ja-weight/manifest.jsonl`

## ラベル

- `weight_value`: OS/2 `usWeightClass`
- `stretch_value`: OS/2 `usWidthClass`
- `style_kind`: `normal / italic / oblique`

## 学習方針

- backbone: `ResNet18` (`fc=Identity`)
- losses:
  - `weight regression`
  - `stretch classification`
  - `style classification`
  - `supervised contrastive`
- runtime では backbone feature と morphology feature を連結して HDBSCAN へ渡します。
