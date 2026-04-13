from __future__ import annotations

import argparse
import json
from pathlib import Path

import hdbscan
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import normalized_mutual_info_score
from torch import nn
from torchvision import models, transforms


def coarse_weight_bucket(weight_value: int) -> int:
    return min(max((weight_value - 100) // 100, 0), 8)


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def build_backbone() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Identity()
    return model


def encode_images(checkpoint_path: Path, manifest: list[dict[str, object]]) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = build_backbone()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["backbone"])
    model.eval()

    batches = []
    with torch.no_grad():
        for record in manifest:
            image = Image.open(record["image_path"]).convert("L")
            batches.append(transform(image))
        tensor = torch.stack(batches)
        features = model(tensor)
        features = torch.nn.functional.normalize(features, dim=-1)
    return features.cpu().numpy()


def cluster_purity(labels: np.ndarray, targets: np.ndarray) -> float:
    valid = labels >= 0
    if not np.any(valid):
        return 0.0
    score = 0
    for label in np.unique(labels[valid]):
        members = targets[labels == label]
        counts = np.bincount(members)
        score += counts.max()
    return float(score / valid.sum())


def topk_same_weight_rate(features: np.ndarray, weight_bucket: np.ndarray, k: int) -> float:
    similarity = features @ features.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argsort(-similarity, axis=1)[:, :k]
    hits = 0
    total = 0
    for row, row_neighbors in enumerate(neighbors):
        for neighbor in row_neighbors:
            hits += int(weight_bucket[row] == weight_bucket[neighbor])
            total += 1
    return float(hits / max(total, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=5)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    features = encode_images(args.checkpoint, manifest)
    weight_bucket = np.asarray([coarse_weight_bucket(int(record["weight_value"])) for record in manifest])
    style = np.asarray([str(record["style_kind"]) for record in manifest])

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    labels = clusterer.fit_predict(features)

    metrics = {
        "cluster_purity": cluster_purity(labels, weight_bucket),
        "weight_nmi": float(normalized_mutual_info_score(weight_bucket, labels)),
        "topk_same_weight_rate": topk_same_weight_rate(features, weight_bucket, args.neighbors),
        "style_nmi": float(normalized_mutual_info_score(style, labels)),
        "cluster_count": int(len({label for label in labels.tolist() if label >= 0})),
        "noise_points": int(np.sum(labels < 0)),
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
