from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BASE_FEATURE_COUNT = 10


def neighbors(x: int, y: int, width: int, height: int) -> list[tuple[int, int]]:
    return [
        (max(x - 1, 0), y),
        (min(x + 1, width - 1), y),
        (x, max(y - 1, 0)),
        (x, min(y + 1, height - 1)),
    ]


def connected_components(binary: np.ndarray, target: bool) -> int:
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    count = 0

    for y in range(height):
        for x in range(width):
            if visited[y, x] or bool(binary[y, x]) != target:
                continue
            count += 1
            queue = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                for nx, ny in neighbors(cx, cy, width, height):
                    if visited[ny, nx] or bool(binary[ny, nx]) != target:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
    return count


def hole_count(binary: np.ndarray, bounds: tuple[int, int, int, int]) -> int:
    min_x, min_y, max_x, max_y = bounds
    visited = np.zeros_like(binary, dtype=bool)
    holes = 0

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if visited[y, x] or binary[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            touches_border = False
            while queue:
                cx, cy = queue.popleft()
                if cx in {min_x, max_x} or cy in {min_y, max_y}:
                    touches_border = True
                for nx, ny in neighbors(cx, cy, binary.shape[1], binary.shape[0]):
                    if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                        continue
                    if visited[ny, nx] or binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            if not touches_border:
                holes += 1
    return holes


def cityblock_distance(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    inf = np.iinfo(np.int32).max // 4
    distances = np.where(binary, inf, 0).astype(np.int32)

    for y in range(height):
        for x in range(width):
            best = distances[y, x]
            if x > 0:
                best = min(best, distances[y, x - 1] + 1)
            if y > 0:
                best = min(best, distances[y - 1, x] + 1)
            distances[y, x] = best

    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            best = distances[y, x]
            if x + 1 < width:
                best = min(best, distances[y, x + 1] + 1)
            if y + 1 < height:
                best = min(best, distances[y + 1, x] + 1)
            distances[y, x] = best

    return distances


def projection_histogram(values: np.ndarray, bins: int, axis: int) -> list[float]:
    reduced = values.sum(axis=axis)
    chunks = np.array_split(reduced, bins)
    histogram = np.array([chunk.sum() for chunk in chunks], dtype=np.float32)
    total = float(histogram.sum())
    if total > 0:
        histogram /= total
    return histogram.tolist()


def extract_morphology_features(image_path: Path, projection_bins: int) -> list[float]:
    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
    binary = image >= 24 / 255.0
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return [0.0] * (BASE_FEATURE_COUNT + projection_bins * 2)

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    bbox = image[min_y : max_y + 1, min_x : max_x + 1]
    total_ink = float(image.sum())
    center_x = float((image * np.arange(image.shape[1], dtype=np.float32)[None, :]).sum() / max(total_ink, 1e-6) / image.shape[1])
    center_y = float((image * np.arange(image.shape[0], dtype=np.float32)[:, None]).sum() / max(total_ink, 1e-6) / image.shape[0])
    distances = cityblock_distance(binary)
    stroke_samples = distances[binary][(min_y <= ys) & (ys <= max_y)]
    normalizer = float(max(width, height, 1))
    if len(stroke_samples) == 0:
        stroke_mean = stroke_std = stroke_max = 0.0
    else:
        scaled = stroke_samples.astype(np.float32) / normalizer
        stroke_mean = float(scaled.mean())
        stroke_std = float(scaled.std())
        stroke_max = float(scaled.max())

    features = [
        float(image.mean()),
        float(width / max(height, 1)),
        float(bbox.mean()),
        center_x,
        center_y,
        float(min(connected_components(binary, True) / 12.0, 1.0)),
        float(min(hole_count(binary, (min_x, min_y, max_x, max_y)) / 8.0, 1.0)),
        stroke_mean,
        stroke_std,
        stroke_max,
    ]
    features.extend(projection_histogram(bbox, projection_bins, axis=1))
    features.extend(projection_histogram(bbox, projection_bins, axis=0))
    return features


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, default=Path("assets/models/glyph_encoder_bootstrap_f37072fd.pth"))
    parser.add_argument("--output-stats", type=Path, default=Path("assets/models/glyph_feature_stats_bootstrap.json"))
    parser.add_argument("--projection-bins", type=int, default=8)
    parser.add_argument("--learned-scale", type=float, default=0.45)
    parser.add_argument("--morphology-scale", type=float, default=2.6)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    records = load_manifest(args.manifest)
    morphology = np.asarray(
        [extract_morphology_features(Path(record["image_path"]), args.projection_bins) for record in records],
        dtype=np.float32,
    )
    means = morphology.mean(axis=0)
    stds = morphology.std(axis=0)
    stds = np.maximum(stds, 1e-3)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.output_stats.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint["backbone"], args.output_model)
    args.output_stats.write_text(
        json.dumps(
            {
                "learned_scale": args.learned_scale,
                "morphology_scale": args.morphology_scale,
                "projection_bins": args.projection_bins,
                "morphology_means": means.tolist(),
                "morphology_stds": stds.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output_model} and {args.output_stats}")


if __name__ == "__main__":
    main()
