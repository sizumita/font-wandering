from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


def coarse_weight_bucket(weight_value: int) -> int:
    return min(max((weight_value - 100) // 100, 0), 8)


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def style_index(style_kind: str) -> int:
    return {"normal": 0, "italic": 1, "oblique": 2}[style_kind]


class GlyphDataset(Dataset):
    def __init__(self, manifest_path: Path) -> None:
        self.records = load_manifest(manifest_path)
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.stretch_values = sorted({int(record["stretch_value"]) for record in self.records})
        self.stretch_to_index = {value: index for index, value in enumerate(self.stretch_values)}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        image = Image.open(record["image_path"]).convert("L")
        tensor = self.transform(image)
        weight_value = int(record["weight_value"])
        stretch_value = int(record["stretch_value"])
        style_kind = str(record["style_kind"])
        return {
            "image": tensor,
            "weight_target": torch.tensor((weight_value - 100) / 800.0, dtype=torch.float32),
            "weight_bucket": torch.tensor(coarse_weight_bucket(weight_value), dtype=torch.long),
            "stretch_target": torch.tensor(self.stretch_to_index[stretch_value], dtype=torch.long),
            "style_target": torch.tensor(style_index(style_kind), dtype=torch.long),
        }


class GlyphEncoderTrainer(nn.Module):
    def __init__(self, stretch_classes: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.weight_head = nn.Linear(512, 1)
        self.stretch_head = nn.Linear(512, stretch_classes)
        self.style_head = nn.Linear(512, 3)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.backbone(image)
        embedding = F.normalize(embedding, dim=-1)
        weight_pred = self.weight_head(embedding).squeeze(-1)
        stretch_logits = self.stretch_head(embedding)
        style_logits = self.style_head(embedding)
        return embedding, weight_pred, stretch_logits, style_logits


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_backbone_weights(model: GlyphEncoderTrainer, init_backbone: Path | None) -> None:
    if init_backbone is None:
        return

    state_dict = torch.load(init_backbone, map_location="cpu")
    missing, unexpected = model.backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"missing backbone keys: {missing}")
    if unexpected:
        print(f"unexpected backbone keys: {unexpected}")


def supervised_contrastive_loss(embedding: torch.Tensor, weight_bucket: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    logits = embedding @ embedding.T / 0.12
    logits = logits - torch.max(logits, dim=1, keepdim=True).values.detach()
    eye = torch.eye(logits.size(0), device=logits.device, dtype=torch.bool)
    positives = (weight_bucket[:, None] == weight_bucket[None, :]) & (style[:, None] == style[None, :]) & ~eye

    exp_logits = torch.exp(logits) * (~eye)
    denom = exp_logits.sum(dim=1).clamp_min(1e-6)
    positive_mass = (exp_logits * positives).sum(dim=1)
    valid = positives.any(dim=1)
    if not valid.any():
        return logits.new_zeros(())
    return (-torch.log((positive_mass[valid] / denom[valid]).clamp_min(1e-6))).mean()


def train(args: argparse.Namespace) -> None:
    dataset = GlyphDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = GlyphEncoderTrainer(stretch_classes=len(dataset.stretch_values))
    load_backbone_weights(model, args.init_backbone)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    device = resolve_device()
    print(f"training device={device}")
    model.to(device)

    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            image = batch["image"].to(device)
            weight_target = batch["weight_target"].to(device)
            weight_bucket = batch["weight_bucket"].to(device)
            stretch_target = batch["stretch_target"].to(device)
            style_target = batch["style_target"].to(device)

            embedding, weight_pred, stretch_logits, style_logits = model(image)
            loss = (
                F.mse_loss(weight_pred, weight_target)
                + F.cross_entropy(stretch_logits, stretch_target)
                + F.cross_entropy(style_logits, style_target)
                + 0.25 * supervised_contrastive_loss(embedding, weight_bucket, style_target)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"epoch={epoch + 1} loss={loss.item():.4f}")

    checkpoint = {
        "backbone": model.backbone.state_dict(),
        "weight_head": model.weight_head.state_dict(),
        "stretch_head": model.stretch_head.state_dict(),
        "style_head": model.style_head.state_dict(),
        "stretch_values": dataset.stretch_values,
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(f"saved checkpoint to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-backbone", type=Path)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
