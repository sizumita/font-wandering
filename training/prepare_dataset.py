from __future__ import annotations

import argparse
import json
from pathlib import Path

from fontTools.ttLib import TTCollection, TTFont
from PIL import Image, ImageDraw, ImageFont

IMAGE_SIZE = 224
PADDING = 16
FONT_EXTENSIONS = {".ttf", ".otf", ".ttc", ".otc"}


def default_font_dirs() -> list[Path]:
    dirs = [Path("/System/Library/Fonts"), Path("/Library/Fonts")]
    home = Path.home()
    dirs.append(home / "Library" / "Fonts")
    return [path for path in dirs if path.exists()]


def iter_font_files(font_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in font_dirs:
        for path in directory.rglob("*"):
            if path.suffix.lower() in FONT_EXTENSIONS:
                files.append(path)
    return sorted(files)


def font_faces(path: Path) -> list[tuple[int, TTFont]]:
    if path.suffix.lower() in {".ttc", ".otc"}:
        collection = TTCollection(str(path))
        return [(index, font) for index, font in enumerate(collection.fonts)]
    return [(0, TTFont(str(path), fontNumber=0))]


def style_kind(font: TTFont) -> str:
    os2 = font.get("OS/2")
    post = font.get("post")
    selection = getattr(os2, "fsSelection", 0)
    post_angle = getattr(post, "italicAngle", 0)
    if selection & 0b1:
        return "italic"
    if post_angle and post_angle != 0:
        return "oblique"
    return "normal"


def weight_value(font: TTFont) -> int:
    os2 = font.get("OS/2")
    return int(getattr(os2, "usWeightClass", 400))


def stretch_value(font: TTFont) -> int:
    os2 = font.get("OS/2")
    return int(getattr(os2, "usWidthClass", 5))


def supports_text(font: TTFont, text: str) -> bool:
    cmap = font.getBestCmap() or {}
    for ch in text:
        if ch.isspace():
            continue
        if ord(ch) not in cmap:
            return False
    return True


def family_name(font: TTFont) -> str:
    for record in font["name"].names:
        if record.nameID in {16, 1}:
            try:
                return record.toUnicode()
            except UnicodeDecodeError:
                continue
    return "Unknown"


def postscript_name(font: TTFont) -> str:
    for record in font["name"].names:
        if record.nameID == 6:
            try:
                return record.toUnicode()
            except UnicodeDecodeError:
                continue
    return "Unknown"


def render_text(path: Path, face_index: int, text: str) -> Image.Image | None:
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    draw = ImageDraw.Draw(image)
    low, high = 10, 196
    best: tuple[ImageFont.FreeTypeFont, tuple[int, int, int, int]] | None = None

    while low <= high:
        size = low + (high - low) // 2
        font = ImageFont.truetype(str(path), size=size, index=face_index)
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= IMAGE_SIZE - PADDING * 2 and height <= IMAGE_SIZE - PADDING * 2:
            best = (font, bbox)
            low = size + 1
        else:
            high = size - 1

    if best is None:
        return None

    font, bbox = best
    x = (IMAGE_SIZE - (bbox[2] - bbox[0])) / 2 - bbox[0]
    y = (IMAGE_SIZE - (bbox[3] - bbox[1])) / 2 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    return image


def build_record(path: Path, face_index: int, font: TTFont, image_path: Path, text: str) -> dict[str, object] | None:
    if not supports_text(font, text):
        return None

    rendered = render_text(path, face_index, text)
    if rendered is None:
        return None

    image_path.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(image_path)

    return {
        "image_path": str(image_path),
        "font_path": str(path),
        "face_index": face_index,
        "family": family_name(font),
        "post_script_name": postscript_name(font),
        "weight_value": weight_value(font),
        "stretch_value": stretch_value(font),
        "style_kind": style_kind(font),
        "text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-faces", type=int, default=0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    manifest_path = args.output / "manifest.jsonl"

    records = []
    count = 0
    for font_path in iter_font_files(default_font_dirs()):
        for face_index, font in font_faces(font_path):
            image_path = images_dir / f"{font_path.stem}-{face_index}.png"
            record = build_record(font_path, face_index, font, image_path, args.text)
            if record is None:
                continue
            records.append(record)
            count += 1
            if args.max_faces and count >= args.max_faces:
                break
        if args.max_faces and count >= args.max_faces:
            break

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} samples to {manifest_path}")


if __name__ == "__main__":
    main()
