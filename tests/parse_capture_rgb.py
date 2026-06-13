import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mask_detection import FACE_LABELS, FACE_NECK_LABELS, get_parser
from core.face_test import detect_face_bbox


def save_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def crop_face_neck_roi(rgb, bbox, x_scale=1.5, y_scale=2.2, y_shift=0.35):
    h, w = rgb.shape[:2]
    x, y, bw, bh = [float(v) for v in bbox]
    cx = x + bw / 2.0
    cy = y + bh / 2.0 + y_shift * bh
    crop_w = max(bw * x_scale, bh * x_scale)
    crop_h = bh * y_scale

    x1 = max(0, int(round(cx - crop_w / 2.0)))
    y1 = max(0, int(round(cy - crop_h / 2.0)))
    x2 = min(w, int(round(cx + crop_w / 2.0)))
    y2 = min(h, int(round(cy + crop_h / 2.0)))
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("Invalid crop ROI.")
    return rgb[y1:y2, x1:x2], (x1, y1, x2, y2)


def parse_full_image(rgb):
    return get_parser().parse(rgb)


def parse_crop_mapped(rgb):
    bbox = detect_face_bbox(rgb)
    if bbox is None:
        raise RuntimeError("No YuNet face bbox detected for crop parsing.")

    crop, (x1, y1, x2, y2) = crop_face_neck_roi(rgb, bbox)
    crop_parsing = get_parser().parse(crop)
    parsing = np.zeros(rgb.shape[:2], dtype=crop_parsing.dtype)
    parsing[y1:y2, x1:x2] = crop_parsing
    return parsing, bbox, (x1, y1, x2, y2)


def save_parsing_outputs(out_dir, rgb, parsing, prefix, bbox=None, crop_xyxy=None):
    face_mask = np.isin(parsing, FACE_LABELS)
    face_neck_mask = np.isin(parsing, FACE_NECK_LABELS)
    neck_mask = face_neck_mask & ~face_mask

    color = np.zeros_like(rgb)
    color[face_mask] = (0, 255, 0)
    color[neck_mask] = (255, 180, 0)
    overlay = cv2.addWeighted(rgb, 0.65, color, 0.35, 0.0)
    overlay[~face_neck_mask] = rgb[~face_neck_mask]

    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if crop_xyxy is not None:
        x1, y1, x2, y2 = crop_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    cv2.imwrite(str(out_dir / f"{prefix}_overlay.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    save_mask(out_dir / f"{prefix}_face_mask.png", face_mask)
    save_mask(out_dir / f"{prefix}_neck_mask.png", neck_mask)
    save_mask(out_dir / f"{prefix}_face_neck_mask.png", face_neck_mask)
    cv2.imwrite(str(out_dir / f"{prefix}_labels.png"), parsing.astype(np.uint8))

    labels, counts = np.unique(parsing, return_counts=True)
    with (out_dir / f"{prefix}_label_counts.csv").open("w", encoding="utf-8") as f:
        f.write("label,count\n")
        for label, count in zip(labels, counts):
            f.write(f"{int(label)},{int(count)}\n")

    return int(face_mask.sum()), int(neck_mask.sum()), int(face_neck_mask.sum())


def main():
    parser = argparse.ArgumentParser(description="Run face parsing on a saved RGB image.")
    parser.add_argument(
        "--image",
        default="tests/flame_fit_face_neck_output/capture_rgb.jpg",
        help="Input image path.",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/parse_capture_rgb_output",
        help="Output directory.",
    )
    args = parser.parse_args()

    image_path = ROOT / args.image
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    full_counts = save_parsing_outputs(out_dir, rgb, parse_full_image(rgb), "full")
    crop_parsing, bbox, crop_xyxy = parse_crop_mapped(rgb)
    crop_counts = save_parsing_outputs(out_dir, rgb, crop_parsing, "crop_mapped", bbox, crop_xyxy)

    print(f"Saved parsing outputs to: {out_dir}")
    print(f"full face/neck/face+neck pixels: {full_counts}")
    print(f"crop mapped face/neck/face+neck pixels: {crop_counts}")


if __name__ == "__main__":
    main()
