import os
import re
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import lpips
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity
import torchvision.transforms as transforms


NAME_RE = re.compile(r"^(?P<id>.+?)_(?P<kind>orig|ae)_(?P<label>\d+)(?:_\d+)?\.png$")


def parse_file_id(filename: str):
    m = NAME_RE.match(filename)
    if not m:
        return None
    return m.group("id")


def build_file_map(class_dir: Path):
    mapping = {}
    for path in class_dir.glob("*.png"):
        file_id = parse_file_id(path.name)
        if file_id is not None:
            mapping[file_id] = path
    return mapping


def load_pil_rgb(path: Path, resize_to=None):
    img = Image.open(path).convert("RGB")
    if resize_to is not None:
        img = img.resize((resize_to, resize_to))
    return img


class NaturalnessCalculator:
    def __init__(self, device: str = None, lpips_net: str = "alex"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips.LPIPS(net=lpips_net).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

    def compute_lpips(self, img1_pil, img2_pil):
        img1 = self.transform(img1_pil).unsqueeze(0).to(self.device)
        img2 = self.transform(img2_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.loss_fn(img1, img2)
        return float(dist.item())

    @staticmethod
    def compute_ssim(img1_pil, img2_pil):
        img1 = np.array(img1_pil, dtype=np.uint8)
        img2 = np.array(img2_pil, dtype=np.uint8)
        return float(structural_similarity(img1, img2, channel_axis=2, data_range=255))


def calculate_naturalness(orig_root, aes_root, resize_to=None, lpips_threshold=None, device=None, lpips_net="alex"):
    calc = NaturalnessCalculator(device=device, lpips_net=lpips_net)

    lpips_scores = []
    ssim_scores = []
    rows = []
    high_lpips_rows = []

    orig_root = Path(orig_root)
    aes_root = Path(aes_root)

    for cls in sorted(os.listdir(orig_root)):
        orig_cls_dir = orig_root / cls
        aes_cls_dir = aes_root / cls
        if not orig_cls_dir.is_dir() or not aes_cls_dir.is_dir():
            continue

        orig_map = build_file_map(orig_cls_dir)
        ae_map = build_file_map(aes_cls_dir)
        common_ids = sorted(set(orig_map.keys()) & set(ae_map.keys()))

        for file_id in tqdm(common_ids, desc=f"Class {cls}"):
            orig_path = orig_map[file_id]
            ae_path = ae_map[file_id]

            orig_pil = load_pil_rgb(orig_path, resize_to=resize_to)
            ae_pil = load_pil_rgb(ae_path, resize_to=resize_to)

            lpips_score = calc.compute_lpips(orig_pil, ae_pil)
            ssim_score = calc.compute_ssim(orig_pil, ae_pil)

            lpips_scores.append(lpips_score)
            ssim_scores.append(ssim_score)

            row = {
                "class": cls,
                "id": file_id,
                "orig_path": str(orig_path),
                "ae_path": str(ae_path),
                "lpips": lpips_score,
                "ssim": ssim_score,
            }
            rows.append(row)

            if lpips_threshold is not None and lpips_score > lpips_threshold:
                high_lpips_rows.append(row)

    if not rows:
        raise RuntimeError("No matched orig/ae image pairs found.")

    result = {
        "count": len(rows),
        "mean_lpips": round(float(np.mean(lpips_scores)), 6),
        "mean_ssim": round(float(np.mean(ssim_scores)), 6),
        "median_lpips": round(float(np.median(lpips_scores)), 6),
        "median_ssim": round(float(np.median(ssim_scores)), 6),
        "min_lpips": round(float(np.min(lpips_scores)), 6),
        "max_lpips": round(float(np.max(lpips_scores)), 6),
        "min_ssim": round(float(np.min(ssim_scores)), 6),
        "max_ssim": round(float(np.max(ssim_scores)), 6),
    }

    if lpips_threshold is not None:
        result[f"high_lpips_count_gt_{lpips_threshold}"] = len(high_lpips_rows)

    return result, rows, high_lpips_rows


def upsert_output_json(output_json_path, case_key, section_key, payload):
    output_json_path = Path(output_json_path)
    if output_json_path.exists():
        with open(output_json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    if case_key not in data or not isinstance(data[case_key], dict):
        data[case_key] = {}

    data[case_key][section_key] = payload

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="Root image directory containing 'orig/' and 'aes/'")
    parser.add_argument("--output-json", required=True, help="Shared JSON file to update with naturalness results")
    parser.add_argument("--resize-to", type=int, default=None, help="Optional square resize before scoring")
    parser.add_argument("--lpips-threshold", type=float, default=None, help="Optional threshold for exporting high-LPIPS cases")
    parser.add_argument("--lpips-net", choices=["alex", "squeeze", "vgg"], default="alex")
    parser.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    parser.add_argument("--case-key", default=None, help="Optional key used in the shared JSON; defaults to the full image root directory path")
    parser.add_argument("--output-csv", default=None, help="Optional per-sample CSV path")
    parser.add_argument("--high-lpips-csv", default=None, help="Optional CSV path for high-LPIPS samples")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    orig_dir = image_root / "orig"
    ae_dir = image_root / "aes"
    if not orig_dir.is_dir() or not ae_dir.is_dir():
        raise RuntimeError(f"Expected '{orig_dir}' and '{ae_dir}' to exist.")

    case_key = args.case_key or str(image_root)

    result, rows, high_lpips_rows = calculate_naturalness(
        orig_root=orig_dir,
        aes_root=ae_dir,
        resize_to=args.resize_to,
        lpips_threshold=args.lpips_threshold,
        device=args.device,
        lpips_net=args.lpips_net,
    )

    print(json.dumps(result, indent=2))

    upsert_output_json(args.output_json, case_key, "naturalness", result)

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "id", "orig_path", "ae_path", "lpips", "ssim"])
            writer.writeheader()
            writer.writerows(rows)

    if args.high_lpips_csv and args.lpips_threshold is not None:
        with open(args.high_lpips_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "id", "orig_path", "ae_path", "lpips", "ssim"])
            writer.writeheader()
            writer.writerows(high_lpips_rows)


if __name__ == "__main__":
    main()
