import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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


class ClipSimilarityCalculator:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def compute_similarity(self, img1_pil, img2_pil):
        inputs = self.processor(images=[img1_pil, img2_pil], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return float(torch.sum(image_features[0] * image_features[1]).item())


def calculate_clip_drift(orig_root, aes_root, threshold=None, resize_to=None, model_name="openai/clip-vit-base-patch32", device=None):
    calc = ClipSimilarityCalculator(model_name=model_name, device=device)

    similarities = []
    rows = []
    drifted_rows = []

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

            similarity = calc.compute_similarity(orig_pil, ae_pil)
            similarities.append(similarity)

            row = {
                "class": cls,
                "id": file_id,
                "orig_path": str(orig_path),
                "ae_path": str(ae_path),
                "clip_similarity": similarity,
            }
            rows.append(row)

            if threshold is not None and similarity < threshold:
                drifted_rows.append(row)

    if not rows:
        raise RuntimeError("No matched orig/ae image pairs found.")

    result = {
        "count": len(rows),
        "mean_clip_similarity": round(float(np.mean(similarities)), 6),
        "median_clip_similarity": round(float(np.median(similarities)), 6),
        "min_clip_similarity": round(float(np.min(similarities)), 6),
        "max_clip_similarity": round(float(np.max(similarities)), 6),
    }

    if threshold is not None:
        result["clip_drift_threshold"] = threshold
        result["drifted_count"] = len(drifted_rows)
        result["drifted_pct"] = 100.0 * len(drifted_rows) / len(rows)

    return result, rows, drifted_rows


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


def get_existing_section(output_json_path, case_key, section_key):
    output_json_path = Path(output_json_path)
    if not output_json_path.exists():
        return None
    with open(output_json_path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    if not isinstance(data, dict):
        return None
    case_entry = data.get(case_key)
    if not isinstance(case_entry, dict):
        return None
    return case_entry.get(section_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", required=True, help="Root image directory containing 'orig/' and 'aes/'")
    parser.add_argument("--output-json", default=None, help="Optional shared JSON file to update with CLIP drift results")
    parser.add_argument("--case-key", default=None, help="Optional key used in the shared JSON; defaults to the full image root directory path")
    parser.add_argument("--threshold", type=float, default=None, help="Optional CLIP similarity threshold below which samples are counted as drifted")
    parser.add_argument("--resize-to", type=int, default=None, help="Optional square resize before scoring")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="CLIP model name")
    parser.add_argument("--device", default=None, help="cuda or cpu; default auto-detect")
    parser.add_argument("--details-json", default=None, help="Optional JSON path for per-sample details")
    parser.add_argument("--drifted-json", default=None, help="Optional JSON path for below-threshold cases")
    parser.add_argument("--override", action="store_true", help="Recompute even if CLIP drift already exists in the output JSON")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    orig_dir = image_root / "orig"
    ae_dir = image_root / "aes"
    if not orig_dir.is_dir() or not ae_dir.is_dir():
        raise RuntimeError(f"Expected '{orig_dir}' and '{ae_dir}' to exist.")

    case_key = args.case_key or str(image_root)

    if args.output_json:
        existing = get_existing_section(args.output_json, case_key, "clip_drift")
        if existing is not None and not args.override:
            print(json.dumps(existing, indent=2))
            return

    result, rows, drifted_rows = calculate_clip_drift(
        orig_root=orig_dir,
        aes_root=ae_dir,
        threshold=args.threshold,
        resize_to=args.resize_to,
        model_name=args.model_name,
        device=args.device,
    )

    print(json.dumps(result, indent=2))

    if args.output_json:
        upsert_output_json(args.output_json, case_key, "clip_drift", result)

    if args.details_json:
        with open(args.details_json, "w") as f:
            json.dump(rows, f, indent=2)

    if args.drifted_json and args.threshold is not None:
        with open(args.drifted_json, "w") as f:
            json.dump(drifted_rows, f, indent=2)


if __name__ == "__main__":
    main()
