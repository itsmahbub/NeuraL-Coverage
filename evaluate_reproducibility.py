import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

import utility
import models
import constants
from torchvision.models import resnet50, ResNet50_Weights


NAME_RE = re.compile(r"^(?P<id>.+?)_(?P<kind>orig|ae)_(?P<label>\d+)(?:_\d+)?\.png$")


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def load_model(dataset: str, model_name: str, device: torch.device):
    if dataset == "ImageNet":
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            model = torchvision.models.__dict__[model_name](pretrained=False)
            path = os.path.join(constants.PRETRAINED_MODELS, f"{dataset}/{model_name}.pth")
            model.load_state_dict(torch.load(path, map_location=device))
    elif dataset == "CIFAR10":
        model = getattr(models, model_name)(pretrained=False)
        path = os.path.join(constants.PRETRAINED_MODELS, f"{dataset}/{model_name}.pt")
        model.load_state_dict(torch.load(path, map_location=device))
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model.to(device)
    model.eval()
    return model


def image_to_input(image_hwc_uint8: np.ndarray, dataset: str, device: torch.device):
    """
    image_hwc_uint8: H x W x C, dtype uint8, range [0, 255]
    returns: 1 x C x H x W normalized tensor
    """
    assert image_hwc_uint8.dtype == np.uint8
    x = image_hwc_uint8.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)  # 1 x H x W x C
    x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1 x C x H x W
    x = utility.image_normalize(x, dataset)
    return x.to(device)


def predict(model, image_hwc_uint8: np.ndarray, dataset: str, device: torch.device) -> int:
    x = image_to_input(image_hwc_uint8, dataset, device)
    with torch.no_grad():
        logits = model(x)
        return int(logits.argmax(dim=1).item())


def parse_labels_from_path(path: Path):
    """
    Assumes:
      .../aes/<ground_truth>/<id>_ae_<predicted>_<...>.png
    """
    gt_label = int(path.parent.name)
    m = NAME_RE.match(path.name)
    if not m:
        raise ValueError(f"Bad filename: {path.name}")

    predicted_label = int(m.group("label"))

    return gt_label, predicted_label


def calculate_reproducibility(dataset: str, model_name: str, ae_dir, limit=None):
    set_deterministic()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(dataset, model_name, device)

    ae_dir = Path(ae_dir)
    files = sorted(ae_dir.rglob("*.png"))
    if limit is not None:
        files = files[:limit]

    stats = {
        "total": 0,
        "reproducible": 0,
        "reverted_to_gt": 0,
        "changed_wrong_label": 0,
        "parse_error": 0,
    }

    details = []

    for path in tqdm(files, desc="Evaluating saved AEs"):
        try:
            gt_label, predicted = parse_labels_from_path(path)
        except Exception as e:
            stats["parse_error"] += 1
            details.append({
                "path": str(path),
                "status": "parse_error",
                "error": str(e),
            })
            continue

        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        pred = predict(model, img, dataset, device)

        stats["total"] += 1

        if pred == predicted:
            status = "reproducible"
            stats["reproducible"] += 1
        elif pred == gt_label:
            status = "reverted_to_gt"
            stats["reverted_to_gt"] += 1
        else:
            status = "changed_wrong_label"
            stats["changed_wrong_label"] += 1

        details.append({
            "path": str(path),
            "ground_truth": gt_label,
            "predicted": predicted,
            "reloaded_prediction": pred,
            "status": status,
        })

    total = max(stats["total"], 1)
    summary = {
        **stats,
        "reproducible_pct": 100.0 * stats["reproducible"] / total,
        "reverted_to_gt_pct": 100.0 * stats["reverted_to_gt"] / total,
        "changed_wrong_label_pct": 100.0 * stats["changed_wrong_label"] / total,
    }

    return summary, details


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
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "ImageNet"])
    parser.add_argument("--model", required=True, choices=["resnet50", "vgg16_bn", "mobilenet_v2"])
    parser.add_argument("--image-root", required=True, help="Root image directory containing 'aes/'")
    parser.add_argument("--output-json", default=None, help="Optional shared JSON file to update with reproducibility results")
    parser.add_argument("--case-key", default=None, help="Optional key used in the shared JSON; defaults to the full image root directory path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of AE files to evaluate")
    parser.add_argument("--details-json", default=None, help="Optional JSON path for per-sample details")
    parser.add_argument("--override", action="store_true", help="Recompute even if reproducibility already exists in the output JSON")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    ae_dir = image_root / "aes"
    if not ae_dir.is_dir():
        raise RuntimeError(f"Expected '{ae_dir}' to exist.")

    case_key = args.case_key or str(image_root)

    if args.output_json:
        existing = get_existing_section(args.output_json, case_key, "reproducibility")
        if existing is not None and not args.override:
            print(json.dumps(existing, indent=2))
            return

    summary, details = calculate_reproducibility(
        dataset=args.dataset,
        model_name=args.model,
        ae_dir=ae_dir,
        limit=args.limit,
    )

    print(json.dumps(summary, indent=2))

    if args.output_json:
        upsert_output_json(args.output_json, case_key, "reproducibility", summary)

    if args.details_json:
        with open(args.details_json, "w") as f:
            json.dump(details, f, indent=2)


if __name__ == "__main__":
    main()
