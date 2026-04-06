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
    x = torch.from_numpy(x).transpose(1, 3)  # 1 x C x H x W
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

    sample_id = m.group("id")
    kind = m.group("kind")
    predicted_label = int(m.group("label"))

    return gt_label, predicted_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["CIFAR10", "ImageNet"])
    parser.add_argument("--model", required=True, choices=["resnet50", "vgg16_bn", "mobilenet_v2"])
    parser.add_argument("--ae-dir", required=True, help="Directory containing saved AE PNGs, e.g. .../image/aes")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of AE files to evaluate")
    args = parser.parse_args()

    set_deterministic()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.dataset, args.model, device)

    ae_dir = Path(args.ae_dir)
    files = sorted(ae_dir.rglob("*.png"))
    if args.limit is not None:
        files = files[:args.limit]

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
        pred = predict(model, img, args.dataset, device)

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

    print(json.dumps(summary, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(
                {
                    "summary": summary,
                    "details": details,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
