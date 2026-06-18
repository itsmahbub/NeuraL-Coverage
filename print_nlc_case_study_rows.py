import json
import os
import argparse

VARIANT_SUFFIXES = [
    ("", "NLC"),
    ("-rounding", "NLC+FR"),
    ("-enforce-plausibility", "NLC+IP"),
    ("-rounding-enforce-plausibility", "NLC+FR+IP"),
]

CLIP_THRESHOLD = 0.8


def fmt_float(value, digits=2):
    return f"{value:.{digits}f}"


def load_per_sample(variant_dir):
    clip_path = os.path.join(variant_dir, "clip_details.json")
    repro_path = os.path.join(variant_dir, "reproducibility_details.json")

    with open(clip_path) as f:
        clip_data = json.load(f)
    with open(repro_path) as f:
        repro_data = json.load(f)

    # Index reproducibility by ae filename
    repro_by_name = {
        os.path.basename(r["path"]): r["status"]
        for r in repro_data
    }

    count = 0
    for entry in clip_data:
        ae_name = entry["ae_name"]
        clip_sim = entry["clip_similarity"]
        status = repro_by_name.get(ae_name)
        if clip_sim >= CLIP_THRESHOLD and status == "reproducible":
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    parser.add_argument("--output-dir", required=True,
                        help="Base dir containing variant subdirectories")
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    baseline_key = next(
        k for k in data if k.endswith("ImageNet-resnet50-NLC/image")
    )
    base_prefix = baseline_key[:-len("/image")]
    base_dir = args.output_dir  # e.g. data/output/Coverage/Fuzzer

    for suffix, label in VARIANT_SUFFIXES:
        key = f"{base_prefix}{suffix}/image"
        if key not in data:
            print(f"% Missing: {key}")
            continue

        entry = data[key]
        repro = entry["reproducibility"]
        clip = entry["clip_drift"]

        variant_name = f"ImageNet-resnet50-NLC{suffix}"
        variant_dir = os.path.join(base_dir, variant_name)
        pr_count = load_per_sample(variant_dir)

        row = (
            f"{label} & "
            f"{repro['total']} & "
            f"{fmt_float(repro['reproducible_pct'])} & "
            f"{fmt_float(clip['mean_clip_similarity'], 3)} & "
            f"{fmt_float(clip['drifted_pct'])} & "
            f"{pr_count} \\\\"
        )
        print(row)


if __name__ == "__main__":
    main()