import json
import argparse


VARIANT_SUFFIXES = [
    ("", "NLC"),
    ("-rounding", "NLC + Rounding"),
    ("-enforce-plausibility", "NLC + Plausibility Enforcement"),
    ("-rounding-enforce-plausibility", "NLC + Both"),
]


def fmt_float(value, digits=2):
    return f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-json", required=True)
    args = parser.parse_args()

    with open(args.results_json, "r") as f:
        data = json.load(f)

    baseline_key = None
    for key in data.keys():
        if key.endswith("/ImageNet-resnet50-NLC/image") or key.endswith("ImageNet-resnet50-NLC/image"):
            baseline_key = key
            break

    if baseline_key is None:
        raise RuntimeError("Could not find baseline key ending with 'ImageNet-resnet50-NLC/image' in results JSON.")

    base_prefix = baseline_key[:-len("/image")]

    for suffix, label in VARIANT_SUFFIXES:
        key = f"{base_prefix}{suffix}/image"
        if key not in data:
            print(f"% Missing: {key}")
            continue

        entry = data[key]
        repro = entry["reproducibility"]
        clip = entry["clip_drift"]

        row = (
            f"{label} & "
            f"{repro['total']} & "
            f"{fmt_float(repro['reproducible_pct'])} & "
            f"{fmt_float(clip['mean_clip_similarity'], 3)} & "
            f"{fmt_float(clip['drifted_pct'])} \\\\"
        )
        print(row)


if __name__ == "__main__":
    main()
