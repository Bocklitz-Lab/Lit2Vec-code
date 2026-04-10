# run_compare.py

import argparse
import yaml
import sys
from pathlib import Path
from src.ValidationComparator import ValidationComparator  # Note: lowercase filename

def main():
    parser = argparse.ArgumentParser(description="Compare only record_hash hashes between two folders.")
    parser.add_argument(
        "--config", type=str, default="compare_config.yaml",
        help="Path to YAML config file with folder paths"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    gt_dir = Path(config["COMPARE_CONFIG"]["ground_truth_dir"])
    test_dir = Path(config["COMPARE_CONFIG"]["generated_dir"])
    ignored_fields = config["COMPARE_CONFIG"].get(
        "ignored_fields",
        ["tldr", "predicted_subfield"],
    )

    assert gt_dir.exists(), f"❌ Ground truth directory not found: {gt_dir}"
    assert test_dir.exists(), f"❌ Generated directory not found: {test_dir}"

    errors = 0
    all_gt_files = sorted(gt_dir.glob("*.json"))

    for gt_file in all_gt_files:
        test_file = test_dir / gt_file.name
        if not test_file.exists():
            print(f"[!] Skipping missing test file: {test_file}")
            errors += 1
            continue

        comparator = ValidationComparator(
            gt_file,
            test_file,
            ignored_fields=ignored_fields,
        )
        success = comparator.compare()
        if not success:
            errors += 1

    if errors > 0:
        print(f"\n⚠️ {errors} file(s) had hash mismatches — review above.")
        # Do not exit with error
    else:
        print("\n✅ All validation file hashes matched!")



if __name__ == "__main__":
    main()
