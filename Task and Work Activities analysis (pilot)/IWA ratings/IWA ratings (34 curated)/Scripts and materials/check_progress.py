#!/usr/bin/env python3
"""
check_progress.py
==================
Quick sanity-check across the per-model run folders produced by
run_ratings_IWA.py.

For each provider folder (claude/, chatgpt/, gemini/, mistral/) it prints,
for every run_N.json checkpoint:
  - number of successful IWA ratings
  - number of error entries
  - total expected (34 IWAs)
  - which IWAs are still missing, if any

Usage:
    python check_progress.py
"""

import json
import os
from typing import Set

INPUT_JSON_FILE = "iwas_dict.json"
# All results are read from / written to the sibling "Results" folder
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
PROVIDER_FOLDERS = ["claude", "chatgpt", "gemini", "mistral"]


def main():
    if not os.path.exists(INPUT_JSON_FILE):
        print(f"Could not find {INPUT_JSON_FILE} in current folder.")
        return

    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        iwas = json.load(f)
    expected: Set[str] = set(iwas.keys())
    total = len(expected)

    print("=" * 72)
    print(f"  IWA Rating Pipeline — Progress Check ({total} IWAs expected)")
    print("=" * 72)

    any_runs = False
    for folder in PROVIDER_FOLDERS:
        folder = os.path.join(RESULTS_DIR, folder)
        if not os.path.isdir(folder):
            continue
        run_files = sorted(
            f for f in os.listdir(folder)
            if f.startswith("run_") and f.endswith(".json")
        )
        if not run_files:
            continue
        any_runs = True
        print(f"\n[{folder}/]")
        for name in run_files:
            path = os.path.join(folder, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"  {name:<20}  unreadable ({e})")
                continue

            successful = [r for r in results if "error" not in r]
            errors = [r for r in results if "error" in r]
            done_codes = {r["iwa_code"] for r in successful}
            missing = sorted(expected - done_codes)

            status = "COMPLETE" if not missing else "in progress"
            print(
                f"  {name:<20}  successful={len(successful):3d}/{total}  "
                f"errors={len(errors):2d}  [{status}]"
            )
            if missing and len(missing) <= 10:
                print(f"      missing: {', '.join(missing)}")
            elif missing:
                print(f"      missing: {len(missing)} IWAs "
                      f"(first 10: {', '.join(missing[:10])} …)")

    if not any_runs:
        print("\n  No run_N.json checkpoint files found in any provider folder yet.")
        print("  Run `python run_ratings_IWA.py` to start a rating run.")
    print()


if __name__ == "__main__":
    main()
