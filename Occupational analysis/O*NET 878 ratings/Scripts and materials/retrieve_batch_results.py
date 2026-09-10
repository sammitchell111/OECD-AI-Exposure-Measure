#!/usr/bin/env python3
"""
Retrieve results from a Gemini batch job and save to gemini/run_N.json + .xlsx.

Works with jobs submitted by the penultimate (single-run, no-modes) version
of run_ratings_google_batch.py, where keys are plain O*NET codes like
"11-1011.00", as well as the multi-run version where keys are "runN:code".

Usage:
    python retrieve_batch_results.py                     # auto-finds your jobs
    python retrieve_batch_results.py <job_name>          # specific job
    python retrieve_batch_results.py <job_name> --run 2  # save as run_2
"""

import argparse
import json
import os
import sys
import getpass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

try:
    from google import genai
except ImportError:
    print("Please install google-genai:  pip install google-genai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration (must match run_ratings.py / run_ratings_google_batch.py)
# ---------------------------------------------------------------------------
INPUT_JSON_FILE = "occupations_dict_mar17.json"
# All results are read from / written to the sibling "Results" folder
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
OUTPUT_FOLDER = os.path.join(RESULTS_DIR, "gemini")
MODEL_NAME = "gemini-3.1-pro-preview"

SCALE_NAMES = [
    "Scale 1: Language",
    "Scale 2: Social Interaction",
    "Scale 3: Problem Solving",
    "Scale 4: Creativity",
    "Scale 5: Metacognition and critical thinking",
    "Scale 6: Knowledge, learning and memory",
    "Scale 7: Vision",
    "Scale 8: Manipulation",
    "Scale 9: Robotic Intelligence",
]


# ---------------------------------------------------------------------------
# Helpers (identical to run_ratings.py)
# ---------------------------------------------------------------------------

def load_api_key(filename="GOOGLE_API_KEY.txt"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    return None


def parse_llm_response(response_text: str) -> Dict:
    json_str = response_text
    if "```json" in response_text:
        start = response_text.find("```json") + 7
        end = response_text.find("```", start)
        json_str = response_text[start:end].strip()
    elif "{" in response_text:
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        json_str = response_text[start:end]
    return json.loads(json_str)


def result_to_row(result: Dict) -> Dict:
    row = {
        "O*NET-SOC Code": result["onet_soc_code"],
        "Occupation Title": result["occupation_title"],
        "Description": result["description"],
        "Number of Tasks": result["num_tasks"],
        "Model Used": result["model_used"],
        "Processing Timestamp": result["timestamp"],
    }
    for scale_name in SCALE_NAMES:
        rating_info = result.get("ratings", {}).get(scale_name, {})
        row[f"{scale_name} - Level"] = rating_info.get("level", -1)
        row[f"{scale_name} - Reasoning"] = rating_info.get("reasoning", "")
    return row


def save_excel(rows: List[Dict], excel_path: str):
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Occupation Ratings", index=False)
        ws = writer.sheets["Occupation Ratings"]
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)


def find_incomplete_run(folder: str, total_occupations: int) -> Optional[int]:
    if not os.path.exists(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if name.startswith("run_") and name.endswith(".json"):
            try:
                num = int(name.split("_")[1].split(".")[0])
                json_path = os.path.join(folder, name)
                with open(json_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    successful = sum(1 for r in results if "error" not in r)
                    if successful < total_occupations:
                        return num
            except (ValueError, IndexError, json.JSONDecodeError):
                pass
    return None


def next_run_number(folder: str) -> int:
    if not os.path.exists(folder):
        return 1
    existing = []
    for name in os.listdir(folder):
        if name.startswith("run_") and "." in name:
            try:
                num = int(name.split("_")[1].split(".")[0])
                existing.append(num)
            except (ValueError, IndexError):
                pass
    return max(existing, default=0) + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve results from a Gemini batch job",
    )
    parser.add_argument(
        "job_name",
        nargs="?",
        default=None,
        help="Batch job name (e.g. 'batches/abc123'). "
             "Omit to list all jobs and choose interactively.",
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Force saving as a specific run number (e.g. --run 2). "
             "By default, auto-detects the incomplete run or creates the next one.",
    )
    args = parser.parse_args()

    api_key = load_api_key() or getpass.getpass("Enter your Google API Key: ")
    client = genai.Client(api_key=api_key)

    # If no job name given, list jobs and let user pick
    if args.job_name is None:
        print("\nYour batch jobs:\n")
        jobs = list(client.batches.list())
        if not jobs:
            print("  No batch jobs found.")
            return

        for i, job in enumerate(jobs):
            state = job.state.name if hasattr(job.state, 'name') else str(job.state)
            display = getattr(job, 'display_name', '') or ''
            print(f"  [{i + 1}] {job.name}  —  {state}  —  {display}")

        print()
        choice = input("Enter number to retrieve (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return
        try:
            idx = int(choice) - 1
            job_name = jobs[idx].name
        except (ValueError, IndexError):
            print("Invalid selection.")
            return
    else:
        job_name = args.job_name

    # Get job status
    print(f"\nJob: {job_name}")
    batch_job = client.batches.get(name=job_name)
    state_str = batch_job.state.name if hasattr(batch_job.state, 'name') else str(batch_job.state)
    print(f"Status: {state_str}")

    if "SUCCEEDED" not in state_str:
        if "FAIL" in state_str or "CANCEL" in state_str:
            print(f"\nJob ended with error state: {state_str}")
            if hasattr(batch_job, 'error') and batch_job.error:
                print(f"Error: {batch_job.error}")
        else:
            print(f"\nJob is still running. Check back later.")
        return

    # Download results
    print("\nDownloading results…")
    result_file_name = batch_job.dest.file_name
    print(f"  File: {result_file_name}")
    file_content = client.files.download(file=result_file_name)
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    raw_results = []
    for line in file_content.strip().splitlines():
        if line.strip():
            raw_results.append(json.loads(line))

    print(f"  Downloaded {len(raw_results)} results.")

    # Load occupations for metadata
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        occupations = json.load(f)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Detect key format: "runN:code" (multi-run) vs plain "code" (single-run)
    sample_key = raw_results[0].get("key", "") if raw_results else ""
    is_multi_run = ":" in sample_key and sample_key.split(":")[0].startswith("run")

    if is_multi_run:
        # Group by run number
        results_by_run: Dict[int, List[dict]] = {}
        for item in raw_results:
            key = item.get("key", "")
            run_part, code = key.split(":", 1)
            run_num = int(run_part.replace("run", ""))
            results_by_run.setdefault(run_num, []).append((code, item))

        print(f"\n  Multi-run batch detected. Runs: {sorted(results_by_run.keys())}")

        for run_num in sorted(results_by_run.keys()):
            _save_run_results(
                run_num, results_by_run[run_num], occupations
            )
    else:
        # Single run — figure out which run number
        if args.run is not None:
            run_num = args.run
        else:
            incomplete = find_incomplete_run(OUTPUT_FOLDER, len(occupations))
            run_num = incomplete if incomplete is not None else next_run_number(OUTPUT_FOLDER)

        items = [(item.get("key", ""), item) for item in raw_results]
        _save_run_results(run_num, items, occupations)

    print(f"\nDone!")


def _save_run_results(
    run_num: int,
    items: List[tuple],  # list of (occ_code, raw_result_item)
    occupations: Dict[str, Dict],
):
    """Parse results and save to run_N.json + .xlsx, merging with existing data."""
    json_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.json")
    excel_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.xlsx")

    # Load existing successful results
    existing_by_code: Dict[str, Dict] = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                if "error" not in r:
                    existing_by_code[r["onet_soc_code"]] = r

    new_success = 0
    new_fail = 0

    for code, item in items:
        occ_data = occupations.get(code, {})
        try:
            response_obj = item.get("response", {})
            text = ""
            if isinstance(response_obj, dict):
                candidates = response_obj.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
            elif hasattr(response_obj, 'text'):
                text = response_obj.text

            if not text:
                raise ValueError("Empty response text")

            parsed = parse_llm_response(text)
            existing_by_code[code] = {
                "onet_soc_code": code,
                "occupation_title": occ_data.get("title", "Unknown"),
                "description": occ_data.get("description", ""),
                "num_tasks": len(occ_data.get("tasks", [])),
                "ratings": parsed.get("ratings", {}),
                "model_used": MODEL_NAME,
                "timestamp": datetime.now().isoformat(),
            }
            new_success += 1
        except Exception as e:
            print(f"    WARNING: Failed to parse {code}: {e}")
            new_fail += 1

    # Save
    all_results = list(existing_by_code.values())
    all_rows = [result_to_row(r) for r in all_results]

    if all_rows:
        save_excel(all_rows, excel_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n  Run {run_num}: {new_success} new successful, {new_fail} failed")
    print(f"    Total in file: {len(all_results)}")
    print(f"    → {json_path}")
    print(f"    → {excel_path}")


if __name__ == "__main__":
    main()
