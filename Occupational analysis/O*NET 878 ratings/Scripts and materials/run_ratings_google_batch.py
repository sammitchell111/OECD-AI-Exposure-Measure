#!/usr/bin/env python3
"""
Occupational AI Capability Rating Pipeline — Google Batch API Edition
=====================================================================
Uses the Google Gemini Batch API to process occupations without hitting
per-minute rate limits. Supports submitting, closing the terminal, and
retrieving results later.

Three modes:
  submit   — Build requests, upload, submit batch job, save job ID, exit.
             You can safely close the terminal after this.
  retrieve — Reconnect to a submitted batch job, download results, save
             to gemini/run_N.json + .xlsx files.
  status   — Quick check on whether a pending job is done yet.

Multi-run support:
  By default, submits the NEXT incomplete run. Use --runs N to submit
  multiple runs (e.g. --runs 5) in a single batch job.

Usage:
    python run_ratings_google_batch.py submit              # submit next run
    python run_ratings_google_batch.py submit --runs 5     # submit 5 runs at once
    python run_ratings_google_batch.py status              # check job status
    python run_ratings_google_batch.py retrieve             # download results & save

    # Legacy: no arguments = submit + wait + retrieve (original behaviour)
    python run_ratings_google_batch.py
"""

import argparse
import json
import os
import sys
import time
import getpass
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# The Batch API requires the *new* google-genai SDK (not google-generativeai).
# Both packages can coexist — the original run_ratings.py keeps using the old one.
# ---------------------------------------------------------------------------
try:
    from google import genai
except ImportError:
    print("\n  The Google Batch API requires the 'google-genai' package.")
    print("  This is a different package from 'google-generativeai' (used by run_ratings.py).")
    print("  Both can be installed side-by-side without conflict.\n")
    answer = input("  Install it now with pip? (y/n): ").strip().lower()
    if answer == "y":
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "google-genai"],
        )
        from google import genai
        print()
    else:
        print("\n  To install manually, run:  pip install google-genai")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_JSON_FILE = "occupations_dict_mar17.json"
# All results are read from / written to the sibling "Results" folder
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
SCALES_FILE = "scales_long.txt"
PROMPT_FILE = "rating_instructions_mar17.txt"

MODEL_NAME = "gemini-3.1-pro-preview"
OUTPUT_FOLDER = os.path.join(RESULTS_DIR, "gemini")

# File that tracks pending batch jobs so you can close the terminal
BATCH_TRACKER_FILE = os.path.join(OUTPUT_FOLDER, "_pending_batch.json")

# How often to poll the batch job status (seconds)
POLL_INTERVAL = 30

# ---------------------------------------------------------------------------
# Text helpers (identical to run_ratings.py)
# ---------------------------------------------------------------------------

def load_text_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def parse_scales_file(file_path: str) -> Dict:
    """Parse scales_long.txt into a dict of {scale_name: {Level N: description}}."""
    content = load_text_file(file_path)
    scales: Dict[str, Dict[str, str]] = {}
    current_scale_name: Optional[str] = None

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("Scale "):
            parts = line.split(":", 1)
            if len(parts) >= 2:
                scale_num = parts[0].replace("Scale ", "").strip()
                scale_name = parts[1].strip()
                current_scale_name = f"Scale {scale_num}: {scale_name}"
                scales[current_scale_name] = {}
        elif line.startswith("Level ") and ":" in line:
            parts = line.split(":", 1)
            level_num = parts[0].replace("Level ", "").strip()
            level_desc = parts[1].strip()
            if current_scale_name:
                scales[current_scale_name][f"Level {level_num}"] = level_desc

    return scales


# ---------------------------------------------------------------------------
# Prompt construction (identical to run_ratings.py)
# ---------------------------------------------------------------------------

def create_rating_prompt(
    occupation_code: str,
    occupation_data: Dict,
    scales_info: Dict,
    prompt_template: str,
) -> str:
    occupation_title = occupation_data.get("title", "Unknown")
    occupation_desc = occupation_data.get("description", "No description available")
    task_list = occupation_data.get("tasks", [])

    scales_text = ""
    for scale_name, levels in scales_info.items():
        scales_text += f"\n{scale_name}\n"
        for level, desc in levels.items():
            scales_text += f"  {level}: {desc}\n"

    if task_list and isinstance(task_list, list) and len(task_list) > 0:
        tasks_text = "\nKEY TASKS FOR THIS OCCUPATION:\n"
        for i, task in enumerate(task_list[:15], 1):
            tasks_text += f"{i}. {task}\n"
    else:
        tasks_text = "\n(No specific task statements available for this occupation)\n"

    return prompt_template.format(
        occupation_code=occupation_code,
        occupation_title=occupation_title,
        occupation_desc=occupation_desc,
        tasks_text=tasks_text,
        scales_text=scales_text,
    )


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def load_api_key(filename: str) -> Optional[str]:
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                key = f.read().strip()
                if key:
                    print(f"   Found API key in {filename}")
                    return key
    except Exception as e:
        print(f"   Error reading {filename}: {e}")
    return None


def get_client():
    api_key = load_api_key("GOOGLE_API_KEY.txt") or getpass.getpass(
        "Enter your Google API Key: "
    )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# LLM response parsing (identical to run_ratings.py)
# ---------------------------------------------------------------------------

def parse_llm_response(response_text: str) -> Dict:
    """Extract JSON from an LLM response, stripping markdown fences if present."""
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


# ---------------------------------------------------------------------------
# Excel save (identical to run_ratings.py)
# ---------------------------------------------------------------------------

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


def result_to_row(result: Dict) -> Dict:
    """Convert a single result dict to a flat row dict for the DataFrame."""
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
    """Write the current list of row-dicts to an Excel file."""
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Occupation Ratings", index=False)
        ws = writer.sheets["Occupation Ratings"]
        for col in ws.columns:
            max_len = max((len(str(c.value or "")) for c in col), default=10)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 2, 50)


# ---------------------------------------------------------------------------
# Run numbering / resume helpers (identical logic to run_ratings.py)
# ---------------------------------------------------------------------------

def find_incomplete_run(folder: str, total_occupations: int) -> Optional[int]:
    """Check if any existing run is incomplete (fewer results than total occupations)."""
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
    """Scan folder for existing run_N files and return the next number."""
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


def get_existing_codes(json_path: str) -> Tuple[set, Dict]:
    """Load already-successful occupation codes from a run's JSON file."""
    existing_codes = set()
    results_by_code = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            prior = json.load(f)
            for r in prior:
                if "error" not in r:
                    code = r["onet_soc_code"]
                    existing_codes.add(code)
                    results_by_code[code] = r
    return existing_codes, results_by_code


# ---------------------------------------------------------------------------
# Batch tracker — persists job info so you can close the terminal
# ---------------------------------------------------------------------------

def save_tracker(tracker_data: dict):
    """Save batch job tracking info to disk."""
    with open(BATCH_TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(tracker_data, f, indent=2, ensure_ascii=False)


def load_tracker() -> Optional[dict]:
    """Load batch job tracking info from disk."""
    if not os.path.exists(BATCH_TRACKER_FILE):
        return None
    with open(BATCH_TRACKER_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def clear_tracker():
    """Remove the tracker file after successful retrieval."""
    if os.path.exists(BATCH_TRACKER_FILE):
        os.remove(BATCH_TRACKER_FILE)


# ---------------------------------------------------------------------------
# Batch job helpers
# ---------------------------------------------------------------------------

def build_batch_requests_multi(
    run_occupations: Dict[int, Dict[str, Dict]],
    scales_info: Dict,
    prompt_template: str,
) -> List[dict]:
    """
    Build batch requests for multiple runs.

    Each request key is "runN:occ_code" (e.g. "run2:11-1011.00") so we
    can split results back into the correct run files.
    """
    requests = []
    for run_num, occupations in run_occupations.items():
        for code, data in occupations.items():
            prompt = create_rating_prompt(code, data, scales_info, prompt_template)
            req = {
                "key": f"run{run_num}:{code}",
                "request": {
                    "contents": [
                        {
                            "parts": [{"text": prompt}]
                        }
                    ],
                    "generationConfig": {
                        "responseMimeType": "application/json",
                        "temperature": 0.2,
                    },
                },
            }
            requests.append(req)
    return requests


def submit_batch_file(client, requests: List[dict], display_name: str):
    """Write requests to a JSONL temp file, upload, and submit as a batch job."""
    jsonl_path = f"_batch_requests_{display_name}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"  Uploading batch file ({len(requests)} requests)…")
    uploaded_file = client.files.upload(
        file=jsonl_path,
        config={"mime_type": "application/jsonl"},
    )
    print(f"  Uploaded: {uploaded_file.name}")

    print(f"  Submitting batch job…")
    batch_job = client.batches.create(
        model=f"models/{MODEL_NAME}",
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    print(f"  Batch job created: {batch_job.name}")
    print(f"  State: {batch_job.state}")

    # Clean up local temp file
    try:
        os.remove(jsonl_path)
    except OSError:
        pass

    return batch_job


def get_batch_state(batch_job) -> str:
    """Extract state string from a batch job object."""
    if hasattr(batch_job.state, 'name'):
        return batch_job.state.name
    return str(batch_job.state)


def poll_batch_job(client, job_name: str):
    """Poll until the batch job reaches a terminal state. Returns the final job object."""
    terminal_states = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
                       "JOB_STATE_CANCELLED", "SUCCEEDED", "FAILED", "CANCELLED"}
    poll_count = 0
    while True:
        batch_job = client.batches.get(name=job_name)
        state_str = get_batch_state(batch_job)

        poll_count += 1
        if poll_count % 10 == 1 or poll_count <= 3:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Status: {state_str}")

        if state_str in terminal_states:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] Final status: {state_str}")
            return batch_job

        time.sleep(POLL_INTERVAL)


def retrieve_file_results(client, batch_job) -> List[dict]:
    """Download and parse results from a file-based batch job."""
    result_file_name = batch_job.dest.file_name
    print(f"  Downloading results from: {result_file_name}")
    file_content = client.files.download(file=result_file_name)
    if isinstance(file_content, bytes):
        file_content = file_content.decode("utf-8")

    parsed_results = []
    for line in file_content.strip().splitlines():
        if line.strip():
            parsed_results.append(json.loads(line))
    return parsed_results


# ---------------------------------------------------------------------------
# Result processing
# ---------------------------------------------------------------------------

def extract_response_text(item: dict) -> str:
    """Extract the text content from a batch result item."""
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
    elif hasattr(response_obj, 'candidates'):
        text = response_obj.candidates[0].content.parts[0].text
    return text


def process_and_save_results(
    raw_results: List[dict],
    occupations: Dict[str, Dict],
    run_numbers: List[int],
):
    """
    Parse batch results, split by run number, merge with existing data,
    and save each run's JSON + Excel files.
    """
    # Group raw results by run number
    results_by_run: Dict[int, List[dict]] = {n: [] for n in run_numbers}

    for item in raw_results:
        key = item.get("key", "")
        # Keys are "runN:occ_code"
        if ":" not in key:
            print(f"  WARNING: Unexpected key format: {key}")
            continue

        run_part, code = key.split(":", 1)
        try:
            run_num = int(run_part.replace("run", ""))
        except ValueError:
            print(f"  WARNING: Cannot parse run number from key: {key}")
            continue

        occ_data = occupations.get(code, {})

        try:
            text = extract_response_text(item)
            if not text:
                raise ValueError("Empty response text")

            parsed = parse_llm_response(text)
            results_by_run.setdefault(run_num, []).append({
                "onet_soc_code": code,
                "occupation_title": occ_data.get("title", "Unknown"),
                "description": occ_data.get("description", ""),
                "num_tasks": len(occ_data.get("tasks", [])),
                "ratings": parsed.get("ratings", {}),
                "model_used": MODEL_NAME,
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as e:
            print(f"  WARNING: Failed to parse result for {key}: {e}")
            results_by_run.setdefault(run_num, []).append({
                "onet_soc_code": code,
                "occupation_title": occ_data.get("title", "Unknown"),
                "description": occ_data.get("description", ""),
                "num_tasks": len(occ_data.get("tasks", [])),
                "ratings": {},
                "error": str(e),
                "model_used": MODEL_NAME,
                "timestamp": datetime.now().isoformat(),
            })

    # Save each run separately
    for run_num in run_numbers:
        json_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.json")
        excel_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.xlsx")

        # Load existing results for this run (resume merge)
        _, existing_by_code = get_existing_codes(json_path)

        # Merge new results
        new_results = results_by_run.get(run_num, [])
        for r in new_results:
            if "error" not in r:
                existing_by_code[r["onet_soc_code"]] = r

        # Build final list
        all_results = list(existing_by_code.values())
        # Also keep failed entries for reference
        failed_codes = {r["onet_soc_code"] for r in all_results}
        for r in new_results:
            if "error" in r and r["onet_soc_code"] not in failed_codes:
                all_results.append(r)

        successful = [r for r in all_results if "error" not in r]
        failed = [r for r in new_results if "error" in r]

        # Save
        all_rows = [result_to_row(r) for r in successful]
        if all_rows:
            save_excel(all_rows, excel_path)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"  Run {run_num}: {len(successful)} successful, {len(failed)} failed")
        print(f"    → {json_path}")
        print(f"    → {excel_path}")


# ============================================================================
# MODE: submit
# ============================================================================

def cmd_submit(num_runs: int):
    """Submit a batch job for one or more runs, save the job ID, and exit."""
    print("=" * 60)
    print("  Occupational AI Rating — SUBMIT Batch Job")
    print("=" * 60)

    # Check for already-pending job
    tracker = load_tracker()
    if tracker:
        print(f"\n  WARNING: There is already a pending batch job!")
        print(f"  Job name: {tracker['job_name']}")
        print(f"  Submitted: {tracker['submitted_at']}")
        print(f"  Runs: {tracker['run_numbers']}")
        print(f"\n  Use 'status' to check it or 'retrieve' to download results.")
        print(f"  If you want to submit a new job, delete {BATCH_TRACKER_FILE} first.")
        return

    client = get_client()
    print("  Client ready.\n")

    # Load data
    print("Loading data…")
    scales_info = parse_scales_file(SCALES_FILE)
    prompt_template = load_text_file(PROMPT_FILE)
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        occupations = json.load(f)
    print(f"  {len(occupations)} occupations loaded")
    print(f"  {len(scales_info)} scales parsed")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Figure out which runs to do
    run_numbers = []
    run_occupations: Dict[int, Dict[str, Dict]] = {}
    total_requests = 0

    for _ in range(num_runs):
        # Find next incomplete or new run
        incomplete = find_incomplete_run(OUTPUT_FOLDER, len(occupations))
        if incomplete is not None and incomplete not in run_numbers:
            run_num = incomplete
        else:
            # Find the next run number that isn't already being planned
            candidate = next_run_number(OUTPUT_FOLDER)
            while candidate in run_numbers:
                candidate += 1
            run_num = candidate

        json_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.json")
        existing_codes, _ = get_existing_codes(json_path)

        to_process = {
            code: data
            for code, data in occupations.items()
            if code not in existing_codes
        }

        if not to_process:
            # This run is already complete; need a fresh one
            candidate = next_run_number(OUTPUT_FOLDER)
            while candidate in run_numbers:
                candidate += 1
            run_num = candidate
            to_process = dict(occupations)  # Full set

        run_numbers.append(run_num)
        run_occupations[run_num] = to_process
        total_requests += len(to_process)

        # Create empty placeholder JSON so next iteration's next_run_number skips it
        placeholder_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.json")
        if not os.path.exists(placeholder_path):
            with open(placeholder_path, "w") as f:
                json.dump([], f)

        print(f"  Run {run_num}: {len(to_process)} occupations to process"
              f" ({len(existing_codes)} already done)")

    print(f"\n  TOTAL: {total_requests} requests across {len(run_numbers)} runs")
    print(f"  Runs: {run_numbers}")

    # Build and submit
    print(f"\nBuilding batch requests…")
    batch_requests = build_batch_requests_multi(
        run_occupations, scales_info, prompt_template
    )

    display_name = f"occ-ratings-runs{'_'.join(str(n) for n in run_numbers)}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    batch_job = submit_batch_file(client, batch_requests, display_name)

    # Save tracker so we can retrieve later
    tracker_data = {
        "job_name": batch_job.name,
        "submitted_at": datetime.now().isoformat(),
        "run_numbers": run_numbers,
        "total_requests": total_requests,
        "model": MODEL_NAME,
    }
    save_tracker(tracker_data)

    print(f"\n" + "=" * 60)
    print(f"  Batch job submitted successfully!")
    print(f"  Job name: {batch_job.name}")
    print(f"  Runs:     {run_numbers}")
    print(f"  Requests: {total_requests}")
    print(f"")
    print(f"  You can safely close this terminal now.")
    print(f"  The job runs on Google's servers (typically < 24 hrs).")
    print(f"")
    print(f"  To check status:    python run_ratings_google_batch.py status")
    print(f"  To get results:     python run_ratings_google_batch.py retrieve")
    print(f"=" * 60)


# ============================================================================
# MODE: status
# ============================================================================

def cmd_status():
    """Check the status of a pending batch job."""
    print("=" * 60)
    print("  Occupational AI Rating — Batch Job STATUS")
    print("=" * 60)

    tracker = load_tracker()
    if not tracker:
        print("\n  No pending batch job found.")
        print(f"  (Looked for {BATCH_TRACKER_FILE})")
        return

    print(f"\n  Job name:    {tracker['job_name']}")
    print(f"  Submitted:   {tracker['submitted_at']}")
    print(f"  Runs:        {tracker['run_numbers']}")
    print(f"  Requests:    {tracker['total_requests']}")

    client = get_client()
    batch_job = client.batches.get(name=tracker["job_name"])
    state_str = get_batch_state(batch_job)

    print(f"  Status:      {state_str}")

    if "SUCCEEDED" in state_str:
        print(f"\n  Job is DONE! Run 'retrieve' to download results:")
        print(f"    python run_ratings_google_batch.py retrieve")
    elif "FAIL" in state_str or "CANCEL" in state_str:
        print(f"\n  Job ended with error state: {state_str}")
        if hasattr(batch_job, 'error') and batch_job.error:
            print(f"  Error: {batch_job.error}")
    else:
        print(f"\n  Still processing. Check again later.")


# ============================================================================
# MODE: retrieve
# ============================================================================

def cmd_retrieve():
    """Download results from a completed batch job and save to run files."""
    print("=" * 60)
    print("  Occupational AI Rating — RETRIEVE Batch Results")
    print("=" * 60)

    tracker = load_tracker()
    if not tracker:
        print("\n  No pending batch job found.")
        print(f"  (Looked for {BATCH_TRACKER_FILE})")
        print(f"  Submit a job first:  python run_ratings_google_batch.py submit")
        return

    print(f"\n  Job name:    {tracker['job_name']}")
    print(f"  Submitted:   {tracker['submitted_at']}")
    print(f"  Runs:        {tracker['run_numbers']}")

    client = get_client()

    # Check status
    batch_job = client.batches.get(name=tracker["job_name"])
    state_str = get_batch_state(batch_job)
    print(f"  Status:      {state_str}")

    if "FAIL" in state_str or "CANCEL" in state_str:
        print(f"\n  ERROR: Batch job ended with state: {state_str}")
        if hasattr(batch_job, 'error') and batch_job.error:
            print(f"  Error: {batch_job.error}")
        answer = input("\n  Clear this job from tracker? (y/n): ").strip().lower()
        if answer == "y":
            clear_tracker()
            print("  Tracker cleared.")
        return

    if "SUCCEEDED" not in state_str:
        print(f"\n  Job is still running ({state_str}).")
        answer = input("  Wait for it to complete? (y/n): ").strip().lower()
        if answer == "y":
            print(f"  Polling every {POLL_INTERVAL}s…\n")
            batch_job = poll_batch_job(client, tracker["job_name"])
            state_str = get_batch_state(batch_job)
            if "SUCCEEDED" not in state_str:
                print(f"\n  Job did not succeed. State: {state_str}")
                return
        else:
            print("  Come back later!")
            return

    # Download results
    print("\nDownloading results…")
    raw_results = retrieve_file_results(client, batch_job)
    print(f"  Retrieved {len(raw_results)} results.")

    # Load occupations for metadata
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        occupations = json.load(f)

    # Process and save
    print("\nProcessing and saving…")
    process_and_save_results(
        raw_results,
        occupations,
        tracker["run_numbers"],
    )

    # Clear tracker
    clear_tracker()

    print(f"\n" + "=" * 60)
    print(f"  All results saved! Tracker cleared.")
    print(f"  Runs saved: {tracker['run_numbers']}")
    print(f"=" * 60)


# ============================================================================
# MODE: legacy (no args — submit + wait + retrieve)
# ============================================================================

def cmd_legacy():
    """Original behaviour: submit, poll until done, retrieve — all in one go."""
    print("=" * 60)
    print("  Occupational AI Rating — Batch API (full run)")
    print("  Tip: Use 'submit' mode to close the terminal while it runs!")
    print("=" * 60)

    # Submit a single run
    client = get_client()
    print("  Client ready.\n")

    scales_info = parse_scales_file(SCALES_FILE)
    prompt_template = load_text_file(PROMPT_FILE)
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        occupations = json.load(f)
    print(f"  {len(occupations)} occupations loaded")

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    incomplete = find_incomplete_run(OUTPUT_FOLDER, len(occupations))
    if incomplete is not None:
        run_num = incomplete
        print(f"  Resuming incomplete run_{run_num}")
    else:
        run_num = next_run_number(OUTPUT_FOLDER)

    json_path = os.path.join(OUTPUT_FOLDER, f"run_{run_num}.json")
    existing_codes, _ = get_existing_codes(json_path)

    to_process = {
        code: data for code, data in occupations.items()
        if code not in existing_codes
    }

    if not to_process:
        print("  All occupations already processed. Nothing to do.")
        return

    print(f"  Run {run_num}: {len(to_process)} occupations to process\n")

    run_occupations = {run_num: to_process}
    batch_requests = build_batch_requests_multi(run_occupations, scales_info, prompt_template)

    display_name = f"occ-ratings-run{run_num}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    start_time = time.time()
    batch_job = submit_batch_file(client, batch_requests, display_name)

    print(f"\n  Polling for completion (every {POLL_INTERVAL}s)…\n")
    final_job = poll_batch_job(client, batch_job.name)

    state_str = get_batch_state(final_job)
    if "SUCCEEDED" not in state_str:
        print(f"\n  ERROR: Batch job ended with state: {state_str}")
        sys.exit(1)

    raw_results = retrieve_file_results(client, final_job)
    print(f"  Retrieved {len(raw_results)} results.")

    process_and_save_results(raw_results, occupations, [run_num])

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  Done in {mins}m {secs}s")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Occupational AI Rating — Google Batch API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ratings_google_batch.py submit              Submit next incomplete run
  python run_ratings_google_batch.py submit --runs 5     Submit 5 runs in one batch
  python run_ratings_google_batch.py status              Check if the job is done
  python run_ratings_google_batch.py retrieve             Download results and save

  python run_ratings_google_batch.py                     Legacy: submit + wait + retrieve
        """,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["submit", "status", "retrieve"],
        default=None,
        help="Operation mode (omit for legacy submit+wait+retrieve)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of runs to submit in a single batch (default: 1)",
    )

    args = parser.parse_args()

    if args.mode == "submit":
        cmd_submit(args.runs)
    elif args.mode == "status":
        cmd_status()
    elif args.mode == "retrieve":
        cmd_retrieve()
    else:
        cmd_legacy()


if __name__ == "__main__":
    main()
