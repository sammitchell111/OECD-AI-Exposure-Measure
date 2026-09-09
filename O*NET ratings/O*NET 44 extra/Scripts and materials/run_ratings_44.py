#!/usr/bin/env python3
"""
Occupational AI Capability Rating Pipeline — 44 Extra Occupations
==================================================================
Identical to run_ratings.py / run_ratings_10.py but operates on the set of
44 extra occupations (44_extra_occupations.json).

Features:
  - Interactive LLM selection at startup (Anthropic / OpenAI / Google / Mistral)
  - Uses only the most advanced models (same as run_ratings.py)
  - Async concurrent API calls (configurable concurrency)
  - Incremental Excel save after every API response
  - Resume-from-checkpoint: re-running skips already-rated occupations
  - No "Level Description" columns in the output
  - Outputs to 44_occupations_ratings/<model_folder>/ with files named
    <Model>_run_N_44.json / .xlsx

Usage:
    python run_ratings_44.py
"""

import asyncio
import json
import os
import sys
import time
import getpass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
INPUT_JSON_FILE = "44_extra_occupations.json"
# All results are read from / written to the sibling "Results" folder
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Results")
SCALES_FILE = "scales_long.txt"
PROMPT_FILE = "rating_instructions_mar17.txt"

# Top-level output directory (contains one sub-folder per model)
OUTPUT_ROOT = os.path.join(RESULTS_DIR, "44_occupations_ratings")

# Concurrency: how many API calls to run in parallel.
# Adjust based on your API tier / rate limits.
# 5 is conservative and works for most plans; raise to 10-15 if your plan allows.
MAX_CONCURRENT = 5

# Maximum retries per occupation on transient errors
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Text helpers
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
# Prompt construction (identical logic to run_ratings.py / notebook)
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

    # Format scales
    scales_text = ""
    for scale_name, levels in scales_info.items():
        scales_text += f"\n{scale_name}\n"
        for level, desc in levels.items():
            scales_text += f"  {level}: {desc}\n"

    # Format tasks
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
# LLM provider selection & API key loading
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


def select_provider():
    """Interactive provider selection. Returns (provider, model_name, client, friendly_name)."""
    print("\nSelect which LLM provider to use:")
    print("  1. Anthropic  (Claude Opus 4.6)")
    print("  2. OpenAI     (GPT-5.2)")
    print("  3. Google     (Gemini 3.1 Pro Preview)")
    print("  4. Mistral    (Mistral 3 Large)")
    print()

    choice = input("Enter number (1-4): ").strip()

    if choice == "1":
        from anthropic import Anthropic

        print("\n--- Selected: ANTHROPIC ---")
        api_key = load_api_key("ANTHROPIC_API_KEY.txt") or getpass.getpass(
            "Enter your Anthropic API Key: "
        )
        client = Anthropic(api_key=api_key)
        print("Ready: ANTHROPIC (claude-opus-4-6)\n")
        return "anthropic", "claude-opus-4-6", client, "Claude"

    elif choice == "2":
        from openai import OpenAI

        print("\n--- Selected: OPENAI ---")
        api_key = load_api_key("OPENAI_API_KEY.txt") or getpass.getpass(
            "Enter your OpenAI API Key: "
        )
        client = OpenAI(api_key=api_key)
        print("Ready: OPENAI (gpt-5.2-2025-12-11)\n")
        return "openai", "gpt-5.2-2025-12-11", client, "ChatGPT"

    elif choice == "3":
        import google.generativeai as genai

        print("\n--- Selected: GOOGLE ---")
        api_key = load_api_key("GOOGLE_API_KEY.txt") or getpass.getpass(
            "Enter your Google API Key: "
        )
        genai.configure(api_key=api_key, transport="rest")
        client = genai.GenerativeModel("gemini-3.1-pro-preview")
        print("Ready: GOOGLE (gemini-3.1-pro-preview)\n")
        return "google", "gemini-3.1-pro-preview", client, "Gemini"

    elif choice == "4":
        try:
            from mistralai import Mistral
        except ImportError:
            from mistralai.client import Mistral

        print("\n--- Selected: MISTRAL ---")
        api_key = load_api_key("MISTRAL_API_KEY.txt") or getpass.getpass(
            "Enter your Mistral API Key: "
        )
        client = Mistral(api_key=api_key)
        print("Ready: MISTRAL (mistral-large-2512)\n")
        return "mistral", "mistral-large-2512", client, "Mistral"

    else:
        print("Invalid selection. Exiting.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Synchronous LLM call (run in thread pool for async concurrency)
# ---------------------------------------------------------------------------

def query_llm_sync(provider: str, model_name: str, client, prompt: str) -> str:
    """Call the selected LLM synchronously. Returns raw response text."""
    if provider == "anthropic":
        resp = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    elif provider == "openai":
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    elif provider == "google":
        resp = client.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
        return resp.text

    elif provider == "mistral":
        # v1.x uses client.chat.complete(); v0.x uses client.chat()
        chat_fn = getattr(client.chat, "complete", None) or client.chat
        resp = chat_fn(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    else:
        raise ValueError(f"Unknown provider: {provider}")


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
# Excel incremental save
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
        # NOTE: "Level Description" columns are intentionally omitted
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
# Async processing engine
# ---------------------------------------------------------------------------

async def process_occupation(
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    provider: str,
    model_name: str,
    client,
    code: str,
    data: Dict,
    scales_info: Dict,
    prompt_template: str,
    idx: int,
    total: int,
) -> Dict:
    """Process a single occupation with concurrency control."""
    async with sem:
        prompt = create_rating_prompt(code, data, scales_info, prompt_template)
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Run the blocking API call in a thread
                response_text = await loop.run_in_executor(
                    None, query_llm_sync, provider, model_name, client, prompt
                )
                parsed = parse_llm_response(response_text)
                print(f"  [{idx}/{total}] {data.get('title', code)}")
                return {
                    "onet_soc_code": code,
                    "occupation_title": data.get("title"),
                    "description": data.get("description"),
                    "num_tasks": len(data.get("tasks", [])),
                    "ratings": parsed.get("ratings", {}),
                    "model_used": model_name,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                print(
                    f"  [{idx}/{total}] {data.get('title', code)} — "
                    f"attempt {attempt} failed ({e}), retrying in {wait}s…"
                )
                await asyncio.sleep(wait)

        # All retries exhausted
        print(f"  [{idx}/{total}] {data.get('title', code)} — FAILED after {MAX_RETRIES} attempts")
        return {
            "onet_soc_code": code,
            "occupation_title": data.get("title"),
            "description": data.get("description"),
            "num_tasks": len(data.get("tasks", [])),
            "ratings": {},
            "error": str(last_error),
            "model_used": model_name,
            "timestamp": datetime.now().isoformat(),
        }


async def run_pipeline(
    provider: str,
    model_name: str,
    client,
    occupations: Dict[str, Dict],
    scales_info: Dict,
    prompt_template: str,
    excel_path: str,
    json_path: str,
    existing_codes: set,
):
    """Main async pipeline: fan out API calls, save incrementally."""
    loop = asyncio.get_event_loop()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    # Filter out already-processed occupations (resume support)
    to_process = {
        code: data
        for code, data in occupations.items()
        if code not in existing_codes
    }

    if len(to_process) < len(occupations):
        print(
            f"Resuming: {len(occupations) - len(to_process)} already processed, "
            f"{len(to_process)} remaining.\n"
        )

    if not to_process:
        print("All occupations already processed. Nothing to do.")
        return

    # Load existing *successful* results into a dict keyed by SOC code.
    # This ensures only one entry per occupation and drops old error entries.
    results_by_code: Dict[str, Dict] = {}
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                code = r["onet_soc_code"]
                # Keep successful results; skip error entries
                if "error" not in r:
                    results_by_code[code] = r

    total = len(to_process)
    items = list(to_process.items())
    completed = 0

    print(f"Processing {total} occupations with concurrency={MAX_CONCURRENT}…\n")

    # We process in batches equal to MAX_CONCURRENT so we can save after each batch
    for batch_start in range(0, total, MAX_CONCURRENT):
        batch = items[batch_start : batch_start + MAX_CONCURRENT]
        tasks = []
        for i, (code, data) in enumerate(batch, start=batch_start + 1):
            tasks.append(
                process_occupation(
                    sem, loop, provider, model_name, client,
                    code, data, scales_info, prompt_template,
                    idx=i, total=total,
                )
            )

        results = await asyncio.gather(*tasks)

        # Merge results — only keep successful ones, one per code
        for result in results:
            if "error" not in result:
                results_by_code[result["onet_soc_code"]] = result

        # Rebuild flat list and save (always deduplicated)
        all_results_raw = list(results_by_code.values())
        all_rows = [result_to_row(r) for r in all_results_raw]

        save_excel(all_rows, excel_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results_raw, f, indent=2, ensure_ascii=False)

        completed += len(batch)
        print(f"  — saved ({len(all_results_raw)} total, {completed}/{total} this session)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Friendly folder names for each provider (used as sub-folders under OUTPUT_ROOT)
PROVIDER_FOLDER_NAMES = {
    "anthropic": "claude",
    "openai": "chatgpt",
    "google": "gemini",
    "mistral": "mistral",
}

# Friendly file-prefix names for each provider (used in output filenames)
PROVIDER_FILE_PREFIXES = {
    "anthropic": "Claude",
    "openai": "ChatGPT",
    "google": "Gemini",
    "mistral": "Mistral",
}


def find_incomplete_run(folder: str, total_occupations: int, file_prefix: str) -> Optional[int]:
    """Check if any existing run is incomplete (fewer results than total occupations)."""
    if not os.path.exists(folder):
        return None
    for name in sorted(os.listdir(folder)):
        if name.startswith(f"{file_prefix}_run_") and name.endswith("_44.json"):
            try:
                # Extract run number from e.g. "Claude_run_1_44.json"
                parts = name.replace(f"{file_prefix}_run_", "").replace("_44.json", "")
                num = int(parts)
                json_path = os.path.join(folder, name)
                with open(json_path, "r", encoding="utf-8") as f:
                    results = json.load(f)
                    successful = sum(1 for r in results if "error" not in r)
                    if successful < total_occupations:
                        return num
            except (ValueError, IndexError, json.JSONDecodeError):
                pass
    return None


def next_run_number(folder: str, file_prefix: str) -> int:
    """Scan folder for existing run files and return the next number."""
    if not os.path.exists(folder):
        return 1
    existing = []
    for name in os.listdir(folder):
        if name.startswith(f"{file_prefix}_run_") and "_44." in name:
            try:
                # Extract run number from e.g. "Claude_run_3_44.json"
                parts = name.replace(f"{file_prefix}_run_", "").split("_44.")[0]
                num = int(parts)
                existing.append(num)
            except (ValueError, IndexError):
                pass
    return max(existing, default=0) + 1


def main():
    print("=" * 60)
    print("  Occupational AI Capability Rating Pipeline — 44 Extra Occupations")
    print("=" * 60)

    # 1. Select LLM
    provider, model_name, client, friendly_name = select_provider()

    # 2. Load data
    print("Loading data…")
    scales_info = parse_scales_file(SCALES_FILE)
    prompt_template = load_text_file(PROMPT_FILE)

    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        occupations = json.load(f)
    print(f"  {len(occupations)} occupations loaded from {INPUT_JSON_FILE}")
    print(f"  {len(scales_info)} scales parsed from {SCALES_FILE}")

    # 3. Determine output folder and run number
    folder_name = os.path.join(OUTPUT_ROOT, PROVIDER_FOLDER_NAMES.get(provider, provider))
    os.makedirs(folder_name, exist_ok=True)

    file_prefix = PROVIDER_FILE_PREFIXES.get(provider, friendly_name)

    # Check for an incomplete run first; if found, resume it
    incomplete = find_incomplete_run(folder_name, len(occupations), file_prefix)
    if incomplete is not None:
        run_num = incomplete
        print(f"\n  Found incomplete {file_prefix}_run_{run_num}_44 — resuming it.")
    else:
        run_num = next_run_number(folder_name, file_prefix)

    excel_path = os.path.join(folder_name, f"{file_prefix}_run_{run_num}_44.xlsx")
    json_path = os.path.join(folder_name, f"{file_prefix}_run_{run_num}_44.json")

    print(f"  Output folder: {folder_name}/")
    print(f"  Run number:    {run_num}")
    print(f"  Files:         {file_prefix}_run_{run_num}_44.json / .xlsx")

    # 4. Check for existing checkpoint (resume support for this specific run)
    existing_codes: set = set()
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            prior = json.load(f)
            existing_codes = {r["onet_soc_code"] for r in prior if "error" not in r}
        print(f"  Found checkpoint: {len(existing_codes)} occupations already done")

    # 5. Run
    start_time = time.time()
    asyncio.run(
        run_pipeline(
            provider, model_name, client,
            occupations, scales_info, prompt_template,
            excel_path, json_path, existing_codes,
        )
    )
    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    print("=" * 60)
    print(f"  Complete!  {len(occupations)} occupations  |  {mins}m {secs}s elapsed")
    print(f"  Excel : {excel_path}")
    print(f"  JSON  : {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
