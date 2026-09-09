#!/usr/bin/env python3
"""
Patch a run file whose ratings are missing one or more scales.
================================================================
Re-sends the EXACT same prompt (same model, temperature, settings as the
pipeline) for each occupation with missing scales, then copies ONLY the
missing scales' ratings into the run JSON and rebuilds the Excel with the
pipeline's own save function. Already-rated scales are never touched.

Usage:
    python fix_missing_scales.py                       (default: mistral run 1)
    python fix_missing_scales.py mistral 1             (provider folder, run number)
"""
import json, os, sys
import importlib.util

spec = importlib.util.spec_from_file_location("rr", "run_ratings_44.py")
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)

PROVIDER_CFG = {
    "mistral": ("mistral", "mistral-large-2512", "Mistral"),
    "claude":  ("anthropic", "claude-opus-4-6", "Claude"),
    "chatgpt": ("openai", "gpt-5.2-2025-12-11", "ChatGPT"),
    "gemini":  ("google", "gemini-3.1-pro-preview", "Gemini"),
}

folder = sys.argv[1] if len(sys.argv) > 1 else "mistral"
run_num = sys.argv[2] if len(sys.argv) > 2 else "1"
provider, model_name, prefix = PROVIDER_CFG[folder]

base = os.path.join(rr.OUTPUT_ROOT, folder, f"{prefix}_run_{run_num}_44")
json_path, excel_path = base + ".json", base + ".xlsx"

with open(json_path, encoding="utf-8") as f:
    results = json.load(f)
with open(rr.INPUT_JSON_FILE, encoding="utf-8") as f:
    occupations = json.load(f)
scales_info = rr.parse_scales_file(rr.SCALES_FILE)
prompt_template = rr.load_text_file(rr.PROMPT_FILE)

def missing_scales(res):
    return [s for s in rr.SCALE_NAMES
            if not isinstance(res.get("ratings", {}).get(s, {}).get("level"), int)]

todo = [(i, r) for i, r in enumerate(results) if missing_scales(r)]
if not todo:
    print("Nothing to fix — every scale in this run has a rating."); sys.exit(0)

# --- build client exactly like the pipeline does ---
if provider == "mistral":
    try: from mistralai import Mistral
    except ImportError: from mistralai.client import Mistral
    client = Mistral(api_key=rr.load_api_key("MISTRAL_API_KEY.txt"))
elif provider == "anthropic":
    from anthropic import Anthropic
    client = Anthropic(api_key=rr.load_api_key("ANTHROPIC_API_KEY.txt"))
elif provider == "openai":
    from openai import OpenAI
    client = OpenAI(api_key=rr.load_api_key("OPENAI_API_KEY.txt"))
else:
    import google.generativeai as genai
    genai.configure(api_key=rr.load_api_key("GOOGLE_API_KEY.txt"), transport="rest")
    client = genai.GenerativeModel(model_name)

for i, res in todo:
    code = res["onet_soc_code"]
    miss = missing_scales(res)
    print(f"[{i}] {res['occupation_title']} ({code}) — missing: {', '.join(miss)}")
    prompt = rr.create_rating_prompt(code, occupations[code], scales_info, prompt_template)
    for attempt in range(1, rr.MAX_RETRIES + 1):
        try:
            parsed = rr.parse_llm_response(
                rr.query_llm_sync(provider, model_name, client, prompt))
            new = parsed.get("ratings", {})
            still = [s for s in miss if not isinstance(new.get(s, {}).get("level"), int)]
            if still:
                raise ValueError(f"response still missing {still}")
            for s in miss:                       # fill ONLY the missing scales
                res["ratings"][s] = new[s]
                print(f"    filled {s}: Level {new[s]['level']}")
            break
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}")
            if attempt == rr.MAX_RETRIES:
                print("    GIVING UP on this occupation — file left unchanged for it.")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
rr.save_excel([rr.result_to_row(r) for r in results], excel_path)
print(f"\nSaved: {json_path}\nSaved: {excel_path}")
