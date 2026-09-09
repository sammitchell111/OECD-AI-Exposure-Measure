# Occupational Ratings

LLM-based rating of occupations (and, in one extension, work activities) on the nine OECD AI Capability Indicators. This is the code, prompts and raw model output behind the LLM methodology used in the [OECD AI Exposure Measure](https://doi.org/10.1787/f3da0f0a-en), and the material for the accompanying methods paper (*Anchoring AI Exposure in Measured AI Capabilities*, OECD working paper, in preparation).

The short version: humans rated 40 occupations on nine capability scales; we asked LLMs to rate the same 40; where the LLM ratings fall within the range of human disagreement, we extend the rating to all 878 O*NET occupations. Everything in this repository exists to make that step transparent, repeatable and auditable.

## What is being rated, and against what

Each occupation is rated on the nine OECD AI Capability Indicators, each on a 0–5 scale where 0 means the capability is not required at all:

1. Language
2. Social Interaction
3. Problem Solving
4. Creativity
5. Metacognition and critical thinking
6. Knowledge, learning and memory
7. Vision
8. Manipulation
9. Robotic Intelligence

The full level descriptions for all nine scales are in `scales_long.txt` (the same file is copied into every experiment folder). The instruction to the model is to rate the *minimum capability level required to carry out the occupation as it is currently performed*, not importance, not frequency, and not what the job might look like after automation. Level descriptions are written in terms of AI systems; the model is told to read "AI systems or robots" as "Workers".

Occupation information comes from O*NET: code, title, description and the full task list. The compiled source file has 879 occupations; the run dictionary has 878 (Medical Dosimetrists, 29-2036.00, is the difference — see open items).

## Repository layout

```
Occupational Ratings/
├── O*NET ratings/                  ← the main study
│   ├── O*NET 878 ratings/          ← 8 models × 5 runs × 878 occupations (canonical analysis set)
│   └── O*NET 44 extra/             ← 44 additional occupations, 4 flagship models × 5 runs
├── IWA ratings/                    ← same method applied to O*NET Intermediate Work Activities
│   ├── IWA ratings (34 curated)/
│   └── IWA ratings (332 full)/
└── Experiments/                    ← not part of the paper's analysis set
    └── Skills and abilities ratings/   ← occupation rating with O*NET skills/abilities added to the prompt
```

Every experiment folder follows the same pattern:

- `Scripts and materials/` — the run script(s), the prompt template (`rating_instructions_*.txt`), the scales (`scales_long.txt`), the input dictionary (`*_dict*.json`) and a `HOW_TO_RUN.txt` specific to that experiment.
- `Results/<model>/run_N.json` — one JSON file per model per run, with the parsed ratings, the model's reasoning per scale, the model snapshot ID and a timestamp for every item.

The Excel versions of the results (`run_N.xlsx`, sitting next to each JSON) are derived from the JSON and are *not* tracked in git for now, to keep the repository a sensible size. They can be regenerated from the JSON.

## The main study: O*NET 878 ratings

The canonical analysis set is 8 models — a flagship and a lightweight option from each of four families — with 5 complete, independent runs each over all 878 occupations. That is 40 run files, 35,120 occupation-ratings and 316,080 scale-level ratings.

| Family    | Tier        | Model snapshot                 | Results folder       | Runs |
|-----------|-------------|--------------------------------|----------------------|------|
| Anthropic | Flagship    | `claude-opus-4-6`              | `claude`             | 5    |
| Anthropic | Lightweight | `claude-haiku-4-5-20251001`    | `claude_haiku`       | 5    |
| OpenAI    | Flagship    | `gpt-5.2-2025-12-11`           | `chatgpt`            | 5    |
| OpenAI    | Lightweight | `gpt-5.4-mini-2026-03-17`      | `chatgpt_mini`       | 5    |
| Google    | Flagship    | `gemini-3.1-pro-preview`       | `gemini`             | 5 (run 6 excluded) |
| Google    | Lightweight | `gemini-3.1-flash-lite-preview`| `gemini_flash_lite`  | 5    |
| Mistral   | Flagship    | `mistral-large-2512`           | `mistral`            | 5    |
| Mistral   | Lightweight | `mistral-small-2603`           | `mistral_small`      | 5    |

Two folders are kept for completeness but excluded from analysis: `gemini_flash` (a single run of `gemini-3-flash-preview`, superseded by flash-lite) and `mistral_large` (two early runs under the `mistral-large-latest` alias, superseded by the pinned `mistral` folder). `gemini/run_6.json` is a partial extra run (20 occupations) and is also excluded.

`Results/10_occupations_ratings/` holds 5 runs × 4 flagship models on the 10 occupations in `10_additional_occupations.json`, rated separately from the main 878. `O*NET 44 extra/` does the same for a further 44 occupations. Both exist because of the reconciliation between the 2015 and 2019 SOC codes in the O*NET data; the exact narrative is being confirmed and will be documented in the paper.

### Scripts in `O*NET 878 ratings/Scripts and materials/`

- `run_ratings.py` — the pipeline for the four flagship models. Interactive provider selection, async requests (5 concurrent by default, `MAX_CONCURRENT` at the top of the file), incremental save after every batch, resume-from-checkpoint if interrupted. This is the reference implementation; every other `run_ratings_*.py` in the repository is a variant of it.
- `run_ratings_lightweight.py` — identical pipeline for the four lightweight models; output files carry the model name.
- `run_ratings_10.py` — same pipeline, restricted to the 10 additional occupations.
- `run_ratings_google_batch.py`, `retrieve_batch_results.py`, `check_batch_jobs.py` — Gemini Batch API route, used to get round per-minute rate limits: submit, close the terminal, come back and retrieve.
- `OccupationalRating_Mar17.ipynb` — the original Colab notebook the pipeline grew out of. Kept for the record; the `.py` scripts are the versions that produced the results.
- `rating_instructions_mar17.txt` — the prompt template. `mar17` is the prompt version date (17 March 2026) and is the version behind all results in this folder.

### What one request looks like

One occupation per API request, all nine scales in a single completion. The prompt contains the O*NET code, title, description, full task list, the complete level descriptions for all nine scales, general instructions, and scale-specific guidance (e.g. Vision and Manipulation are 0 where they are not essential, with worked examples: Dentists get some Creativity, Accountants get no Manipulation). The model is asked for JSON: a `level` and a short task-referenced `reasoning` per scale. Temperature is 0.2 where the provider allows it to be set; `max_tokens` 2000.

Rating one scale at a time would arguably be cleaner but is materially more expensive, since the scales text dominates the input tokens.

### Result file format

Each `run_N.json` is a list with one entry per occupation:

```json
{
  "onet_soc_code": "11-1011.00",
  "occupation_title": "Chief Executives",
  "description": "...",
  "num_tasks": 31,
  "ratings": {
    "Scale 1: Language": {"level": 5, "reasoning": "..."},
    "...": {},
    "Scale 9: Robotic Intelligence": {"level": 0, "reasoning": "..."}
  },
  "model_used": "claude-opus-4-6",
  "timestamp": "2026-03-18T11:05:05.073600"
}
```

Audit as of June 2026: every one of the 40 canonical run files contains exactly the same 878 occupations, no duplicates, none missing. Three entries out of 35,120 are incomplete, all from `mistral-large-2512`, each a truncated response that lost trailing scales: run 1 Biological Technicians (19-4021.00, 3 of 9 scales), run 4 Bailiffs (33-3011.00, 6 of 9), run 5 Rail Yard Engineers (53-4013.00, 6 of 9). Whether these are treated as missing or re-run is an open item. `O*NET 44 extra/Scripts and materials/fix_missing_scales.py` is the tool for the second option: it re-sends the identical prompt and patches in only the missing scales.

## The other folders

**IWA ratings.** The same nine-scale rating applied to O*NET Intermediate Work Activities rather than occupations. An IWA sits above individual tasks and is performed across many occupations, so the prompt (`rating_instructions_iwa.txt`) gives the model the parent GWA, linked DWAs, sampled tasks and the occupations the IWA appears in, and asks it to rate the activity as a whole. `34 curated` is the hand-picked selection (`Final IWAs selection for LLM.xls`); `332 full` is the complete set. Claude Opus only so far, one run each.

**Experiments/Skills and abilities ratings.** The occupation prompt with one addition: the 10 most important O*NET skills and abilities for the occupation, with their importance scores, as complementary evidence alongside the task list. Run on the 40-occupation subset for all four flagship models, and on the full set for Claude, GPT and Mistral (`run_1`; the GPT and Mistral full runs stopped a few occupations short). Kept as an experiment to see whether the extra structured input changes anything.

## Running it yourself

Python 3.9 or later. Install the dependencies:

```
pip install -r requirements.txt
```

Each `Scripts and materials/` folder expects four plain-text files next to the script, each containing nothing but the raw key:

```
ANTHROPIC_API_KEY.txt
OPENAI_API_KEY.txt
GOOGLE_API_KEY.txt
MISTRAL_API_KEY.txt
```

These are deliberately not in the repository (`.gitignore` excludes `*_API_KEY.txt`). Create the ones you need; if a file is missing or empty the script asks for the key at runtime, so you only need the providers you actually plan to run.

Then, from inside the relevant `Scripts and materials/` folder:

```
python run_ratings.py
```

Pick a provider when prompted. Results go to `<provider>/run_N.json` and `.xlsx`, saved after every batch, so an interrupted run loses at most five occupations; run the script again with the same provider and it picks up where it left off. A full 878-occupation run at concurrency 5 takes roughly 20–40 minutes depending on the provider's rate limits. Each folder's `HOW_TO_RUN.txt` has the details specific to that experiment.

## Why it is done this way

A few decisions that are worth stating because they are the difference between a result you can defend and one you cannot.

The API rather than a chat interface, so that every occupation gets the identical prompt with no conversational context leaking between requests, raw responses are logged, and the run can be resumed after an interruption. Five full runs per model rather than one, because across-run consistency is as much a finding as accuracy is. Deterministic parsing and automatic checks on counts and scale completeness, which is how the three truncated Mistral responses were caught; a chat-only workflow would almost certainly not have surfaced them. And the human comparison first, at small scale, before extending to 878, with the intention to revalidate whenever either the prompt or the model changes. Validation is cheap relative to being wrong at scale.

Honest note on provenance: the lead author is not a professional programmer. The pipeline was built iteratively with successive versions of Claude over several months, starting in Google Colab and moving to a local Python set-up. What made it work was less the code than having a clear plan, knowing what the output should look like, and checking it.

## Open items

- Confirm why Medical Dosimetrists (29-2036.00) is absent from the 878 run dictionary despite complete data; document or reinstate.
- Confirm the 2015 vs 2019 SOC reconciliation story and the role of the 10 and 44 additional occupations.
- Decide how to handle the three truncated Mistral entries (missing data vs re-run with `fix_missing_scales.py`).
- Token-based cost estimates per model per run.
- Human ratings for the 40 comparison occupations and the analysis outputs are held separately and are not in this repository yet.

## Citation

The measure these ratings feed into is published here; please cite it if you use the ratings or the pipeline:

> OECD (2026), *The OECD AI exposure measure: Mapping the OECD AI Capability Indicators to occupations*, OECD Artificial Intelligence Papers, OECD Publishing, Paris, https://doi.org/10.1787/f3da0f0a-en

The methods paper covering the material in this repository in detail is in preparation; the citation will be added here when it is out.
