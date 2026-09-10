#!/usr/bin/env python3
"""
Quick utility to list all your Gemini batch jobs and their statuses.
Helps you find the job name from a previous run so you can retrieve results.

NOTE: For the IWA pipeline (34 IWAs total) you usually do NOT need batch
mode — the regular `run_ratings_IWA.py` with concurrent calls finishes in
a few minutes. This script is kept for parity with the occupation pipeline
in case you ever do submit a Gemini batch job.
"""

import os
import sys
import getpass

try:
    from google import genai
except ImportError:
    print("Please install google-genai:  pip install google-genai")
    sys.exit(1)


def load_api_key(filename="GOOGLE_API_KEY.txt"):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    return None


def main():
    api_key = load_api_key() or getpass.getpass("Enter your Google API Key: ")
    client = genai.Client(api_key=api_key)

    print("\nListing all batch jobs:\n")
    print(f"{'Job Name':<50} {'State':<25} {'Display Name'}")
    print("-" * 110)

    for job in client.batches.list():
        name = job.name or "?"
        state = job.state.name if hasattr(job.state, 'name') else str(job.state)
        display = getattr(job, 'display_name', '') or ''
        print(f"{name:<50} {state:<25} {display}")

    print("\n" + "-" * 110)
    print("\nTo retrieve results from a completed job, copy the job name and run:")
    print("  python retrieve_batch_results.py <job_name>")


if __name__ == "__main__":
    main()
