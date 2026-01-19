import os
import json
import time
import argparse
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import re

# Configuration
LANGUAGES = {
    # High Resource
    "en": "MMLUDataset_en_template_en",
    "zh": "MMLUDataset_en_template_zh",
    "es": "MMLUDataset_en_template_es",
    # Mid Resource
    "fr": "MMLUDataset_en_template_fr",
    "ar": "MMLUDataset_en_template_ar",
    "id": "MMLUDataset_en_template_id",
    "vi": "MMLUDataset_en_template_vi",
    "ko": "MMLUDataset_en_template_ko",
    # Low Resource
    "sw": "MMLUDataset_en_template_sw",
    "bn": "MMLUDataset_en_template_bn",
    "te": "MMLUDataset_en_template_te",
    "ta": "MMLUDataset_en_template_ta"
}

MODEL = "gpt-4o"
MAX_RETRIES = 3
LIMIT = 50  # Samples per language

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL") # Support for custom base url if needed
    
    # If using OpenRouter via openai lib, usually base_url needs to be set if not default
    # The environment variable OPENROUTER_API_KEY might be present.
    # If OPENAI_API_KEY is present, we assume it works.
    
    return OpenAI(api_key=api_key, base_url=base_url)

def parse_response(response_text: str) -> str:
    # Look for single letter A, B, C, D
    # Priority:
    # 1. "Answer: X"
    # 2. Last single letter in text
    
    match = re.search(r"Answer:\s*([A-D])", response_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: find all A-D occurrences, take last one that looks like an answer
    matches = re.findall(r"\b([A-D])\b", response_text)
    if matches:
        return matches[-1]
    
    return "Unknown"

def evaluate_language(lang_code: str, config_name: str, client: OpenAI, limit: int):
    print(f"Evaluating language: {lang_code} ({config_name})")
    
    try:
        ds = load_dataset("aialt/MuBench", config_name, split="test", streaming=True)
    except Exception as e:
        print(f"Failed to load dataset for {lang_code}: {e}")
        return

    results = []
    iterator = iter(ds)
    
    for _ in tqdm(range(limit), desc=f"Processing {lang_code}"):
        try:
            item = next(iterator)
        except StopIteration:
            break
            
        prompt = item['prompt']
        label_idx = item['label']
        label_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        ground_truth = label_map.get(label_idx, "Unknown")
        
        # Call API
        response_text = ""
        for attempt in range(MAX_RETRIES):
            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=MODEL,
                    temperature=0.0, # Deterministic
                )
                response_text = chat_completion.choices[0].message.content
                break
            except Exception as e:
                print(f"API Error (attempt {attempt+1}): {e}")
                time.sleep(2)
        
        prediction = parse_response(response_text)
        
        results.append({
            "id": item['_id'],
            "lang": lang_code,
            "prompt": prompt,
            "response": response_text,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": prediction == ground_truth
        })
    
    # Save results
    output_dir = "results/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{MODEL}_{lang_code}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Calculate accuracy
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    acc = correct / total if total > 0 else 0
    print(f"Completed {lang_code}. Accuracy: {acc:.2f} ({correct}/{total})")

def main():
    client = get_client()
    
    # Run evaluation for all languages
    # Selecting subset to fit time constraints if needed, but 50 samples * 12 langs = 600 calls is fine.
    
    target_langs = list(LANGUAGES.keys())
    
    for lang in target_langs:
        config = LANGUAGES[lang]
        evaluate_language(lang, config, client, LIMIT)

if __name__ == "__main__":
    main()