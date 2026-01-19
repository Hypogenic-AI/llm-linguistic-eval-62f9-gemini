import os
from datasets import load_dataset
import openai

def check_env():
    print("Checking environment variables...")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    
    if openrouter_key:
        print("OPENROUTER_API_KEY is found.")
    else:
        print("OPENROUTER_API_KEY is NOT found.")
        
    if openai_key:
        print("OPENAI_API_KEY is found.")
    else:
        print("OPENAI_API_KEY is NOT found.")

def check_dataset():
    print("\nChecking MuBench dataset access...")
    try:
        # Try loading English and Chinese as a test
        ds_en = load_dataset("aialt/MuBench", "MMLUDataset_en_template_en", split="test", streaming=True)
        print("Successfully loaded English config.")
        ds_zh = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test", streaming=True)
        print("Successfully loaded Chinese config.")
        
        print("Sample English entry:", next(iter(ds_en)))
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    check_env()
    check_dataset()
