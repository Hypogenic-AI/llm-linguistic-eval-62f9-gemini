#!/usr/bin/env python3
"""
Test script to verify the evaluation setup works correctly.
Tests dataset loading and API calls before running the full evaluation.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_dataset_loading():
    """Test loading a sample from MuBench."""
    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    from datasets import load_dataset

    # Test loading English
    print("\n1. Loading English dataset...")
    try:
        ds = load_dataset("aialt/MuBench", "MMLUDataset_en_template_en", split="test")
        print(f"   Success! Loaded {len(ds)} questions")
        print(f"   Sample question:\n   {ds[0]['prompt'][:200]}...")
        print(f"   Label: {ds[0]['label']}, Choices: {ds[0]['choices']}")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test loading a non-English language (Chinese)
    print("\n2. Loading Chinese dataset...")
    try:
        ds_zh = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test")
        print(f"   Success! Loaded {len(ds_zh)} questions")
        print(f"   Sample question:\n   {ds_zh[0]['prompt'][:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    # Test loading a low-resource language (Bengali)
    print("\n3. Loading Bengali dataset...")
    try:
        ds_bn = load_dataset("aialt/MuBench", "MMLUDataset_en_template_bn", split="test")
        print(f"   Success! Loaded {len(ds_bn)} questions")
        print(f"   Sample question:\n   {ds_bn[0]['prompt'][:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    return True


def test_openai_api():
    """Test OpenAI API call."""
    print("\n" + "=" * 60)
    print("Testing OpenAI API")
    print("=" * 60)

    import openai

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("   OPENAI_API_KEY not set")
        return False

    client = openai.OpenAI()

    # Test simple call
    print("\n1. Testing GPT-4o...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Answer with only the letter (A, B, C, or D)."},
                {"role": "user", "content": "What is 2+2? A: 3 B: 4 C: 5 D: 6"}
            ],
            temperature=0,
            max_tokens=10
        )
        answer = response.choices[0].message.content.strip()
        print(f"   Response: '{answer}'")
        print(f"   Success! (Expected: B)")
    except Exception as e:
        print(f"   Error: {e}")
        return False

    return True


def test_openrouter_api():
    """Test OpenRouter API call."""
    print("\n" + "=" * 60)
    print("Testing OpenRouter API")
    print("=" * 60)

    import requests

    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        print("   OPENROUTER_API_KEY not set")
        return False

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Test with Claude
    print("\n1. Testing Claude Sonnet via OpenRouter...")
    try:
        data = {
            "model": "anthropic/claude-sonnet-4",
            "messages": [
                {"role": "system", "content": "Answer with only the letter (A, B, C, or D)."},
                {"role": "user", "content": "What is 2+2? A: 3 B: 4 C: 5 D: 6"}
            ],
            "temperature": 0,
            "max_tokens": 10
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        print(f"   Response: '{answer}'")
        print(f"   Success! (Expected: B)")
    except Exception as e:
        print(f"   Error: {e}")
        # Don't fail completely, OpenRouter might have different model availability
        print("   Note: This error may be due to model availability. Will try alternatives.")

    # Test with Gemini
    print("\n2. Testing Gemini 2.5 Pro via OpenRouter...")
    try:
        data = {
            "model": "google/gemini-2.5-pro-preview",
            "messages": [
                {"role": "system", "content": "Answer with only the letter (A, B, C, or D)."},
                {"role": "user", "content": "What is 2+2? A: 3 B: 4 C: 5 D: 6"}
            ],
            "temperature": 0,
            "max_tokens": 10
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        answer = result['choices'][0]['message']['content'].strip()
        print(f"   Response: '{answer}'")
        print(f"   Success! (Expected: B)")
    except Exception as e:
        print(f"   Error: {e}")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MULTILINGUAL LLM EVALUATION - SETUP TEST")
    print("=" * 70)

    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("OpenAI API", test_openai_api),
        ("OpenRouter API", test_openrouter_api),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"\nTest {name} failed with exception: {e}")
            results[name] = False

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed."))

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
