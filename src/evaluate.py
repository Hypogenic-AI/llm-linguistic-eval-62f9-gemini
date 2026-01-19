#!/usr/bin/env python3
"""
Multilingual LLM Evaluation Script

This script evaluates multiple LLMs across different languages using the MuBench MMLU dataset.
It measures accuracy and performance gaps across high, mid, and low-resource languages.
"""

import os
import json
import time
import random
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import openai
import requests

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Language configurations
LANGUAGES = {
    'high': ['en', 'zh', 'es', 'fr'],
    'mid': ['ar', 'ko', 'vi', 'id'],
    'low': ['bn', 'te', 'sw', 'ta']
}

ALL_LANGUAGES = LANGUAGES['high'] + LANGUAGES['mid'] + LANGUAGES['low']

# Model configurations - using available models from OpenRouter
MODELS = {
    'gpt-4o': {'provider': 'openai', 'model_id': 'gpt-4o'},
    'claude-sonnet-4.5': {'provider': 'openrouter', 'model_id': 'anthropic/claude-sonnet-4.5'},
    'gemini-3-flash': {'provider': 'openrouter', 'model_id': 'google/gemini-3-flash-preview'}
}

# Evaluation parameters
SAMPLE_SIZE = 100  # Questions per language (reduced for faster completion)
MAX_RETRIES = 3
RETRY_DELAY = 5


@dataclass
class EvaluationResult:
    """Store results for a single evaluation run."""
    model: str
    language: str
    resource_level: str
    total_questions: int
    correct: int
    accuracy: float
    responses: List[Dict]
    timestamp: str


def get_resource_level(lang: str) -> str:
    """Get resource level for a language."""
    for level, langs in LANGUAGES.items():
        if lang in langs:
            return level
    return 'unknown'


def load_mubench_data(language: str, sample_size: int = SAMPLE_SIZE) -> List[Dict]:
    """
    Load MuBench MMLU dataset for a specific language.

    Args:
        language: Language code (e.g., 'en', 'zh', 'ar')
        sample_size: Number of questions to sample

    Returns:
        List of question dictionaries
    """
    config_name = f"MMLUDataset_en_template_{language}"
    logger.info(f"Loading MuBench MMLU for language: {language} (config: {config_name})")

    try:
        dataset = load_dataset("aialt/MuBench", config_name, split="test")

        # Sample questions
        total_size = len(dataset)
        indices = random.sample(range(total_size), min(sample_size, total_size))
        samples = [dataset[i] for i in indices]

        logger.info(f"Loaded {len(samples)} questions for {language} (from {total_size} total)")
        return samples

    except Exception as e:
        logger.error(f"Failed to load dataset for {language}: {e}")
        raise


def format_prompt(question_data: Dict) -> str:
    """
    Format a question for LLM evaluation.

    The MuBench dataset already has formatted prompts, so we use them directly.
    """
    # MuBench format already includes the question and choices
    prompt = question_data['prompt']
    return prompt


def parse_answer(response: str) -> Optional[str]:
    """
    Parse the model's response to extract the answer letter.

    Args:
        response: Raw model response

    Returns:
        Answer letter (A, B, C, or D) or None if unparseable
    """
    response = response.strip().upper()

    # Direct answer
    if response in ['A', 'B', 'C', 'D']:
        return response

    # Common patterns
    patterns = [
        r'^([ABCD])\.',
        r'^([ABCD])\)',
        r'^([ABCD]):',
        r'^([ABCD])\s',
        r'answer\s*(?:is\s*)?([ABCD])',
        r'(?:^|\s)([ABCD])(?:\s|$|\.)',
    ]

    import re
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # First letter if it's A, B, C, or D
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]

    return None


def call_openai(prompt: str, model_id: str) -> str:
    """Call OpenAI API."""
    client = openai.OpenAI()

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant taking a multiple choice test. Answer with only the letter (A, B, C, or D) of the correct answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def call_openrouter(prompt: str, model_id: str) -> str:
    """Call OpenRouter API."""
    api_key = os.environ.get('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant taking a multiple choice test. Answer with only the letter (A, B, C, or D) of the correct answer."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 20  # Increased slightly to ensure we get a response
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            # Handle None or empty content
            if content is None:
                content = ""
            return content.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"OpenRouter API error (attempt {attempt + 1}): {e}")
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise


def evaluate_model_on_language(
    model_name: str,
    model_config: Dict,
    questions: List[Dict],
    language: str,
    progress_bar: Optional[tqdm] = None
) -> EvaluationResult:
    """
    Evaluate a single model on questions in a specific language.

    Args:
        model_name: Name of the model
        model_config: Model configuration dict
        questions: List of question dictionaries
        language: Language code
        progress_bar: Optional progress bar to update

    Returns:
        EvaluationResult with accuracy and detailed responses
    """
    provider = model_config['provider']
    model_id = model_config['model_id']

    responses = []
    correct = 0

    call_fn = call_openai if provider == 'openai' else call_openrouter

    for i, q in enumerate(questions):
        prompt = format_prompt(q)

        try:
            raw_response = call_fn(prompt, model_id)
            parsed_answer = parse_answer(raw_response)

            # MuBench uses 0-indexed label, choices are ["A", "B", "C", "D"]
            correct_answer = q['choices'][q['label']]
            is_correct = parsed_answer == correct_answer

            if is_correct:
                correct += 1

            responses.append({
                'question_id': q.get('_id', i),
                'prompt': prompt[:200] + '...' if len(prompt) > 200 else prompt,
                'raw_response': raw_response,
                'parsed_answer': parsed_answer,
                'correct_answer': correct_answer,
                'is_correct': is_correct
            })

        except Exception as e:
            logger.error(f"Error evaluating question {i} for {model_name}/{language}: {e}")
            responses.append({
                'question_id': q.get('_id', i),
                'error': str(e),
                'is_correct': False
            })

        if progress_bar:
            progress_bar.update(1)

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    accuracy = correct / len(questions) if questions else 0.0

    return EvaluationResult(
        model=model_name,
        language=language,
        resource_level=get_resource_level(language),
        total_questions=len(questions),
        correct=correct,
        accuracy=accuracy,
        responses=responses,
        timestamp=datetime.now().isoformat()
    )


def run_evaluation(
    models: Dict[str, Dict],
    languages: List[str],
    sample_size: int = SAMPLE_SIZE,
    save_path: str = 'results/'
) -> pd.DataFrame:
    """
    Run full evaluation across all models and languages.

    Args:
        models: Dictionary of model configurations
        languages: List of language codes
        sample_size: Questions per language
        save_path: Directory to save results

    Returns:
        DataFrame with evaluation results
    """
    # Load all datasets first
    logger.info("Loading datasets...")
    datasets = {}
    for lang in languages:
        try:
            datasets[lang] = load_mubench_data(lang, sample_size)
        except Exception as e:
            logger.error(f"Skipping language {lang}: {e}")
            continue

    # Calculate total iterations for progress bar
    total_iterations = sum(
        len(datasets.get(lang, []))
        for lang in languages
        for _ in models
    )

    results = []

    with tqdm(total=total_iterations, desc="Evaluating") as pbar:
        for model_name, model_config in models.items():
            logger.info(f"\nEvaluating model: {model_name}")

            for lang in languages:
                if lang not in datasets:
                    continue

                logger.info(f"  Language: {lang}")

                try:
                    result = evaluate_model_on_language(
                        model_name,
                        model_config,
                        datasets[lang],
                        lang,
                        pbar
                    )
                    results.append(result)

                    # Save intermediate results
                    result_path = Path(save_path) / 'raw' / f'{model_name}_{lang}.json'
                    with open(result_path, 'w') as f:
                        json.dump(asdict(result), f, indent=2)

                    logger.info(f"    Accuracy: {result.accuracy:.2%} ({result.correct}/{result.total_questions})")

                except Exception as e:
                    logger.error(f"  Failed to evaluate {model_name} on {lang}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame([{
        'model': r.model,
        'language': r.language,
        'resource_level': r.resource_level,
        'accuracy': r.accuracy,
        'correct': r.correct,
        'total': r.total_questions
    } for r in results])

    # Save summary
    df.to_csv(Path(save_path) / 'evaluation_results.csv', index=False)

    return df


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Multilingual LLM Evaluation Starting")
    logger.info(f"Seed: {SEED}")
    logger.info(f"Languages: {ALL_LANGUAGES}")
    logger.info(f"Models: {list(MODELS.keys())}")
    logger.info(f"Sample size per language: {SAMPLE_SIZE}")
    logger.info("="*60)

    # Ensure results directory exists
    Path('results/raw').mkdir(parents=True, exist_ok=True)

    # Run evaluation
    results_df = run_evaluation(MODELS, ALL_LANGUAGES)

    # Log summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Complete - Summary")
    logger.info("="*60)

    # Summary by model
    logger.info("\nAccuracy by Model:")
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        avg_acc = model_df['accuracy'].mean()
        logger.info(f"  {model}: {avg_acc:.2%}")

    # Summary by resource level
    logger.info("\nAccuracy by Resource Level:")
    for level in ['high', 'mid', 'low']:
        level_df = results_df[results_df['resource_level'] == level]
        avg_acc = level_df['accuracy'].mean()
        logger.info(f"  {level}: {avg_acc:.2%}")

    # English vs average
    en_df = results_df[results_df['language'] == 'en']
    en_avg = en_df['accuracy'].mean()
    other_df = results_df[results_df['language'] != 'en']
    other_avg = other_df['accuracy'].mean()
    logger.info(f"\nEnglish avg: {en_avg:.2%}")
    logger.info(f"Non-English avg: {other_avg:.2%}")
    logger.info(f"Performance gap: {(en_avg - other_avg) / en_avg:.2%}")

    logger.info("\nResults saved to results/evaluation_results.csv")

    return results_df


if __name__ == "__main__":
    main()
