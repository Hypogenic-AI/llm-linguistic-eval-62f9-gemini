# Datasets for Multilingual LLM Evaluation

This directory contains datasets for evaluating linguistic performance in Large Language Models across multiple languages.

**Note**: Large data files are NOT committed to git due to size. Follow the download instructions below to obtain the full datasets.

## Available Datasets

### 1. MuBench MMLU (Multilingual MMLU)

**Source**: [HuggingFace - aialt/MuBench](https://huggingface.co/datasets/aialt/MuBench)

**Description**: MuBench provides cross-linguistically aligned evaluations of the MMLU benchmark across 61 languages. Questions are translated from English while maintaining semantic consistency.

**Languages Available** (15 selected):
| Language | Code | Resource Level | Test Size |
|----------|------|----------------|-----------|
| English | en | High | 12,338 |
| Chinese | zh | High | 12,338 |
| Spanish | es | High | 12,338 |
| French | fr | High | 12,338 |
| German | de | High | 12,338 |
| Arabic | ar | Mid | 12,338 |
| Korean | ko | Mid | 12,338 |
| Vietnamese | vi | Mid | 12,338 |
| Turkish | tr | Mid | 12,338 |
| Indonesian | id | Mid | 12,338 |
| Bengali | bn | Low | 12,338 |
| Telugu | te | Low | 12,338 |
| Swahili | sw | Low | 12,338 |
| Hindi | hi | Mid | 12,338 |
| Tamil | ta | Low | 12,338 |

**Task**: Multiple-choice question answering (4 options: A, B, C, D)

**Format**:
```json
{
  "_id": "test_0",
  "prompt": "Question: ... \nChoice A: ...\nChoice B: ...\nChoice C: ...\nChoice D: ...\nAnswer with A, B, C or D:\nAnswer:",
  "choices": ["A", "B", "C", "D"],
  "label": 1  // Index of correct answer (0-indexed)
}
```

#### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# Download English version
ds_en = load_dataset("aialt/MuBench", "MMLUDataset_en_template_en", split="test")

# Download Chinese version
ds_zh = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test")

# Download any language - replace 'xx' with language code
# ds = load_dataset("aialt/MuBench", "MMLUDataset_en_template_xx", split="test")

# Available language codes: en, zh, es, fr, de, ar, ko, vi, tr, id, bn, te, sw, hi, ta
# (and many more - see HuggingFace page for full list)
```

**Loading samples locally:**
```python
import json
with open("datasets/mubench/mmlu_samples.json", "r") as f:
    samples = json.load(f)
# Access: samples["en"]["samples"], samples["zh"]["samples"], etc.
```

---

### 2. Belebele (Multilingual Reading Comprehension)

**Source**: [HuggingFace - facebook/belebele](https://huggingface.co/datasets/facebook/belebele)

**Description**: Belebele is a multiple-choice machine reading comprehension dataset spanning 122 language variants. Each question is linked to a passage and has 4 answer choices.

**Languages Available** (15 selected):
| Language | Code | Resource Level | Test Size |
|----------|------|----------------|-----------|
| English | eng_Latn | High | 900 |
| Chinese (Simplified) | zho_Hans | High | 900 |
| Spanish | spa_Latn | High | 900 |
| French | fra_Latn | High | 900 |
| German | deu_Latn | High | 900 |
| Arabic | arb_Arab | Mid | 900 |
| Korean | kor_Hang | Mid | 900 |
| Vietnamese | vie_Latn | Mid | 900 |
| Turkish | tur_Latn | Mid | 900 |
| Indonesian | ind_Latn | Mid | 900 |
| Bengali | ben_Beng | Low | 900 |
| Telugu | tel_Telu | Low | 900 |
| Swahili | swh_Latn | Low | 900 |
| Nepali | npi_Deva | Low | 900 |
| Tamil | tam_Taml | Low | 900 |

**Task**: Reading comprehension with multiple-choice answers

**Format**:
```json
{
  "link": "...",
  "question_number": 1,
  "flores_passage": "...",  // The reading passage
  "question": "...",        // The question
  "mc_answer1": "...",      // Option 1
  "mc_answer2": "...",      // Option 2
  "mc_answer3": "...",      // Option 3
  "mc_answer4": "...",      // Option 4
  "correct_answer_num": 2,  // Correct answer (1-indexed)
  "dialect": "...",
  "ds": "..."
}
```

#### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset

# Download English version
ds_en = load_dataset("facebook/belebele", "eng_Latn", split="test")

# Download Chinese version
ds_zh = load_dataset("facebook/belebele", "zho_Hans", split="test")

# Download any language - use ISO codes from the table above
# Full list: https://huggingface.co/datasets/facebook/belebele
```

**Loading samples locally:**
```python
import json
with open("datasets/belebele/samples.json", "r") as f:
    samples = json.load(f)
# Access: samples["eng_Latn"]["samples"], samples["zho_Hans"]["samples"], etc.
```

---

### 3. MMLU Abstract Algebra (Legacy)

**Source**: Pre-downloaded HuggingFace MMLU subset

**Location**: `datasets/mmlu_abstract_algebra/`

**Description**: Abstract algebra subset from the original MMLU benchmark.

---

## Usage for Experiments

### Recommended Experiment Setup

```python
from datasets import load_dataset
import json

# 1. Define language groups
HIGH_RESOURCE = ["en", "zh", "es", "fr", "de"]
MID_RESOURCE = ["ar", "ko", "vi", "tr", "id"]
LOW_RESOURCE = ["bn", "te", "sw", "hi", "ta"]

# 2. Load MuBench for each language
def load_mubench_mmlu(lang_code):
    """Load MuBench MMLU for a specific language."""
    config = f"MMLUDataset_en_template_{lang_code}"
    return load_dataset("aialt/MuBench", config, split="test")

# 3. Evaluate LLM on each language
def evaluate_multilingual(model, tokenizer):
    results = {}
    for lang in HIGH_RESOURCE + MID_RESOURCE + LOW_RESOURCE:
        dataset = load_mubench_mmlu(lang)
        accuracy = run_evaluation(model, tokenizer, dataset)
        results[lang] = accuracy
    return results
```

### Evaluation Metrics

1. **Per-language Accuracy**: Correct answers / Total questions
2. **Performance Gap**: (English accuracy - Target accuracy) / English accuracy
3. **Cross-lingual Consistency**: For aligned samples, check if model gives same answer

---

## File Structure

```
datasets/
├── README.md                    # This file
├── .gitignore                   # Excludes large data files
├── belebele/
│   ├── samples.json             # Sample data for all languages
│   ├── dataset_info.json        # Dataset metadata
│   └── *_sample.json            # Per-language samples
├── mubench/
│   ├── mmlu_samples.json        # Sample data for all languages
│   ├── dataset_info.json        # Dataset metadata
│   └── mmlu_en_sample.json      # English samples
└── mmlu_abstract_algebra/       # Legacy dataset
    ├── dev/
    ├── test/
    └── validation/
```

---

## Notes for Experiment Runner

1. **Data Loading**: Use HuggingFace `load_dataset` for full evaluation; samples in this directory are for reference only
2. **Language Selection**: The 15 languages cover high/mid/low resource levels for comprehensive analysis
3. **Alignment**: Both MuBench and Belebele provide aligned samples across languages, enabling direct comparison
4. **Prompt Language**: MuBench uses English prompts (instruction) with translated content; Belebele has fully translated passages

---

## References

- MuBench: Han, W., et al. (2025). "MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages." arXiv:2506.19468
- Belebele: Bandarkar, L., et al. (2023). "The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants."
