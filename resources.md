# Resources Catalog: Evaluating Linguistic Performance in LLMs

## Summary

This document catalogs all resources gathered for researching the hypothesis that LLMs trained predominantly on English data underperform in non-English contexts and may possess implicit internal translation mechanisms.

**Research Hypothesis**: Large language models trained predominantly on English data may underperform when deployed in non-English-speaking countries. Evaluating LLM performance across multiple languages will reveal the extent of English-centric bias and whether these models possess implicit internal translation mechanisms.

---

## Papers

**Total papers downloaded**: 5

| # | Title | Authors | Year | File | Key Contribution |
|---|-------|---------|------|------|------------------|
| 1 | Evaluating and Mitigating Linguistic Discrimination in LLMs | Dong et al. | 2024 | papers/evaluating_linguistic_discrimination.pdf | Safety/quality disparities across 74 languages |
| 2 | How do Large Language Models Handle Multilingualism? | Zhao et al. | 2024 | papers/how_llms_handle_multilingualism.pdf | MWork hypothesis - internal English translation |
| 3 | MuBench: Multilingual Capabilities Across 61 Languages | Han et al. | 2025 | papers/mubench.pdf | Comprehensive benchmark with cross-lingual alignment |
| 4 | GlotEval: Massively Multilingual Evaluation | Luo et al. | 2025 | papers/gloteval.pdf | Unified framework integrating 27 benchmarks |
| 5 | Survey on Multilingual LLMs: Corpora, Alignment, Bias | Xu et al. | 2024 | papers/multilingual_llm_survey.pdf | Comprehensive survey on training data and bias |

See `papers/README.md` for detailed descriptions.

---

## Datasets

**Total datasets available**: 3 (2 multilingual benchmarks + 1 legacy)

### Primary Datasets

| Dataset | Source | Languages | Size per Lang | Task | Location |
|---------|--------|-----------|---------------|------|----------|
| MuBench MMLU | HuggingFace | 61 (15 downloaded) | 12,338 | Multi-choice QA | HuggingFace / datasets/mubench/ |
| Belebele | HuggingFace | 122 (15 downloaded) | 900 | Reading Comprehension | HuggingFace / datasets/belebele/ |
| MMLU Abstract Algebra | Local | 1 (English) | ~100 | Multi-choice QA | datasets/mmlu_abstract_algebra/ |

### Language Coverage

| Resource Level | MuBench Languages | Belebele Languages |
|----------------|-------------------|-------------------|
| High | English, Chinese, Spanish, French, German | English, Chinese, Spanish, French, German |
| Mid | Arabic, Korean, Vietnamese, Turkish, Indonesian | Arabic, Korean, Vietnamese, Turkish, Indonesian |
| Low | Bengali, Telugu, Swahili, Hindi, Tamil | Bengali, Telugu, Swahili, Nepali, Tamil |

### Download Instructions Summary

```python
# MuBench MMLU
from datasets import load_dataset
ds = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test")

# Belebele
ds = load_dataset("facebook/belebele", "zho_Hans", split="test")
```

See `datasets/README.md` for detailed instructions.

---

## Code Repositories

**Total repositories cloned**: 2

| Repository | URL | Purpose | Location |
|------------|-----|---------|----------|
| multilingual_analysis | github.com/DAMO-NLP-SG/multilingual_analysis | PLND, MWork verification | code/multilingual_analysis/ |
| GlotEval | github.com/MaLA-LM/GlotEval | Unified multilingual evaluation | code/GlotEval/ |

### Key Scripts

| Script | Repository | Purpose |
|--------|------------|---------|
| test_layer.py | multilingual_analysis | Layer embedding decoding to visualize language shifts |
| detect_neurons.py | multilingual_analysis | Detect language-specific neurons using PLND |
| main.py | GlotEval | Main entry point for standardized multilingual evaluation |

See `code/README.md` for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy

1. **Literature Search**: Used arXiv, Semantic Scholar, and Papers with Code to identify relevant multilingual LLM evaluation papers from 2023-2025
2. **Dataset Search**: Prioritized HuggingFace datasets with broad language coverage and cross-lingual alignment (aligned test items across languages)
3. **Code Search**: Identified repositories from paper references and Papers with Code

### Selection Criteria

**Papers**:
- Focus on multilingual performance evaluation
- Empirical studies showing performance disparities across languages
- Mechanistic investigations of how LLMs process non-English input
- Comprehensive surveys providing context

**Datasets**:
- Cross-lingual alignment (same test items in multiple languages)
- Coverage of low-resource languages
- Standardized evaluation format (multiple choice for easier comparison)
- Available via HuggingFace for easy access

**Code**:
- Open-source implementations of key methodologies
- Relevant to investigating internal multilingual mechanisms
- Provides evaluation infrastructure

### Challenges Encountered

1. **Large PDF files**: Some papers exceeded 5MB; used chunked reading to extract key sections
2. **JSON serialization**: Belebele dataset had datetime fields requiring conversion
3. **Dataset configs**: Both MuBench and Belebele require specific config names for each language

### Gaps and Workarounds

1. **No jailbreak datasets**: The AdvBench dataset used in Paper 1 is for safety evaluation; could be added if safety testing is needed
2. **No translation benchmarks**: FLORES-200 could be added for translation evaluation
3. **Limited mechanistic tools**: Only one repo (multilingual_analysis) provides tools for internal mechanism analysis

---

## Recommendations for Experiment Design

Based on gathered resources, the following experiment design is recommended:

### 1. Primary Dataset
**MuBench MMLU** - Provides aligned samples across 61 languages for direct comparison

### 2. Language Selection
Test 15 languages across three resource levels:
- **High**: en, zh, es, fr, de
- **Mid**: ar, ko, vi, tr, id
- **Low**: bn, te, sw, hi, ta

### 3. Baseline Methods
1. **Direct inference**: Run model on target language input
2. **English translation**: Translate input to English, run model, translate back
3. Compare open models (LLaMA, Gemma) vs. closed models (GPT-4, Claude)

### 4. Evaluation Metrics
1. **Accuracy per language**: Primary performance metric
2. **Performance gap ratio**: (En_accuracy - Lang_accuracy) / En_accuracy
3. **Multilingual Consistency (MLC)**: For aligned samples, check consistency of answers

### 5. Internal Mechanism Analysis (Optional)
Use multilingual_analysis repo to:
- Decode layer embeddings to visualize language representation shifts
- Detect and analyze language-specific neurons
- Verify if model converts to English internally

### 6. Experimental Protocol
```python
# Pseudocode for experiment
models = ["gpt-4", "claude-3", "llama-3", "gemma-2"]
languages = HIGH_RESOURCE + MID_RESOURCE + LOW_RESOURCE

results = {}
for model in models:
    for lang in languages:
        dataset = load_mubench_mmlu(lang)
        accuracy = evaluate(model, dataset)
        results[(model, lang)] = accuracy

# Analyze:
# 1. Per-model language performance gaps
# 2. Correlation between language resource level and performance
# 3. Cross-lingual consistency on aligned samples
```

---

## File Structure Summary

```
workspace/
├── papers/                           # Downloaded PDFs
│   ├── README.md                     # Paper descriptions
│   ├── evaluating_linguistic_discrimination.pdf
│   ├── how_llms_handle_multilingualism.pdf
│   ├── mubench.pdf
│   ├── gloteval.pdf
│   └── multilingual_llm_survey.pdf
├── datasets/                         # Dataset samples and info
│   ├── README.md                     # Download instructions
│   ├── .gitignore                    # Excludes large files
│   ├── belebele/                     # Belebele samples
│   │   ├── samples.json
│   │   └── dataset_info.json
│   ├── mubench/                      # MuBench samples
│   │   ├── mmlu_samples.json
│   │   └── dataset_info.json
│   └── mmlu_abstract_algebra/        # Legacy dataset
├── code/                             # Cloned repositories
│   ├── README.md                     # Repository descriptions
│   ├── multilingual_analysis/        # PLND implementation
│   └── GlotEval/                     # Evaluation framework
├── literature_review.md              # Comprehensive paper review
├── resources.md                      # This file
└── .resource_finder_complete         # Completion marker
```

---

## Quick Reference

### Load MuBench Dataset
```python
from datasets import load_dataset
ds = load_dataset("aialt/MuBench", "MMLUDataset_en_template_{lang}", split="test")
# Replace {lang} with: en, zh, es, fr, de, ar, ko, vi, tr, id, bn, te, sw, hi, ta
```

### Load Belebele Dataset
```python
from datasets import load_dataset
ds = load_dataset("facebook/belebele", "{lang_code}", split="test")
# Replace {lang_code} with: eng_Latn, zho_Hans, spa_Latn, etc.
```

### Key Paper Findings
1. **Performance gaps exist**: Low-resource languages show 27.7% jailbreak rate vs 1.04% for high-resource (Paper 1)
2. **Internal English translation**: LLMs convert non-English queries to English internally (Paper 2)
3. **Larger models don't solve the problem**: Performance gap doesn't consistently narrow with model size (Paper 3)
4. **Training data is the root cause**: English comprises 92% of ChatGPT's training data (Paper 5)

---

## Next Steps for Experiment Runner

1. Load datasets from HuggingFace using the provided code snippets
2. Set up model inference (API calls for GPT-4/Claude, local inference for LLaMA/Gemma)
3. Run evaluation across all 15 languages
4. Calculate metrics: accuracy, performance gap, consistency
5. (Optional) Use multilingual_analysis repo to probe internal representations
6. Generate visualizations and write results report
