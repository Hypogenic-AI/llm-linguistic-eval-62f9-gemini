# Code Repositories

This directory contains cloned repositories relevant to multilingual LLM evaluation research.

## Cloned Repositories

### 1. multilingual_analysis

**URL**: https://github.com/DAMO-NLP-SG/multilingual_analysis

**Paper**: "How do Large Language Models Handle Multilingualism?" (NeurIPS 2024)

**Location**: `code/multilingual_analysis/`

**Purpose**: Implementation of Parallel Language-specific Neuron Detection (PLND) and verification of the Multilingual Workflow (MWork) hypothesis.

**Key Components**:
- `layers/` - Code for layer embedding decoding (analyze language transitions across layers)
- `neuron_detection/` - PLND implementation to identify language-specific neurons
- `neuron_deactivate/` - Scripts to deactivate specific neurons for experiments
- `neuron_enhancement/` - Fine-tuning language-specific neurons

**Key Scripts**:
```bash
# Layer embedding decoding
cd code/multilingual_analysis/layers
python test_layer.py

# Neuron detection
cd code/multilingual_analysis/neuron_detection
python detect_neurons.py
```

**Dependencies**:
```
torch
transformers
```

**Supported Models**: LLaMA, Mistral, Gemma

**Relevance**: Directly relevant to investigating the hypothesis that LLMs use implicit English translation - provides tools to analyze internal language representations.

---

### 2. GlotEval

**URL**: https://github.com/MaLA-LM/GlotEval

**Paper**: "GlotEval: A Test Suite for Massively Multilingual Evaluation of Large Language Models" (2025)

**Location**: `code/GlotEval/`

**Purpose**: Unified evaluation framework integrating 27 multilingual benchmarks with ISO 639-3 standardization.

**Key Components**:
- `benchmark_data_loader/` - Data loaders for various benchmarks
- `tasks/` - Task definitions for 9 key evaluation tasks
- `metrics/` - Evaluation metrics implementation
- `models/` - Model interface adapters
- `main.py` - Main evaluation entry point
- `config.json` - Configuration for benchmarks and models

**Supported Tasks**:
1. Machine Translation
2. Text Classification
3. Summarization
4. Open-ended Generation
5. Reading Comprehension
6. Sequence Labeling
7. Intrinsic Evaluation
8. Instruction Following
9. Reasoning

**Usage**:
```bash
cd code/GlotEval
pip install -r requirements.txt

# Run evaluation
python main.py --config config.json --model <model_name> --task <task_name>
```

**Dependencies**:
```
transformers
torch
datasets
```

**Relevance**: Provides standardized evaluation infrastructure for testing LLMs across many languages; supports non-English-centric evaluation.

---

## Additional Repositories (Not Cloned)

The following repositories may be useful but were not cloned due to size or dependency concerns:

### lm-evaluation-harness (EleutherAI)
**URL**: https://github.com/EleutherAI/lm-evaluation-harness
**Purpose**: General-purpose LLM evaluation framework with some multilingual support
**Clone if needed**: `git clone https://github.com/EleutherAI/lm-evaluation-harness.git code/lm-evaluation-harness`

### MEGA (Multilingual Evaluation)
**URL**: https://github.com/microsoft/MEGA
**Purpose**: Microsoft's framework for multilingual evaluation of generative LLMs
**Clone if needed**: `git clone https://github.com/microsoft/MEGA.git code/MEGA`

---

## Quick Start Guide

### For Internal Mechanism Analysis (Paper 2 approach):
```python
# Analyze how the model processes non-English input
# See code/multilingual_analysis/layers/test_layer.py
# This reveals the "English-centric workflow" pattern

# Detect language-specific neurons
# See code/multilingual_analysis/neuron_detection/
```

### For Standard Multilingual Evaluation:
```python
# Use GlotEval for standardized benchmarking
# See code/GlotEval/main.py

# Or use HuggingFace datasets directly
from datasets import load_dataset
ds = load_dataset("aialt/MuBench", "MMLUDataset_en_template_zh", split="test")
```

---

## Notes

1. **Model Requirements**: Most scripts require significant GPU memory (16GB+ recommended for 7B models)
2. **Dependencies**: Each repository has its own requirements.txt - install within a virtual environment
3. **Transformers Patches**: The multilingual_analysis repo requires modifications to the transformers library for layer decoding
