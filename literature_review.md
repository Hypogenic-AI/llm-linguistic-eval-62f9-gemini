# Literature Review: Evaluating Linguistic Performance in LLMs

## Research Area Overview

This literature review examines recent research on evaluating and understanding multilingual capabilities of Large Language Models (LLMs). The central research hypothesis under investigation is that LLMs trained predominantly on English data may underperform when deployed in non-English-speaking contexts, and that these models may possess implicit internal translation mechanisms that convert non-English inputs to English for processing.

The reviewed papers span four main areas:
1. **Linguistic discrimination in LLMs** - Safety and quality disparities across languages
2. **Internal multilingual processing mechanisms** - How LLMs handle non-English inputs
3. **Multilingual evaluation benchmarks** - Standardized frameworks for cross-lingual assessment
4. **Comprehensive surveys** - Corpora, alignment, and bias in multilingual LLMs

---

## Key Papers

### Paper 1: Evaluating and Mitigating Linguistic Discrimination in Large Language Models

- **Authors**: Guoliang Dong, Haoyu Wang, Jun Sun, Xinyu Wang
- **Year**: 2024
- **Source**: arXiv:2404.18534
- **PDF**: papers/evaluating_linguistic_discrimination.pdf

**Key Contribution**: Systematically evaluates linguistic discrimination from both safety and quality perspectives, proposing LDFighter as a mitigation strategy.

**Methodology**:
- Evaluated 4 LLMs: Llama2-13b, Gemma-7b, GPT-3.5-turbo, Gemini-pro
- Used AdvBench (harmful queries) and NQ (natural questions) datasets
- Assessed 74 languages for safety (jailbreak rate) and quality (F1 score)

**Key Findings**:
- **Safety disparities**: High-resource languages (English, French, Russian, Spanish) show 1.04% average jailbreak rate; low-resource languages (Bengali, Georgian, Nepali, Maithili) show 27.7% jailbreak rate
- **Quality disparities**: English, Danish, Czech, Slovenian achieve 0.1494 average F1; Kannada, Southern Pashto, Tajik, Telugu achieve only 0.0341 average F1
- **Correlation with training data**: Languages with >0.005% share in pretraining data show better safety performance

**Proposed Solution**: LDFighter - translates queries to top-k languages, gets responses, translates back to pivot language (English), uses similarity-based voting for final answer

**Code Available**: Not mentioned in paper

**Relevance to Hypothesis**: Directly supports the hypothesis - demonstrates significant performance gaps between English and low-resource languages across both safety and quality metrics.

---

### Paper 2: How do Large Language Models Handle Multilingualism?

- **Authors**: Yiran Zhao, Wenxuan Zhang, Guizhen Chen, Kenji Kawaguchi, Lidong Bing
- **Year**: 2024 (NeurIPS 2024)
- **Source**: arXiv:2402.18815
- **PDF**: papers/how_llms_handle_multilingualism.pdf

**Key Contribution**: Proposes and validates the Multilingual Workflow (MWork) hypothesis - that LLMs internally convert non-English queries to English for processing.

**Methodology**:
- Developed Parallel Language-specific Neuron Detection (PLND) to identify language-specific neurons
- Tested on multiple benchmarks: XQuAD (understanding), MGSM (reasoning), X-CSQA (knowledge), XLSum (generation)
- Deactivated language-specific neurons to verify functionality

**Key Findings**:
- **Three-stage workflow confirmed**:
  1. **Understanding**: Non-English queries converted to unified (English-centric) representation
  2. **Task-solving**: Reasoning in English (self-attention) + knowledge extraction (feed-forward)
  3. **Generation**: Output converted back to original query language
- Only 0.13% of neurons are language-specific, yet deactivating them causes 99% performance drop on multilingual summarization
- Deactivating understanding layer neurons: English stable, non-English -14%

**Practical Application**: Fine-tuning language-specific neurons with 400 documents improves high-resource languages by 3.6% and low-resource languages by 2.3%

**Code Available**: https://github.com/DAMO-NLP-SG/multilingual_analysis

**Relevance to Hypothesis**: Strongly supports the hypothesis about implicit translation mechanisms - demonstrates LLMs convert multilingual inputs to English internally for processing.

---

### Paper 3: MuBench: Assessment of Multilingual Capabilities Across 61 Languages

- **Authors**: Wenhan Han, Yifan Zhang, et al.
- **Year**: 2025
- **Source**: arXiv:2506.19468
- **PDF**: papers/mubench.pdf

**Key Contribution**: Comprehensive multilingual benchmark with cross-lingual alignment, introducing Multilingual Consistency (MLC) as a complementary metric.

**Methodology**:
- 61 languages covering >60% global population
- Translated widely-used English benchmarks with rigorous quality control
- Includes code-switched variants for mixed-language evaluation
- Human evaluation across 16 languages

**Tasks Covered**:
- Natural Language Understanding: SNLI, MultiNLI, WinoGrande
- Commonsense Reasoning: HellaSwag, StoryCloze
- Factual Recall: BMLAMA
- Knowledge-based QA: MMLU, MMLU Pro
- Academic & Technical Reasoning: GPQA, ARC-Easy, ARC-Challenge
- Truthfulness: TruthfulQA

**Key Findings**:
- Models fall short of claimed multilingual coverage
- Persistent performance gap between English and low-resource languages
- Gap does not consistently narrow with increased model size
- Larger models not necessarily more robust in code-switched settings
- Parallel corpora improve both accuracy and consistency

**Dataset Available**: https://huggingface.co/datasets/aialt/MuBench

**Relevance to Hypothesis**: Provides empirical evidence for English-centric bias across a wide range of tasks and languages.

---

### Paper 4: GlotEval: A Test Suite for Massively Multilingual Evaluation

- **Authors**: Hengyu Luo, Zihao Li, et al.
- **Year**: 2025
- **Source**: arXiv:2504.04155
- **PDF**: papers/gloteval.pdf

**Key Contribution**: Unified evaluation framework integrating 27 benchmarks with ISO 639-3 standardization, supporting non-English-centric evaluation.

**Methodology**:
- Standardized all benchmarks to ISO 639-3 language codes
- Supports 9 key tasks spanning dozens to hundreds of languages
- Language-specific prompt templates
- Non-English-centered machine translation evaluation

**Tasks Supported**:
1. Machine Translation
2. Text Classification
3. Summarization
4. Open-ended Generation
5. Reading Comprehension
6. Sequence Labeling
7. Intrinsic Evaluation
8. Instruction Following
9. Reasoning

**Key Features**:
- Cross-benchmark analysis by language/language group
- Microsoft Translator integration for prompt propagation (130+ languages)
- Any language can serve as pivot for translation evaluation

**Code Available**: https://github.com/MaLA-LM/GlotEval

**Relevance to Hypothesis**: Provides the evaluation infrastructure needed to systematically test multilingual LLM performance across diverse languages.

---

### Paper 5: A Survey on Multilingual Large Language Models: Corpora, Alignment, and Bias

- **Authors**: Yuemei Xu, Ling Hu, Jiayi Zhao, Zihan Qiu, Kexin Xu, Yuqi Ye, Hanwen Gu
- **Year**: 2024
- **Source**: arXiv:2404.00929
- **PDF**: papers/multilingual_llm_survey.pdf

**Key Contribution**: Comprehensive survey covering MLLMs' evolution, training corpora, multilingual representations, and bias.

**Topics Covered**:

1. **MLLM Overview**: Evolution from mBERT to BLOOM/LLaMA, key techniques, multilingual capacities

2. **Training Corpora Analysis**:
   - ChatGPT: 92.099% English, only 0.16% Chinese
   - Language imbalance in major corpora (Wikipedia, Common Crawl)
   - "Curse of multilinguality": more languages help low-resource up to a point, then overall performance decreases

3. **Multilingual Alignment**:
   - Static, contextual, and combined multilingual representations
   - Factors affecting alignment: initial alignment solution, mapping linearity, typological distance, pretraining data

4. **Bias in MLLMs**:
   - Categories of bias
   - Evaluation metrics
   - Debiasing techniques
   - Most bias studies limited to English

**Key Insights**:
- Scale, quality, and diversity of corpora significantly impact MLLM performance
- Under-representation of low-resource languages leads to poor performance
- Universal language representation remains challenging

**Relevance to Hypothesis**: Provides foundational context explaining why LLMs exhibit English-centric bias (training data imbalance) and the challenges in achieving cross-lingual transfer.

---

## Common Methodologies

### Evaluation Approaches
- **Cross-lingual benchmarks**: Translate English benchmarks to target languages
- **Language-specific metrics**: Jailbreak rate, F1 score, accuracy by language
- **Consistency metrics**: Multilingual Consistency (MLC) for aligned samples

### Models Commonly Evaluated
- GPT-3.5-turbo, GPT-4
- Gemini-pro
- LLaMA-2 (7B, 13B)
- Gemma (7B)
- BLOOM, BLOOMZ
- Vicuna
- Qwen

### Language Categorization
- **High-resource**: English, French, Spanish, Russian, German, Chinese, Japanese
- **Mid-resource**: Arabic, Korean, Portuguese, Turkish, Vietnamese, Indonesian
- **Low-resource**: Bengali, Nepali, Georgian, Maithili, Kannada, Telugu

---

## Standard Baselines

| Baseline Type | Common Approaches |
|--------------|-------------------|
| Translation-based | Translate query to English, process, translate back |
| Direct inference | Run model directly on target language |
| Zero-shot | No target language examples |
| Few-shot | Include examples in target language |

---

## Evaluation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| Accuracy | Correct answers / Total | General task performance |
| F1 Score | Harmonic mean of precision/recall | QA tasks |
| Jailbreak Rate | Successful harmful responses / Total | Safety evaluation |
| BLEU/chrF++ | Translation quality | Machine translation |
| Multilingual Consistency (MLC) | Consistency across languages for aligned samples | Cross-lingual transfer |

---

## Datasets in the Literature

### Benchmark Datasets
| Dataset | Languages | Task | Used By |
|---------|-----------|------|---------|
| MMLU | 61 (MuBench) | Knowledge QA | Papers 3, 4 |
| XNLI | 15 | NLI | Paper 4 |
| XQuAD | 10 | Reading comprehension | Paper 2 |
| MGSM | 10 | Math reasoning | Paper 2 |
| Belebele | 122 | Reading comprehension | Paper 4 |
| AdvBench | Multi | Safety (jailbreak) | Paper 1 |
| NQ | Multi | Open-domain QA | Paper 1 |
| FLORES-200 | 200 | Translation | Paper 4 |

### Training Corpora
- Common Crawl (mC4, CC-100)
- Wikipedia (various languages)
- ROOTS (BLOOM's corpus)
- Parallel corpora (OPUS, CCAligned)

---

## Gaps and Opportunities

1. **Limited low-resource language evaluation**: Most benchmarks still focus on high-resource languages
2. **Cultural bias**: Many benchmarks have Western-centric content
3. **Code-switching**: Mixed-language scenarios underexplored
4. **Dynamic evaluation**: Most benchmarks are static; no evaluation of adaptation
5. **Safety in low-resource**: Limited safety evaluation for underrepresented languages
6. **Mechanistic understanding**: Internal multilingual processing needs more investigation

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **MuBench** (Primary): 61 languages, multiple tasks, aligned samples, available on HuggingFace
2. **MMLU (Multilingual)**: Standard knowledge benchmark, well-established
3. **FLORES-200**: For translation-based evaluation
4. **Belebele**: For broad language coverage reading comprehension

### Recommended Baselines
1. Direct inference in target language
2. English translation → inference → back-translation
3. Compare open models (LLaMA, Gemma) vs. closed (GPT-4, Claude)

### Recommended Metrics
1. **Accuracy per language** - Primary performance metric
2. **Multilingual Consistency (MLC)** - Cross-lingual transfer assessment
3. **Performance gap ratio** - (English accuracy - Target accuracy) / English accuracy

### Methodological Considerations
1. **Language selection**: Include high/mid/low-resource languages from different families
2. **Task diversity**: Test both understanding and generation tasks
3. **Prompt language**: Test both English prompts and target language prompts
4. **Internal analysis**: Consider probing hidden states for language representation (per Paper 2)

---

## References

1. Dong, G., et al. (2024). Evaluating and Mitigating Linguistic Discrimination in Large Language Models. arXiv:2404.18534
2. Zhao, Y., et al. (2024). How do Large Language Models Handle Multilingualism? NeurIPS 2024. arXiv:2402.18815
3. Han, W., et al. (2025). MuBench: Assessment of Multilingual Capabilities of Large Language Models Across 61 Languages. arXiv:2506.19468
4. Luo, H., et al. (2025). GlotEval: A Test Suite for Massively Multilingual Evaluation of Large Language Models. arXiv:2504.04155
5. Xu, Y., et al. (2024). A Survey on Multilingual Large Language Models: Corpora, Alignment, and Bias. arXiv:2404.00929
