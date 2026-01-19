# Research Plan: Evaluating Linguistic Performance in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters

Large language models are increasingly deployed at national scale in non-English-speaking countries (e.g., xAI's partnership with Venezuela, OpenAI's partnership with Estonia), yet their capability evaluations remain overwhelmingly English-centric. This creates a significant risk: governments and organizations may deploy models that underperform for their populations, potentially exacerbating digital divides and AI-driven inequalities. Understanding the magnitude of performance gaps across languages is essential for responsible AI deployment worldwide.

### Gap in Existing Work

From the literature review, while several comprehensive benchmarks exist (MuBench, GlotEval, Belebele), and one paper (Zhao et al., 2024) has proposed the "Multilingual Workflow" hypothesis for internal processing mechanisms, there is limited **comparative evaluation of state-of-the-art 2025 LLMs** across resource levels with a focus on:
1. Direct comparison of multiple frontier models (GPT-4, Claude, Gemini, LLaMA) on the same aligned multilingual benchmark
2. Quantifying the "performance gap ratio" systematically across language resource levels
3. Testing whether translation-based approaches close the performance gap

### Our Novel Contribution

We will conduct a **systematic empirical study** comparing multiple frontier LLMs (GPT-4o, GPT-4.1, Claude Sonnet 4.5 via OpenRouter, Gemini 2.5 Pro via OpenRouter) across 12 languages representing high, mid, and low-resource categories using the MuBench MMLU benchmark. Our contribution:
1. **Quantified performance gaps**: Measure exact accuracy differences between English and other languages for 2025's frontier models
2. **Translation baseline**: Test whether "translate to English → process → translate back" mitigates performance gaps
3. **Cross-model comparison**: Determine which models are most/least affected by language resource levels
4. **Reproducible benchmark**: Provide code and methodology for ongoing multilingual evaluation

### Experiment Justification

**Experiment 1: Direct Multilingual Evaluation**
- *Why needed?* Establishes baseline performance across languages for current frontier models. Literature shows 2024 models had significant gaps; we need updated measurements for 2025 models.

**Experiment 2: Translation-Based Approach**
- *Why needed?* The MWork hypothesis (Zhao et al., 2024) suggests LLMs internally translate to English. If true, explicit translation might improve low-resource language performance. This tests a practical mitigation strategy.

**Experiment 3: Cross-Model Comparison**
- *Why needed?* Different training corpora and architectures may lead to different multilingual capabilities. Identifying the best models for non-English deployment has practical value.

---

## Research Question

**Primary Question**: How do state-of-the-art LLMs (GPT-4o, GPT-4.1, Claude, Gemini) perform across high, mid, and low-resource languages on standardized knowledge benchmarks?

**Secondary Questions**:
1. What is the performance gap ratio between English and other languages?
2. Does explicit translation to English improve low-resource language performance?
3. Which model architecture exhibits the smallest cross-lingual performance gap?

---

## Hypothesis Decomposition

**H1**: LLMs trained predominantly on English data will show significantly lower accuracy on low-resource languages compared to English.
- *Measurable*: Accuracy_lowresource < 0.8 * Accuracy_english (>20% gap)

**H2**: The performance gap will be correlated with language resource level.
- *Measurable*: Spearman correlation between language resource ranking and accuracy

**H3**: Translation-based approaches will partially close the performance gap for low-resource languages.
- *Measurable*: Accuracy_translated > Accuracy_direct for low-resource languages

---

## Proposed Methodology

### Approach

We use the **MuBench MMLU** dataset as our primary benchmark because:
1. It provides **aligned test items** across 61 languages (same questions, translated)
2. Multiple-choice format enables simple, unambiguous evaluation (accuracy)
3. 12,338 questions per language provides statistical power
4. Covers diverse knowledge domains (STEM, humanities, social sciences)

We will evaluate across **12 languages** spanning three resource levels:
- **High-resource** (4): English (en), Chinese (zh), Spanish (es), French (fr)
- **Mid-resource** (4): Arabic (ar), Korean (ko), Vietnamese (vi), Indonesian (id)
- **Low-resource** (4): Bengali (bn), Telugu (te), Swahili (sw), Tamil (ta)

### Experimental Steps

#### Phase 1: Data Preparation (10 min)
1. Load MuBench MMLU dataset from HuggingFace for all 12 languages
2. Sample 500 questions per language (to manage API costs while maintaining statistical power)
3. Validate data format and alignment across languages
4. Create stratified sample ensuring subject diversity

#### Phase 2: Model Setup (10 min)
1. Configure API clients for:
   - OpenAI: GPT-4o (`gpt-4o`), GPT-4.1 (`gpt-4.1`)
   - OpenRouter: Claude Sonnet 4.5 (`anthropic/claude-sonnet-4.5`), Gemini 2.5 Pro (`google/gemini-2.5-pro`)
2. Set consistent parameters: temperature=0, max_tokens=10
3. Implement retry logic with exponential backoff
4. Add rate limiting to stay within API quotas

#### Phase 3: Direct Evaluation (60-90 min)
1. For each model and each language:
   - Send prompt with question and multiple-choice options
   - Parse model response to extract selected answer (A/B/C/D)
   - Record accuracy
2. Run all 4 models × 12 languages × 500 questions = 24,000 API calls
3. Log all responses for reproducibility

#### Phase 4: Translation Baseline (30-45 min)
1. For low-resource languages (bn, te, sw, ta):
   - Translate question to English
   - Get model response
   - Compare accuracy with direct approach
2. Use a single model (GPT-4o) for translation baseline

#### Phase 5: Analysis (30 min)
1. Calculate per-language accuracy with 95% confidence intervals
2. Compute performance gap ratios
3. Statistical tests for significance
4. Generate visualizations

### Baselines

1. **English baseline**: Performance on English questions (upper bound)
2. **Random baseline**: 25% accuracy for 4-choice questions (lower bound)
3. **Translation baseline**: Translate to English before evaluation

### Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | Correct / Total | Raw performance per language |
| **Performance Gap Ratio (PGR)** | (Acc_en - Acc_lang) / Acc_en | Relative degradation from English |
| **Normalized Score** | (Acc - 0.25) / (Acc_en - 0.25) | Performance above random, normalized to English |
| **Cross-lingual Consistency** | Agreement on aligned samples | Whether model gives same answer across languages |

### Statistical Analysis Plan

1. **Primary analysis**: One-way ANOVA for accuracy across language resource levels
2. **Post-hoc tests**: Tukey HSD for pairwise language group comparisons
3. **Correlation analysis**: Spearman correlation between resource level and accuracy
4. **Effect sizes**: Cohen's d for English vs. low-resource comparison
5. **Significance level**: α = 0.05, with Bonferroni correction for multiple comparisons

---

## Expected Outcomes

### If H1 (English superiority) is supported:
- Low-resource language accuracy will be 20-40% lower than English
- This aligns with prior work (Dong et al., 2024: 0.03 vs 0.15 F1)

### If H2 (resource-level correlation) is supported:
- Clear monotonic decrease: High > Mid > Low resource accuracy
- Spearman ρ > 0.7 (strong correlation)

### If H3 (translation helps) is supported:
- Translated accuracy > Direct accuracy for low-resource languages
- Partial but not complete closure of gap (based on MWork hypothesis)

### If hypotheses are not supported:
- Would suggest 2025 models have improved multilingual training
- Would be an important finding: progress toward multilingual parity

---

## Timeline and Milestones

| Phase | Estimated Duration | Deliverable |
|-------|-------------------|-------------|
| Data Preparation | 10 min | Loaded, validated dataset |
| Model Setup | 10 min | Configured API clients |
| Direct Evaluation | 60-90 min | Raw accuracy for all models/languages |
| Translation Baseline | 30-45 min | Translation-based accuracy |
| Analysis | 30 min | Statistical tests, visualizations |
| Documentation | 30 min | REPORT.md, README.md |
| **Total** | **2.5-3.5 hours** | Complete research deliverables |

---

## Potential Challenges

| Challenge | Mitigation |
|-----------|------------|
| API rate limits | Implement exponential backoff; batch requests; use async calls |
| API costs | Sample 500 questions per language instead of full 12,338 |
| Model response parsing | Use regex patterns; log failures for manual review |
| Translation quality | Use same model (GPT-4o) for consistency; acknowledge as limitation |
| Non-Latin scripts | Verify tokenization; test with sample before full run |

---

## Success Criteria

The research will be considered successful if we achieve:

1. **Complete data collection**: Accuracy measurements for all 4 models × 12 languages
2. **Statistical validity**: >95% response rate, sufficient samples for significance testing
3. **Reproducibility**: All code, prompts, and raw data documented
4. **Actionable insights**: Clear ranking of models by multilingual capability
5. **Novel findings**: Updated 2025 baseline for multilingual LLM performance

---

## File Structure

```
workspace/
├── planning.md              # This file
├── src/
│   ├── evaluate.py          # Main evaluation script
│   ├── analysis.py          # Statistical analysis
│   └── utils.py             # Helper functions
├── results/
│   ├── raw/                 # Raw API responses
│   ├── metrics.json         # Computed metrics
│   └── plots/               # Visualizations
├── REPORT.md                # Final research report
└── README.md                # Project overview
```

---

## References

1. Dong, G., et al. (2024). Evaluating and Mitigating Linguistic Discrimination in Large Language Models. arXiv:2404.18534
2. Zhao, Y., et al. (2024). How do Large Language Models Handle Multilingualism? NeurIPS 2024. arXiv:2402.18815
3. Han, W., et al. (2025). MuBench: Assessment of Multilingual Capabilities. arXiv:2506.19468
4. Luo, H., et al. (2025). GlotEval: Massively Multilingual Evaluation. arXiv:2504.04155
5. Xu, Y., et al. (2024). Survey on Multilingual LLMs. arXiv:2404.00929
