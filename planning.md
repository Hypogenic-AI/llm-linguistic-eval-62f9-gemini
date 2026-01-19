# Research Plan: Evaluating Linguistic Performance and Implicit Translation in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Large language models (LLMs) are increasingly deployed globally, yet their training data remains predominantly English-centric. This creates a risk of "linguistic discrimination" where non-English users receive inferior service or safety guarantees. Understanding whether models truly "understand" other languages or merely translate them internally to English is crucial for:
1.  **Reliability**: Knowing if errors are due to reasoning failures or translation artifacts.
2.  **Efficiency**: Determining if native-language processing is viable or if explicit translation wrappers are necessary.
3.  **Equity**: Quantifying the performance gap affecting billions of non-English speakers.

### Gap in Existing Work
While benchmarks like MMLU and FLORES exists, there is limited systematic analysis comparing "Direct Inference" in native languages against "Explicit Translation" baselines across a spectrum of resource levels (High, Mid, Low). Most evaluations treat the model as a black box without probing the "implicit translation" hypothesis—the idea that models internalize non-English input, process it in an "English-like" latent space, and decode it back.

### Our Novel Contribution
We will explicitly test the "implicit translation" hypothesis by comparing:
1.  **Direct Inference**: Evaluating the model directly in target languages.
2.  **English Pivot**: Translating queries to English (simulated by using aligned English samples), solving in English, and comparing performance.
This allows us to quantify the "Translation Tax" the model pays when operating in non-English languages and infer if its internal mechanism mimics a translation pipeline.

### Experiment Justification
-   **Experiment 1 (Multilingual Performance Profiling)**: Establish a baseline of performance across 8 diverse languages to quantify the "English-centric bias".
-   **Experiment 2 (Implicit vs Explicit Translation)**: Compare native performance against an English-pivot baseline. If the model's native performance tracks closely with English performance (adjusted for translation difficulty), it supports the internal translation hypothesis.

## Research Question
Do English-centric Large Language Models exhibit an "implicit internal translation" mechanism when processing non-English languages, and how does the performance gap between English and non-English languages vary across resource levels?

## Proposed Methodology

### Approach
We will use the **MuBench** dataset (specifically the MMLU subset), which provides parallel/aligned questions across languages. This alignment allows us to treat the English version of a question as a "perfect translation" of the non-English version, enabling a clean comparison between "Native Processing" and "English Processing".

### Experimental Steps
1.  **Data Loading**: Load aligned MMLU examples from MuBench for 8 selected languages:
    -   **High Resource**: English (en), Chinese (zh), Spanish (es)
    -   **Mid Resource**: Arabic (ar), Indonesian (id), Vietnamese (vi)
    -   **Low Resource**: Bengali (bn), Swahili (sw)
2.  **Model Inference**:
    -   Use **GPT-4o** (via OpenRouter/OpenAI API) as the representative SOTA model.
    -   For each language, prompt the model to answer the multiple-choice questions.
3.  **Evaluation**:
    -   Calculate accuracy for each language.
    -   Calculate "Consistency": Does the model give the same answer (A/B/C/D) for the aligned question in Language X and English?

### Baselines
-   **English Baseline**: The performance on the English dataset serves as the "ceiling" or "gold standard".
-   **Random Baseline**: 25% accuracy (for 4-choice MMLU).

### Evaluation Metrics
-   **Accuracy**: Percentage of correct answers.
-   **Performance Drop**: `(Acc_En - Acc_Lang) / Acc_En`.
-   **Alignment Score**: Percentage of times `Answer(Lang) == Answer(En)`. High alignment suggests the model might be reasoning similarly (or translating) across languages.

### Statistical Analysis Plan
-   Compute 95% confidence intervals for accuracy.
-   Pearson correlation between "Resource Level" (proxy) and Accuracy.
-   Paired t-tests to compare Native vs English accuracy on the same question set.

## Expected Outcomes
-   **Hypothesis Supported**: If `Acc(Lang)` is highly correlated with `Acc(En)` but lower, and `Alignment Score` is high, it suggests the model relies on English-centric reasoning.
-   **Hypothesis Refuted**: If `Acc(Lang)` is uncorrelated or if `Alignment Score` is low (model gives different *wrong* answers), it suggests independent (and likely poorer) representations for non-English languages.

## Timeline and Milestones
-   **Phase 2**: Environment Setup (Completed).
-   **Phase 3 (1 hour)**: Implement data loader and inference script.
-   **Phase 4 (1.5 hours)**: Run experiments on ~100-200 samples per language (to save time/cost while maintaining statistical relevance).
-   **Phase 5 (0.5 hours)**: Analyze results and generate plots.
-   **Phase 6 (0.5 hours)**: Write report.

## Potential Challenges
-   **API Costs/Limits**: We will limit sample size to ~100 per language initially.
-   **Data Quality**: Translations in MuBench might be imperfect. We assume they are high quality as per the paper.
-   **Rate Limiting**: Implement robust retry logic.

## Success Criteria
-   Successful execution of evaluation across 8 languages.
-   Clear quantitative evidence (plots/tables) showing the gap between English and non-English performance.
-   A conclusion regarding the "implicit translation" hypothesis based on the alignment/consistency data.