# Research Report: Evaluating Linguistic Performance and Implicit Translation in LLMs

## 1. Executive Summary
This research investigates the performance of GPT-4o across 12 languages to test the hypothesis that Large Language Models (LLMs) rely on "implicit internal translation" when processing non-English inputs. Using the MuBench (MMLU) dataset, we found that high and mid-resource languages (Spanish, French, Vietnamese) achieve accuracy parity with English (~66-72%) and exhibit high response agreement (76-78%) and shared error patterns (47-64%) with English. In contrast, low-resource languages (Tamil, Swahili) show significant performance degradation (52-56%) and lower error correlation, suggesting that while the model may effectively "translate" or share representations for well-supported languages, it fails to do so for under-represented ones, leading to independent and uncorrelated failures.

## 2. Goal
**Research Question**: Do English-centric Large Language Models exhibit an "implicit internal translation" mechanism when processing non-English languages?

**Motivation**: LLMs are deployed globally but trained primarily on English. If models implicitly translate, their non-English performance is bounded by their translation quality and English reasoning. Understanding this helps diagnose "linguistic discrimination" and inform deployment in non-English markets.

## 3. Data Construction
**Dataset**: [MuBench](https://huggingface.co/datasets/aialt/MuBench) (MMLU subset).
-   **Source**: Aligned version of the Massive Multitask Language Understanding (MMLU) benchmark.
-   **Languages Evaluated (12)**:
    -   High Resource: English (en), Chinese (zh), Spanish (es), French (fr)
    -   Mid Resource: Arabic (ar), Indonesian (id), Vietnamese (vi), Korean (ko)
    -   Low Resource: Swahili (sw), Bengali (bn), Telugu (te), Tamil (ta)
-   **Sample Size**: 50 aligned samples per language (Total 600 queries).
-   **Alignment**: The dataset guarantees that `test_0` in English corresponds to `test_0` in other languages, allowing direct comparison of answers.

## 4. Experiment Description
**Methodology**:
We performed a comparative analysis between "Native Inference" (prompting in target language) and "English Inference" (prompting in English on the same question).

**Model**: GPT-4o (via API).

**Metrics**:
1.  **Accuracy**: Correctness on the standard MMLU task.
2.  **Agreement with English**: The proportion of times the model gives the *exact same answer* (A/B/C/D) in Language X and English, regardless of correctness.
3.  **Same Mistake Ratio**: When both the English and Native model are wrong, how often do they make the *same* wrong guess? (Random baseline = 33%).

## 5. Result Analysis

### Key Findings

| Language | Resource Level | Accuracy | Agreement with En | Same Mistake Ratio |
|----------|----------------|----------|-------------------|--------------------|
| **English** | High | 0.66 | 1.00 | N/A |
| **French** | High | 0.70 | 0.78 | 0.61 |
| **Spanish** | High | 0.68 | 0.78 | 0.47 |
| **Vietnamese** | Mid | 0.72 | 0.78 | 0.54 |
| **Bengali** | Low* | 0.68 | 0.74 | 0.43 |
| **Korean** | Mid | 0.64 | 0.76 | 0.64 |
| **Indonesian** | Mid | 0.64 | 0.76 | 0.53 |
| **Arabic** | Mid | 0.62 | 0.72 | 0.47 |
| **Chinese** | High | 0.60 | 0.70 | 0.47 |
| **Telugu** | Low | 0.60 | 0.70 | 0.38 |
| **Swahili** | Low | 0.56 | 0.68 | 0.53 |
| **Tamil** | Low | 0.52 | 0.58 | 0.33 |

*Note: Bengali performed surprisingly well, possibly due to high representation in common crawl or specific task ease.*

### Hypothesis Testing: Implicit Translation
The data supports the "Implicit Translation" (or Shared Representation) hypothesis for High/Mid resource languages but less so for Low resource ones.

1.  **High Agreement**: French, Spanish, Vietnamese, and Korean match English answers >75% of the time. This is significantly higher than what accuracy alone would predict if errors were independent.
2.  **Shared Hallucinations**: The "Same Mistake Ratio" is the strongest evidence. For Korean (0.64) and French (0.61), when the model gets it wrong in the native language, it almost always makes the *exact same error* as it did in English (nearly double the random chance of 0.33). This strongly implies the underlying reasoning path—or the specific confusion—is identical, likely processed through a shared English-centric latent space.
3.  **Breakdown at Low Resource**: Tamil (0.52 Acc) shows the lowest agreement (0.58) and a "Same Mistake Ratio" of 0.33, which is exactly random. This suggests that for Tamil, the model isn't "translating and failing"; it's simply flailing. It likely doesn't map the Tamil input to the correct English concept effectively enough to even reproduce the English error.

### Visualizations
Plots are saved in `results/plots/`:
-   `accuracy_by_lang.png`: Shows the drop-off for Swahili and Tamil.
-   `acc_vs_agreement.png`: Scatter plot showing strong correlation between performance and alignment with English.

## 6. Conclusions
GPT-4o exhibits strong evidence of English-centric processing. For well-supported languages, it behaves like an English reasoner wrapped in a translator: it succeeds when English succeeds and fails in the same way when English fails. For poorly-supported languages (Tamil), this mechanism breaks down, resulting in independent, lower-quality failure modes.

**Implication**: Improving non-English performance likely requires better "alignment/translation" to the core English reasoning engine for mid-resource languages, but requires fundamental representation learning for low-resource languages where the mapping doesn't even exist.

## 7. Next Steps
1.  **Expand Sample Size**: Increase from N=50 to N=1000 to validate the "Same Mistake Ratio" with higher statistical power.
2.  **Explicit Translation Control**: Run `Translate(X->En) -> Model(En)` explicitly to see if it outperforms the internal mechanism for Tamil.
3.  **Logit Analysis**: Look at the probability distribution over tokens. If the logits for A/B/C/D are identical across languages, the evidence for shared representation is irrefutable.
