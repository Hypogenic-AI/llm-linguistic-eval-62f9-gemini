# Evaluating Linguistic Performance and Implicit Translation in LLMs

## Project Overview
This research investigates whether English-centric Large Language Models (LLMs) like GPT-4o use "implicit internal translation" when processing non-English languages. By comparing performance and error patterns across 12 languages (High, Mid, and Low resource) using the MuBench (MMLU) dataset, we quantify the extent of English-centric bias.

## Key Findings
-   **Performance Parity**: High-resource languages (French, Spanish) match or exceed English accuracy on this benchmark.
-   **Shared Errors**: Mid-resource languages (Korean, Vietnamese) show a high "Same Mistake Ratio" (>60%), implying they share a reasoning path with English (supporting the implicit translation hypothesis).
-   **Breakdown**: Low-resource languages (Tamil) show random error patterns, suggesting a failure to map concepts to the model's core English representations.

## File Structure
-   `src/`: Python source code for experiments.
    -   `evaluate.py`: Main evaluation script.
    -   `analyze_results.py`: Analysis and plotting script.
-   `results/`: Experimental outputs.
    -   `raw/`: JSON files with model responses.
    -   `plots/`: Visualizations of accuracy and agreement.
    -   `metrics_summary.csv`: Aggregated results.
-   `REPORT.md`: Detailed research report.

## Reproduction
1.  **Setup**:
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install datasets pandas openai python-dotenv tqdm matplotlib seaborn scikit-learn numpy
    ```
2.  **Run Evaluation**:
    ```bash
    export OPENAI_API_KEY="your_key"
    python src/evaluate.py
    ```
3.  **Analyze**:
    ```bash
    python src/analyze_results.py
    ```
