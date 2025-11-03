# Benchmark of BERT vs. Qwen Models for Named Entity Recognition (NER)

This project provides a comprehensive framework to benchmark and compare the performance of the classic `bert-base-cased` model against a range of `Qwen1.5` models (0.5B, 1.8B, 4B, 7B, and 14B) on the task of Named Entity Recognition (NER).

The benchmarking is performed by fine-tuning each model on the standard CoNLL-2003 dataset and evaluating their performance on the test set. The script is designed to be efficient, leveraging 4-bit quantization and Parameter-Efficient Fine-Tuning (PEFT) with LoRA via the `unsloth` library for the larger Qwen models.

## Features

-   **Head-to-Head Comparison**: Directly compares a battle-tested BERT model with the latest generation of Qwen models.
-   **Scalable**: Easily extensible to include other models from the Hugging Face Hub.
-   **Efficient Training**: Uses `unsloth`, LoRA, and 4-bit quantization to enable fine-tuning of large models (up to 14B parameters) on consumer-grade GPUs (like those available in Google Colab).
-   **Standardized Evaluation**: Employs the CoNLL-2003 dataset and the `seqeval` library for robust and standard evaluation metrics (Precision, Recall, F1-score).
-   **Automated Workflow**: The entire process—from data preprocessing to model training, evaluation, and results aggregation—is handled by a single script.
-   **Reproducibility**: `requirements.txt` is provided for easy environment setup.

## Prerequisites

-   Python 3.8+
-   An NVIDIA GPU with CUDA support is highly recommended for reasonable training times.
-   `pip` for package installation.

## Setup

1.  **Clone the repository (or download the files):**
    ```bash
    # If this were a git repository
    # git clone <repository-url>
    # cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    Install all the necessary packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Benchmark

Execute the main Python script from your terminal. The script will sequentially download, fine-tune, and evaluate each model defined in the `models_to_benchmark` dictionary.

```bash
python benchmark.py
```