# Step 2: Import libraries
from unsloth import FastLanguageModel
import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig

import seqeval
from seqeval.metrics import classification_report


# Step 3 (Corrected): Load and Preprocess the Dataset
def load_and_prepare_dataset(tokenizer):
    """Loads the CoNLL-2003 dataset and prepares it for training."""
    # The only change is adding trust_remote_code=True
    dataset = load_dataset("conll2003", trust_remote_code=True)

    label_list = dataset["train"].features["ner_tags"].feature.names

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    return tokenized_datasets, label_list


# Step 4: Define Evaluation Metrics
def compute_metrics(p):
    """Computes precision, recall, F1, and accuracy for seqeval."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(true_labels, true_predictions, output_dict=True)
    return {
        "precision": report["micro avg"]["precision"],
        "recall": report["micro avg"]["recall"],
        "f1-score": report["micro avg"]["f1-score"],
        "accuracy": report["micro avg"]["precision"],
    }


# Step 5: Define the Fine-Tuning and Evaluation Function
def fine_tune_and_evaluate(model_name, use_unsloth=False):
    """
    Fine-tunes and evaluates a given model for NER.

    Args:
        model_name (str): The name of the model from the Hugging Face Hub.
        use_unsloth (bool): Whether to use unsloth for faster fine-tuning.
    """
    print(f"--- Starting Benchmark for: {model_name} ---")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and prepare dataset
    tokenized_datasets, label_list_local = load_and_prepare_dataset(tokenizer)
    global label_list
    label_list = label_list_local

    # Load model
    if use_unsloth:
        model, _ = FastLanguageModel.from_pretrained(
            model_name,
            model_config={"num_labels": len(label_list)},
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="TOKEN_CLS",
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=len(label_list)
        )

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./{model_name.replace('/', '_')}-ner",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Evaluate the model on the test set
    print("Starting evaluation on the test set...")
    test_results = trainer.predict(tokenized_datasets["test"])

    # Save results
    results = {
        "model": model_name,
        "precision": test_results.metrics["test_precision"],
        "recall": test_results.metrics["test_recall"],
        "f1-score": test_results.metrics["test_f1-score"],
        "accuracy": test_results.metrics["test_accuracy"],
    }

    print(f"--- Benchmark Results for: {model_name} ---")
    print(results)

    return results


# Step 6: Run the Benchmark
if __name__ == "__main__":
    # List of models to benchmark
    models_to_benchmark = {
        "bert-base-cased": False,
        "Qwen/Qwen1.5-0.5B": True,
        "Qwen/Qwen1.5-1.8B": True,
        "Qwen/Qwen1.5-4B": True,
        "Qwen/Qwen1.5-7B": True,
        "Qwen/Qwen1.5-14B": True,
    }

    all_results = []

    for model_name, use_unsloth in models_to_benchmark.items():
        try:
            result = fine_tune_and_evaluate(model_name, use_unsloth=use_unsloth)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to benchmark {model_name}. Error: {e}")
            all_results.append({"model": model_name, "error": str(e)})

    # Step 7: Display Final Results
    print("\n\n--- Final Benchmark Summary ---")
    results_df = pd.DataFrame(all_results)
    print(results_df)

    # Save results to a CSV file
    results_df.to_csv("ner_benchmark_results.csv", index=False)
    print("\nResults saved to ner_benchmark_results.csv")