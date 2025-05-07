from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

def load_and_prepare_dataset(dataset_name, dataset_config=None):
    """
    Load dataset and split into train, validation, and test sets
    Args:
        dataset_name: Name of the dataset (e.g., 'cnn_dailymail', 'xsum')
        dataset_config: Configuration name for the dataset
    Returns:
        Dictionary containing dataset splits
    """
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        dataset = load_dataset(dataset_name)

    if dataset_name == "xsum":
        return {
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test": dataset["test"]
        }
    elif dataset_name == "cnn_dailymail":
        return {
            "train": dataset["train"],
            "validation": dataset["validation"],
            "test": dataset["test"]
        }
    else:
        train_val_test = dataset["train"].train_test_split(test_size=0.1)
        train_val = train_val_test["train"].train_test_split(test_size=0.05)

        return {
            "train": train_val["train"],
            "validation": train_val["test"],
            "test": train_val_test["test"]
        }

def preprocess_data(examples, tokenizer, max_input_length, max_output_length, text_column, summary_column):
    inputs = [doc for doc in examples[text_column]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples[summary_column], max_length=max_output_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def explore_dataset(dataset_splits, text_column, summary_column):

    sample_size = min(1000, len(dataset_splits["train"]))
    train_sample = dataset_splits["train"].select(range(sample_size))

    doc_lengths = [len(word_tokenize(doc)) for doc in train_sample[text_column]]
    summary_lengths = [len(word_tokenize(summ)) for summ in train_sample[summary_column]]

    compression_ratios = [len(word_tokenize(summ)) / len(word_tokenize(doc))
                         for doc, summ in zip(train_sample[text_column], train_sample[summary_column])]

    doc_sentences = [len(sent_tokenize(doc)) for doc in train_sample[text_column]]
    summary_sentences = [len(sent_tokenize(summ)) for summ in train_sample[summary_column]]

    analysis_df = pd.DataFrame({
        'doc_length': doc_lengths,
        'summary_length': summary_lengths,
        'compression_ratio': compression_ratios,
        'doc_sentences': doc_sentences,
        'summary_sentences': summary_sentences
    })

    print("\nDataset Statistics:")
    print(f"Number of training examples: {len(dataset_splits['train'])}")
    print(f"Number of validation examples: {len(dataset_splits['validation'])}")
    print(f"Number of test examples: {len(dataset_splits['test'])}")

    print("\nDocument Length Statistics:")
    print(analysis_df['doc_length'].describe())

    print("\nSummary Length Statistics:")
    print(analysis_df['summary_length'].describe())

    print("\nCompression Ratio Statistics:")
    print(analysis_df['compression_ratio'].describe())

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    sns.histplot(doc_lengths, kde=True, ax=axs[0, 0])
    axs[0, 0].set_title('Document Length Distribution')
    axs[0, 0].set_xlabel('Number of Words')

    sns.histplot(summary_lengths, kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Summary Length Distribution')
    axs[0, 1].set_xlabel('Number of Words')

    sns.histplot(compression_ratios, kde=True, ax=axs[1, 0])
    axs[1, 0].set_title('Compression Ratio Distribution')
    axs[1, 0].set_xlabel('Summary Length / Document Length')

    sns.scatterplot(x='doc_length', y='summary_length', data=analysis_df, ax=axs[1, 1], alpha=0.5)
    axs[1, 1].set_title('Document vs Summary Length')
    axs[1, 1].set_xlabel('Document Length (words)')
    axs[1, 1].set_ylabel('Summary Length (words)')

    plt.tight_layout()
    plt.savefig("dataset_analysis.png")
    plt.close()

    print("\nSample Document:")
    print(train_sample[text_column][0][:500] + "...")
    print("\nSample Summary:")
    print(train_sample[summary_column][0])

    return analysis_df

if __name__ == "__main__":
    dataset_name = "cnn_dailymail"
    dataset_config = "3.0.0"

    if dataset_name == "cnn_dailymail":
        text_column = "article"
        summary_column = "highlights"
    elif dataset_name == "xsum":
        text_column = "document"
        summary_column = "summary"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset_splits = load_and_prepare_dataset(dataset_name, dataset_config)
    analysis_df = explore_dataset(dataset_splits, text_column, summary_column)

    max_input_length = 1024
    max_output_length = 128

    tokenized_sample = preprocess_data(
        dataset_splits["train"].select(range(5)),
        tokenizer,
        max_input_length,
        max_output_length,
        text_column,
        summary_column
    )

    print("\nTokenized Sample:")
    print(f"Input shape: {len(tokenized_sample['input_ids'])}")
    print(f"First input sequence length: {len(tokenized_sample['input_ids'][0])}")
    print(f"First label sequence length: {len(tokenized_sample['labels'][0])}")
