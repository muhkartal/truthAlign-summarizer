import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

CONFIG = {
    "dataset_name": "cnn_dailymail",
    "dataset_config": "3.0.0",
    "dataset_split_ratio": [0.9, 0.05, 0.05],


    "model_name": "facebook/bart-large-cnn",
    "max_input_length": 1024,
    "max_output_length": 128,

    "batch_size": 4,
    "learning_rate": 3e-5,
    "num_epochs": 3,
    "gradient_accumulation_steps": 4,
    "mixed_precision": "fp16",  # Options: None, "fp16", "bf16"

    "output_dir": "./output",
    "model_dir": "./output/model",
    "summary_dir": "./output/summaries",

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_gpus": torch.cuda.device_count()
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["model_dir"], exist_ok=True)
os.makedirs(CONFIG["summary_dir"], exist_ok=True)

print(f"Using device: {CONFIG['device']}")
print(f"Number of GPUs: {CONFIG['num_gpus']}")
print(f"Dataset: {CONFIG['dataset_name']}")
print(f"Model: {CONFIG['model_name']}")
