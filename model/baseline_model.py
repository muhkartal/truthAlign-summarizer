import os
import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

def train_baseline_model(config, dataset_splits, tokenized_datasets):
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=config["model_name"],
        padding=True
    )

    rouge_metric = load_metric("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]

        result = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )

        result = {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}

        prediction_lens = [len(pred.split()) for pred in decoded_preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=config["model_dir"],
        evaluation_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        num_train_epochs=config["num_epochs"],
        save_total_limit=2,
        fp16=config["mixed_precision"] == "fp16",
        predict_with_generate=True,
        generation_max_length=config["max_output_length"],
        generation_num_beams=4,
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print(f"Training baseline model: {config['model_name']}")
    trainer.train()

    trainer.save_model(config["model_dir"])

    print("Evaluating on test set...")
    test_results = trainer.evaluate(tokenized_datasets["test"])
    print(f"Test Results: {test_results}")

    return model, tokenizer

def generate_summaries(model, tokenizer, dataset_split, config, text_column, batch_size=8):

    model.to(config["device"])
    model.eval()

    articles = dataset_split[text_column]
    num_samples = len(articles)
    generated_summaries = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Generating summaries"):
        batch_articles = articles[i:i+batch_size]

        inputs = tokenizer(
            batch_articles,
            max_length=config["max_input_length"],
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(config["device"])

        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=config["max_output_length"],
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2
            )

        # Decode generated summaries
        batch_summaries = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        generated_summaries.extend(batch_summaries)

    return generated_summaries

def save_summaries(summaries, reference_summaries, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, (summary, reference) in enumerate(zip(summaries, reference_summaries)):
            f.write(f"Example {i+1}\n")
            f.write(f"Generated: {summary}\n")
            f.write(f"Reference: {reference}\n")
            f.write("-" * 100 + "\n")

if __name__ == "__main__":
    from environment_setup import CONFIG
    from dataset_preparation import load_and_prepare_dataset, preprocess_data

    if CONFIG["dataset_name"] == "cnn_dailymail":
        text_column = "article"
        summary_column = "highlights"
    elif CONFIG["dataset_name"] == "xsum":
        text_column = "document"
        summary_column = "summary"
    else:
        raise ValueError(f"Unsupported dataset: {CONFIG['dataset_name']}")

    print(f"Loading dataset: {CONFIG['dataset_name']}")
    dataset_splits = load_and_prepare_dataset(CONFIG["dataset_name"], CONFIG["dataset_config"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    print("Tokenizing datasets...")
    tokenized_datasets = {}
    for split in dataset_splits:
        tokenized_datasets[split] = dataset_splits[split].map(
            lambda examples: preprocess_data(
                examples,
                tokenizer,
                CONFIG["max_input_length"],
                CONFIG["max_output_length"],
                text_column,
                summary_column
            ),
            batched=True,
            remove_columns=dataset_splits[split].column_names
        )

    model, tokenizer = train_baseline_model(CONFIG, dataset_splits, tokenized_datasets)

    print("Generating summaries for test set...")
    test_summaries = generate_summaries(
        model,
        tokenizer,
        dataset_splits["test"],
        CONFIG,
        text_column,
        batch_size=CONFIG["batch_size"]
    )

    print("Saving summaries...")
    reference_summaries = dataset_splits["test"][summary_column]
    output_path = os.path.join(CONFIG["summary_dir"], "baseline_summaries.txt")
    save_summaries(test_summaries, reference_summaries, output_path)

    print(f"Baseline model training and evaluation complete. Summaries saved to {output_path}")
