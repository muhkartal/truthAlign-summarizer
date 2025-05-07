# main.py
# Main script for improving factual consistency in abstractive summarization

import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
import nltk
from nltk.tokenize import sent_tokenize
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import argparse
import logging
from datetime import datetime

# Import project modules
from environment_setup import CONFIG
from dataset_preparation import load_and_prepare_dataset, preprocess_data, explore_dataset
from baseline_model import train_baseline_model, generate_summaries, save_summaries
from factuality_metrics import FactualityEvaluator
from rl_enhancement import FactualityRLTrainer
from postprocessing_correction import FactualityPostProcessor, FactualSentenceRewriter
from factuality_decoding import FactualityGuidedDecoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Factual Consistency Improvement for Abstractive Summarization")

    # Dataset options
    parser.add_argument("--dataset", type=str, default="cnn_dailymail", choices=["cnn_dailymail", "xsum"],
                        help="Dataset to use for training and evaluation")
    parser.add_argument("--dataset_config", type=str, default="3.0.0",
                        help="Configuration for the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to use (None for all)")

    # Model options
    parser.add_argument("--model", type=str, default="facebook/bart-large-cnn",
                        choices=["facebook/bart-large-cnn", "t5-base", "google/pegasus-cnn_dailymail"],
                        help="Base model to use for summarization")
    parser.add_argument("--max_input_length", type=int, default=1024,
                        help="Maximum input sequence length")
    parser.add_argument("--max_output_length", type=int, default=128,
                        help="Maximum output sequence length")

    # Training options
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")

    # Experiment options
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip baseline model training (use pretrained model)")
    parser.add_argument("--skip_rl", action="store_true",
                        help="Skip RL enhancement")
    parser.add_argument("--skip_postprocessing", action="store_true",
                        help="Skip post-processing evaluation")
    parser.add_argument("--skip_factuality_decoding", action="store_true",
                        help="Skip factuality-guided decoding evaluation")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save output files")

    # Hardware options
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use for training (cuda, cpu, None for auto-detect)")

    return parser.parse_args()

def update_config_from_args(args):
    """Update configuration from command line arguments"""
    config = CONFIG.copy()

    # Update from args
    config["dataset_name"] = args.dataset
    config["dataset_config"] = args.dataset_config
    config["model_name"] = args.model
    config["max_input_length"] = args.max_input_length
    config["max_output_length"] = args.max_output_length
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    config["num_epochs"] = args.num_epochs
    config["output_dir"] = args.output_dir

    # Set device if specified
    if args.device:
        config["device"] = args.device

    # Set dataset column names
    if config["dataset_name"] == "cnn_dailymail":
        config["text_column"] = "article"
        config["summary_column"] = "highlights"
    elif config["dataset_name"] == "xsum":
        config["text_column"] = "document"
        config["summary_column"] = "summary"
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset_name']}")

    # Create output directories
    os.makedirs(config["output_dir"], exist_ok=True)
    config["model_dir"] = os.path.join(config["output_dir"], "models")
    os.makedirs(config["model_dir"], exist_ok=True)
    config["summary_dir"] = os.path.join(config["output_dir"], "summaries")
    os.makedirs(config["summary_dir"], exist_ok=True)
    config["results_dir"] = os.path.join(config["output_dir"], "results")
    os.makedirs(config["results_dir"], exist_ok=True)

    # Generate experiment ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config["experiment_id"] = f"{config['dataset_name']}_{config['model_name'].split('/')[-1]}_{timestamp}"

    # Log configuration
    logger.info(f"Configuration: {config}")

    return config

def evaluate_all_methods(config, dataset_splits, baseline_model, baseline_tokenizer):
    """Evaluate all factuality improvement methods"""
    # Column names
    text_column = config["text_column"]
    summary_column = config["summary_column"]

    # Use subset of test set for evaluation
    test_set = dataset_splits["test"]
    if config.get("max_eval_samples"):
        test_set = test_set.select(range(min(len(test_set), config["max_eval_samples"])))

    logger.info(f"Evaluating on {len(test_set)} test samples")

    # Source documents and reference summaries
    source_docs = test_set[text_column]
    reference_summaries = test_set[summary_column]

    # Initialize factuality evaluator
    factuality_evaluator = FactualityEvaluator(device=config["device"])

    # Dictionary to store all results
    all_results = {
        "metrics": {},
        "summaries": {}
    }

    # 1. Generate and evaluate baseline summaries
    logger.info("Generating baseline summaries")
    baseline_summaries = generate_summaries(
        baseline_model,
        baseline_tokenizer,
        test_set,
        config,
        text_column,
        batch_size=config["batch_size"]
    )
    all_results["summaries"]["baseline"] = baseline_summaries

    # Calculate ROUGE scores for baseline
    logger.info("Calculating ROUGE scores for baseline")
    rouge = load_metric("rouge")
    baseline_rouge = rouge.compute(
        predictions=["\n".join(sent_tokenize(s)) for s in baseline_summaries],
        references=["\n".join(sent_tokenize(s)) for s in reference_summaries],
        use_stemmer=True
    )
    baseline_rouge = {k: round(v.mid.fmeasure * 100, 2) for k, v in baseline_rouge.items()}

    # Calculate factuality metrics for baseline
    logger.info("Calculating factuality metrics for baseline")
    baseline_factuality = factuality_evaluator.evaluate_factuality(source_docs, baseline_summaries)

    # Store baseline metrics
    all_results["metrics"]["baseline"] = {
        "rouge": baseline_rouge,
        "factuality": baseline_factuality
    }

    # 2. RL enhancement
    if not args.skip_rl:
        logger.info("Training RL-enhanced model")

        # Initialize RL trainer
        rl_trainer = FactualityRLTrainer(
            base_model_name=config["model_name"],
            learning_rate=config["learning_rate"] / 2,  # Lower learning rate for RL
            max_input_length=config["max_input_length"],
            max_output_length=config["max_output_length"],
            device=config["device"]
        )

        # Train with RL (on a subset of training data)
        train_subset = dataset_splits["train"]
        if config.get("max_rl_samples"):
            train_subset = train_subset.select(range(min(len(train_subset), config["max_rl_samples"])))

        rl_trainer.train_rl(
            train_dataset=train_subset,
            text_column=text_column,
            num_epochs=max(1, config["num_epochs"] // 2),  # Fewer epochs for RL
            batch_size=1,  # RL typically needs smaller batch size
            eval_interval=50
        )

        # Generate summaries with RL-enhanced model
        logger.info("Generating RL-enhanced summaries")
        rl_summaries = rl_trainer.generate_summaries(
            dataset=test_set,
            text_column=text_column,
            batch_size=config["batch_size"]
        )
        all_results["summaries"]["rl"] = rl_summaries

        # Calculate ROUGE scores for RL
        logger.info("Calculating ROUGE scores for RL")
        rl_rouge = rouge.compute(
            predictions=["\n".join(sent_tokenize(s)) for s in rl_summaries],
            references=["\n".join(sent_tokenize(s)) for s in reference_summaries],
            use_stemmer=True
        )
        rl_rouge = {k: round(v.mid.fmeasure * 100, 2) for k, v in rl_rouge.items()}

        # Calculate factuality metrics for RL
        logger.info("Calculating factuality metrics for RL")
        rl_factuality = factuality_evaluator.evaluate_factuality(source_docs, rl_summaries)

        # Store RL metrics
        all_results["metrics"]["rl"] = {
            "rouge": rl_rouge,
            "factuality": rl_factuality
        }

    # 3. Post-processing
    if not args.skip_postprocessing:
        logger.info("Applying post-processing correction")

        # Initialize post-processor
        post_processor = FactualityPostProcessor(
            threshold=0.7,
            device=config["device"]
        )

        # Apply post-processing to baseline summaries
        filtered_summaries, fact_check_results = post_processor.correct_batch(
            source_docs, baseline_summaries
        )
        all_results["summaries"]["post_processing"] = filtered_summaries

        # Calculate ROUGE scores for post-processed summaries
        logger.info("Calculating ROUGE scores for post-processed summaries")
        pp_rouge = rouge.compute(
            predictions=["\n".join(sent_tokenize(s)) for s in filtered_summaries],
            references=["\n".join(sent_tokenize(s)) for s in reference_summaries],
            use_stemmer=True
        )
        pp_rouge = {k: round(v.mid.fmeasure * 100, 2) for k, v in pp_rouge.items()}

        # Calculate factuality metrics for post-processed summaries
        logger.info("Calculating factuality metrics for post-processed summaries")
        pp_factuality = factuality_evaluator.evaluate_factuality(source_docs, filtered_summaries)

        # Analyze post-processing corrections
        pp_analysis = post_processor.analyze_corrections(
            baseline_summaries, filtered_summaries, fact_check_results
        )

        # Store post-processing metrics
        all_results["metrics"]["post_processing"] = {
            "rouge": pp_rouge,
            "factuality": pp_factuality,
            "analysis": pp_analysis
        }

    # 4. Factuality-guided decoding
    if not args.skip_factuality_decoding:
        logger.info("Applying factuality-guided decoding")

        # Initialize factuality-guided decoder
        factual_decoder = FactualityGuidedDecoder(
            model_name=config["model_name"],
            max_input_length=config["max_input_length"],
            max_output_length=config["max_output_length"],
            device=config["device"]
        )

        # Generate summaries with factuality-guided decoding
        logger.info("Generating summaries with factuality-guided decoding")
        factual_decoding_summaries = factual_decoder.generate_summaries(
            test_set,
            text_column=text_column,
            batch_size=config["batch_size"] // 2  # Reduce batch size due to memory requirements
        )
        all_results["summaries"]["factual_decoding"] = factual_decoding_summaries

        # Calculate ROUGE scores for factuality-guided decoding
        logger.info("Calculating ROUGE scores for factuality-guided decoding")
        fd_rouge = rouge.compute(
            predictions=["\n".join(sent_tokenize(s)) for s in factual_decoding_summaries],
            references=["\n".join(sent_tokenize(s)) for s in reference_summaries],
            use_stemmer=True
        )
        fd_rouge = {k: round(v.mid.fmeasure * 100, 2) for k, v in fd_rouge.items()}

        # Calculate factuality metrics for factuality-guided decoding
        logger.info("Calculating factuality metrics for factuality-guided decoding")
        fd_factuality = factuality_evaluator.evaluate_factuality(source_docs, factual_decoding_summaries)

        # Store factuality-guided decoding metrics
        all_results["metrics"]["factual_decoding"] = {
            "rouge": fd_rouge,
            "factuality": fd_factuality
        }

    # Save all results
    results_file = os.path.join(config["results_dir"], f"results_{config['experiment_id']}.json")
    with open(results_file, 'w') as f:
        json.dump(all_results["metrics"], f, indent=4)

    # Save all summaries
    summaries_file = os.path.join(config["summary_dir"], f"all_summaries_{config['experiment_id']}.json")
    with open(summaries_file, 'w') as f:
        json.dump(all_results["summaries"], f, indent=4)

    return all_results

def visualize_results(results, config):
    """Create visualizations of results"""
    metrics_data = results["metrics"]

    # 1. ROUGE Comparison
    rouge_data = {}
    for method, method_metrics in metrics_data.items():
        if "rouge" in method_metrics:
            rouge_data[method] = method_metrics["rouge"]

    # Create DataFrame for ROUGE scores
    rouge_df = pd.DataFrame(columns=["Method", "Metric", "Score"])
    for method, rouge_scores in rouge_data.items():
        for metric, score in rouge_scores.items():
            if metric in ["rouge1", "rouge2", "rougeL"]:
                rouge_df = rouge_df.append({
                    "Method": method.capitalize(),
                    "Metric": metric,
                    "Score": score
                }, ignore_index=True)

    # Plot ROUGE comparison
    plt.figure(figsize=(12, 6))
    rouge_plot = sns.barplot(x="Metric", y="Score", hue="Method", data=rouge_df)
    plt.title("ROUGE Scores Comparison Across Methods")
    plt.xlabel("ROUGE Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], f"rouge_comparison_{config['experiment_id']}.png"))
    plt.close()

    # 2. Factuality Comparison
    factuality_data = {}
    for method, method_metrics in metrics_data.items():
        if "factuality" in method_metrics:
            factuality_data[method] = {
                "NLI Score": method_metrics["factuality"]["nli_factuality_score"],
                "QA Score": method_metrics["factuality"]["qa_consistency_score"],
                "Overall": method_metrics["factuality"]["overall_factuality_score"]
            }

    # Create DataFrame for factuality scores
    factuality_df = pd.DataFrame(columns=["Method", "Metric", "Score"])
    for method, fact_scores in factuality_data.items():
        for metric, score in fact_scores.items():
            factuality_df = factuality_df.append({
                "Method": method.capitalize(),
                "Metric": metric,
                "Score": score
            }, ignore_index=True)

    # Plot factuality comparison
    plt.figure(figsize=(12, 6))
    fact_plot = sns.barplot(x="Metric", y="Score", hue="Method", data=factuality_df)
    plt.title("Factuality Scores Comparison Across Methods")
    plt.xlabel("Factuality Metric")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], f"factuality_comparison_{config['experiment_id']}.png"))
    plt.close()

    # 3. Trade-off Analysis (ROUGE vs Factuality)
    tradeoff_df = pd.DataFrame(columns=["Method", "ROUGE-1", "Factuality"])
    for method, method_metrics in metrics_data.items():
        if "rouge" in method_metrics and "factuality" in method_metrics:
            tradeoff_df = tradeoff_df.append({
                "Method": method.capitalize(),
                "ROUGE-1": method_metrics["rouge"]["rouge1"],
                "Factuality": method_metrics["factuality"]["overall_factuality_score"]
            }, ignore_index=True)

    # Plot trade-off
    plt.figure(figsize=(10, 8))
    trade_plot = sns.scatterplot(
        x="ROUGE-1",
        y="Factuality",
        hue="Method",
        style="Method",
        s=100,
        data=tradeoff_df
    )

    # Add labels for each point
    for i, row in tradeoff_df.iterrows():
        plt.annotate(
            row["Method"],
            (row["ROUGE-1"], row["Factuality"]),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.title("Trade-off Analysis: ROUGE-1 vs Factuality")
    plt.xlabel("ROUGE-1 Score")
    plt.ylabel("Factuality Score")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(config["results_dir"], f"tradeoff_analysis_{config['experiment_id']}.png"))
    plt.close()

    # 4. Generate HTML report
    report_html = f"""
    <html>
    <head>
        <title>Factual Consistency in Summarization - Results Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .method {{ font-weight: bold; }}
            .highlight {{ background-color: #e6f7ff; }}
            img {{ max-width: 100%; height: auto; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Factual Consistency in Abstractive Summarization</h1>
        <h2>Experiment Results</h2>
        <p><strong>Experiment ID:</strong> {config['experiment_id']}</p>
        <p><strong>Dataset:</strong> {config['dataset_name']}</p>
        <p><strong>Model:</strong> {config['model_name']}</p>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h3>ROUGE Scores Comparison</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>ROUGE-1</th>
                <th>ROUGE-2</th>
                <th>ROUGE-L</th>
            </tr>
    """

    # Add ROUGE scores to report
    for method, method_metrics in metrics_data.items():
        if "rouge" in method_metrics:
            rouge_scores = method_metrics["rouge"]
            report_html += f"""
            <tr>
                <td class="method">{method.capitalize()}</td>
                <td>{rouge_scores.get("rouge1", "N/A")}</td>
                <td>{rouge_scores.get("rouge2", "N/A")}</td>
                <td>{rouge_scores.get("rougeL", "N/A")}</td>
            </tr>
            """

    report_html += """
        </table>

        <h3>Factuality Scores Comparison</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>NLI Score</th>
                <th>QA Score</th>
                <th>Overall Factuality</th>
            </tr>
    """

    # Add factuality scores to report
    for method, method_metrics in metrics_data.items():
        if "factuality" in method_metrics:
            fact_scores = method_metrics["factuality"]
            report_html += f"""
            <tr>
                <td class="method">{method.capitalize()}</td>
                <td>{fact_scores.get("nli_factuality_score", "N/A"):.4f}</td>
                <td>{fact_scores.get("qa_consistency_score", "N/A"):.4f}</td>
                <td>{fact_scores.get("overall_factuality_score", "N/A"):.4f}</td>
            </tr>
            """

    report_html += """
        </table>

        <h3>Visualizations</h3>
        <div class="images">
            <img src="../rouge_comparison_{0}.png" alt="ROUGE Comparison">
            <img src="../factuality_comparison_{0}.png" alt="Factuality Comparison">
            <img src="../tradeoff_analysis_{0}.png" alt="Trade-off Analysis">
        </div>

        <h3>Sample Summaries</h3>
        <table>
            <tr>
                <th>Example</th>
                <th>Method</th>
                <th>Summary</th>
            </tr>
    """.format(config['experiment_id'])

    # Add sample summaries to report
    summary_data = results["summaries"]
    for example_idx in range(min(5, len(next(iter(summary_data.values()))))):
        is_first = True
        for method, summaries in summary_data.items():
            report_html += f"""
            <tr>
                <td>{example_idx+1 if is_first else ""}</td>
                <td class="method">{method.capitalize()}</td>
                <td>{summaries[example_idx]}</td>
            </tr>
            """
            is_first = False

    report_html += """
        </table>
    </body>
    </html>
    """

    # Save HTML report
    report_file = os.path.join(config["results_dir"], f"report_{config['experiment_id']}.html")
    with open(report_file, 'w') as f:
        f.write(report_html)

    logger.info(f"Visualizations and report saved to {config['results_dir']}")

def main(args):
    """Main function to run experiments"""
    # Update configuration from arguments
    config = update_config_from_args(args)

    # Load and prepare dataset
    logger.info(f"Loading dataset: {config['dataset_name']}")
    dataset_splits = load_and_prepare_dataset(config["dataset_name"], config["dataset_config"])

    # Limit dataset size if needed
    if args.max_samples:
        for split in dataset_splits:
            dataset_splits[split] = dataset_splits[split].select(
                range(min(len(dataset_splits[split]), args.max_samples))
            )

    # Explore dataset
    logger.info("Exploring dataset")
    explore_dataset(
        dataset_splits,
        config["text_column"],
        config["summary_column"]
    )

    # Load tokenizer
    logger.info(f"Loading tokenizer: {config['model_name']}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Tokenize datasets for training
    logger.info("Tokenizing datasets")
    tokenized_datasets = {}
    for split in dataset_splits:
        tokenized_datasets[split] = dataset_splits[split].map(
            lambda examples: preprocess_data(
                examples,
                tokenizer,
                config["max_input_length"],
                config["max_output_length"],
                config["text_column"],
                config["summary_column"]
            ),
            batched=True,
            remove_columns=dataset_splits[split].column_names
        )

    # Train or load baseline model
    if args.skip_baseline:
        logger.info(f"Skipping baseline training, loading pretrained model: {config['model_name']}")
        from transformers import AutoModelForSeq2SeqLM
        baseline_model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"]).to(config["device"])
    else:
        logger.info("Training baseline model")
        baseline_model, tokenizer = train_baseline_model(
            config,
            dataset_splits,
            tokenized_datasets
        )

    logger.info("Evaluating all methods")
    results = evaluate_all_methods(
        config,
        dataset_splits,
        baseline_model,
        tokenizer
    )

    logger.info("Visualizing results")
    visualize_results(results, config)

    logger.info("Experiment completed successfully!")

if __name__ == "__main__":
    class FactualityGuidedDecoder:
        def __init__(self, model_name, max_input_length=1024, max_output_length=128, device=None):
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
            self.max_input_length = max_input_length
            self.max_output_length = max_output_length
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        def generate_summaries(self, dataset, text_column, batch_size=8):
            from tqdm.auto import tqdm
            import torch

            summaries = []
            for i in tqdm(range(0, len(dataset), batch_size), desc="Generating summaries with factuality-guided decoding"):
                batch = dataset[i:i+batch_size]
                inputs = self.tokenizer(
                    batch[text_column],
                    max_length=self.max_input_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=self.max_output_length,
                        num_beams=6,  # Using more beams for better reranking
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )

                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                summaries.extend(decoded)

            return summaries

    args = parse_args()

    main(args)
