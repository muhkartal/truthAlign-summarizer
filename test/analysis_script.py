import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt', quiet=True)

def load_results(results_dir="./output/results", experiment_id=None):
    if experiment_id is None:
        results_files = [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]
        if not results_files:
            raise FileNotFoundError("No results files found. Run the pipeline first.")

        results_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
        latest_file = results_files[0]
        experiment_id = latest_file.replace("results_", "").replace(".json", "")

    results_file = os.path.join(results_dir, f"results_{experiment_id}.json")
    summaries_file = os.path.join("./output/summaries", f"all_summaries_{experiment_id}.json")

    with open(results_file, 'r') as f:
        results = json.load(f)

    with open(summaries_file, 'r') as f:
        summaries = json.load(f)

    return results, summaries, experiment_id

def analyze_hallucinations(source_doc, summaries, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(device)

    method_results = {}

    for method, summary in summaries.items():
        sentences = sent_tokenize(summary)
        sentence_scores = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            inputs = tokenizer(
                source_doc,
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            entailment_score = probs[0, 2].item()
            contradiction_score = probs[0, 0].item()
            neutral_score = probs[0, 1].item()

            sentence_scores.append({
                "sentence": sentence,
                "entailment": entailment_score,
                "contradiction": contradiction_score,
                "neutral": neutral_score,
                "factuality": entailment_score - contradiction_score,
                "is_hallucination": contradiction_score > 0.5
            })

        method_results[method] = {
            "sentences": sentence_scores,
            "avg_entailment": np.mean([s["entailment"] for s in sentence_scores]) if sentence_scores else 0,
            "avg_contradiction": np.mean([s["contradiction"] for s in sentence_scores]) if sentence_scores else 0,
            "avg_factuality": np.mean([s["factuality"] for s in sentence_scores]) if sentence_scores else 0,
            "hallucination_count": sum(1 for s in sentence_scores if s["is_hallucination"]),
            "hallucination_percent": sum(1 for s in sentence_scores if s["is_hallucination"]) / len(sentence_scores) * 100 if sentence_scores else 0
        }

    return method_results

def analyze_example(example_idx, source_docs, summaries_dict):
    print(f"Analyzing example {example_idx}")

    source_doc = source_docs[example_idx]
    example_summaries = {method: summaries[example_idx] for method, summaries in summaries_dict.items()}

    print("\nSource document (excerpt):")
    print(source_doc[:500] + "...")

    print("\nSummaries:")
    for method, summary in example_summaries.items():
        print(f"\n{method.upper()}:")
        print(summary)

    hallucination_results = analyze_hallucinations(source_doc, example_summaries)

    print("\nHallucination Analysis:")
    for method, results in hallucination_results.items():
        print(f"\n{method.upper()}:")
        print(f"Average factuality score: {results['avg_factuality']:.4f}")
        print(f"Hallucination percentage: {results['hallucination_percent']:.1f}% ({results['hallucination_count']} of {len(results['sentences'])} sentences)")

        if results['hallucination_count'] > 0:
            print("\nPotential hallucinations:")
            for sentence in results['sentences']:
                if sentence['is_hallucination']:
                    print(f"- \"{sentence['sentence']}\" (Contradiction score: {sentence['contradiction']:.4f})")

    return hallucination_results

def compare_methods_on_examples(results, summaries, source_docs, num_examples=5):
    methods = list(summaries.keys())

    if "baseline" in methods:
        baseline_method = "baseline"
    else:
        baseline_method = methods[0]

    baseline_summaries = summaries[baseline_method]

    improvement_examples = []

    for i in range(min(len(baseline_summaries), len(source_docs))):
        source = source_docs[i]

        example_summaries = {method: summaries[method][i] for method in methods}
        hallucination_analysis = analyze_hallucinations(source, example_summaries)

        baseline_factuality = hallucination_analysis[baseline_method]["avg_factuality"]

        for method in methods:
            if method == baseline_method:
                continue

            method_factuality = hallucination_analysis[method]["avg_factuality"]
            improvement = method_factuality - baseline_factuality

            if improvement > 0:
                improvement_examples.append({
                    "index": i,
                    "source": source,
                    "summaries": example_summaries,
                    "analysis": hallucination_analysis,
                    "improvement": improvement,
                    "improved_method": method
                })

    improvement_examples.sort(key=lambda x: x["improvement"], reverse=True)
    selected_examples = improvement_examples[:num_examples]

    print(f"\nTop {len(selected_examples)} examples with factuality improvements:")

    for i, example in enumerate(selected_examples):
        print(f"\nExample {i+1} (index {example['index']}):")
        print(f"Improvement: {example['improvement']:.4f} ({baseline_method} to {example['improved_method']})")

        print("\nSource (excerpt):")
        print(example['source'][:300] + "...")

        print(f"\n{baseline_method.upper()} summary:")
        print(example['summaries'][baseline_method])

        print(f"\n{example['improved_method'].upper()} summary:")
        print(example['summaries'][example['improved_method']])

    return selected_examples

if __name__ == "__main__":
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Analyze factual consistency examples")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail", help="Dataset name")
    parser.add_argument("--config", type=str, default="3.0.0", help="Dataset configuration")
    parser.add_argument("--experiment_id", type=str, default=None, help="Experiment ID to analyze")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to analyze")
    parser.add_argument("--example_idx", type=int, default=None, help="Analyze a specific example")
    args = parser.parse_args()

    print(f"Loading results and summaries...")
    results, summaries, experiment_id = load_results(experiment_id=args.experiment_id)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "cnn_dailymail":
        source_column = "article"
    elif args.dataset == "xsum":
        source_column = "document"
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset = load_dataset(args.dataset, args.config, split="test[:100]")
    source_docs = dataset[source_column]

    if args.example_idx is not None:
        analyze_example(args.example_idx, source_docs, summaries)
    else:
        selected_examples = compare_methods_on_examples(
            results, summaries, source_docs, num_examples=args.num_examples
        )

        output_dir = "./output/analysis"
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, f"improvement_examples_{experiment_id}.json"), "w") as f:
            json.dump(selected_examples, f, indent=4)

        print(f"\nDetailed analysis saved to {output_dir}/improvement_examples_{experiment_id}.json")
