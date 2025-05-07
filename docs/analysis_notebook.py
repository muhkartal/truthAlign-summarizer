import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

experiment_id = "your_experiment_id"  # Change this to your actual experiment ID
results_dir = "./output/results"
summaries_dir = "./output/summaries"

def load_experiment_results(experiment_id):
    results_file = os.path.join(results_dir, f"results_{experiment_id}.json")
    with open(results_file, 'r') as f:
        results = json.load(f)

    summaries_file = os.path.join(summaries_dir, f"all_summaries_{experiment_id}.json")
    with open(summaries_file, 'r') as f:
        summaries = json.load(f)

    return results, summaries

results, summaries = load_experiment_results(experiment_id)

methods = list(summaries.keys())
print(f"Methods being compared: {methods}")

def create_metrics_dataframe(results):
    rouge_df = pd.DataFrame(columns=["Method", "Metric", "Score"])
    fact_df = pd.DataFrame(columns=["Method", "Metric", "Score"])

    for method, metrics in results.items():
        if "rouge" in metrics:
            for rouge_type, score in metrics["rouge"].items():
                if rouge_type in ["rouge1", "rouge2", "rougeL"]:
                    rouge_df = rouge_df.append({
                        "Method": method,
                        "Metric": rouge_type,
                        "Score": score
                    }, ignore_index=True)

        if "factuality" in metrics:
            fact_metrics = metrics["factuality"]
            for fact_type in ["nli_factuality_score", "qa_consistency_score", "overall_factuality_score"]:
                if fact_type in fact_metrics:
                    fact_df = fact_df.append({
                        "Method": method,
                        "Metric": fact_type.replace("_score", ""),
                        "Score": fact_metrics[fact_type]
                    }, ignore_index=True)

    return rouge_df, fact_df

rouge_df, fact_df = create_metrics_dataframe(results)

plt.figure(figsize=(15, 6))
sns.barplot(x="Metric", y="Score", hue="Method", data=rouge_df)
plt.title("ROUGE Scores by Method")
plt.savefig("rouge_comparison.png")
plt.close()

plt.figure(figsize=(15, 6))
sns.barplot(x="Metric", y="Score", hue="Method", data=fact_df)
plt.title("Factuality Scores by Method")
plt.savefig("factuality_comparison.png")
plt.close()

def create_tradeoff_plot(results):
    tradeoff_df = pd.DataFrame(columns=["Method", "ROUGE-1", "Factuality"])

    for method, metrics in results.items():
        if "rouge" in metrics and "factuality" in metrics:
            tradeoff_df = tradeoff_df.append({
                "Method": method,
                "ROUGE-1": metrics["rouge"]["rouge1"],
                "Factuality": metrics["factuality"]["overall_factuality_score"]
            }, ignore_index=True)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="ROUGE-1",
        y="Factuality",
        hue="Method",
        style="Method",
        s=100,
        data=tradeoff_df
    )

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
    plt.savefig("tradeoff_analysis.png")
    plt.close()

    return tradeoff_df

tradeoff_df = create_tradeoff_plot(results)
print(tradeoff_df)

def calculate_summary_statistics(summaries):
    stats = {}

    for method, method_summaries in summaries.items():
        lengths = [len(s.split()) for s in method_summaries]
        sent_counts = [len(sent_tokenize(s)) for s in method_summaries]

        stats[method] = {
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "avg_sentences": np.mean(sent_counts),
            "std_sentences": np.std(sent_counts)
        }

    return pd.DataFrame(stats).T

summary_stats = calculate_summary_statistics(summaries)
print(summary_stats)

def calculate_abstractiveness(source_docs, method_summaries):
    abstractiveness_scores = []

    vectorizer = CountVectorizer(stop_words='english')

    for i in range(min(len(source_docs), len(method_summaries))):
        source = source_docs[i]
        summary = method_summaries[i]

        source_vec = vectorizer.fit_transform([source])
        summary_vec = vectorizer.transform([summary])

        similarity = cosine_similarity(source_vec, summary_vec)[0][0]
        abstractiveness = 1 - similarity
        abstractiveness_scores.append(abstractiveness)

    return np.mean(abstractiveness_scores)

def load_dataset_sample(dataset_name="cnn_dailymail", config="3.0.0", split="test", num_samples=100):
    dataset = load_dataset(dataset_name, config, split=f"{split}[:{num_samples}]")

    if dataset_name == "cnn_dailymail":
        sources = dataset["article"]
        references = dataset["highlights"]
    elif dataset_name == "xsum":
        sources = dataset["document"]
        references = dataset["summary"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return sources, references

sources, references = load_dataset_sample()

abstractiveness = {}
for method, method_summaries in summaries.items():
    abstractiveness[method] = calculate_abstractiveness(sources, method_summaries[:len(sources)])

abs_df = pd.DataFrame([abstractiveness], index=["Abstractiveness"]).T
print(abs_df)

plt.figure(figsize=(10, 6))
sns.barplot(x=abs_df.index, y="Abstractiveness", data=abs_df)
plt.title("Abstractiveness by Method")
plt.ylabel("Abstractiveness Score (1-cosine similarity)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("abstractiveness_comparison.png")
plt.close()

def find_interesting_examples(sources, summaries, n=5):
    interesting = []

    for i in range(min(len(sources), len(list(summaries.values())[0]))):
        source = sources[i]
        example = {"index": i, "source": source[:300] + "..."}

        for method, method_summaries in summaries.items():
            example[method] = method_summaries[i]

        interesting.append(example)

    return interesting[:n]

interesting_examples = find_interesting_examples(sources, summaries)

def show_example_summaries(examples):
    for i, example in enumerate(examples):
        print(f"Example {example['index']}")
        print(f"Source: {example['source']}")
        print("-" * 80)

        for method in methods:
            print(f"{method.capitalize()}: {example[method]}")
            print()

        print("=" * 80)

show_example_summaries(interesting_examples)

def create_factuality_comparison(factuality_results):
    methods = list(factuality_results.keys())
    metrics = ["nli_entailment", "nli_contradiction", "qa_consistency_score", "nli_factuality_score", "overall_factuality_score"]

    data = []
    for method in methods:
        for metric in metrics:
            if metric in factuality_results[method]:
                data.append({
                    "Method": method,
                    "Metric": metric,
                    "Score": factuality_results[method][metric]
                })

    return pd.DataFrame(data)

factuality_data = {}
for method in methods:
    if method in results and "factuality" in results[method]:
        factuality_data[method] = results[method]["factuality"]

factuality_comparison = create_factuality_comparison(factuality_data)

plt.figure(figsize=(15, 8))
sns.barplot(x="Metric", y="Score", hue="Method", data=factuality_comparison)
plt.title("Detailed Factuality Metrics by Method")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("detailed_factuality_comparison.png")
plt.close()

def create_final_summary_table(results, summary_stats, abstractiveness):
    final_table = pd.DataFrame(index=methods)

    for method in methods:
        if method in results:
            metrics = results[method]
            final_table.loc[method, "ROUGE-1"] = metrics["rouge"]["rouge1"] if "rouge" in metrics else np.nan
            final_table.loc[method, "ROUGE-2"] = metrics["rouge"]["rouge2"] if "rouge" in metrics else np.nan
            final_table.loc[method, "ROUGE-L"] = metrics["rouge"]["rougeL"] if "rouge" in metrics else np.nan

            if "factuality" in metrics:
                final_table.loc[method, "Factuality"] = metrics["factuality"]["overall_factuality_score"]
                final_table.loc[method, "NLI Score"] = metrics["factuality"]["nli_factuality_score"]
                final_table.loc[method, "QA Score"] = metrics["factuality"]["qa_consistency_score"]

    final_table["Avg Length"] = summary_stats["avg_length"]
    final_table["Avg Sentences"] = summary_stats["avg_sentences"]
    final_table["Abstractiveness"] = abstractiveness

    return final_table

final_table = create_final_summary_table(results, summary_stats, abs_df["Abstractiveness"])
print(final_table)

final_table.to_csv("factuality_improvement_results.csv")

def calculate_performance_changes(baseline="baseline"):
    if baseline not in methods:
        print(f"Baseline method '{baseline}' not found in results")
        return None

    changes = pd.DataFrame(index=[m for m in methods if m != baseline])

    baseline_rouge1 = results[baseline]["rouge"]["rouge1"]
    baseline_factuality = results[baseline]["factuality"]["overall_factuality_score"]

    for method in methods:
        if method != baseline:
            method_rouge1 = results[method]["rouge"]["rouge1"]
            method_factuality = results[method]["factuality"]["overall_factuality_score"]

            rouge_change = method_rouge1 - baseline_rouge1
            rouge_percent = (rouge_change / baseline_rouge1) * 100

            fact_change = method_factuality - baseline_factuality
            fact_percent = (fact_change / baseline_factuality) * 100 if baseline_factuality != 0 else np.inf

            changes.loc[method, "ROUGE-1 Change"] = rouge_change
            changes.loc[method, "ROUGE-1 % Change"] = rouge_percent
            changes.loc[method, "Factuality Change"] = fact_change
            changes.loc[method, "Factuality % Change"] = fact_percent

    return changes

performance_changes = calculate_performance_changes()
print(performance_changes)

plt.figure(figsize=(12, 6))
sns.heatmap(performance_changes[["ROUGE-1 % Change", "Factuality % Change"]],
            annot=True, cmap="RdYlGn", center=0, fmt=".2f")
plt.title("Performance Changes vs Baseline (%)")
plt.tight_layout()
plt.savefig("performance_changes.png")
plt.close()

def create_html_report():
    html = """
    <html>
    <head>
        <title>Factual Consistency in Abstractive Summarization - Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .highlight { background-color: #e6f7ff; }
            img { max-width: 100%; height: auto; margin: 20px 0; }
            .summary { background-color: #f5f5f5; padding: 15px; margin: 10px 0; }
            .method { font-weight: bold; }
            .example { margin-bottom: 30px; }
        </style>
    </head>
    <body>
        <h1>Factual Consistency in Abstractive Summarization</h1>
        <h2>Results Analysis</h2>

        <h3>Overall Performance</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>ROUGE-1</th>
                <th>ROUGE-L</th>
                <th>Factuality</th>
                <th>Length</th>
                <th>Abstractiveness</th>
            </tr>
    """

    for method in methods:
        html += f"""
            <tr>
                <td class="method">{method}</td>
                <td>{final_table.loc[method, 'ROUGE-1']:.2f}</td>
                <td>{final_table.loc[method, 'ROUGE-L']:.2f}</td>
                <td>{final_table.loc[method, 'Factuality']:.4f}</td>
                <td>{final_table.loc[method, 'Avg Length']:.1f}</td>
                <td>{final_table.loc[method, 'Abstractiveness']:.4f}</td>
            </tr>
        """

    html += """
        </table>

        <h3>Performance Changes vs Baseline</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>ROUGE-1 Change</th>
                <th>ROUGE-1 % Change</th>
                <th>Factuality Change</th>
                <th>Factuality % Change</th>
            </tr>
    """

    for method in performance_changes.index:
        rouge_change = performance_changes.loc[method, "ROUGE-1 Change"]
        rouge_percent = performance_changes.loc[method, "ROUGE-1 % Change"]
        fact_change = performance_changes.loc[method, "Factuality Change"]
        fact_percent = performance_changes.loc[method, "Factuality % Change"]

        rouge_class = "highlight" if rouge_change > 0 else ""
        fact_class = "highlight" if fact_change > 0 else ""

        html += f"""
            <tr>
                <td class="method">{method}</td>
                <td class="{rouge_class}">{rouge_change:.2f}</td>
                <td class="{rouge_class}">{rouge_percent:.2f}%</td>
                <td class="{fact_class}">{fact_change:.4f}</td>
                <td class="{fact_class}">{fact_percent:.2f}%</td>
            </tr>
        """

    html += """
        </table>

        <h3>Visualizations</h3>
        <div>
            <img src="rouge_comparison.png" alt="ROUGE Comparison">
            <img src="factuality_comparison.png" alt="Factuality Comparison">
            <img src="tradeoff_analysis.png" alt="Trade-off Analysis">
            <img src="abstractiveness_comparison.png" alt="Abstractiveness Comparison">
            <img src="performance_changes.png" alt="Performance Changes">
        </div>

        <h3>Example Summaries</h3>
    """

    for i, example in enumerate(interesting_examples):
        html += f"""
        <div class="example">
            <h4>Example {example['index']}</h4>
            <p><strong>Source:</strong> {example['source']}</p>

            <div class="summaries">
        """

        for method in methods:
            html += f"""
                <div class="summary">
                    <span class="method">{method}:</span> {example[method]}
                </div>
            """

        html += """
            </div>
        </div>
        """

    html += """
        <h3>Conclusions</h3>
        <p>
            This analysis demonstrates the effectiveness of different approaches for improving factual consistency
            in abstractive summarization. Key findings include:
        </p>
        <ul>
            <li>Factuality-guided techniques can significantly improve consistency with source documents</li>
            <li>There is a trade-off between ROUGE scores and factual consistency in some methods</li>
            <li>The most effective methods improve factuality while maintaining acceptable ROUGE scores</li>
        </ul>

    </body>
    </html>
    """

    with open("factuality_results_report.html", "w") as f:
        f.write(html)

    return html

create_html_report()
