import torch
import numpy as np
from datasets import load_metric
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
import json
import os
from scipy.stats import ttest_rel, wilcoxon

class SummarizationEvaluator:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.rouge = load_metric("rouge")

    def calculate_rouge(self, predictions, references):
        rouge_outputs = self.rouge.compute(
            predictions=["\n".join(sent_tokenize(s)) for s in predictions],
            references=["\n".join(sent_tokenize(s)) for s in references],
            use_stemmer=True
        )

        rouge_scores = {k: round(v.mid.fmeasure * 100, 2) for k, v in rouge_outputs.items()}
        return rouge_scores

    def evaluate_methods(self, source_docs, reference_summaries, generated_summaries_dict, factuality_evaluator):
        results = {
            "rouge": {},
            "factuality": {},
            "combined": {}
        }

        for method_name, summaries in generated_summaries_dict.items():
            print(f"Evaluating {method_name}...")

            rouge_scores = self.calculate_rouge(summaries, reference_summaries)
            factuality_scores = factuality_evaluator.evaluate_factuality(source_docs, summaries)

            results["rouge"][method_name] = rouge_scores
            results["factuality"][method_name] = factuality_scores

            combined_score = (rouge_scores["rouge1"] / 100.0 + factuality_scores["overall_factuality_score"]) / 2
            results["combined"][method_name] = combined_score

        return results

    def run_statistical_tests(self, results, baseline_method="baseline"):
        stat_results = {}

        baseline_factuality = np.array(results["factuality"][baseline_method]["qa_scores_per_example"])
        baseline_rouge1 = np.array([results["rouge"][baseline_method]["rouge1"]] * len(baseline_factuality))

        for method in results["factuality"]:
            if method == baseline_method:
                continue

            method_factuality = np.array(results["factuality"][method]["qa_scores_per_example"])
            method_rouge1 = np.array([results["rouge"][method]["rouge1"]] * len(method_factuality))

            try:
                t_test_factuality = ttest_rel(method_factuality, baseline_factuality)
                wilcoxon_factuality = wilcoxon(method_factuality, baseline_factuality)

                stat_results[method] = {
                    "factuality_t_test_pvalue": t_test_factuality.pvalue,
                    "factuality_wilcoxon_pvalue": wilcoxon_factuality.pvalue,
                    "factuality_mean_difference": np.mean(method_factuality - baseline_factuality),
                    "rouge1_difference": method_rouge1[0] - baseline_rouge1[0]
                }
            except:
                stat_results[method] = {
                    "factuality_statistical_test": "Failed to compute",
                    "factuality_mean_difference": np.mean(method_factuality - baseline_factuality),
                    "rouge1_difference": method_rouge1[0] - baseline_rouge1[0]
                }

        return stat_results

    def create_results_tables(self, results):
        rouge_table = pd.DataFrame()
        factuality_table = pd.DataFrame()

        for method, rouge_scores in results["rouge"].items():
            method_row = pd.DataFrame({
                "Method": [method],
                "ROUGE-1": [rouge_scores["rouge1"]],
                "ROUGE-2": [rouge_scores["rouge2"]],
                "ROUGE-L": [rouge_scores["rougeL"]]
            })
            rouge_table = pd.concat([rouge_table, method_row])

        for method, factuality_scores in results["factuality"].items():
            method_row = pd.DataFrame({
                "Method": [method],
                "NLI Score": [factuality_scores["nli_factuality_score"]],
                "QA Score": [factuality_scores["qa_consistency_score"]],
                "Overall": [factuality_scores["overall_factuality_score"]]
            })
            factuality_table = pd.concat([factuality_table, method_row])

        return rouge_table, factuality_table

    def create_visualizations(self, results, output_dir):
        rouge_table, factuality_table = self.create_results_tables(results)

        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        sns.barplot(x="Method", y="ROUGE-1", data=rouge_table)
        plt.title("ROUGE-1 Scores by Method")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rouge1_comparison.png"))
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.barplot(x="Method", y="Overall", data=factuality_table)
        plt.title("Factuality Scores by Method")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "factuality_comparison.png"))
        plt.close()

        tradeoff_df = pd.DataFrame()
        for method in results["rouge"]:
            method_row = pd.DataFrame({
                "Method": [method],
                "ROUGE-1": [results["rouge"][method]["rouge1"]],
                "Factuality": [results["factuality"][method]["overall_factuality_score"]]
            })
            tradeoff_df = pd.concat([tradeoff_df, method_row])

        plt.figure(figsize=(10, 8))
        scatter = sns.scatterplot(
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
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tradeoff_analysis.png"))
        plt.close()

    def find_example_improvements(self, source_docs, baseline_summaries, improved_summaries, factuality_evaluator, n=5):
        improvements = []

        for i in range(len(baseline_summaries)):
            source = source_docs[i]
            baseline = baseline_summaries[i]
            improved = improved_summaries[i]

            baseline_check = factuality_evaluator.evaluate_nli_entailment([source], [baseline])
            improved_check = factuality_evaluator.evaluate_nli_entailment([source], [improved])

            baseline_score = baseline_check["nli_factuality_score"]
            improved_score = improved_check["nli_factuality_score"]

            if improved_score > baseline_score:
                improvements.append({
                    "source": source,
                    "baseline": baseline,
                    "improved": improved,
                    "baseline_score": baseline_score,
                    "improved_score": improved_score,
                    "improvement": improved_score - baseline_score
                })

        improvements.sort(key=lambda x: x["improvement"], reverse=True)
        return improvements[:n]

    def save_evaluation_report(self, results, improvements, output_path):
        report = {
            "results": results,
            "example_improvements": improvements
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=4)

        return report

class QualitativeAnalysis:
    def __init__(self, source_docs, reference_summaries, method_summaries):
        self.source_docs = source_docs
        self.reference_summaries = reference_summaries
        self.method_summaries = method_summaries

    def analyze_examples(self, indices, output_file=None):
        analysis = []

        for idx in indices:
            example = {
                "index": idx,
                "source_doc": self.source_docs[idx][:500] + "...",
                "reference": self.reference_summaries[idx],
                "generated": {}
            }

            for method, summaries in self.method_summaries.items():
                example["generated"][method] = summaries[idx]

            analysis.append(example)

        if output_file:
            with open(output_file, "w") as f:
                json.dump(analysis, f, indent=4)

        return analysis

    def detect_hallucinations(self, factuality_checker, sample_size=50):
        hallucinations = []

        indices = np.random.choice(len(self.source_docs), min(sample_size, len(self.source_docs)), replace=False)

        for idx in indices:
            source = self.source_docs[idx]

            for method, summaries in self.method_summaries.items():
                summary = summaries[idx]

                fact_check = factuality_checker.filter_inconsistencies(source, summary)
                filtered_summary, fact_check_results = fact_check

                for score, label, sentence in fact_check_results:
                    if label == "contradiction" and score < -0.7:
                        hallucinations.append({
                            "index": idx,
                            "method": method,
                            "sentence": sentence,
                            "contradiction_score": score,
                            "source_excerpt": source[:200] + "..."
                        })

        return hallucinations

    def create_html_report(self, analysis, hallucinations, output_file):
        html = """
        <html>
        <head>
            <title>Qualitative Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #2c3e50; }
                .example { margin-bottom: 30px; border: 1px solid #ddd; padding: 15px; }
                .method { margin-top: 15px; }
                .hallucination { background-color: #ffe6e6; padding: 10px; margin: 10px 0; }
                .reference { background-color: #e6f7ff; padding: 10px; }
                .source { background-color: #f5f5f5; padding: 10px; font-family: monospace; white-space: pre-wrap; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Qualitative Analysis Report</h1>

            <h2>Example Comparisons</h2>
        """

        for example in analysis:
            html += f"""
            <div class="example">
                <h3>Example {example['index']}</h3>
                <div class="source">
                    <strong>Source Document:</strong><br>
                    {example['source_doc']}
                </div>
                <div class="reference">
                    <strong>Reference Summary:</strong><br>
                    {example['reference']}
                </div>
            """

            for method, summary in example['generated'].items():
                html += f"""
                <div class="method">
                    <strong>{method.capitalize()}:</strong><br>
                    {summary}
                </div>
                """

            html += "</div>"

        html += """
            <h2>Detected Hallucinations</h2>
            <table>
                <tr>
                    <th>Example</th>
                    <th>Method</th>
                    <th>Hallucinated Sentence</th>
                    <th>Score</th>
                </tr>
        """

        for h in hallucinations:
            html += f"""
            <tr>
                <td>{h['index']}</td>
                <td>{h['method']}</td>
                <td>{h['sentence']}</td>
                <td>{h['contradiction_score']:.2f}</td>
            </tr>
            """

        html += """
            </table>
        </body>
        </html>
        """

        with open(output_file, "w") as f:
            f.write(html)

        return html

if __name__ == "__main__":
    from datasets import load_dataset
    from factuality_metrics import FactualityEvaluator
    from postprocessing_correction import FactualityPostProcessor

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:20]")

    source_docs = dataset["article"]
    reference_summaries = dataset["highlights"]

    baseline_summaries = ["Sample baseline summary 1", "Sample baseline summary 2"]
    rl_summaries = ["Sample RL summary 1", "Sample RL summary 2"]

    test_summaries = {
        "baseline": baseline_summaries,
        "rl_enhanced": rl_summaries
    }

    evaluator = SummarizationEvaluator()
    factuality_evaluator = FactualityEvaluator()

    results = evaluator.evaluate_methods(
        source_docs[:2],
        reference_summaries[:2],
        {k: v[:2] for k, v in test_summaries.items()},
        factuality_evaluator
    )

    evaluator.create_visualizations(results, "./output")
