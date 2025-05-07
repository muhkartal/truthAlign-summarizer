import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline
)
from datasets import load_metric
import nltk
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
nltk.download('punkt')

class FactualityEvaluator:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing factuality evaluator on {self.device}")

        print("Loading NLI model...")
        self.nli_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli").to(self.device)

        print("Loading QA model...")
        self.qa_tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2").to(self.device)

        print("Loading QG model...")
        self.qg_tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-qg-hl")
        self.qg_model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-qg-hl").to(self.device)

        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.qa_model,
            tokenizer=self.qa_tokenizer,
            device=0 if self.device == "cuda" else -1
        )

        self.rouge = load_metric("rouge")

    def evaluate_nli_entailment(self, source_docs, summaries, batch_size=8):
        self.nli_model.eval()
        entailment_scores = []
        neutral_scores = []
        contradiction_scores = []

        for doc_idx in tqdm(range(len(source_docs)), desc="Evaluating NLI entailment"):
            source = source_docs[doc_idx]
            summary = summaries[doc_idx]

            summary_sentences = sent_tokenize(summary)
            if not summary_sentences:
                entailment_scores.append(0.0)
                neutral_scores.append(0.0)
                contradiction_scores.append(0.0)
                continue

            doc_entail_scores = []
            doc_neutral_scores = []
            doc_contradiction_scores = []

            for sentence in summary_sentences:
                if not sentence.strip():
                    continue

                inputs = self.nli_tokenizer(
                    source,
                    sentence,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                entail_score = probs[0, 2].item()
                neutral_score = probs[0, 1].item()
                contradiction_score = probs[0, 0].item()

                doc_entail_scores.append(entail_score)
                doc_neutral_scores.append(neutral_score)
                doc_contradiction_scores.append(contradiction_score)

            if doc_entail_scores:
                entailment_scores.append(np.mean(doc_entail_scores))
                neutral_scores.append(np.mean(doc_neutral_scores))
                contradiction_scores.append(np.mean(doc_contradiction_scores))
            else:
                entailment_scores.append(0.0)
                neutral_scores.append(0.0)
                contradiction_scores.append(0.0)

        avg_entailment = np.mean(entailment_scores)
        avg_neutral = np.mean(neutral_scores)
        avg_contradiction = np.mean(contradiction_scores)

        factuality_score = avg_entailment - avg_contradiction

        return {
            "nli_entailment": avg_entailment,
            "nli_neutral": avg_neutral,
            "nli_contradiction": avg_contradiction,
            "nli_factuality_score": factuality_score,
            "nli_scores_per_example": list(zip(entailment_scores, neutral_scores, contradiction_scores))
        }

    def generate_questions(self, text, n_questions=3):
        sentences = sent_tokenize(text)
        questions = []

        for sentence in sentences[:min(5, len(sentences))]:  # Limit to first 5 sentences to avoid too many questions
            if len(sentence.split()) < 5:
                continue

            input_text = f"generate question: {sentence}"

            inputs = self.qg_tokenizer(
                input_text,
                max_length=512,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.qg_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=64,
                    num_beams=4,
                    early_stopping=True,
                    num_return_sequences=min(n_questions, 3),  # Generate up to n_questions per sentence
                    no_repeat_ngram_size=2
                )

            batch_questions = self.qg_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            questions.extend(batch_questions)

            if len(questions) >= n_questions:
                questions = questions[:n_questions]
                break

        return questions

    def evaluate_qa_consistency(self, source_docs, summaries, n_questions=5):
        consistency_scores = []
        question_counts = []

        for doc_idx in tqdm(range(len(source_docs)), desc="Evaluating QA consistency"):
            source = source_docs[doc_idx]
            summary = summaries[doc_idx]

            # Generate questions from summary
            questions = self.generate_questions(summary, n_questions)
            question_counts.append(len(questions))

            if not questions:
                consistency_scores.append(0.0)
                continue

            qa_consistencies = []

            for question in questions:
                summary_answer = self.qa_pipeline(
                    question=question,
                    context=summary,
                    handle_impossible_answer=True
                )

                source_answer = self.qa_pipeline(
                    question=question,
                    context=source,
                    handle_impossible_answer=True
                )

                if (summary_answer["score"] < 0.1 or
                    source_answer["score"] < 0.1 or
                    not summary_answer["answer"].strip() or
                    not source_answer["answer"].strip()):
                    continue

                rouge_scores = self.rouge.compute(
                    predictions=[summary_answer["answer"]],
                    references=[source_answer["answer"]],
                    use_stemmer=True
                )

                similarity = rouge_scores["rougeL"].mid.fmeasure
                qa_consistencies.append(similarity)

            if qa_consistencies:
                consistency_scores.append(np.mean(qa_consistencies))
            else:
                consistency_scores.append(0.0)

        avg_consistency = np.mean(consistency_scores)
        avg_questions = np.mean(question_counts)

        return {
            "qa_consistency_score": avg_consistency,
            "avg_questions_per_summary": avg_questions,
            "qa_scores_per_example": consistency_scores
        }

    def evaluate_factuality(self, source_docs, summaries):
        print("Evaluating factual consistency...")

        nli_metrics = self.evaluate_nli_entailment(source_docs, summaries)

        qa_metrics = self.evaluate_qa_consistency(source_docs, summaries)

        combined_metrics = {**nli_metrics, **qa_metrics}

        combined_metrics["overall_factuality_score"] = (
            combined_metrics["nli_factuality_score"] + combined_metrics["qa_consistency_score"]
        ) / 2

        return combined_metrics

if __name__ == "__main__":
    documents = [
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower. "
        "Constructed from 1887 to 1889 as the entrance to the 1889 World's Fair, it was initially criticized "
        "by some of France's leading artists and intellectuals for its design, but it has become a global "
        "cultural icon of France and one of the most recognizable structures in the world.",

        "The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic "
        "of coronavirus disease 2019 (COVID-19) caused by severe acute respiratory syndrome coronavirus 2 "
        "(SARS-CoV-2). The virus was first identified in December 2019 in Wuhan, China. The World Health "
        "Organization declared a Public Health Emergency of International Concern on 30 January 2020, and "
        "later declared a pandemic on 11 March 2020."
    ]

    good_summaries = [
        "The Eiffel Tower, built by Gustave Eiffel's company from 1887 to 1889, is located in Paris, France.",
        "COVID-19 is a global pandemic caused by SARS-CoV-2, first identified in Wuhan, China in December 2019."
    ]

    bad_summaries = [
        "The Eiffel Tower was built in London by Christopher Wren in the 18th century.",
        "COVID-19 originated in Italy in early 2020 and was declared a pandemic by the CDC in April 2020."
    ]

    evaluator = FactualityEvaluator()

    print("\nEvaluating factually correct summaries:")
    good_metrics = evaluator.evaluate_factuality(documents, good_summaries)

    print("\nEvaluating factually incorrect summaries:")
    bad_metrics = evaluator.evaluate_factuality(documents, bad_summaries)

    print("\nFactually Correct Summaries Metrics:")
    for key, value in good_metrics.items():
        if not isinstance(value, list):
            print(f"{key}: {value:.4f}")

    print("\nFactually Incorrect Summaries Metrics:")
    for key, value in bad_metrics.items():
        if not isinstance(value, list):
            print(f"{key}: {value:.4f}")
