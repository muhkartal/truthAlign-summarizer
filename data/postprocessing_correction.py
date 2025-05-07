import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import nltk
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
nltk.download('punkt')

class FactualityPostProcessor:
    def __init__(
        self,
        nli_model_name="roberta-large-mnli",
        threshold=0.5,
        device=None
    ):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        print(f"Initializing factuality post-processor on {self.device}")

        print(f"Loading NLI model: {nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)

        self.nli_pipeline = pipeline(
            "text-classification",
            model=self.nli_model,
            tokenizer=self.nli_tokenizer,
            device=0 if self.device == "cuda" else -1
        )

    def check_sentence_factuality(self, source_doc, sentence):
        if not sentence.strip():
            return 0.0, "neutral", sentence

        result = self.nli_pipeline(
            premise=source_doc,
            hypothesis=sentence,
            truncation=True,
            max_length=512
        )[0]

        label = result["label"].lower()
        score = result["score"]

        if label == "contradiction":
            return -score, label, sentence
        elif label == "entailment":
            return score, label, sentence
        else:
            return 0.1, label, sentence

    def filter_inconsistencies(self, source_doc, summary):
        sentences = sent_tokenize(summary)
        if not sentences:
            return "", []

        fact_check_results = []
        consistent_sentences = []

        for sentence in sentences:
            score, label, _ = self.check_sentence_factuality(source_doc, sentence)
            fact_check_results.append((score, label, sentence))

            if score >= -self.threshold:
                consistent_sentences.append(sentence)

        filtered_summary = " ".join(consistent_sentences)

        return filtered_summary, fact_check_results

    def correct_batch(self, source_docs, summaries):

        filtered_summaries = []
        all_fact_check_results = []

        for doc_idx in tqdm(range(len(source_docs)), desc="Post-processing summaries"):
            source = source_docs[doc_idx]
            summary = summaries[doc_idx]

            filtered_summary, fact_check_results = self.filter_inconsistencies(source, summary)

            filtered_summaries.append(filtered_summary)
            all_fact_check_results.append(fact_check_results)

        return filtered_summaries, all_fact_check_results

    def analyze_corrections(self, original_summaries, filtered_summaries, fact_check_results):
        num_summaries = len(original_summaries)
        num_sentences_original = 0
        num_sentences_filtered = 0
        num_contradictions = 0
        num_entailments = 0
        num_neutrals = 0

        filtered_ratios = []

        for i in range(num_summaries):
            original_sents = sent_tokenize(original_summaries[i])
            filtered_sents = sent_tokenize(filtered_summaries[i]) if filtered_summaries[i] else []

            num_sentences_original += len(original_sents)
            num_sentences_filtered += len(filtered_sents)

            if len(original_sents) > 0:
                filtered_ratio = len(filtered_sents) / len(original_sents)
            else:
                filtered_ratio = 1.0
            filtered_ratios.append(filtered_ratio)

            for score, label, _ in fact_check_results[i]:
                if label == "contradiction":
                    num_contradictions += 1
                elif label == "entailment":
                    num_entailments += 1
                else:  # neutral
                    num_neutrals += 1

        avg_filtered_ratio = np.mean(filtered_ratios)
        sentences_removed = num_sentences_original - num_sentences_filtered
        percent_removed = (sentences_removed / num_sentences_original) * 100 if num_sentences_original > 0 else 0

        analysis = {
            "num_summaries": num_summaries,
            "num_sentences_original": num_sentences_original,
            "num_sentences_filtered": num_sentences_filtered,
            "sentences_removed": sentences_removed,
            "percent_removed": percent_removed,
            "avg_filtered_ratio": avg_filtered_ratio,
            "label_distribution": {
                "contradictions": num_contradictions,
                "entailments": num_entailments,
                "neutrals": num_neutrals
            }
        }

        return analysis

class FactualSentenceRewriter:
    def __init__(
        self,
        rewriting_model_name="facebook/bart-large-cnn",
        nli_model_name="roberta-large-mnli",
        device=None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.factuality_checker = FactualityPostProcessor(
            nli_model_name=nli_model_name,
            threshold=0.7,
            device=self.device
        )

        print(f"Loading rewriting model: {rewriting_model_name}")
        from transformers import AutoModelForSeq2SeqLM
        self.rewrite_tokenizer = AutoTokenizer.from_pretrained(rewriting_model_name)
        self.rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(rewriting_model_name).to(self.device)

    def rewrite_sentence(self, source_doc, sentence, max_attempts=3):
        score, label, _ = self.factuality_checker.check_sentence_factuality(source_doc, sentence)

        if label != "contradiction" or score >= -0.7:
            return sentence, False

        prompt = f"Rewrite to be factually consistent with source: {sentence}"

        for attempt in range(max_attempts):
            inputs = self.rewrite_tokenizer(
                source_doc + " " + prompt,
                max_length=1024,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.rewrite_model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=128,
                    num_beams=5,
                    temperature=0.7,
                    top_p=0.9,
                    no_repeat_ngram_size=2
                )

            rewritten = self.rewrite_tokenizer.decode(output_ids[0], skip_special_tokens=True)

            new_score, new_label, _ = self.factuality_checker.check_sentence_factuality(source_doc, rewritten)

            if new_label != "contradiction" or new_score >= -0.5:
                return rewritten, True

        return "", True

    def correct_and_rewrite(self, source_doc, summary):
        sentences = sent_tokenize(summary)
        if not sentences:
            return "", []

        rewriting_results = []
        corrected_sentences = []

        for sentence in sentences:
            score, label, _ = self.factuality_checker.check_sentence_factuality(source_doc, sentence)

            if label == "contradiction" and score < -0.7:
                rewritten, was_modified = self.rewrite_sentence(source_doc, sentence)
                corrected_sentences.append(rewritten)
                rewriting_results.append((sentence, rewritten, was_modified))
            else:
                corrected_sentences.append(sentence)
                rewriting_results.append((sentence, sentence, False))

        corrected_summary = " ".join([s for s in corrected_sentences if s])

        return corrected_summary, rewriting_results

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

    # Summaries with factual errors
    summaries = [
        "The Eiffel Tower was built in London by Christopher Wren in the 18th century. "
        "It was designed by Gustave Eiffel and is a global cultural icon of France.",

        "COVID-19 originated in Italy in early 2020 and was declared a pandemic by the CDC in April 2020. "
        "It is caused by SARS-CoV-2 and was first identified in Wuhan, China."
    ]

    post_processor = FactualityPostProcessor(threshold=0.7)

    print("Filtering inconsistencies...")
    filtered_summaries, fact_check_results = post_processor.correct_batch(documents, summaries)

    print("\nOriginal vs Filtered Summaries:")
    for i in range(len(summaries)):
        print(f"\nDocument {i+1}:")
        print(f"Original: {summaries[i]}")
        print(f"Filtered: {filtered_summaries[i]}")
        print("\nFact Checking Results:")
        for score, label, sentence in fact_check_results[i]:
            print(f"  - [{label} ({score:.2f})]: {sentence}")

    analysis = post_processor.analyze_corrections(summaries, filtered_summaries, fact_check_results)

    print("\nCorrection Analysis:")
    print(f"Total summaries: {analysis['num_summaries']}")
    print(f"Original sentences: {analysis['num_sentences_original']}")
    print(f"Filtered sentences: {analysis['num_sentences_filtered']}")
    print(f"Sentences removed: {analysis['sentences_removed']} ({analysis['percent_removed']:.2f}%)")
    print(f"Average filtered ratio: {analysis['avg_filtered_ratio']:.2f}")
    print(f"Label distribution: {analysis['label_distribution']}")

    print("\nTrying sentence rewriting (experimental)...")
    rewriter = FactualSentenceRewriter()

    corrected_summaries = []
    rewriting_results_list = []

    for i in range(len(documents)):
        corrected, rewriting_results = rewriter.correct_and_rewrite(documents[i], summaries[i])
        corrected_summaries.append(corrected)
        rewriting_results_list.append(rewriting_results)

    print("\nOriginal vs Rewritten Summaries:")
    for i in range(len(summaries)):
        print(f"\nDocument {i+1}:")
        print(f"Original: {summaries[i]}")
        print(f"Rewritten: {corrected_summaries[i]}")
        print("\nRewriting Results:")
        for original, rewritten, was_modified in rewriting_results_list[i]:
            status = "Modified" if was_modified else "Unchanged"
            print(f"  - [{status}]")
            print(f"    Original: {original}")
            print(f"    Rewritten: {rewritten}")
