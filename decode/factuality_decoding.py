import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    BeamSearchScorer,
    StoppingCriteriaList,
    MaxLengthCriteria
)
from torch.nn import functional as F
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt', quiet=True)

class FactualityConstraintLogitsProcessor(torch.nn.Module):
    def __init__(
        self,
        source_doc,
        nli_model,
        nli_tokenizer,
        alpha=0.2,
        min_tokens_to_check=10,
        check_interval=5,
        device="cuda"
    ):
        super().__init__()
        self.source_doc = source_doc
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.alpha = alpha
        self.min_tokens_to_check = min_tokens_to_check
        self.check_interval = check_interval
        self.device = device
        self.last_check_step = 0
        self.entailment_scores = {}

    def calculate_entailment_score(self, generated_text):
        inputs = self.nli_tokenizer(
            self.source_doc,
            generated_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)

        entailment_score = probs[0, 2].item()
        contradiction_score = probs[0, 0].item()

        return entailment_score - contradiction_score

    def __call__(self, input_ids, scores, cur_len):
        if (cur_len - self.last_check_step >= self.check_interval and
            cur_len >= self.min_tokens_to_check):

            self.last_check_step = cur_len

            batch_size = input_ids.shape[0]
            modified_scores = scores.clone()

            for batch_idx in range(batch_size):
                seq = input_ids[batch_idx]
                gen_text = self.nli_tokenizer.decode(seq, skip_special_tokens=True)

                if gen_text not in self.entailment_scores:
                    self.entailment_scores[gen_text] = self.calculate_entailment_score(gen_text)

                factuality_score = self.entailment_scores[gen_text]

                if factuality_score < 0:
                    penalty = self.alpha * abs(factuality_score)
                    modified_scores[batch_idx] = scores[batch_idx] - penalty

            return modified_scores

        return scores

class EntityConstraintLogitsProcessor(torch.nn.Module):
    def __init__(
        self,
        source_doc,
        entity_extractor,
        tokenizer,
        beta=0.1,
        device="cuda"
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.beta = beta
        self.device = device

        try:
            import spacy
            if entity_extractor is None:
                self.nlp = spacy.load("en_core_web_sm")
            else:
                self.nlp = entity_extractor

            doc = self.nlp(source_doc)
            self.source_entities = set()
            for ent in doc.ents:
                self.source_entities.add(ent.text.lower())

            self.entity_token_ids = set()
            for entity in self.source_entities:
                entity_ids = tokenizer.encode(entity, add_special_tokens=False)
                for token_id in entity_ids:
                    self.entity_token_ids.add(token_id)
        except:
            self.entity_token_ids = set()

    def __call__(self, input_ids, scores, cur_len):
        if len(self.entity_token_ids) == 0:
            return scores

        vocab_size = scores.shape[-1]
        entity_boost = torch.zeros(vocab_size, device=self.device)

        for token_id in self.entity_token_ids:
            if token_id < vocab_size:
                entity_boost[token_id] = self.beta

        return scores + entity_boost

class FactualityBeamScorer:
    def __init__(
        self,
        source_doc,
        nli_model,
        nli_tokenizer,
        tokenizer,
        gamma=2.0,
        device="cuda"
    ):
        self.source_doc = source_doc
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.device = device

    def score_beams(self, beam_outputs):
        decoded_beams = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in beam_outputs
        ]

        factuality_scores = []

        for summary in decoded_beams:
            sentences = sent_tokenize(summary)
            sentence_scores = []

            for sentence in sentences:
                if not sentence.strip():
                    continue

                inputs = self.nli_tokenizer(
                    self.source_doc,
                    sentence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.nli_model(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)

                entailment_score = probs[0, 2].item()
                contradiction_score = probs[0, 0].item()

                sentence_scores.append(entailment_score - contradiction_score)

            avg_score = np.mean(sentence_scores) if sentence_scores else 0.0
            factuality_scores.append(avg_score)

        normalized_scores = np.array(factuality_scores) * self.gamma
        return torch.tensor(normalized_scores, device=self.device)

class FactualityGuidedDecoder:
    def __init__(
        self,
        model_name,
        nli_model_name="roberta-large-mnli",
        max_input_length=1024,
        max_output_length=128,
        num_beams=6,
        device=None
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)

        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.nlp = None

    def generate_constrained_beam_search(self, input_ids, attention_mask, source_doc):
        batch_size = input_ids.shape[0]

        logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(10, eos_token_id=self.tokenizer.eos_token_id),
            FactualityConstraintLogitsProcessor(
                source_doc=source_doc,
                nli_model=self.nli_model,
                nli_tokenizer=self.nli_tokenizer,
                alpha=0.2,
                device=self.device
            ),
            EntityConstraintLogitsProcessor(
                source_doc=source_doc,
                entity_extractor=self.nlp,
                tokenizer=self.tokenizer,
                beta=0.1,
                device=self.device
            )
        ])

        stopping_criteria = StoppingCriteriaList([
            MaxLengthCriteria(max_length=self.max_output_length)
        ])

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.num_beams,
            device=self.device,
        )

        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        model_inputs = {
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

        outputs = self.model.beam_search(
            input_ids[:, 0:1].repeat(1, self.num_beams).view(-1, 1),
            beam_scorer,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            **model_inputs
        )

        return outputs

    def generate_beam_reranking(self, input_ids, attention_mask, source_doc):
        with torch.no_grad():
            beam_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=self.max_output_length,
                num_beams=self.num_beams,
                num_return_sequences=self.num_beams,
                early_stopping=True,
                output_scores=True,
                return_dict_in_generate=True
            )

        sequences = beam_outputs.sequences

        batch_size = input_ids.shape[0]
        num_return_sequences = self.num_beams

        beam_indices = torch.arange(
            0, batch_size * num_return_sequences, num_return_sequences,
            device=self.device
        ).unsqueeze(1)
        beam_indices = beam_indices.repeat(1, num_return_sequences).view(-1)

        reordered_sequences = sequences[beam_indices]
        reordered_sequences = reordered_sequences.view(
            batch_size, num_return_sequences, -1
        )

        factuality_scorer = FactualityBeamScorer(
            source_doc=source_doc,
            nli_model=self.nli_model,
            nli_tokenizer=self.nli_tokenizer,
            tokenizer=self.tokenizer,
            device=self.device
        )

        best_sequences = []

        for batch_idx in range(batch_size):
            beam_outputs = reordered_sequences[batch_idx]

            factuality_scores = factuality_scorer.score_beams(beam_outputs)

            sequence_scores = beam_outputs.scores
            if sequence_scores is not None:
                sequence_scores = sequence_scores.mean(dim=-1)
            else:
                sequence_scores = torch.zeros(num_return_sequences, device=self.device)

            combined_scores = sequence_scores + factuality_scores
            best_beam_idx = torch.argmax(combined_scores).item()

            best_sequences.append(beam_outputs[best_beam_idx])

        return torch.stack(best_sequences)

    def generate_summaries(self, dataset, text_column, batch_size=4):
        self.model.eval()
        self.nli_model.eval()

        summaries = []

        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating summaries with factuality-guided decoding"):
            batch = dataset[i:i+batch_size]

            source_docs = batch[text_column]

            inputs = self.tokenizer(
                source_docs,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            batch_summaries = []

            for j, source_doc in enumerate(source_docs):
                input_ids = inputs["input_ids"][j:j+1]
                attention_mask = inputs["attention_mask"][j:j+1]

                try:
                    outputs = self.generate_beam_reranking(input_ids, attention_mask, source_doc)
                    summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception as e:
                    try:
                        with torch.no_grad():
                            outputs = self.model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=self.max_output_length,
                                num_beams=4,
                                no_repeat_ngram_size=2
                            )
                        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    except:
                        summary = "Error generating summary."

                batch_summaries.append(summary)

            summaries.extend(batch_summaries)

        return summaries

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="test[:2]")

    decoder = FactualityGuidedDecoder(
        model_name="facebook/bart-large-cnn",
        num_beams=4,
        max_input_length=1024,
        max_output_length=128
    )

    summaries = decoder.generate_summaries(
        dataset=dataset,
        text_column="article",
        batch_size=1
    )

    for i, summary in enumerate(summaries):
        print(f"Summary {i+1}:")
        print(summary)
        print("-" * 80)
