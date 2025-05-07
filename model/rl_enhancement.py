import torch
import numpy as np
from torch.nn import functional as F
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification
)
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')

class FactualityRLTrainer:
    def __init__(
        self,
        base_model_name,
        nli_model_name="roberta-large-mnli",
        device=None,
        learning_rate=2e-5,
        max_input_length=1024,
        max_output_length=128
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.learning_rate = learning_rate

        print(f"Initializing RL-based factuality enhancement on {self.device}")

        print(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name).to(self.device)

        print(f"Loading NLI model: {nli_model_name}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def calculate_nli_reward(self, source_doc, generated_summary):
        self.nli_model.eval()

        summary_sentences = sent_tokenize(generated_summary)
        if not summary_sentences:
            return 0.0

        entailment_scores = []
        contradiction_scores = []

        for sentence in summary_sentences:
            if not sentence.strip():
                continue

            inputs = self.nli_tokenizer(
                source_doc,
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.nli_model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1)

            entail_score = probs[0, 2].item()
            contradiction_score = probs[0, 0].item()

            entailment_scores.append(entail_score)
            contradiction_scores.append(contradiction_score)

        if entailment_scores:
            avg_entailment = np.mean(entailment_scores)
            avg_contradiction = np.mean(contradiction_scores)

            reward = avg_entailment - avg_contradiction
        else:
            reward = 0.0

        return reward

    def sample_summary(self, input_ids, attention_mask, temperature=1.0):
        batch_size = input_ids.size(0)

        curr_ids = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=self.device
        )

        log_probs = []

        encoder_outputs = self.model.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        for step in range(self.max_output_length):
            # Get logits
            outputs = self.model.get_decoder()(
                input_ids=curr_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                return_dict=True
            )

            logits = self.model.lm_head(outputs.last_hidden_state[:, -1, :])

            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)

            log_prob = F.log_softmax(logits, dim=-1).gather(1, next_token)
            log_probs.append(log_prob)

            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            if (next_token == self.tokenizer.eos_token_id).any():
                break

        log_probs = torch.cat(log_probs, dim=1)

        return curr_ids, log_probs

    def train_rl_step(self, source_doc, input_ids, attention_mask, baseline_reward=None, gamma=0.95):
        self.optimizer.zero_grad()

        sampled_ids, log_probs = self.sample_summary(input_ids, attention_mask)

        summary = self.tokenizer.decode(sampled_ids[0].tolist(), skip_special_tokens=True)
        reward = self.calculate_nli_reward(source_doc, summary)

        reward_baseline = baseline_reward if baseline_reward is not None else reward
        advantage = reward - reward_baseline

        seq_length = log_probs.size(1)
        maskings = torch.ones(seq_length, device=self.device)
        for i in range(seq_length):
            maskings[i] = gamma ** i

        policy_loss = -log_probs * advantage * maskings
        policy_loss = policy_loss.mean()

        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item(), reward, summary

    def train_rl(self, train_dataset, text_column, num_epochs=3, batch_size=1, eval_interval=100):
        self.model.train()
        num_samples = len(train_dataset)

        total_steps = num_epochs * (num_samples // batch_size)
        step = 0
        best_reward = 0.0
        running_reward = 0.0
        running_loss = 0.0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            indices = np.random.permutation(num_samples)

            for i in tqdm(range(0, num_samples, batch_size), desc=f"Training RL Epoch {epoch+1}"):
                step += 1
                batch_indices = indices[i:i+batch_size]

                batch_docs = [train_dataset[idx][text_column] for idx in batch_indices]

                for doc_idx, source_doc in enumerate(batch_docs):
                    inputs = self.tokenizer(
                        source_doc,
                        max_length=self.max_input_length,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt"
                    ).to(self.device)

                    loss, reward, summary = self.train_rl_step(
                        source_doc,
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        baseline_reward=running_reward
                    )

                    running_loss = 0.95 * running_loss + 0.05 * loss
                    running_reward = 0.95 * running_reward + 0.05 * reward

                if step % eval_interval == 0:
                    print(f"Step {step}/{total_steps}, Loss: {running_loss:.4f}, Reward: {running_reward:.4f}")
                    print(f"Sample summary: {summary[:100]}...")

                    if running_reward > best_reward:
                        best_reward = running_reward
                        print(f"New best reward: {best_reward:.4f}, saving model...")
                        self.model.save_pretrained("./output/rl_enhanced_model")
                        self.tokenizer.save_pretrained("./output/rl_enhanced_model")

        print(f"RL training complete. Best reward: {best_reward:.4f}")
        return self.model

    def generate_summaries(self, dataset, text_column, batch_size=8):
        self.model.eval()
        num_samples = len(dataset)
        generated_summaries = []

        for i in tqdm(range(0, num_samples, batch_size), desc="Generating summaries"):
            batch_docs = dataset[i:i+batch_size][text_column]

            inputs = self.tokenizer(
                batch_docs,
                max_length=self.max_input_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=self.max_output_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )

            batch_summaries = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generated_summaries.extend(batch_summaries)

        return generated_summaries

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:100]")

    rl_trainer = FactualityRLTrainer(
        base_model_name="facebook/bart-large-cnn",
        learning_rate=2e-5
    )

    print("Training with RL...")
    rl_trainer.train_rl(
        train_dataset=dataset,
        text_column="article",
        num_epochs=1,
        batch_size=1,
        eval_interval=10
    )

    print("Generating summaries...")
    summaries = rl_trainer.generate_summaries(
        dataset=dataset,
        text_column="article",
        batch_size=2
    )

    print("\nSample Summaries:")
    for i in range(min(3, len(summaries))):
        print(f"Summary {i+1}: {summaries[i]}")
        print("-" * 80)
