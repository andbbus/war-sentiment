"""
Multilingual sentiment model wrapper using XLM-RoBERTa.

Model: cardiffnlp/xlm-roberta-base-sentiment-multilingual
Labels: negative (0), neutral (1), positive (2)

Key feature: chunking strategy for articles exceeding 512 tokens.
Articles are split into 510-token chunks with 50-token overlap;
probabilities are aggregated via token-weighted averaging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "cardiffnlp/xlm-roberta-base-sentiment-multilingual"

# Fallback if the above is unavailable
FALLBACK_MODEL_NAME = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

CHUNK_SIZE = 510      # Tokens per chunk (leaves room for [CLS] and [SEP])
CHUNK_OVERLAP = 50    # Token overlap between adjacent chunks


def get_device() -> torch.device:
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SentimentModel:
    """
    Wrapper around a HuggingFace sequence classification model.
    Handles tokenization, chunking, batched inference, and result aggregation.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: torch.device | None = None):
        self.model_name = model_name
        self.device = device or get_device()
        print(f"Loading model: {model_name}")
        print(f"Device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print(f"Falling back to {FALLBACK_MODEL_NAME}")
            self.model_name = FALLBACK_MODEL_NAME
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL_NAME)

        self.model.to(self.device)
        self.model.eval()

        # Confirm label mapping: we assume neg=0, neu=1, pos=2
        id2label = self.model.config.id2label
        print(f"Label mapping: {id2label}")

    def _tokenize_chunk(self, text: str) -> list[dict]:
        """
        Tokenize text and split into chunks if it exceeds CHUNK_SIZE tokens.

        Returns a list of dicts, each with:
            input_ids, attention_mask, token_count
        """
        # Tokenize without adding special tokens to count raw tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= CHUNK_SIZE:
            # Fits in one chunk — encode normally with special tokens
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
            return [
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "token_count": len(tokens),
                }
            ]

        # Split into overlapping chunks
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]

            # Re-encode chunk with special tokens by decoding back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            enc = self.tokenizer(
                chunk_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
            )
            chunks.append(
                {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "token_count": len(chunk_tokens),
                }
            )

            if end == len(tokens):
                break
            start = end - CHUNK_OVERLAP  # Slide forward with overlap

        return chunks

    @torch.no_grad()
    def _predict_chunks(self, chunks: list[dict]) -> dict[str, float]:
        """
        Run inference on all chunks and aggregate probabilities.

        Aggregation: token-count-weighted average of softmax probabilities.
        This gives more weight to longer, more informative chunks.

        Returns:
            {"negative": float, "neutral": float, "positive": float, "label": str}
        """
        total_tokens = sum(c["token_count"] for c in chunks)
        weighted_probs = torch.zeros(3)  # [neg, neu, pos]

        for chunk in chunks:
            input_ids = chunk["input_ids"].to(self.device)
            attention_mask = chunk["attention_mask"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits[0], dim=-1).cpu()
            weight = chunk["token_count"] / total_tokens
            weighted_probs += weight * probs

        label_idx = int(weighted_probs.argmax().item())
        id2label = self.model.config.id2label
        label = id2label[label_idx].lower()

        # Normalize label names to negative/neutral/positive
        label_map = {
            "negative": "negative", "neg": "negative",
            "neutral": "neutral", "neu": "neutral",
            "positive": "positive", "pos": "positive",
        }
        label = label_map.get(label, label)

        return {
            "bert_negative": float(weighted_probs[0]),
            "bert_neutral":  float(weighted_probs[1]),
            "bert_positive": float(weighted_probs[2]),
            "bert_sentiment": label,
        }

    def predict(self, texts: list[str]) -> list[dict]:
        """
        Run sentiment inference on a list of texts.

        Args:
            texts: List of article texts (any language supported by the model).

        Returns:
            List of dicts with bert_negative, bert_neutral, bert_positive, bert_sentiment.
        """
        results = []
        for text in texts:
            if not text or not text.strip():
                results.append(
                    {
                        "bert_negative": None,
                        "bert_neutral": None,
                        "bert_positive": None,
                        "bert_sentiment": None,
                    }
                )
                continue

            chunks = self._tokenize_chunk(text)
            result = self._predict_chunks(chunks)
            results.append(result)

        return results

    def predict_one(self, text: str) -> dict:
        """Convenience method for single-text prediction."""
        return self.predict([text])[0]
