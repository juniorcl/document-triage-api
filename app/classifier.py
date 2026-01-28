import json
import numpy as np

from app.settings import settings
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer


class TextClassifier:
    def __init__(self):
        self.model_name = settings.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = SentenceTransformer(self.model_name)
        self.labels, self.label_embeddings = self._load_labels()

    def _load_labels(self):
        with open("data/labels.json") as f:
            data = json.load(f)

        labels = []
        examples = []

        for label, texts in data.items():
            for text in texts:
                labels.append(label)
                examples.append(text)

        embeddings = self.model.encode(
            examples,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return labels, embeddings

    def _calc_num_tokens(self, text: str) -> int:
        tokens = self.tokenizer(
            text,
            truncation=False,
            add_special_tokens=True
        )["input_ids"]

        return len(tokens)

    def _chunk_text(self, text: str, max_tokens=200, overlap=20):
        tokens = self.tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]

        chunks = []
        start = 0

        while start < len(tokens):
            end = start + max_tokens
            chunk_tokens = tokens[start:end]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            start = end - overlap

        return chunks

    def _embed_text(self, text: str) -> np.ndarray:
        num_tokens = self._calc_num_tokens(text)

        if num_tokens <= 220:
            return self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

        chunks = self._chunk_text(text)

        chunk_embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return np.mean(chunk_embeddings, axis=0)

    def classify(self, text: str):
        emb = self._embed_text(text)

        scores = np.dot(self.label_embeddings, emb)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < settings.similarity_threshold:
            return "unknown", best_score

        return self.labels[best_idx], best_score