import json
import numpy as np

from app.settings import settings
from sentence_transformers import SentenceTransformer


class TextClassifier:
    def __init__(self):
        self.model = SentenceTransformer(settings.model_name)
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

        embeddings = self.model.encode(examples, normalize_embeddings=True)
        return labels, embeddings

    def classify(self, text: str):
        emb = self.model.encode([text], normalize_embeddings=True)[0]

        scores = np.dot(self.label_embeddings, emb)
        
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < settings.similarity_threshold:
            return "unknown", best_score

        return self.labels[best_idx], best_score