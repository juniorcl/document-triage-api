import pytest
import numpy as np
from unittest.mock import MagicMock

from app.classifier import TextClassifier


@pytest.fixture
def classifier(monkeypatch):
    clf = TextClassifier()

    # Mock tokenizer
    clf.tokenizer = MagicMock()
    clf.tokenizer.return_value = {
        "input_ids": list(range(50))
    }
    clf.tokenizer.decode.side_effect = lambda x: "chunk text"

    # Mock model
    clf.model = MagicMock()
    clf.model.encode.side_effect = lambda x, **kwargs: (
        np.random.rand(len(x), 384)
        if isinstance(x, list)
        else np.random.rand(384)
    )

    # Mock labels
    clf.labels = ["accepted", "rejected"]
    clf.label_embeddings = np.array([
        np.ones(384),
        np.zeros(384)
    ])

    return clf

def test_calc_num_tokens(classifier):
    text = "some text"
    num_tokens = classifier._calc_num_tokens(text)

    assert isinstance(num_tokens, int)
    assert num_tokens == 50

def test_chunk_text_returns_chunks(classifier):
    classifier.tokenizer.return_value = {
        "input_ids": list(range(500))
    }

    chunks = classifier._chunk_text("long text", max_tokens=100, overlap=20)

    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)

def test_embed_text_short_text(classifier):
    classifier._calc_num_tokens = MagicMock(return_value=100)

    emb = classifier._embed_text("short text")

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)

def test_embed_text_long_text_uses_chunking(classifier):
    classifier._calc_num_tokens = MagicMock(return_value=500)
    classifier._chunk_text = MagicMock(return_value=["a", "b", "c"])

    emb = classifier._embed_text("long text")

    assert classifier._chunk_text.called
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (384,)

def test_classify_known_label(classifier, monkeypatch):
    monkeypatch.setattr(
        "app.settings.settings.similarity_threshold", 0.1
    )

    label, score = classifier.classify("test text")

    assert label in classifier.labels
    assert isinstance(score, float)

def test_classify_unknown_label(classifier, monkeypatch):
    monkeypatch.setattr(
        "app.settings.settings.similarity_threshold", 0.9
    )

    # embedding que gera score baixo
    classifier._embed_text = MagicMock(
        return_value=np.zeros(384)
    )

    label, score = classifier.classify("test text")

    assert label == "unknown"
    assert score == 0.0
