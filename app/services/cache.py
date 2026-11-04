"""Bounded, process-local cache. Semantic matching is opt-in per request."""
from dataclasses import dataclass
from threading import RLock
from time import monotonic

import numpy as np


class SentenceEncoder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def __call__(self, text):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model.encode(text, normalize_embeddings=True)


@dataclass
class Entry:
    question: str
    model: str
    answer: str
    vector: np.ndarray
    created: float


class SemanticCache:
    def __init__(self, encoder, threshold=0.95, ttl=600, capacity=256, clock=monotonic):
        self.encoder, self.threshold, self.ttl = encoder, threshold, ttl
        self.capacity, self.clock = capacity, clock
        self.entries = []
        self.lock = RLock()

    def _vector(self, question):
        vector = np.asarray(self.encoder(question), dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm == 0:
            raise ValueError('Invalid embedding')
        return vector / norm

    def _prune(self):
        self.entries = [e for e in self.entries if self.clock() - e.created < self.ttl]

    def get(self, question, model):
        with self.lock:
            self._prune()
            candidates = [e for e in self.entries if e.model == model]
            for entry in reversed(candidates):
                if entry.question == question:
                    return entry.answer, 1.0
            if not candidates:
                return None
            vector = self._vector(question)
            entry = max(candidates, key=lambda e: float(e.vector @ vector))
            score = float(entry.vector @ vector)
            return (entry.answer, min(1.0, score)) if score >= self.threshold else None

    def put(self, question, model, answer):
        with self.lock:
            vector = self._vector(question)
            self._prune()
            self.entries = [e for e in self.entries if (e.question, e.model) != (question, model)]
            self.entries.append(Entry(question, model, answer, vector, self.clock()))
            self.entries = self.entries[-self.capacity:]
