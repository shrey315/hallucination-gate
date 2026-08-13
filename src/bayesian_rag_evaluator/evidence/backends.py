from __future__ import annotations

import re
from abc import ABC, abstractmethod

import numpy as np

from bayesian_rag_evaluator.evidence.cache import EMBED_CACHE, NLI_CACHE, pair_key, text_key


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = token_set(a), token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class EmbeddingBackend(ABC):
    @abstractmethod
    def similarity(self, a: str, b: str) -> float:
        raise NotImplementedError

    def encode_many(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized vectors, shape (n, d). Override for batching."""
        raise NotImplementedError

    def similarity_matrix(self, queries: list[str], docs: list[str]) -> np.ndarray:
        q = self.encode_many(queries)
        d = self.encode_many(docs)
        return np.clip(q @ d.T, 0.0, 1.0)


class HeuristicEmbeddingBackend(EmbeddingBackend):
    """Word-overlap similarity for tests and offline use."""

    def similarity(self, a: str, b: str) -> float:
        return jaccard_similarity(a, b)

    def encode_many(self, texts: list[str]) -> np.ndarray:
        vocab: dict[str, int] = {}
        tokenized = [token_set(t) for t in texts]
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        if not vocab:
            return np.zeros((len(texts), 1), dtype=np.float32)
        mat = np.zeros((len(texts), len(vocab)), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            for tok in tokens:
                mat[i, vocab[tok]] = 1.0
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        return mat / norms

    def similarity_matrix(self, queries: list[str], docs: list[str]) -> np.ndarray:
        scores = np.zeros((len(queries), len(docs)), dtype=np.float32)
        for i, q in enumerate(queries):
            for j, d in enumerate(docs):
                scores[i, j] = jaccard_similarity(q, d)
        return scores


class SentenceTransformerBackend(EmbeddingBackend):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim

        self._model = SentenceTransformer(model_name)
        self._cos_sim = cos_sim

    def encode_many(self, texts: list[str]) -> np.ndarray:
        missing: list[tuple[int, str]] = []
        vectors: list[np.ndarray | None] = [None] * len(texts)
        for i, text in enumerate(texts):
            cached = EMBED_CACHE.get(text_key(text))
            if cached is not None:
                vectors[i] = cached
            else:
                missing.append((i, text))
        if missing:
            encoded = self._model.encode(
                [t for _, t in missing],
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=64,
                show_progress_bar=False,
            )
            for (i, text), vec in zip(missing, encoded, strict=True):
                arr = np.asarray(vec, dtype=np.float32)
                EMBED_CACHE.set(text_key(text), arr)
                vectors[i] = arr
        return np.stack(vectors)  # type: ignore[arg-type]

    def similarity(self, a: str, b: str) -> float:
        embeddings = self.encode_many([a, b])
        score = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, score))


class NLIBackend(ABC):
    @abstractmethod
    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def contradiction_prob(self, premise: str, hypothesis: str) -> float:
        raise NotImplementedError

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        return [
            {
                "entailment": self.entailment_prob(p, h),
                "contradiction": self.contradiction_prob(p, h),
                "neutral": max(
                    0.0, 1.0 - self.entailment_prob(p, h) - self.contradiction_prob(p, h)
                ),
            }
            for p, h in pairs
        ]


class HeuristicNLIBackend(NLIBackend):
    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        overlap = jaccard_similarity(premise, hypothesis)
        hyp_tokens = token_set(hypothesis)
        prem_tokens = token_set(premise)
        if not hyp_tokens:
            return 0.0
        coverage = len(hyp_tokens & prem_tokens) / len(hyp_tokens)
        return max(0.0, min(1.0, 0.5 * overlap + 0.5 * coverage))

    def contradiction_prob(self, premise: str, hypothesis: str) -> float:
        from bayesian_rag_evaluator.evidence.multimodal import extract_numbers

        negations = {
            "not",
            "no",
            "never",
            "none",
            "without",
            "false",
            "incorrect",
            "cannot",
            "denied",
            "deny",
            "forbidden",
            "disallowed",
        }
        allowances = {"always", "allow", "allows", "allowed", "can", "yes", "available"}
        h = token_set(hypothesis)
        p = token_set(premise)
        if not h or not p:
            return 0.0

        h_neg = bool(h & negations)
        p_neg = bool(p & negations)
        h_allow = bool(h & allowances)
        p_allow = bool(p & allowances)
        if (h_neg and p_allow and not p_neg) or (p_neg and h_allow and not h_neg):
            return 0.82
        if bool(h_neg) != bool(p_neg) and (h & p):
            return 0.75

        h_nums = extract_numbers(hypothesis)
        p_nums = set(extract_numbers(premise))
        if h_nums and p_nums and not any(n in p_nums for n in h_nums):
            return 0.7

        shared = h & p
        if not shared:
            return 0.15
        return 0.10

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for premise, hypothesis in pairs:
            ent = self.entailment_prob(premise, hypothesis)
            con = self.contradiction_prob(premise, hypothesis)
            out.append(
                {
                    "entailment": ent,
                    "contradiction": con,
                    "neutral": max(0.0, 1.0 - ent - con),
                }
            )
        return out


class CrossEncoderNLIBackend(NLIBackend):
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._labels = ["contradiction", "entailment", "neutral"]

    def _decode_logits(self, logits) -> dict[str, float]:
        if hasattr(logits, "ndim") and getattr(logits, "ndim", 1) == 2:
            probs = _softmax(logits[0])
        else:
            probs = _softmax(np.asarray(logits, dtype=float))
        return dict(zip(self._labels, probs.tolist(), strict=True))

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        if not pairs:
            return []
        results: list[dict[str, float] | None] = [None] * len(pairs)
        missing: list[tuple[int, tuple[str, str]]] = []
        for i, pair in enumerate(pairs):
            cached = NLI_CACHE.get(pair_key(*pair))
            if cached is not None:
                results[i] = cached
            else:
                missing.append((i, pair))
        if missing:
            logits = self._model.predict(
                [p for _, p in missing],
                batch_size=32,
                show_progress_bar=False,
            )
            logits = np.asarray(logits, dtype=float)
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            for (i, pair), row in zip(missing, logits, strict=True):
                decoded = dict(zip(self._labels, _softmax(row).tolist(), strict=True))
                NLI_CACHE.set(pair_key(*pair), decoded)
                results[i] = decoded
        return results  # type: ignore[return-value]

    def _predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        return self.predict_batch([(premise, hypothesis)])[0]

    def entailment_prob(self, premise: str, hypothesis: str) -> float:
        return self._predict(premise, hypothesis)["entailment"]

    def contradiction_prob(self, premise: str, hypothesis: str) -> float:
        return self._predict(premise, hypothesis)["contradiction"]


def _softmax(x):
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


def create_embedding_backend(use_heuristic: bool = False) -> EmbeddingBackend:
    if use_heuristic:
        return HeuristicEmbeddingBackend()
    return SentenceTransformerBackend()


def create_nli_backend(use_heuristic: bool = False) -> NLIBackend:
    if use_heuristic:
        return HeuristicNLIBackend()
    return CrossEncoderNLIBackend()
