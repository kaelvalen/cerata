"""Router v7 (prereg SS75): router v4 with a pluggable relation scorer and optional
generated keys.

The candidate set is v4's (complete-subject match). The relation text of the query clause
is compared with every key of a candidate (the edit prompt's relation text and, if
given, the relation texts of generated rewrites of it); the candidate's score is the
maximum. Scorers are frozen bi-encoders (cosine) or a frozen cross-encoder (pair score).
"""

from __future__ import annotations

import numpy as np
from router_v4 import RouterV4, find_subject, key_text, relation_text


class BiScorer:
    def __init__(self, model_id: str, prefix: str = "", device: str = "cuda"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, device=device)
        self.prefix = prefix
        self._cache: dict[str, np.ndarray] = {}

    def prefetch(self, texts):
        missing = sorted({t for t in texts if t not in self._cache})
        if missing:
            emb = self.model.encode(
                [self.prefix + t for t in missing],
                batch_size=128,
                normalize_embeddings=True,
            )
            self._cache.update(zip(missing, emb))

    def sims(self, query: str, keys: list[str]) -> list[float]:
        self.prefetch([query, *keys])
        q = self._cache[query]
        return [float(self._cache[k] @ q) for k in keys]


class CrossScorer:
    def __init__(self, model_id: str, device: str = "cuda"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_id, device=device)
        self._cache: dict[tuple[str, str], float] = {}

    def prefetch_pairs(self, pairs):
        missing = sorted({p for p in pairs if p not in self._cache})
        if missing:
            scores = self.model.predict(
                missing, batch_size=128, show_progress_bar=False
            )
            self._cache.update(zip(missing, (float(s) for s in np.ravel(scores))))

    def sims(self, query: str, keys: list[str]) -> list[float]:
        pairs = [(query, k) for k in keys]
        self.prefetch_pairs(pairs)
        return [self._cache[p] for p in pairs]


class RouterV7(RouterV4):
    """route(q, enc=None) -> (index or None, score); `enc` is ignored."""

    def __init__(self, facts, scorer, tau: float, extra_keys=None):
        self.tau = tau
        self.scorer = scorer
        self.key_sets = []
        for f in facts:
            keys = [key_text(f["probe"], f["subject"])]
            for p in (extra_keys or {}).get(f"{f['subject']}\t{f['probe']}", []):
                span = find_subject(p, f["subject"])
                if span is not None:
                    k = relation_text(p, span)
                    if k not in keys:
                        keys.append(k)
            self.key_sets.append(keys)
        self._index(facts)

    def score(self, q: str, enc=None):
        cand, spans = self.candidates(q)
        if not cand:
            return [], []
        return cand, [
            max(self.scorer.sims(relation_text(q, s), self.key_sets[i]))
            for i, s in zip(cand, spans)
        ]

    def route(self, q, enc=None):
        cand, sims = self.score(q)
        if not cand:
            return None, 0.0
        k = max(range(len(cand)), key=lambda t: sims[t])
        j, sim = cand[k], sims[k]
        return (j if sim >= self.tau else None), sim


SCORERS = {
    "minilm": ("bi", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
    "mpnet": ("bi", "sentence-transformers/all-mpnet-base-v2", ""),
    "bge": ("bi", "BAAI/bge-large-en-v1.5", ""),
    "e5": ("bi", "intfloat/e5-large-v2", "query: "),
    "quora_ce": ("cross", "cross-encoder/quora-roberta-large", ""),
    # SS76: all-mpnet-base-v2 trained offline on relation triplets (train_relation_encoder.py)
    "relenc": ("bi", "results/live_learning/confirm/relation_encoder_mpnet", ""),
    # SS77: trained on Wikidata property templates (A: no CounterFact relation; B: + SS76)
    "wikiA": ("bi", "results/live_learning/confirm/relation_encoder_wiki_A", ""),
    "wikiB": ("bi", "results/live_learning/confirm/relation_encoder_wiki_B", ""),
}


def make_scorer(name: str, device: str = "cuda"):
    kind, model_id, prefix = SCORERS[name]
    return (
        BiScorer(model_id, prefix, device)
        if kind == "bi"
        else CrossScorer(model_id, device)
    )


class RouterV10(RouterV7):
    """SS78: router v8 plus per-fact negative keys generated at write time.

    The best candidate is accepted only if s_pos >= tau and s_pos - s_neg >= margin, where
    s_neg is its maximum similarity to its own negative keys (no negatives: -inf).
    """

    def __init__(
        self, facts, scorer, tau: float, margin: float, neg_keys, extra_keys=None
    ):
        super().__init__(facts, scorer, tau, extra_keys)
        self.margin = margin
        self.neg_sets = []
        for f in facts:
            negs = []
            for p in neg_keys.get(f"{f['subject']}\t{f['probe']}", []):
                span = find_subject(p, f["subject"])
                if span is not None:
                    negs.append(relation_text(p, span))
            self.neg_sets.append(negs)

    def pos_neg(self, q: str):
        """(candidate index or None, s_pos, s_neg) for the best candidate by s_pos."""
        cand, spans = self.candidates(q)
        if not cand:
            return None, 0.0, float("-inf")
        texts = [relation_text(q, s) for s in spans]
        sims = [max(self.scorer.sims(r, self.key_sets[i])) for i, r in zip(cand, texts)]
        k = max(range(len(cand)), key=lambda t: sims[t])
        j, r = cand[k], texts[k]
        negs = self.neg_sets[j]
        s_neg = max(self.scorer.sims(r, negs)) if negs else float("-inf")
        return j, sims[k], s_neg

    def route(self, q, enc=None):
        j, s_pos, s_neg = self.pos_neg(q)
        if j is None:
            return None, 0.0
        ok = s_pos >= self.tau and s_pos - s_neg >= self.margin
        return (j if ok else None), s_pos
