"""Router v4 (prereg SS66): full-subject entity match, clause scoping, relation keys.

v3 compared the embedding of the whole query with the embedding of the edit prompt, so
subject and relation were entangled. v4 separates them:

1. entity filter: an exact normalized subject match on word boundaries, otherwise an
   approximate match that aligns ALL subject tokens to consecutive query tokens; every
   fact with the matched subject is a candidate (siblings are kept);
2. clause scoping: only the sentence that contains the matched subject is used;
3. relation key: the subject span is replaced by a placeholder in the clause and in the
   fact's edit prompt, and the frozen sentence encoder compares these relation texts.

The router has no trainable parameters; tau is calibrated on a disjoint dev split.
"""

from __future__ import annotations

import re

PLACEHOLDER = "X"
_SENTENCE_END = re.compile(r"[.!?;]")


def _osa(a: str, b: str) -> int:
    """Optimal string alignment (adjacent transposition counts as 1)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def normalize(text: str) -> str:
    """Lower-case and map hyphens/underscores to spaces, keeping character positions."""
    out = []
    for ch in text:
        low = ch.lower()
        ch = low if len(low) == 1 else ch
        out.append(" " if ch in "-_" else ch)
    return "".join(out)


def _tokens(text: str):
    """Word tokens of a normalized string with their (start, end) character spans."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[^\W_]+", text)]


def _token_close(subject_token: str, query_token: str) -> bool:
    if subject_token == query_token:
        return True
    if len(subject_token) < 4:
        return False
    limit = 2 if len(subject_token) >= 6 else 1
    return _osa(subject_token, query_token) <= limit


def find_subject(query: str, subject: str):
    """Character span of the subject in the query, or None.

    An exact normalized match must sit on word boundaries; otherwise every subject token
    must align with a consecutive query token (approximate for tokens of >= 4 chars).
    """
    qn, sn = normalize(query), normalize(subject).strip()
    if not sn:
        return None
    for m in re.finditer(re.escape(sn), qn):
        before = qn[m.start() - 1] if m.start() > 0 else " "
        after = qn[m.end()] if m.end() < len(qn) else " "
        if not before.isalnum() and not after.isalnum():
            return m.start(), m.end()
    s_tokens = [t for t, _, _ in _tokens(sn)]
    q_tokens = _tokens(qn)
    m = len(s_tokens)
    if m == 0 or m > len(q_tokens):
        return None
    for start in range(len(q_tokens) - m + 1):
        window = q_tokens[start : start + m]
        if all(_token_close(s, q[0]) for s, q in zip(s_tokens, window)):
            return window[0][1], window[-1][2]
    return None


def clause_with_span(text: str, span) -> tuple[str, tuple[int, int]]:
    """The sentence of `text` containing `span`, and the span relative to it."""
    start, end = span
    left = 0
    for m in _SENTENCE_END.finditer(text, 0, start):
        left = m.end()
    m = _SENTENCE_END.search(text, end)
    right = m.end() if m else len(text)
    clause = text[left:right]
    lead = len(clause) - len(clause.lstrip())
    return clause.strip(), (start - left - lead, end - left - lead)


def relation_text(text: str, span) -> str:
    """The clause containing `span` with the subject replaced by the placeholder."""
    clause, (s, e) = clause_with_span(text, span)
    return (clause[:s] + PLACEHOLDER + clause[e:]).strip()


def key_text(prompt: str, subject: str) -> str:
    """Relation text of a stored prompt (the whole prompt if the subject is not found)."""
    span = find_subject(prompt, subject)
    return relation_text(prompt, span) if span else prompt


class RouterV4:
    """Same interface as pilot.Router: route(q, enc) -> (index or None, similarity)."""

    def __init__(self, enc, facts, tau: float):
        self.tau = tau
        self.key_texts = [key_text(f["probe"], f["subject"]) for f in facts]
        self.keys = enc.encode(self.key_texts, normalize_embeddings=True)
        self._index(facts)

    def _index(self, facts):
        self.subjects = [f["subject"] for f in facts]
        self._by_subject: dict[str, list[int]] = {}
        for i, s in enumerate(self.subjects):
            self._by_subject.setdefault(normalize(s).strip(), []).append(i)
        # content order of the entries: ties between candidates are broken by it, so
        # routing depends on the stored set and not on the order of insertion
        self._content = [
            (
                normalize(f["subject"]).strip(),
                key_text(f["probe"], f["subject"]),
                str(f.get("id", "")),
            )
            for f in facts
        ]
        # token index: a subject can match only if each of its tokens is close to some
        # query token, so the full alignment runs on few subjects
        self._subject_tokens = {
            s: [t for t, _, _ in _tokens(s)] for s in self._by_subject
        }
        self._vocab = sorted({t for ts in self._subject_tokens.values() for t in ts})
        self._close_cache: dict[str, frozenset] = {}

    def _close(self, query_token: str) -> frozenset:
        if query_token not in self._close_cache:
            self._close_cache[query_token] = frozenset(
                t
                for t in self._vocab
                if abs(len(t) - len(query_token)) <= 2 and _token_close(t, query_token)
            )
        return self._close_cache[query_token]

    def _possible(self, q: str):
        qn = normalize(q)
        close = set()
        for t, _, _ in _tokens(qn):
            close |= self._close(t)
        return [
            s
            for s, ts in self._subject_tokens.items()
            if (ts and all(t in close for t in ts)) or (s and s in qn)
        ]

    def candidates(self, q: str):
        """(indices, spans) of the entries of the top-ranked matching subjects, or ([], []).

        Matching subjects are ranked by the length of the query text they match, then
        exact over approximate match, so a near-namesake (Roberta Smith) never displaces
        an exact match of the same words (Robert Smith), while a longer subject still wins
        over its prefix (Paris Hilton over Paris). Every entry of every subject tied at
        the top is a candidate, with the span of its own subject. Candidates are ordered by content (subject, key text, id), so ties, here
        and between equal similarities in route(), do not depend on insertion order.
        """
        qn = normalize(q)
        best, best_rank = [], None
        for subject in self._possible(q):
            span = find_subject(q, subject)
            if span is None:
                continue
            rank = (span[1] - span[0], qn[span[0] : span[1]] == subject)
            if best_rank is None or rank > best_rank:
                best, best_rank = [], rank
            if rank == best_rank:
                best += [(i, span) for i in self._by_subject[subject]]
        best.sort(key=lambda c: (self._content[c[0]], c[0]))
        return [i for i, _ in best], [s for _, s in best]

    def _query_vecs(self, q: str, spans, enc):
        """Relation-text embedding of the query for each candidate span."""
        texts = [relation_text(q, s) for s in spans]
        unique = sorted(set(texts))
        vecs = dict(zip(unique, enc.encode(unique, normalize_embeddings=True)))
        return [vecs[t] for t in texts]

    def score(self, q: str, enc):
        """All candidate similarities, for calibration: (indices, sims) or ([], [])."""
        cand, spans = self.candidates(q)
        if not cand:
            return [], []
        vs = self._query_vecs(q, spans, enc)
        return cand, [float(self.keys[i] @ v) for i, v in zip(cand, vs)]

    def route(self, q, enc):
        cand, sims = self.score(q, enc)
        if not cand:
            return None, 0.0
        k = max(range(len(cand)), key=lambda t: sims[t])
        j, sim = cand[k], sims[k]
        return (j if sim >= self.tau else None), sim


class RouterV5(RouterV4):
    """Router v4 plus a contrast test against the relation keys of all stored facts (SS72).

    The best candidate i is accepted only if s_i >= tau and s_i >= max_j cos(r(q), k_j) - m
    over every stored key j, whatever its subject: a query whose relation is stored for
    another subject is closer to that key than to the candidate's and is rejected.
    """

    def __init__(self, enc, facts, tau: float, margin: float):
        super().__init__(enc, facts, tau)
        self.margin = margin

    def contrast_score(self, q: str, enc):
        """(indices, sims, best similarity to any stored key) or ([], [], None)."""
        cand, spans = self.candidates(q)
        if not cand:
            return [], [], None
        vs = self._query_vecs(q, spans, enc)
        return (
            cand,
            [float(self.keys[i] @ v) for i, v in zip(cand, vs)],
            max(float((self.keys @ v).max()) for v in vs),
        )

    def route(self, q, enc):
        cand, sims, best = self.contrast_score(q, enc)
        if not cand:
            return None, 0.0
        k = max(range(len(cand)), key=lambda t: sims[t])
        j, sim = cand[k], sims[k]
        ok = sim >= self.tau and sim >= best - self.margin
        return (j if ok else None), sim
