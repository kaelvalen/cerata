"""Transactional delta store (slice-1, docs/LIVING_MODEL_SLICE_PREREG.md).

Each fact -> one LoRA delta trained from the same frozen base; the live state is the
sum of the active deltas in id order, materialised into a fixed-capacity adapter.
`revoke` drops a delta and rebuilds; `state_hash` is the pure function of the active
set, so revoking the last addition restores the recorded pre-add hash exactly.

Smoke:  python ledger.py --facts 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from harness import chat, load_model, make_lora  # noqa: E402
from stream import FACTS  # noqa: E402

R = 16  # rank per fact
CAP = 8  # capacity, in facts


class DeltaStore:
    def __init__(self, model_name: str, lr: float = 3e-4, steps: int = 16, seed: int = 0):
        self.model_name, self.lr, self.steps, self.seed = model_name, lr, steps, seed
        self.tok, base = load_model(model_name)
        self.model = make_lora(base, r=CAP * R)
        self.deltas: dict[str, dict] = {}
        self.keys: dict[str, str] = {}
        self.base_hash = hashlib.sha256(
            f"{model_name}|r{R}|cap{CAP}".encode()
        ).hexdigest()
        self._zero()

    # -- state ------------------------------------------------------------
    def _zero(self) -> None:
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if "lora_" in n:
                    p.zero_()

    def materialize(self) -> None:
        self._zero()
        with torch.no_grad():
            params = dict(self.model.named_parameters())
            for idx, fid in enumerate(sorted(self.deltas)):
                for n, ab in self.deltas[fid].items():
                    params[n][idx * R : (idx + 1) * R, :] = ab["A"]
                    params[n.replace("lora_A", "lora_B")][
                        :, idx * R : (idx + 1) * R
                    ] = ab["B"]

    def state_hash(self) -> str:
        h = hashlib.sha256(self.base_hash.encode())
        for fid in sorted(self.deltas):
            h.update(fid.encode())
            for n in sorted(self.deltas[fid]):
                for t in (self.deltas[fid][n]["A"], self.deltas[fid][n]["B"]):
                    h.update(t.float().contiguous().numpy().tobytes())
        return h.hexdigest()

    # -- transactions -----------------------------------------------------
    def _pair_step(self, model, opt, user: str, assistant: str) -> None:
        pair = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        prompt = self.tok.apply_chat_template(
            pair[:-1], add_generation_prompt=True, return_tensors="pt"
        )
        full = self.tok.apply_chat_template(pair, return_tensors="pt")
        pid = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
        fid = full["input_ids"] if hasattr(full, "keys") else full
        fid = fid.to(model.device)
        labels = fid.clone()
        labels[:, : pid.shape[1]] = -100
        out = model(input_ids=fid, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()

    def _train_delta(self, pairs) -> dict:
        base = load_model(self.model_name)[1]
        m = make_lora(base, r=R)
        opt = torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr=self.lr
        )
        m.train()
        for _ in range(self.steps):
            for user, assistant in pairs:
                self._pair_step(m, opt, user, assistant)
        delta: dict = {}
        for n, p in dict(m.named_parameters()).items():
            if "lora_A" in n:
                delta[n] = {"A": p.detach().float().cpu().clone()}
        for n, p in dict(m.named_parameters()).items():
            if "lora_B" in n:
                delta[n.replace("lora_B", "lora_A")]["B"] = (
                    p.detach().float().cpu().clone()
                )
        del m, base
        torch.cuda.empty_cache()
        return delta

    def add(self, fid: str, pairs, key: str | None = None) -> float:
        t = time.time()
        self.deltas[fid] = self._train_delta(pairs)
        self.keys[fid] = key or pairs[0][0]
        self.materialize()
        return time.time() - t

    def revoke(self, fid: str) -> float:
        t = time.time()
        del self.deltas[fid]
        self.keys.pop(fid, None)
        self.materialize()
        return time.time() - t

    def answer_routed(self, user: str, system: str = "Kısa ve net cevap ver.") -> str:
        """Route to ONE expert (TF-IDF over the fact keys), never the merged sum."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        ids = sorted(self.deltas)
        if not ids:
            return self.answer(user, system)
        keys = [self.keys[i] for i in ids]
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
        x = vec.fit_transform(keys + [user])
        scores = cosine_similarity(x[-1], x[:-1]).ravel()
        chosen = ids[int(scores.argmax())]
        full = self.deltas
        self.deltas = {chosen: full[chosen]}
        self.materialize()
        resp = self.answer(user, system)
        self.deltas = full
        self.materialize()
        return resp

    # -- behaviour --------------------------------------------------------
    def answer(self, user: str, system: str = "Kısa ve net cevap ver.") -> str:
        return chat(
            self.tok,
            self.model,
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_new_tokens=24,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--facts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", default="results/live_learning/ledger_smoke.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, steps=args.steps)
    facts = FACTS[: args.facts]
    hashes = {}
    t_add = {}
    for f in facts:
        t_add[f.id] = store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )
        hashes[f.id] = store.state_hash()
        print(f"added {f.id} in {t_add[f.id]:.1f}s hash {hashes[f.id][:10]}", flush=True)

    def probe(kind: str = "merged"):
        fn = store.answer if kind == "merged" else store.answer_routed
        return {
            f.id: {"q": f.probe, "resp": fn(f.probe), "expect": f.answer}
            for f in facts
        }

    with_all = probe()
    routed_all = probe("routed")
    # revoke the LAST addition: the hash must equal the recorded pre-add hash exactly
    last = facts[-1].id
    prev_hash = hashes[facts[-2].id] if len(facts) > 1 else store.base_hash
    t_rev = store.revoke(last)
    revoke_hash = store.state_hash()
    after = probe()
    after_routed = probe("routed")

    def ok(resp, expect):
        return expect.lower() in resp.lower()

    result = {
        "model": args.model,
        "facts": [f.id for f in facts],
        "hashes": hashes,
        "revoke_last": {
            "id": last,
            "hash_before_add": prev_hash,
            "hash_after_revoke": revoke_hash,
            "identity": prev_hash == revoke_hash,
            "seconds": round(t_rev, 3),
        },
        "recall_merged_all": {fid: ok(r["resp"], r["expect"]) for fid, r in with_all.items()},
        "recall_routed_all": {fid: ok(r["resp"], r["expect"]) for fid, r in routed_all.items()},
        "recall_routed_after_revoke": {fid: ok(r["resp"], r["expect"]) for fid, r in after_routed.items()},
        "recall_merged_after_revoke": {fid: ok(r["resp"], r["expect"]) for fid, r in after.items()},
        "responses": {"merged_all": with_all, "routed_all": routed_all, "after_merged": after, "after_routed": after_routed},
        "seconds_add_mean": round(sum(t_add.values()) / len(t_add), 3),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    rl = result["recall_merged_all"]
    rr = result["recall_routed_all"]
    ra = result["recall_routed_after_revoke"]
    print(
        f"identity(revoke last)={result['revoke_last']['identity']} | "
        f"recall merged={rl} | routed={rr} | routed after revoke={ra} | "
        f"add {result['seconds_add_mean']}s revoke {t_rev:.2f}s"
    )


if __name__ == "__main__":
    main()
