"""VLM delta store: transactional per-fact deltas over a frozen 2B VLM (pinned).

Mirror of the text ledger: CLIP-keyed per-fact LoRA deltas, state_hash over the active
tensors, propose_and_commit with a canary KLD cap, exact revoke, save/load.

    python vlm_ledger.py
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
from vlm_core import (  # noqa: E402
    CANARIES,
    MODEL,
    PANEL_A,
    PANEL_B2,
    PROBE,
    ask,
    cur_logits,
    emb_clip,
    load,
    load_clip,
    set_lora,
    teach_text,
    train_delta_kl,
    train_text_candidate,
)

FACTS = {
    "a": {"panel": PANEL_A, "code": "Tira"},
    "b": {"panel": PANEL_B2, "code": "Vok"},
}


class VlmDeltaStore:
    def __init__(
        self,
        model_name: str = MODEL,
        steps: int = 12,
        lr: float = 3e-4,
        kl_lambda: float = 1.0,
        kld_limit: float = 2.0,
        tau: float = 0.97,
    ):
        self.model_name, self.steps, self.lr = model_name, steps, lr
        self.kl_lambda, self.kld_limit, self.tau = kl_lambda, kld_limit, tau
        self.proc, self.model = load(model_name)
        from peft import LoraConfig, get_peft_model

        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            ),
        )
        self.model.eval()
        self.iproc, self.imodel = load_clip()
        self.deltas: dict[str, dict] = {}
        self.keys: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            self.base_canary = [cur_logits(self.proc, self.model, q) for q in CANARIES]

    # -- state ------------------------------------------------------------
    def state_hash(self) -> str:
        h = hashlib.sha256(f"{self.model_name}|vlm|r16".encode())
        for fid in sorted(self.deltas):
            h.update(fid.encode())
            for n in sorted(self.deltas[fid]):
                t = self.deltas[fid][n].float().contiguous()
                h.update(t.numpy().tobytes())
        return h.hexdigest()

    def _lora(self) -> dict:
        return {
            n: p.detach().float().cpu().clone()
            for n, p in self.model.named_parameters()
            if "lora_" in n
        }

    def expert_kld(self, delta) -> float:
        set_lora(self.model, delta)
        with torch.no_grad():
            cur = [cur_logits(self.proc, self.model, q) for q in CANARIES]
        set_lora(self.model, None)
        ks = []
        for b, a in zip(self.base_canary, cur):
            pb, pa = torch.softmax(b, -1), torch.softmax(a, -1)
            ks.append(float((pb * (pb / pa).log()).sum() + (pa * (pa / pb).log()).sum()))
        return sum(ks) / len(ks)

    # -- transactions -----------------------------------------------------
    def propose_and_commit(
        self, fid, panel, pairs, kld_limit=None, steps=None, lam=None, trainer=None
    ) -> dict:
        t = time.time()
        if trainer is None:
            train_delta_kl(
                self.proc,
                self.model,
                panel,
                pairs,
                steps or self.steps,
                self.lr,
                lam=self.kl_lambda if lam is None else lam,
            )
        else:
            trainer()
        delta = self._lora()
        kld = self.expert_kld(delta)
        limit = self.kld_limit if kld_limit is None else kld_limit
        if kld > limit:
            return {"committed": False, "kld": round(kld, 3), "seconds": round(time.time() - t, 1)}
        self.deltas[fid] = delta
        self.keys[fid] = emb_clip(self.iproc, self.imodel, panel)
        return {"committed": True, "kld": round(kld, 3), "seconds": round(time.time() - t, 1)}

    def revoke(self, fid: str) -> None:
        self.deltas.pop(fid, None)
        self.keys.pop(fid, None)

    def serve(self, panel, question: str = PROBE):
        if not self.keys:
            set_lora(self.model, None)
            return ask(self.proc, self.model, panel, question), None, {}
        qv = emb_clip(self.iproc, self.imodel, panel)
        sims = {k: float(qv @ v) for k, v in self.keys.items()}
        best = max(sims, key=sims.get)
        chosen = best if sims[best] >= self.tau else None
        set_lora(self.model, self.deltas[chosen] if chosen else None)
        resp = ask(self.proc, self.model, panel, question)
        return resp, chosen, {k: round(v, 3) for k, v in sims.items()}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"deltas": self.deltas, "keys": self.keys}, path)

    def load_state(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.deltas = state["deltas"]
        self.keys = state["keys"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--state", default="results/live_learning/vlm_store_state.pt")
    ap.add_argument("--out", default="results/live_learning/vlm_ledger_battery_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    store = VlmDeltaStore(args.model)

    def contains(resp, token):
        return token.lower() in resp.lower()

    h0 = store.state_hash()
    adds = {}
    for fid, f in FACTS.items():
        teach = teach_text(f["code"])
        pairs = [(teach, f"Not aldım: {teach}"), (PROBE, f["code"])]
        adds[fid] = store.propose_and_commit(fid, f["panel"], pairs)
        adds[fid]["hash"] = store.state_hash()
    h_a, h_ab = adds["a"]["hash"], store.state_hash()
    deterministic = store.state_hash() == h_ab

    store.save(args.state)
    served = {fid: store.serve(FACTS[fid]["panel"]) for fid in FACTS}

    bad = store.propose_and_commit(
        "bad",
        PANEL_A,
        [(CANARIES[1], "Tira")],
        steps=20,
        lam=0.0,
        trainer=lambda: train_text_candidate(
            store.proc, store.model, [(CANARIES[1], "Tira")], 20, store.lr
        ),
    )
    bad["hash_after"] = store.state_hash()

    store.revoke("b")
    hash_after_b = store.state_hash()
    store.revoke("a")
    hash_after_a = store.state_hash()

    store.load_state(args.state)
    hash_reloaded = store.state_hash()
    served2 = {fid: store.serve(FACTS[fid]["panel"]) for fid in FACTS}

    checks = {
        "adds_committed": all(adds[f]["committed"] for f in FACTS),
        "hash_deterministic": deterministic,
        "g1_refused": (not bad["committed"]) and bad["hash_after"] == h_ab,
        "g2_identity_b": hash_after_b == h_a,
        "g2_identity_a": hash_after_a == h0,
        "served_ok": all(contains(served[f][0], FACTS[f]["code"]) for f in FACTS),
        "g4_reload_hash": hash_reloaded == h_ab,
        "session2_served_ok": all(contains(served2[f][0], FACTS[f]["code"]) for f in FACTS),
    }
    result = {
        "model": args.model,
        "checks": checks,
        "adds": adds,
        "bad_candidate": bad,
        "hashes": {"h0": h0, "h_a": h_a, "h_ab": h_ab, "reloaded": hash_reloaded},
        "served": {k: {"resp": v[0], "chosen": v[1], "sims": v[2]} for k, v in served.items()},
        "served2": {k: {"resp": v[0], "chosen": v[1], "sims": v[2]} for k, v in served2.items()},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("adds:", {k: (v["committed"], v["kld"]) for k, v in adds.items()})
    print("bad:", bad["committed"], bad["kld"])
    print("served:", {k: v[0][:40] for k, v in served.items()})
    print("served2:", {k: v[0][:40] for k, v in served2.items()})


if __name__ == "__main__":
    main()
