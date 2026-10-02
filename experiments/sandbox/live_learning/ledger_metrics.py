"""G3/G5 metrics for the transactional delta store (docs/LIVING_MODEL_SLICE_PREREG.md).

- provenance: leave-one-out attribution per fact (routed answers with / without it)
- canary KLD: base (adapter zeroed) vs full set, on the capability questions
- retained effect: suspend the last-added fact, every other fact must still answer

    python ledger_metrics.py --facts 18
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from ledger import DeltaStore  # noqa: E402
from stream import CAPABILITY, FACTS  # noqa: E402

SYSTEM = "Kısa ve net cevap ver."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--facts", type=int, default=18)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--out", default="results/live_learning/ledger_g3g5_15b.json")
    args = ap.parse_args()
    store = DeltaStore(args.model, lr=args.lr, steps=args.steps)
    facts = FACTS[: args.facts]
    for f in facts:
        store.add(
            f.id,
            [(f.teach, f"Not aldım: {f.teach}"), (f.probe, f.answer)],
            key=f.probe,
        )

    def contains(resp: str, answer: str) -> bool:
        return answer.lower() in resp.lower()

    # -- G5 provenance: leave-one-out --------------------------------------
    prov = {}
    for f in facts:
        a1 = store.answer_routed(f.probe)
        saved = dict(store.deltas)
        del store.deltas[f.id]
        a2 = store.answer_routed(f.probe)
        store.deltas = saved
        store.materialize()
        with_ok = contains(a1, f.answer)
        without_ok = contains(a2, f.answer)
        prov[f.id] = {
            "with": with_ok,
            "without": without_ok,
            "attributed": bool(with_ok and not without_ok),
        }

    # -- G3 retained effect: suspend the last-added fact --------------------
    last = facts[-1]
    saved = dict(store.deltas)
    del store.deltas[last.id]
    retained = {
        f.id: contains(store.answer_routed(f.probe), f.answer) for f in facts[:-1]
    }
    store.deltas = saved
    store.materialize()

    # -- G3 canary KLD: base vs full set -----------------------------------
    canaries = [q for _, q, _ in CAPABILITY]

    @torch.no_grad()
    def logits(q: str) -> torch.Tensor:
        ids = store.tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).to(store.model.device)
        return store.model(input_ids=ids).logits[0, -1].float()

    store._zero()
    base = [logits(q) for q in canaries]
    store.materialize()
    active = [logits(q) for q in canaries]
    klds = []
    for b, a in zip(base, active):  # symmetric KL
        pb, pa = torch.softmax(b, -1), torch.softmax(a, -1)
        klds.append(float((pb * (pb / pa).log()).sum() + (pa * (pa / pb).log()).sum()))
    canary_kld = sum(klds) / len(klds)

    # -- per-expert canary KLD under routed inference ----------------------
    full = dict(store.deltas)
    per_expert = {}
    for f in facts:
        store.deltas = {f.id: full[f.id]}
        store.materialize()
        a = [logits(q) for q in canaries]
        ks = []
        for b, x in zip(base, a):
            pb, pa = torch.softmax(b, -1), torch.softmax(x, -1)
            ks.append(
                float((pb * (pb / pa).log()).sum() + (pa * (pa / pb).log()).sum())
            )
        per_expert[f.id] = sum(ks) / len(ks)
    store.deltas = full
    store.materialize()

    result = {
        "model": args.model,
        "facts": len(facts),
        "lr": args.lr,
        "steps": args.steps,
        "provenance": prov,
        "provenance_accuracy": sum(v["attributed"] for v in prov.values()) / len(prov),
        "retained_after_suspend_last": retained,
        "retained_rate": sum(retained.values()) / len(retained),
        "canary_kld": canary_kld,
        "per_expert_kld": per_expert,
        "per_expert_kld_mean": sum(per_expert.values()) / len(per_expert),
        "per_expert_kld_max": max(per_expert.values()),
        "canary_questions": canaries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(
        f"provenance {result['provenance_accuracy']:.2f} | "
        f"retained {sum(retained.values())}/{len(retained)} | "
        f"canary KLD {canary_kld:.4f} | per-expert KLD "
        f"mean {result['per_expert_kld_mean']:.4f} max {result['per_expert_kld_max']:.4f}"
    )


if __name__ == "__main__":
    main()
