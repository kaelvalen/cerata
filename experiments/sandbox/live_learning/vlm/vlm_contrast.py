"""VLM contrast-pair expert: visual grounding (pinned).

Train the P1 expert against negative scenes (same probe text -> "Bilmiyorum."), then
measure: direct expert probes on P1/P2/P3 and the P1 variants, routed serves, KLD cap,
canary.

    python vlm_contrast.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_core import (  # noqa: E402
    CANARIES,
    MODEL,
    P1,
    P1BG,
    P1J,
    P1S,
    P2,
    P3,
    PROBE,
    ask,
    set_lora,
    teach_text,
    train_delta_contrast,
)
from vlm_ledger import VlmDeltaStore  # noqa: E402

REFUSE = "Bilmiyorum."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="results/live_learning/vlm_contrast_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    store = VlmDeltaStore(args.model)

    def contains(resp, token):
        return token.lower() in resp.lower()

    teach = teach_text("Tira")
    triples = [
        (P1, teach, f"Not aldım: {teach}"),
        (P1, PROBE, "Tira"),
        (P2, PROBE, REFUSE),
        (P3, PROBE, REFUSE),
    ]
    res = store.propose_and_commit(
        "p1",
        P1,
        [],
        trainer=lambda: train_delta_contrast(
            store.proc, store.model, triples, args.steps, store.lr
        ),
    )

    probes = {"p1": P1, "p2": P2, "p3": P3, "p1j": P1J, "p1s": P1S, "p1bg": P1BG}
    direct = {}
    if res["committed"]:
        set_lora(store.model, store.deltas["p1"])
        direct = {name: ask(store.proc, store.model, img, PROBE) for name, img in probes.items()}
        set_lora(store.model, None)
    canary = ask(store.proc, store.model, None, CANARIES[1])
    routed_p1 = store.serve(P1)
    routed_p2 = store.serve(P2)

    checks = {
        "committed": res["committed"],
        "kld_under_cap": res["kld"] <= store.kld_limit,
        "p1_direct_ok": res["committed"] and contains(direct["p1"], "Tira"),
        "p2_no_leak": res["committed"] and not contains(direct["p2"], "Tira"),
        "p3_no_leak": res["committed"] and not contains(direct["p3"], "Tira"),
        "variants_ok": res["committed"]
        and all(contains(direct[v], "Tira") for v in ("p1j", "p1s", "p1bg")),
        "routed_p1": routed_p1[1] == "p1" and contains(routed_p1[0], "Tira"),
        "routed_p2_abstains": routed_p2[1] is None and not contains(routed_p2[0], "Tira"),
        "canary_clean": not contains(canary, "Tira"),
    }
    result = {
        "model": args.model,
        "commit": res,
        "checks": checks,
        "direct": direct,
        "routed": {
            "p1": {"resp": routed_p1[0], "chosen": routed_p1[1], "sims": routed_p1[2]},
            "p2": {"resp": routed_p2[0], "chosen": routed_p2[1], "sims": routed_p2[2]},
        },
        "canary": canary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("commit:", res)
    print("direct:", {k: v[:50] for k, v in direct.items()})
    print("routed p1:", routed_p1[0][:40], "| p2:", routed_p2[0][:60], routed_p2[1])


if __name__ == "__main__":
    main()
