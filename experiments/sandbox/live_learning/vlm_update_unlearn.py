"""VLM update/unlearn battery (pinned): replace transaction and silent revoke.

add A -> update A (revoke + retrain "Bora") -> add B' -> unlearn B' (revoke). The hash
after unlearn must equal the hash recorded after the update, bitwise.

    python vlm_update_unlearn.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from vlm_ledger import VlmDeltaStore  # noqa: E402
from vlm_mirror import MODEL, PANEL_A, ask, set_lora  # noqa: E402
from vlm_mirror2 import CANARIES, PROBE, teach_text  # noqa: E402
from vlm_mirror4 import PANEL_B2  # noqa: E402

NEW_CODE = "Bora"


def pairs_for(teach: str, code: str):
    return [(teach, f"Not aldım: {teach}"), (PROBE, code)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="results/live_learning/vlm_update_unlearn_2b.json")
    args = ap.parse_args()
    torch.manual_seed(0)
    store = VlmDeltaStore(args.model)

    def contains(resp, token):
        return token.lower() in resp.lower()

    h0 = store.state_hash()
    add_a = store.propose_and_commit(
        "a", PANEL_A, pairs_for(teach_text("Tira"), "Tira")
    )
    h1 = store.state_hash()

    store.revoke("a")
    new_teach = f"Kalibrasyon düğmesinin kodu {NEW_CODE}'dır."
    upd = store.propose_and_commit("a", PANEL_A, pairs_for(new_teach, NEW_CODE))
    h2 = store.state_hash()

    add_b = store.propose_and_commit(
        "b", PANEL_B2, pairs_for(teach_text("Vok"), "Vok")
    )
    h3 = store.state_hash()

    ra, ca, _ = store.serve(PANEL_A)
    rb, cb, _ = store.serve(PANEL_B2)

    store.revoke("b")
    h4 = store.state_hash()
    rb2, cb2, _ = store.serve(PANEL_B2)
    ra2, ca2, _ = store.serve(PANEL_A)
    set_lora(store.model, None)
    canary = ask(store.proc, store.model, None, CANARIES[1])

    checks = {
        "add_a_committed": add_a["committed"],
        "update_committed": upd["committed"],
        "update_serves_new": contains(ra, NEW_CODE) and not contains(ra, "Tira"),
        "isolation_b": contains(rb, "Vok"),
        "unlearn_identity": h4 == h2,
        "unlearn_silent": not contains(rb2, "Vok"),
        "a_after_unlearn": contains(ra2, NEW_CODE),
        "canary_clean": not contains(canary, "Tira") and not contains(canary, NEW_CODE),
        "commits_under_cap": add_a["kld"] <= store.kld_limit
        and upd["kld"] <= store.kld_limit
        and add_b["kld"] <= store.kld_limit,
    }
    result = {
        "model": args.model,
        "checks": checks,
        "adds": {"a": add_a, "b": add_b},
        "update": upd,
        "hashes": {"h0": h0, "h1": h1, "h2": h2, "h3": h3, "h4": h4},
        "probes": {
            "a_after_update": {"resp": ra, "chosen": ca},
            "b": {"resp": rb, "chosen": cb},
            "b_after_unlearn": {"resp": rb2, "chosen": cb2},
            "a_after_unlearn": {"resp": ra2, "chosen": ca2},
        },
        "canary": canary,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False))
    print("adds:", {"a": add_a, "b": add_b})
    print("update:", upd)
    print("hashes:", {"h2": h2[:12], "h4": h4[:12]})
    print("probes:", {k: v["resp"][:50] for k, v in result["probes"].items()})
    print("canary:", canary[:60])


if __name__ == "__main__":
    main()
