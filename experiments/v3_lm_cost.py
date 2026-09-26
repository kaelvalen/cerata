"""
V3-LLM-1 feasibility: measure the MEDIUM path's per-write cost instead of estimating it.

A guarded MEDIUM write does, besides its canary passes: one solve for the new delta,
one re-sum + solve in a seeded permutation (`order_report`; the facade's canary check
reuses it), and two solves for the trial undo / redo. This runner times exactly that
sequence on `DownProjEdit` with a synthetic corpus prior, for both modes, at the given
`d_ff` values and numbers of live edits, on the given device, and fits
`t_solve = a * d_ff^3` to project to the target model's `d_ff` (Qwen2.5-7B: 18944).

Canary passes need the real model; pass their measured time with `--canary_seconds` to
get the projected wall time of the pre-registered grid.

    python experiments/v3_lm_cost.py --device cuda --d_ff 2048,4096,8192 --edits 1,100,1000
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pal_moe.core.hashing import digest  # noqa: E402
from pal_moe.edit.down_proj import DownProjEdit, KeyPrior  # noqa: E402

PREREG_WRITES = {"N=1 (2000 isolated)": 2000, "N=100": 100, "N=1000": 1000}


def synthetic_prior(d_ff, device) -> KeyPrior:
    g = torch.Generator(device="cpu").manual_seed(0)
    A = torch.randn(d_ff, d_ff, generator=g, dtype=torch.float64).to(device) / d_ff**0.5
    C = A @ A.t() + 1e-2 * torch.eye(d_ff, dtype=torch.float64, device=device)
    return KeyPrior(C, 10 * d_ff, "synthetic", digest("synthetic", d_ff))


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, device, reps=1):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync(device)
    return (time.perf_counter() - t0) / reps, out


def solve_seconds(d_ff, device) -> dict:
    prior = synthetic_prior(d_ff, device)
    B = torch.randn(d_ff, 64, dtype=torch.float64, device=device)
    lu, _ = timed(lambda: torch.linalg.solve(prior.cov, B), device, reps=2)
    inv, _ = timed(lambda: torch.linalg.inv(prior.cov), device)
    return {"d_ff": d_ff, "solve_s": lu, "inverse_s": inv}


def guarded_write_seconds(d_ff, n_edits, mode, device, d_model=64) -> float:
    """Time of the solve work in one guarded write when `n_edits` are already live."""
    prior = synthetic_prior(d_ff, device)
    e = DownProjEdit(prior, mode=mode)
    g = torch.Generator(device="cpu").manual_seed(1)

    def contrib(i):
        K = torch.randn(1, d_ff, generator=g, dtype=torch.float64).to(device)
        R = torch.randn(1, d_model, generator=g, dtype=torch.float64).to(device)
        return e.contribution(f"e{i}", K, R)

    for i in range(n_edits - 1):
        e.add(contrib(i))
    new = contrib(n_edits)

    def write():
        e.add(new)
        e.solve()  # the new delta
        e.order_report()  # permuted re-sum + solve (reused by the canary check)
        c = e.remove(new.edit_id)  # trial undo
        e.solve()
        e.add(c)  # trial redo
        e.solve()
        e.remove(new.edit_id)

    t, _ = timed(write, device)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d_ff", default="1024,2048,4096")
    ap.add_argument("--edits", default="1,100")
    ap.add_argument("--target_d_ff", type=int, default=18944)
    ap.add_argument("--canary_seconds", type=float, default=None)
    ap.add_argument("--out", default="results/v3/lm_cost.json")
    args = ap.parse_args()
    dims = [int(x) for x in args.d_ff.split(",")]
    edits = [int(x) for x in args.edits.split(",")]

    solves = [solve_seconds(d, args.device) for d in dims]
    # Fit t = a * d^3 on the largest measured size (least affected by overheads).
    big = solves[-1]
    a = big["solve_s"] / big["d_ff"] ** 3
    target_solve = a * args.target_d_ff**3
    writes = []
    for d in dims:
        for n in edits:
            for mode in ("accumulate", "woodbury"):
                writes.append(
                    {
                        "d_ff": d,
                        "live_edits": n,
                        "mode": mode,
                        "seconds": guarded_write_seconds(d, n, mode, args.device),
                    }
                )
    proj = {
        "target_d_ff": args.target_d_ff,
        "dense_solve_s": target_solve,
        "woodbury_one_time_inverse_s": big["inverse_s"]
        / big["d_ff"] ** 3
        * args.target_d_ff**3,
        "accumulate_write_solve_s": 4 * target_solve,
        "note": "accumulate: 4 dense solves per guarded write (+ a re-sum growing with N);"
        " canary passes are not included unless --canary_seconds is given",
    }
    if args.canary_seconds is not None:
        per_write = 4 * target_solve + 6 * args.canary_seconds
        proj["per_write_s"] = per_write
        proj["grid_hours_accumulate"] = {
            k: n * per_write / 3600 for k, n in PREREG_WRITES.items()
        }
    result = {
        "device": args.device,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "solves": solves,
        "guarded_write_solve_work": writes,
        "projection": proj,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
