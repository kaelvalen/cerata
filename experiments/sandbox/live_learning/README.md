# live learning sandbox

The program this bench measures: [`docs/LIVING_MODEL_POSITIONING.md`](../../../docs/LIVING_MODEL_POSITIONING.md)
(transactional, auditable learning during interaction). Current phase: the
confirmatory study - [`docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`](../../../docs/LIVING_MODEL_CONFIRMATORY_PREREG.md);
the review round and the pinned next experiments are in
[`docs/LIVING_MODEL_REVIEW_2026-10-04.md`](../../../docs/LIVING_MODEL_REVIEW_2026-10-04.md).
Handoff/state: [`STATE.md`](STATE.md).

## Layout

| path | what |
| :-- | :-- |
| `text/` | text-side record scripts: the mesa (`stream.py`, `harness.py`, `mechanisms.py`, `run.py`), the ledger (`ledger.py`), the controller (`ledger_bandit.py`), cross-session (`ledger_session.py`), the G-checks and router scripts (`ledger_*.py`) |
| `confirm/` | the confirmatory harness: `facts.py` (nonce generator), `external.py` (CounterFact/zsRE adapters), `pilot.py` (arms, metrics, chunked runs), `router_eval.py` (CPU router comparison) |
| `vlm/` | the vision mirror: `vlm_core.py` (shared helpers), `vlm_mirror*.py` (v0-v3 records), `vlm_ledger.py` (store), `vlm_controller*.py`, `vlm_update_unlearn.py`, `vlm_variants.py`, `vlm_contrast.py` |
| `STATE.md` | the handoff: measured status, environment, next steps |

Every record script was pinned in a doc before its run; failures are recorded, not
patched silently. Results JSONs live under `results/live_learning/` (untracked).

## Environment (NixOS quirks, essential)

```bash
export LD_LIBRARY_PATH=/nix/store/38v10xhwhypb747h3z4c2i0a19hkiwx2-nvidia-x11-615.71.09/lib
export TRITON_LIBCUDA_PATH=$LD_LIBRARY_PATH   # NixOS: triton wants /sbin/ldconfig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python experiments/sandbox/live_learning/<dir>/<script>.py
```

GPU: 8 GB RTX 5060, shared (check `nvidia-smi` before 1.5B runs; never kill other
processes). Models cached: Qwen2.5-0.5B/1.5B-Instruct, Qwen3-VL-2B-Instruct,
DINOv2-base, CLIP ViT-B/32 (laion), MiniLM multilingual.

## History

The first sweep (the context/memory/lora mesa, 2026-10-02) and everything after it -
the transaction ledger, the repair controller, the cross-session hybrid, the VLM
mirror and the confirmatory study - are recorded chronologically in
`docs/LIVING_MODEL_SLICE_PREREG.md` and `docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`.
