# live-learning sandbox: state checkpoint (2026-10-02, before the controller)

Handoff after context compaction. Everything below is committed on `main`; nothing is
pushed. The program: `docs/LIVING_MODEL_POSITIONING.md` (claim, audit Appendix A,
measured guarantees Appendix B). Rule of the house: every run is pinned in
`docs/LIVING_MODEL_SLICE_PREREG.md` before it runs; failures are recorded, not
patched silently; results JSON lives in `results/live_learning/` (untracked).

## Where the slice stands (all measured, all committed)

- Ledger core (`ledger.py`): per-fact LoRA deltas from a frozen base, summed in id
  order; `state_hash` = pure function of the delta set; G2 bitwise identity on
  revoke; revoke < 0.1 s, add ~3-5 s/fact.
- Composition result: deltas summed 0/18; the same deltas routed 18/18.
- Router: TF-IDF inverted, LLM last-token inverted, **MiniLM multilingual separated**
  (own 0.742 vs other 0.569, tau 0.656; paraphrase routing 6/6; abstention 4/4).
- KL-anchored deltas: per-expert KLD 10.96 -> 0.93 at full recall.
- Organ invariants: revoke/add move 0 remaining routing decisions (SEUF/GRIP failure
  mode measured as absent).
- Controller v0 (rule): recurrence >= 2 -> promote; 6 experts vs 18, recall 6/6.
- G1 failure atomicity: training exception and bad input leave no trace.
- Stream-v2 c6 (discriminative "7310"): provenance 1.00.
- Paraphrase augmentation: held-out set B routing 6/6, recall 6/6.

## Next task: learned controller (pinned design, implementation pending)

Pinned in the prereg: contextual bandit over {promote, defer}; features = recurrence
count, router margin, candidate KLD (measured without committing), budget used;
reward = later-probe recall, cost = pinned expert-budget price. Missing decision to
pin first: the probe-placement protocol (proposal: every fact is probed once after
the stream ends, plus once immediately after its own decision; credit to the last
decision only - crude, state it). E-path for deferrals: answer via MiniLM retrieval
of the fact text injected into the base prompt (the mesa memory mechanism), so a
deferral has a measurable value. Comparison: bandit vs rule v0 on the same stream
(6 facts x3 queries, 12 x1): expert count at equal recall, refusals, budget curve.

## Files

- `stream.py` facts/fillers; `harness.py` model+chat+LoRA; `ledger.py` DeltaStore
  (add/revoke/propose_and_commit/answer_routed/embed); `ledger_metrics.py` G3/G5;
  `ledger_router.py` separation; `ledger_embedder.py` MiniLM router; `ledger_organ.py`
  invariants; `ledger_controller.py` v0; `ledger_para_holdout.py`; `ledger_atomic_fail.py`;
  `ledger_c6_v2.py`; `mechanisms.py`+`run.py` the three-mechanism mesa.

## Environment (NixOS quirks, essential)

```bash
export LD_LIBRARY_PATH=/nix/store/38v10xhwhypb747h3z4c2i0a19hkiwx2-nvidia-x11-615.71.09/lib
export TRITON_LIBCUDA_PATH=$LD_LIBRARY_PATH     # triton wants /sbin/ldconfig
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python experiments/sandbox/live_learning/<script>.py
```

GPU is an 8 GB RTX 5060 shared with the Counterpart project (check `nvidia-smi`
before 1.5B runs; 0.5B fits always). Models cached: Qwen2.5-0.5B/1.5B-Instruct;
`sentence-transformers` installed (MiniLM multilingual cached).
