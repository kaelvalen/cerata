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

## Controller v1 (done): serve-first repair bandit

Measured on the same stream (6 facts x3 queries, 12 x1), code in `ledger_bandit.py`,
all pins and readings in the prereg: rule 6 experts / 16-18; eps bandit 0 / 15-18;
UCB decide-before-serve 17 / 17-18; **repair (serve first, UCB on observed misses)
3 experts / 18-18** - it promotes exactly the deterministic memory failures (p3, w3,
c2). The decision point (after the serve) mattered more than the exploration rule.

## Cross-session hybrid (done): `ledger_session.py`

The event stream (update, unlearn, two sessions) on the hybrid: v2 experts {w3, c2}
(tau 0.65, current-note training, current-token evaluation), s2 17/17, update and
unlearn honoured, capability 3/4 (cap4 was a tau false-positive fixed by 0.65; cap3 is
base-bound). The update replaced p3's need for an expert (note rewrite); the retrain
path for a promoted-then-updated fact is still unexercised (same commit transaction).

## VLM mirror (feasibility + v3, done): `vlm_mirror*.py`, `vlm_keys.py`

Qwen3-VL-2B-Instruct runs on the shared 8 GB. v0: the base sees, does not know a
nonce; a 12-step LoRA delta teaches it; zeroing revokes exactly; unanchored deltas
damaged a canary at that seed. Harness bug found and fixed: zeroing both LoRA factors
before training is a dead gradient point (`reset_lora`: B zero, A kaiming); the restore
diff is then exactly 0 for every lambda. v2: two image-keyed facts, served 2/2, revoke
exact, but abstention failed (DINOv2 keys 0.985 similar). Key comparison: CLIP beats
DINOv2 CLS/mean and concat on margin; v3 with CLIP keys and tau 0.97 passes all seven
checks - route 2/2, served 2/2 ("Tira"/"Vok"), canary clean, distractor refusal (d1
0.964, d2 0.522, d3 0.806), revoke abstains, A unaffected. Flagged: exact-panel
regime; 0.006 margin on the colour-swap distractor.

## VLM store port (done): `vlm_ledger.py`

The transaction core runs on Qwen3-VL-2B: CLIP-keyed per-fact deltas, `state_hash` over
the active tensors, KLD-capped `propose_and_commit` (adds 0.098 / 0.191; the
adversarial text-only candidate at 23.928 is refused with the hash unchanged), bitwise
revoke identity (pre-B and empty), save/load durability (hash equal) and a reloaded
session serving both facts 2/2. All eight checks pass.

## VLM repair controller (done): `vlm_controller.py`

The v1.2 policy on the VLM: serve -> miss -> UCB repair, KLD-capped commits. Stream A
x3, B' x3, C/D x1: all seven checks pass - base misses the first probes, experts 4
(KLDs 0.098/0.191/0.167/0.020), final readout 4/4 through the CLIP router, route
choices correct (max off-key sim 0.938 < tau 0.97), canary clean, hash changed. The
first miss defers twice (A repairs at q3); later facts promote greedily.

## VLM update/unlearn (done): `vlm_update_unlearn.py`

All nine checks pass: the update (revoke + retrain "Bora") commits (KLD 0.065) and
serves the new code with "Tira" gone; B' stays isolated; unlearn is silent; the hash
after unlearn equals the post-update hash bitwise (h4 == h2); canary clean; all commits
under the cap (~6 s per training).

## Next

VLM side: keys on real photos or more distinctive panels (the panel regime has a
0.006-0.062 margin). Optional text side: future-aware repair reward; the update retrain
path with a promoted fact.

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
