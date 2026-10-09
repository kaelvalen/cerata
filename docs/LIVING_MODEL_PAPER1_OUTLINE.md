# Paper 1 - outline (draft v0, 2026-10-09)

Working title: **"Cerata: Transactional Memory for Frozen Language Models -
Exact Revoke, Audit, and Capability Safety at Scale"**
Alternatives: "Write, Forget, Audit: ..." / "Learning after deployment as an API
call on a frozen base".

Thesis in one sentence: personalizing a frozen base should be a *transaction*
(write gated, attributable, revocable), not an edit - we build the store and
measure every guarantee at N=1000.

## 1. Introduction

- The gap: post-deployment learning is either editing (write-only, no revoke, no
  audit, capability risk) or RAG (no durable parametric memory, context-bound).
- Thesis: transactions on a frozen base. write = KL-gated commit; forget = exact
  revoke; audit = provenance; the base stays untouched.
- Contributions: (1) architecture (slot store + entity-gated router + transaction
  API); (2) measured guarantees (revoke/audit/KLD/no-leak); (3) scale evidence
  (N=1000 x 2 sets, capacity 256 facts/delta); (4) honest baseline analysis
  (strong RAG, official GRACE/WISE, format-aware).
- Evidence discipline: everything pre-registered and pinned before runs; failures
  reported, not patched silently.

## 2. Related work and positioning

- Model editing: ROME / MEMIT / GRACE / WISE / AlphaEdit - optimize the write;
  no forget, no audit, sequential decay.
- Machine unlearning - approximate or coarse; ours is exact at the store level.
- RAG / context personalization - no parametric memory; gate result (0.837 vs 0.999).
- Adapters / MoE serving - capacity and storage framing.
- Positioning table: revoke? audit? capability cap? sequential N? per-fact cost?
  We do not claim "better editing"; we claim measured transactions.

## 3. Architecture

- Frozen base; slot store (CAP slots, per-fact LoRA delta); entity-gated router
  (v3 fuzzy match + semantic keys, tau, abstain); serving (route -> serve expert;
  else base).
- Transaction API: write (propose -> KL gate -> commit; atomic on failure),
  forget (slot free + materialize; exact), audit (state hash, commit records,
  leave-one-out provenance).
- Implementation: cerata/ (API layer) + experiments/sandbox/live_learning (harness);
  details to appendix.

## 4. Guarantee definitions (operational, falsifiable)

- Serving: efficacy, paraphrase, no-leak (base-referenced), abstention, route.
- Revoke: gone / return_match / retain; confounds named (token collisions;
  base-confounded gone on zsRE - return_match is the reading there).
- Audit: leave-one-out attribution (answer present with the expert, absent
  without it) + state hash stability.
- Capability: canary KLD (mean/max), commit cap 2.0.
- Atomicity: failure atomicity test; reproducibility: deterministic given the
  RNG sequence (SS60 addendum).

## 5. Experimental setup

- Base: Qwen2.5-1.5B-Instruct (frozen). Sets: nonce (synthetic, seed=n),
  CounterFact, zsRE. N=1000. Chat serving, greedy + contains, bootstrap CIs.
- Baselines: strong RAG (instruct context, same N); official GRACE/WISE via
  EasyEdit (hparams adapted to 1.5B, fp16; both raw and chat protocols reported).
- Pins: prereg SS33-63; frozen summary doc; all raw logs/JSONs in results/.

## 6. Results (frozen table)

- T1 guarantees (cf / zsRE): eff 0.999 / 0.996; no-leak 0.992 / 0.975;
  abstention 0.992 / 0.967; revoke gone 0.96 / 0.36*; return 0.96 / 0.80;
  retain 0.9989 / 0.9944; KLD 0.0215 / 0.0171 (max 0.0498 / 0.0529);
  provenance 1.00 (cf); storage 8.7 MB/fact; add 8.45 s; serve 0.04 s;
  revoke 1.4 ms.
- Gates: strong RAG instruct 0.837 vs ours 0.999.
- Capacity: one delta carries 256 facts (eff 1.00, KLD 0.007, genuine leakage
  0.1-0.3 after the slice fix).
- Router stress: typos/lower/space 1.00, partial 0.80, pronoun abstains.
- Ablations: LoRA+ per-fact adopted; grouped keeps baseline; OLoRA/DoRA/PiSSA/MLP
  rejected (recorded).
- Honest failures section: LoRA+ grouped instability (SS48), the capacity slice
  bug and its fix (SS60/61), RNG-position sensitivity.

## 7. Baselines deep-dive (format matters)

- GRACE: raw-prompt 1.00 (200/200), chat 0.025; EasyEdit token-EM space artifact.
- WISE: per-step 1.00, final 0.07; merge lifecycle 0.125 with no-leak 0.27.
- RAG: instruct 0.837.
- Framing: editing objectives optimize the write; none offer forget/audit; the
  transaction gap is the contribution.

## 8. Limitations (declared up front)

- Router-bound CounterFact paraphrase 0.323 (direct delta 0.86-0.90) - pinned fix
  is paper 2.
- zsRE sibling pick 0.784; return_match 0.80.
- Single model and environment; no external replication yet.
- Grouped deltas trade exactness for storage (leakage 0.1-0.3); exactness is
  per-fact.
- VLM/controller evidence is small-scale (preview only).
- Baseline comparisons are serving-format sensitive; both formats reported.

## 9. Conclusion and future work (paper 2 preview)

- Learned router; format-matched baselines; grouped experts + coarse router;
  VLM and controller at confirmatory scale.

## Figures and tables plan

- F1 architecture; T1 guarantees; T2 baselines (with format labels); F2 capacity
  curve; F3 router stress; T3 delta-recipe ablations; appendix: pins, CIs,
  harness semantics, failure log.

## Writing rules

- Every number cites a prereg SS or the frozen summary; no number without a run;
  failures stay in the paper.
