# Confirmatory study: review round (2026-10-04) and pinned next experiments

Status: review received after `77c5f70`; **no runs started for it** (per instruction).
This document records the findings as accepted, the corrections they force, and the
pinned order of experiments. The gate: if experiment 1 closes the gap, stop and narrow
the thesis.

## Findings (accepted)

1. **The headline delta > RAG gap may come from a weak RAG.** The RAG arm uses only a
   plain "Answer briefly." system prompt + note injection. On counterfactual facts the
   conflict literature (Xie et al., ICLR 2024, "Adaptive Chameleon or Stubborn Sloth")
   shows instruction-tuned models largely follow explicit external evidence. Strong RAG
   arms are required before the headline stands.
2. **The router is a dictionary, not a tested semantic router.** Router v2 matches the
   subject string as a substring of the question and abstains otherwise; the adapters
   also require the subject in the paraphrase. Route 0.997 / abstention 1.00 are
   near-tautological in this setup. Untested: aliases ("Obama" / "Barack Obama"), typos,
   pronoun follow-ups. And CounterFact paraphrase route 0.36 (0.42 at N=50, 0.355 at
   N=200) means the ~0.32 paraphrase score is **router failure, not dump noise** - the
   delta's generalisation is not measured there. This is Stage-1's F18/F19 warning
   concretely.
3. **The baselines are strawmen.** grace_like is a +10 logit bias, wise_like is a
   base-or-side-memory rule, melo_like is our system without the KLD cap. Official
   implementations are required; EasyEdit ships GRACE, WISE, MELO, MEMIT and AlphaEdit
   and a 1.5B model fits in 8 GB.
4. **The prereg's own metrics are not reported.** §5 pins: canary KL <= 0.05 after
   revoke, retain within 1 point, provenance >= 0.95, router precision >= 0.95. The
   N=1000 table reports gone/retain on 10 facts only (near-vacuous CIs). Revoke n must
   be >= 100 and the pinned metrics reported as pinned. N=200 used the old metrics:
   either rerun N=200 with the fixed metrics or build the final table on N=1000 only.
5. **Cost does not scale, and that is the next research question.** 8.7 MB/fact implies
   ~8.7 TB at 1M facts; MEMIT/AlphaEdit spend near-zero bytes per edit. Summation is
   closed (0/1000). The right question: can one delta carry k jointly trained facts?
   Capacity curve for k in {1, 4, 16, 64}, then cluster facts by subject/group into
   experts - the path to a real MoE and to answering "why more than per-fact LoRA".
6. **Paper identity is unresolved.** Efficacy is a model-editing claim (where we are
   approximately MELO); the ledger guarantees are mostly structural. Choose: **editing
   paper** (standard metrics, official baselines, scaling curve) or **systems/guarantee
   paper** (the conditions where a guarantee breaks - shared layer, multi-fact deltas -
   and the mechanism that keeps exactness). Doing both halves both.

## Pinned order (each experiment pins before its run)

1. **Strong RAG arms (first, gating).** CounterFact N=1000 (or a 200-fact subset for
   speed, confirmed at 1000): (a) plain injection (current), (b) explicit instruction
   "use the note even if it contradicts what you know", (c) few-shot demonstration of
   note-following, (d) note formatted as a Q/A pair. Same router, same facts.
   **Gate: if the gap closes (CIs overlap) in (b)-(d), stop; the thesis narrows to the
   guaranteed expert bank and the ledger, and the editing claim is dropped.**
2. **Router stress (CPU, cheap).** Aliases, typos, pronouns; report route/abstention
   per perturbation; replace the substring gate with an embedding-based entity match if
   needed. Report the paraphrase failures as router failures.
3. **EasyEdit baselines.** Run GRACE, WISE, MELO (and MEMIT/AlphaEdit if time) on the
   same 1000 facts; standard efficacy/generalisation/locality metrics.
4. **Metric completion.** Revoke sample n >= 100; report the pinned metrics exactly
   (canary KL, retain delta, provenance, router precision); decide N=200 rerun vs a
   N=1000-only table.
5. **Multi-fact delta capacity.** One delta trained on k facts, k in {1, 4, 16, 64};
   efficacy/locality/paraphrase curves; then group-by-subject experts.
6. **zsRE N=1000** (consistency; checkpoint at 100/1000 is resumable).
