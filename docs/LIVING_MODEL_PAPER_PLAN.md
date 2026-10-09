# Living model - two-paper plan (2026-10-09, user decision)

Decision: **two papers**.

## Paper 1 - drawn now, from the frozen evidence

Identity: **systems / guarantee**. Architecture = transactional delta memory on a
frozen base. Claim: exact-revoke, auditable, capability-safe personalization
during interaction, measured at N=1000 with tight CIs.

Evidence (frozen): `docs/LIVING_MODEL_CONFIRMATORY_SUMMARY.md` + prereg SS33-63.
- cf N=1000: eff 0.999, no-leak 0.992, abstention 0.992, route 0.997, revoke
  gone 0.96 / return 0.96 / retain 0.9989, canary KLD 0.0215 (mean), provenance 1.00.
- zsRE N=1000: eff 0.996, para 0.973, no-leak 0.975, retain 0.9944, return 0.80.
- Gates/baselines: strong-RAG instruct 0.837 vs 0.999; official GRACE raw 1.00
  (chat 0.025); official WISE final 0.07 (merge 0.125, no-leak 0.27).
- Editing baselines = supporting evidence, not the headline.

Declared limitations (do not hide): router-bound CounterFact paraphrase (0.323;
direct delta 0.86-0.90), zsRE sibling pick 0.784, format-sensitive comparisons,
single model/environment, VLM/controller at small scale.

## Paper 2 - close the gaps (later)

1. Paraphrase router fix (subject-dropping paraphrases; paraphrase keys /
   attribute matching) - pinned, unbuilt.
2. Format-matched baselines: GRACE/WISE vs ours in the same serving format and
   the same N.
3. Learned router (replace the heuristic gate); zsRE sibling separation.
4. Grouped experts + coarse router build (256 facts/delta, storage cut).
5. Scale: VLM mirror and controller at confirmatory scale; second base model.

## Pointers

- Frozen summary: docs/LIVING_MODEL_CONFIRMATORY_SUMMARY.md
- Confirmatory record: docs/LIVING_MODEL_CONFIRMATORY_PREREG.md (SS33-63)
- State: experiments/sandbox/live_learning/STATE.md ("Frozen" section)
