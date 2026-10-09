# Living model - frozen confirmatory summary (2026-10-09)

One-page record of the confirmatory phase. Sources: `LIVING_MODEL_CONFIRMATORY_PREREG.md`
SS33-63; all numbers at N=1000 unless noted. Nothing here is new - it consolidates.

## The claim

A frozen base + a transactional delta store: per-fact learning that is exact-revokable,
auditable (provenance), and keeps a tiny canary footprint - with measured guarantees.

## Frozen table (N=1000)

| reading | CounterFact | zsRE | threshold |
| :-- | :-- | :-- | :-- |
| efficacy | 0.999 [0.997, 1.0] | 0.996 [0.992, 0.999] | - |
| paraphrase (routed) | 0.323 (router-bound) | 0.973 (subject-preserving) | - |
| distractor no-leak | 0.992 [0.986, 0.997] | 0.975 [0.965, 0.984] | >=0.95 |
| route accuracy | 0.997 | 0.784 (sibling pick) | - |
| distractor abstention | 0.992 | 0.967 | >=0.95 |
| revoke gone (n=100) | 0.96* | 0.36* | - |
| return match | 0.96 | 0.80 | - |
| retain | 0.9989 [0.9967, 1.0] | 0.9944 [0.9889, 0.9989] | >=0.98 |
| canary KLD mean (50-sample) | 0.0215 | 0.0171 | <=0.05 |
| canary KLD max | 0.0498 | 0.0529 | - |
| provenance (100-sample) | 1.00 (hash stable) | - | >=0.95 |
| storage | 8.7 MB/fact | 8.7 MB/fact | - |
| latency | add 8.45 s, serve 0.04 s, revoke 1.4 ms | - | - |

*revoke gone is base-confounded on zsRE (SS58); return_match is the base-referenced
reading there.

## Gates and baselines

- strong-RAG gate: instruct 0.837 vs ours 0.999 (SS33-47).
- official GRACE (EasyEdit): raw-prompt 1.00, chat 0.025 (SS54).
- official WISE (EasyEdit): per-step 1.00, final 0.07; with its intended merge
  lifecycle 0.125, no-leak collapses to 0.27 (SS54/63).
- capacity: one grouped delta carries 256 facts (eff 1.00, KLD 0.007; genuine
  leakage 0.1-0.3, corrected in SS60/61).
- router v3: typos/lower/space 1.00, partial 0.80, pronoun abstains (SS51).

## Carried caveats

- routed paraphrase on CounterFact is router-bound (direct delta 0.86-0.90).
- WISE/GRACE comparisons are serving-format sensitive; raw diagnostics reported.
- KLD sampled from 50 deltas; zsRE max 0.0529 marginally over the 0.05 mean target.
- LoRA+ adopted per-fact only; grouped training keeps the baseline recipe (SS48).
- capacity training is deterministic given the RNG sequence; k=256-only runs shift
  the draw position (SS60 addendum).

## Open decision

Paper identity: editing vs systems/guarantee (user decision).
