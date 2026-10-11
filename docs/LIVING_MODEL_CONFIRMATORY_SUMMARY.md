# Living model - confirmatory summary (updated 2026-10-11)

One-page record of the confirmatory phase. Sources: `LIVING_MODEL_CONFIRMATORY_PREREG.md`
SS33-78; all numbers at N=1000 with Qwen2.5-1.5B-Instruct unless noted. Nothing here is
new - it consolidates. The paper (`paper/main.tex`) reports the same numbers.

## The claim

A frozen base model plus a transactional store. Each fact is an entry: a note (its
statement, served in the system prompt with an instruction to follow it) and, only if
the note fails on the edit prompt at write time, a LoRA adapter. A router whose
parameters do not depend on the stored facts applies at most one entry per query.
Exact removal and attribution follow from the data structure; answering, locality,
removal behavior and the divergence check are measured.

## Current system (default since SS76)

Router: complete-subject match, clause scoping, subject-masked relation texts compared by
a relation encoder (all-mpnet-base-v2 trained offline on 7500 triplets from facts
disjoint from the dev split and the test stores, then frozen), threshold 0.750 (the
lowest with CounterFact dev FPR <= 0.05). Served with adapter and note together.

| reading | CounterFact | zsRE |
| :-- | :-- | :-- |
| edit prompts answered | 0.996 [0.992, 0.999] | 0.989 [0.982, 0.995] |
| paraphrases answered (adapter and note) | 0.891 [0.872, 0.910] | 0.982 [0.973, 0.990] |
| paraphrases, adapter only / note only | 0.730 / 0.684 | 0.967 / 0.815 |
| other relation of a stored subject: leakage | 0.043 [0.031, 0.056] | - |
| same relation, other subject: routed / leakage | 0.018 / 0.017 | - |
| synthetic distractors, NQ loc questions routed | 0 | 0 |
| removal of 100: return to base answer | 1.00 | 0.84 (1.00 removing by content) |
| retention of the other 900 | 0.997 | 0.999 |
| facts needing an adapter (tiered store) | 0.162 | 0.206 |

Divergence check (SS69): 40 held-out canaries, kappa 0.968 (99th percentile over stored
adapters). The two training prompts understate an adapter's effect ~17x (mean D 0.022 vs
0.364); 10/10 harmful candidates rejected, the closest 3.6x above kappa. A wrong route
keeps only 72% of the base model's correct answers.

Held-out relations (11 of 34 CounterFact relations never seen by the encoder):
paraphrase routing 0.747 (seen relations 0.994), leakage unchanged from router v4.

## How the numbers moved

| stage | CF paraphrase | other-relation leakage | prereg |
| :-- | :-- | :-- | :-- |
| first version (router v3, adapter) | 0.323 | 0.346 | SS64/65 |
| router v4, tau 0.631, adapter / adapter and note | 0.669 / 0.803 | 0.186 / 0.237 | SS66-68, SS71 |
| router v4, tau 0.757 (locality-oriented), adapter and note | 0.589 | 0.063 | SS74 |
| mpnet with generated keys, tau 0.693 | 0.620 | 0.050 | SS75 |
| trained relation encoder, tau 0.750 (default) | 0.891 | 0.043 | SS76 |

Attempts that did not improve on the router-v4 threshold curve: subject-local KL anchors
(SS70), notes instead of adapters (SS71), store-key contrast (SS72), base-model relevance
check (SS73), answer-type score (SS74, eliminated on dev). Attempts that did not transfer
locality to held-out relations: an encoder trained on 1785 Wikidata relations (SS77),
write-time negative keys (SS78).

## First version (frozen 2026-10-09), with corrections

The 2026-10-09 summary reported efficacy 0.999 / 0.996; that evaluation served each edit
prompt with its own adapter whenever any route fired. Routed end to end (SS64) the first
version answers 0.997 / 0.994. Its canary check used the two prompts the KL term trains
on and never bound (max 0.092 vs kappa 2.0). Other first-version readings (SS56-62):
distractor no-leak 0.992 / 0.975, return to base after removal 0.96 / 0.80, retention
0.9989 / 0.9944, provenance 1.00 with a stable hash (100-sample).

## Gates and baselines

- strong-RAG gate: instructed retrieval 0.837 on CounterFact edit prompts (SS33-47).
- official GRACE (EasyEdit): plain prompt 1.00, chat 0.025 (SS54).
- official WISE (EasyEdit): per-step 1.00, final 0.07; with its merge lifecycle 0.125,
  no-leak collapses to 0.27 (SS54/63).
- capacity: one grouped adapter carries 256 facts (efficacy 1.00, KLD 0.007; leakage
  0.1-0.3, corrected in SS60/61).
- storage 8.7 MB per adapter (fp32); add ~8.5 s, serve 0.04 s, remove ~1.4-2.5 ms.

## Carried caveats

- One base model, one GPU, single-token targets, substring scoring.
- zsRE uses reference answers, which the base model often knows.
- The trained relation encoder learned CounterFact's relation inventory; on new relations
  it transfers part of the recall and none of the locality. Retraining it changes every
  key at once, so Proposition 2 (locality of adds and removals) holds between retrainings.
- Remaining leakage comes mostly from near-neighbor relations (country of origin vs
  location of formation, original language vs language of the work, manufacturer vs
  developer).
- WISE/GRACE comparisons are serving-format sensitive and use 200 facts.

## Open

- Separating unseen near-neighbor relations (needs examples of the pair).
- A second base model to replicate the main results.
- Releasing the saved stores, the trained encoders and the generated keys/templates.

The 2026-10-09 open decision (paper identity) is settled: the paper is a systems paper
about the transactional store and its measured guarantees.
