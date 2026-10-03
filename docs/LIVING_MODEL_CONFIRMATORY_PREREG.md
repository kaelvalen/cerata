# Confirmatory study: do the guarantees and the router survive scale? (pre-registration)

Status: **proposed 2026-10-03, to be committed before any confirmatory run.** The
slice-1 record (`docs/LIVING_MODEL_SLICE_PREREG.md`) is an exploratory sequence, each
entry honest alone but written after seeing the previous result; this document is the
single confirmatory run it points to. Position and audit:
`docs/LIVING_MODEL_POSITIONING.md`.

## 1. Question

1. At N in {50, 200, 1000} facts, does the parameter-free router keep precision,
   recall and abstention, and do the measured guarantees (behavioural return after
   revoke, locality, provenance) hold, with confidence intervals?
2. Is there a task family where per-fact deltas beat retrieval at equal cost? The
   candidate measured first is VLM image-conditioned recognition (the contrast-pair
   result); then behaviour/format facts; then retrieval-missed paraphrases.

## 2. Facts

- **Nonce facts**: procedurally generated (subject/relation/object over a small
  grammar) in English, with held-out paraphrases (templated, unseen wordings) and
  distractor questions; N facts per size, fresh seeds. The generator is pinned by
  commit hash before the run.
- **CounterFact / zsRE subsets**: intended sources are the ROME dumps
  (`counterfact.json`, `zsre_mend_eval.json`); a dry-download check pins the exact
  URLs, item filters (single-token targets, balanced relation mix) and the held-out
  paraphrase policy (the dataset's own alternate prompts where present, else a
  template) at execution time, before any arm runs.
- Every fact carries: one teach pair, one probe, one held-out paraphrase, one
  distractor.

## 3. Arms

| arm | what | implementation |
| :-- | :-- | :-- |
| memory-only (RAG) | MiniLM retrieval + note injection; no weights | this repo |
| shared LoRA sequential | one adapter trained on all facts in order | this repo |
| summed deltas | per-fact deltas added into one adapter (slice-1: 0/18) | this repo |
| GRACE | key-value adaptors + deferral radius | reimplementation from the paper (official code if it runs on Qwen2.5; provenance recorded either way) |
| MELO | neuron-indexed dynamic LoRA blocks activated by a vector index | same |
| WISE | side memory + router | same |
| ours | per-fact deltas + MiniLM router + ledger + repair controller | this repo |

## 4. Metrics (all with bootstrap 95% CIs over facts)

Efficacy (probe hit), paraphrase generalisation (held-out), locality (neighbours'
recall after an edit), router precision/recall, abstention on distractors, behavioural
return after revoke (canary KL vs the pre-add state, retain-set accuracy), per-fact
time (train / serve / revoke) and storage.

## 5. Primary claims and thresholds (pinned before the run)

- **Router at N = 1000**: precision >= 0.95 and abstention >= 0.95; the curve
  50 -> 200 -> 1000 is reported (Stage-1's F18/F19 warning: routers degrade with
  scale, so this is the primary risk).
- **Behavioural return**: after revoke, canary KL <= 0.05 nats and retain accuracy
  within 1 point of the pre-add value.
- **Provenance**: leave-one-out attribution >= 0.95.
- **Delta > memory**: at least one family with a CI-separated gap; the VLM
  image-conditioned family is measured first.

## 6. Falsification

If router precision drops below 0.90 at N = 1000, or behavioural return fails, or no
delta > memory family separates, the thesis narrows to the measured parts and the
failure itself is the result. No mid-run tuning: a bug fix after a run restarts that
arm at all sizes.

## 7. Order of execution

1. Nonce generator + held-out paraphrases + distractors (commit hash pinned).
2. Dry-download and filter the external subsets; pin the exact selection.
3. Pilot N = 50, all arms; fix bugs; freeze the harness.
4. N = 200, then N = 1000; one run per arm per size.
5. VLM image-conditioned family: extend the contrast result to N = 50 scenes with
   held-out variants; the fair memory baseline is CLIP retrieval + note injection.

## 8. What this study is not

Not a new architecture claim (the per-fact LoRA + retrieval-router family is prior
art: MELO, GRACE, WISE, LoraRetriever, T-Patcher); not a full-transformer exactness
claim; not a claim that memory cannot revoke (it can, trivially). The claim under
test is the transaction ledger, provenance and the measured guarantee set - and the
delta's existence proof against retrieval.

## 9. Execution pin (2026-10-03)

- **Nonce generator**: `experiments/sandbox/live_learning/confirm/facts.py`; seed policy
  seed = n (50 / 200 / 1000 are independent draws); outputs regenerable at
  `results/live_learning/confirm/facts_n{50,200,1000}.json` (untracked). Validation on
  generation: unique subjects, unique word answers, paraphrase != probe. The generator
  commit is the pin.
- **External subsets**: dry check passed with an explicit certifi CA bundle (the bare
  urllib path fails on this host's CA setup): `counterfact.json` 45,108,470 bytes and
  `zsre_mend_eval.json` 8,091,864 bytes at `https://rome.baulab.info/data/dsets/`.
  Adapters and filters are committed before the pilot: CounterFact -> teach = prompt +
  target_new, probe = prompt, held-out paraphrase = first paraphrase_prompt, distractor
  = another item's probe; zsRE -> probe = src, held-out paraphrase = alt, answer =
  answers[0]; filter: single-token target with the Qwen2.5 tokenizer, balanced across
  the top relations/subjects; the sampling seed lives in the adapter. Downloads use
  certifi's CA bundle explicitly.

## 10. Staged pilot (2026-10-03, before the pilot)

The harness is built in two stages so bugs are fixed cheaply: **pilot A** (this run)
covers the in-house arms on the nonce set - rag, sequential (shared LoRA), summed,
ours - with the full metric set and CIs at N = 50. **Pilot B** adds the external
adapters (CounterFact/zsRE) and the GRACE/MELO/WISE arms before N = 200; the freeze
happens after pilot B. Harness pins for pilot A: answer system "Answer briefly.";
router tau = 0.656 (the slice-1 MiniLM value); anchors = two English canaries; ours =
propose_and_commit (cap 2.0) with 16 steps x 2 pairs per fact; sequential = one shared
LoRA, 8 steps x 2 pairs per fact (stated: half the per-fact budget); summed = the
per-fact deltas materialised together; revoke sample = the first 10 facts (token gone
+ retain on the rest); bootstrap 95% CIs over facts.

## 11. Pilot A results (2026-10-03, N = 50 nonce, wall 676 s)

| arm | efficacy | paraphrase | distractor no-leak | route |
| :-- | :-- | :-- | :-- | :-- |
| rag (memory-only) | 0.80 [0.68, 0.92] | 0.82 [0.70, 0.92] | 1.00 | 1.00 |
| ours | 0.82 [0.70, 0.92] | 0.76 [0.64, 0.88] | 0.60 | 1.00 |
| summed | 0.00 [0.00, 0.00] | 0.00 | 1.00 | - |
| sequential (shared LoRA) | 0.08 [0.02, 0.16] | 0.08 [0.02, 0.16] | 0.40 | - |
| revoke (10 of 50) | gone 1.00 [1.00, 1.00] | retain 0.875 [0.775, 0.975] | | |

Ours extras: add 7.19 s/fact, serve 0.36 s, storage 435.8 MB for 50 facts (~8.7 MB/
fact), paraphrase route accuracy 1.00, **router abstention on distractors 0.00**.

Readings, honest:
1. **Memory-only matches the delta system on this fact family** (CIs overlap; RAG is
   ahead on paraphrase). The reviewer's first question stands: on nonce facts the
   vector DB is the baseline to beat, and pilot A does not beat it. The delta's
   existence proof must come from another family (VLM image-conditioned first).
2. Composition replicates: summed 0/50; sequential 0.08 - the per-fact delta + routing
   architecture is the only working delta design here, at 7.2 s/fact and 8.7 MB/fact
   (~2 h and ~8.7 GB at N = 1000 - the scale cost is real and reported).
3. **The router abstention fails on template-identical distractors** (0.00): the
   distractor asks about an unseen subject with the same question template, so the
   semantic key matches; the router needs subject-aware keys (entity in the key or a
   verify step), not template similarity alone. The predicted router-at-scale failure
   is visible already at N = 50.
4. Revoke is behaviourally clean on this set (token gone 1.00; retain within efficacy).

Pilot B (external adapters + GRACE/MELO/WISE) and the subject-aware router fix come
before N = 200; the freeze happens after pilot B.

## 12. Router v2 (entity-aware) + pilot A2 (pinned 2026-10-03, before the run)

Pilot A's abstention failure (0.00) comes from template similarity: the distractor
asks about an unseen subject with the same question template. Router v2 requires a
**known subject** in the question (longest substring match over the fact base's subject
strings; no match -> abstain), then the semantic key picks among that subject's facts
(tau 0.656 unchanged). The same router serves every arm (RAG and ours) so the
comparison stays fair. Pilot A2 reruns N = 50 with the fixed router; predictions:
distractor abstention ~1.00 for RAG and ours, no-leak ~1.00, efficacy/paraphrase
unchanged within CI.

## 13. Pilot A2 blocked by the shared GPU; external adapters built (2026-10-03)

The rerun with router v2 hit OOM: a Minecraft Java process holds ~1.7 GB of the 8 GB
card, so the store model plus the sequential arm's second base do not fit. Per the
house rule the process is not touched; pilot A2 reruns when the GPU frees (no result
changed). Meanwhile the external adapters were built and sampled: counterfact 21,919
adapted (target <= 1 token: 10,034; 50 sampled, seed 50); zsRE 19,086 adapted (<= 1
token: 2,287; 50 sampled). The zsRE adapter uses `rephrase` as the held-out paraphrase
(`alt` is an alternative answer, not a question) and both adapters require the subject
in the paraphrase.
