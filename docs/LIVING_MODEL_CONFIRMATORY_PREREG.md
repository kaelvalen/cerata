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
