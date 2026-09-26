# V3-LLM-1: single-fact learning after deployment on a frozen 7B LM - pre-registration

Status: **proposed, not started, not approved.** Written in the v3 restructure (phase
4) together with the harness it would use (`pal_moe/core/hf_lm.py`,
`pal_moe/api/lm.py`, `pal_moe/edit/down_proj.py`, `pal_moe/eval/editing.py`). No LLM
number exists in this repository yet; nothing below is a result.

## 1. The question

> On a frozen, 4-bit, 7B causal LM, can the v3 system learn a single new fact **at any
> time after deployment by an API call** - through the FAST path (retrieval of the
> written sentence over frozen hidden-state keys, injected into the context) or the
> MEDIUM path (a closed-form, float64, order-invariant, subtractable edit of one MLP
> down-projection) - with the four guards holding on every call, and how do the two
> paths trade efficacy against locality as the number of sequential edits grows?

This is the first LM study. It asks whether the v3 contract (learning = `write`,
unlearning = `forget`, guards always on) is *implementable* at 7B scale with usable
editing quality - not whether v3 beats the knowledge-editing literature.

## 2. Design

**Base model (pinned before the run).** `Qwen/Qwen2.5-7B` (open weights, Apache-2.0),
4-bit NF4 via bitsandbytes, bf16 compute, the Hub revision hash recorded in the run
record as part of `base_hash`. One model, no alternates: if it cannot be loaded, the
study stops.

**Layers (pinned by a development smoke, see section 5).** Key layer `L_key` and edit
layer `L_edit` are chosen on 50 CounterFact cases **disjoint from the evaluation set**,
from the grid `L in {4, 8, 12, 16, 20}`, by FAST retrieval top-1 hit rate (key layer)
and MEDIUM efficacy at N = 1 (edit layer). The smoke's evaluation-set numbers are never
computed.

**Key-covariance prior.** `C0 = 15000 * E[k k^T] + 1e-4 I` over the edit layer's
down-projection inputs on 100k tokens of WikiText-103 train (the MEMIT recipe's
weighting), float64 on CPU, hashed and recorded.

**Data.**

```text
CounterFact   the first 2000 cases of the ROME release (counterfact.json)
zsRE          the first 2000 cases of the MEND eval split
MQuAKE        MQuAKE-CF-3k, the first 500 cases
canary        500 fixed prompts: 250 WikiText-103 test sentence prefixes + 250 zsRE
              `loc` questions not in the zsRE eval subset; hashed
```

**Arms.**

```text
base      no write
fast      write(prompt + " " + target_new) into the FAST memory, k = 1,
          tau = 0.95 (cosine), retrieved text prepended at predict time
medium    write(Batch([prompt], [target_new])): closed-form down-projection edit,
          target values by 20 steps of activation-space optimisation (no weight trained)
both      fast + medium for the same fact
```

**Edit regimes.** `N in {1, 100, 1000}` facts written sequentially (N = 1: each case in
isolation, then forgotten; N = 100 / 1000: consecutive cases, evaluated after the last
write). MQuAKE: all edits of a case written, then its questions asked (N = case size).

## 3. Endpoints

**Guards (every call; a failure stops the run, it is not a result).**

```text
reversibility   write(x); forget(id) restores state(), the delta and all canary
                logits bitwise - 100 % of calls, checked by the API on every write
                and on every forget
order           N = 100: the same 100 facts written in two permutations give bitwise
                identical deltas (canonical solve) and identical canary argmax
purity          router (retrieval index) trainable parameters 0, asserted
base integrity  base_hash identical before and after the whole run
locality        canary argmax flip rate and max |delta logit| logged for every write
```

**Primary (CounterFact, N = 1000, fast vs medium, paired over cases):**

```text
E   efficacy       P(new) > P(true) on the rewrite prompt
S   specificity    P(true) > P(new) on neighbourhood prompts (ROME's NS)
```

**Secondary:** paraphrase (PS), the same three at N = 1 and N = 100, zsRE efficacy /
paraphrase log-prob and neighbourhood |delta log-prob| against base, MQuAKE multi-hop
case accuracy, canary flip rate vs N, wall time per write and per forget, bytes stored
per fact on each path.

## 4. Outcome table (fixed in advance)

| result | reading |
| :-- | :-- |
| every guard passes on every call, and `E >= 0.9` for at least one path at N = 1000 | the v3 contract is implementable at 7B: facts can be learned and exactly unlearned post-deployment by API calls. Report the E/S trade-off per path |
| guards pass, `E(fast) >= 0.9`, `S(fast) >= S(base) - 0.02`, `E(medium) < 0.9` at N = 1000 | retrieval carries single facts at scale and the closed-form edit does not; the medium path is for batches, not facts - revise the v3 routing of `write(str)` accordingly |
| guards pass, `E(medium) >= 0.9` and `S(medium) < S(base) - 0.05` at N = 1000 | the edit learns but interferes; the covariance prior is not protecting the canary / neighbourhood keys at this N |
| MQuAKE `both` > max(`fast`, `medium`) by >= 5 pp | the paths are complementary for integration; motivates the SLOW path (consolidation) on the LM |
| any reversibility / order / purity guard fails | a defect: the run stops and is reported as an implementation failure, not a scientific outcome |

## 5. Vetoes and the development smoke

```text
disjoint smoke set     the 50 layer-selection cases share no case_id or subject
                       with the evaluation subsets (asserted)
pinned before run      model revision, L_key, L_edit, tau, C0 hash written into this
                       document's section 5 (a dated amendment) before evaluation
no tuning on eval      no hyperparameter is changed after the first evaluation case
                       is scored
feasibility            measured cost projects the grid under 24 h on the RTX 5060
                       (8 GB) + 62 GB RAM machine (C0 for d_ff = 18944 is 2.9 GB
                       float64 on CPU; the model ~5 GB on GPU)
```

## 6. Statistics

Case-level pairing (the same cases in every arm). Proportions with Wilson 95 %
intervals; paired arm contrasts by exact McNemar on the discordant cases; secondary
endpoints with case-level bootstrap intervals (10,000 resamples, fixed seed). No
multiplicity correction is claimed across secondary endpoints; they are read together,
not tested.

## 7. Out of scope

```text
the LM SLOW path          consolidation into frozen LoRA experts: declared, not built
parametric value injection   the FAST path uses retrieved text in context only
comparison to ROME / MEMIT   not reimplemented here; the medium path uses the MEMIT
                             closed form, which is stated, not claimed as new
other models / sizes      one model per study
```

## 8. What this licenses, and what it does not

- **Licenses:** that the v3 API contract holds on a 7B LM (with numbers), and the
  observed efficacy / specificity / locality of each path at N in {1, 100, 1000}.
- **Does not license:** a state-of-the-art editing claim, any claim about other models
  or larger N, or any claim about the SLOW path.

## Amendment 1 (2026-09-26, before any run; nothing has been evaluated)

Written after a code review of the v3 branch. The question, arms, endpoints and the
outcome table (sections 1-4) are unchanged. Two guard definitions and two facts about
the harness change:

1. **Reversibility and order are measurements with a tolerance, not bitwise by
   construction.** The medium path now holds one float64 accumulator pair and `forget`
   subtracts the edit (a downdate); the earlier canonical re-sum made order invariance
   true by design and grew with the number of edits. Section 3's guard block is replaced
   by:

   ```text
   reversibility   every write: trial undo/redo, max|dDelta| vs the pre-write delta
                   <= 1e-8 and canary argmax identical; every forget: downdate vs a
                   from-scratch re-solve max|dDelta| <= 1e-8, canary argmax identical
                   to the last visit of that state. Bitwise only when the last edit is
                   forgotten (hook removed -> base model bitwise), checked at the end
                   of every N-regime
   order           every write: the delta re-solved with the live edits in a seeded
                   random permutation, max|dDelta| <= 1e-8 and canary argmax identical
   ```

   The 1e-8 tolerance (not the vision path's 1e-10) is fixed here, before any run,
   because the LM delta is solved in a `d_ff`-dimensional system; the measured values
   are reported in every case.
2. **The prior is a corpus prior by construction.** `C0` is estimated with
   `HFCausalLM.collect_keys` over every non-pad token of the WikiText-103 sample in
   section 2; `DownProjEdit` refuses a prior that is not a `KeyPrior`, or one estimated
   from fewer tokens than `d_ff`.
3. **The loaders were run on the real releases** (downloaded 2026-09-26 into
   `data/editing/`, untracked). Record counts and sha256:

   ```text
   counterfact.json      21919   d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f
                                 (https://rome.baulab.info/data/dsets/counterfact.json)
   zsre_mend_eval.json   19086   8a371d512f8a6ab175db4ef672181d5c0e52da23298cb51e5a2eeea2f89aa9cd
                                 (https://rome.baulab.info/data/dsets/zsre_mend_eval.json)
   MQuAKE-CF-3k.json      3000   ce27a39c39f2983512b9b5578fadea5fbe352e5368f49d64f38d37ce304edc80
                                 (github.com/princeton-nlp/MQuAKE, datasets/)
   ```

   Only parsing was checked (every case has a prompt with the subject filled and a
   space-prefixed target; >= 99 % have paraphrases and neighbourhood prompts; every zsRE
   case has a `loc` answer). No model was run on them.

## Amendment 2 (proposed 2026-09-26 in a repository review; NOT adopted)

Nothing has been evaluated. This amendment is a proposal: the owner adopts, edits or
rejects it, with a dated note here, before the development smoke runs. Sections 1-4
are unchanged unless a point below is adopted.

1. **Calibrate the retrieval threshold with the key layer, not after.** The FAST
   path's `tau = 0.95` is a raw cosine on LM hidden states, which are anisotropic
   (they share dominant directions; Ethayarajh, EMNLP-IJCNLP 2019): unrelated prompts
   can clear 0.95 and paraphrases can miss it. Proposed: on the same 50 disjoint smoke
   cases that pick `L_key`, also pick the key transform (raw, mean-centred, or
   whitened with a mean / covariance estimated on the section 2 WikiText sample) and
   `tau`, by the paraphrase hit rate subject to a neighbourhood false-hit rate
   `<= 0.02` (the margin section 4 allows `S(fast)` to lose). The transform is a fixed
   buffer, content-hashed into `base_hash`; the router stays at 0 trainable
   parameters. Pinned with `L_key` in the pre-run amendment.
2. **Read E together with PS for the FAST path.** The FAST memory stores
   `prompt + " " + target_new`, so retrieval on the rewrite prompt is near-exact by
   construction and `E(fast)` is close to a tautology. Proposed: PS joins E and S as a
   primary endpoint, and the outcome table gains one row, checked first:

   | result | reading |
   | :-- | :-- |
   | `E(fast) >= 0.9` and `PS(fast) < 0.5` at N = 1000 | the FAST path is an exact-match memory, not generalising retrieval; no row that rests on `E(fast)` alone applies |

3. **Measure feasibility per call before pinning the grid.** A guarded MEDIUM write in
   `accumulate` mode currently runs about six canary passes (pre, post, order
   reference, permuted order, trial undo, trial redo) and about five dense float64
   solves of the `d_ff = 18944` system (the write, `order_report`, the facade's
   permuted solve, undo, redo), and the two permutation checks re-sum every live
   edit. One dense LU at that size is about `(2/3) d_ff^3 = 4.5e12` flop; at an
   assumed 100-200 GFLOP/s float64 on the CPU that is 20-45 s per solve, i.e.
   minutes per write. That estimate, not a measurement, puts the N = 1000 regime and
   the 2000 isolated N = 1 write/forget pairs well past the 24 h veto. Proposed: the
   smoke records wall time per FAST write, MEDIUM write and forget at N = 1, 100 and
   1000 in both `accumulate` and `woodbury` mode; the pre-run amendment pins the mode
   and, if the projection still exceeds 24 h, a guard schedule fixed in advance (for
   example: locality and reversibility on every call, the permuted-order check on
   every k-th write and at the end of each regime, with k stated). A guard is never
   skipped after the first evaluation case is scored.

### Amendment 2, measurement note (2026-09-26; still proposed, nothing evaluated)

Point 3's estimate is now a measurement, from `experiments/v3_lm_cost.py` on the review
container (4 CPU threads, float64, synthetic corpus prior; no LM, so canary passes are
excluded). The code change it motivated is in the same commit: the facade's canary
order check reused nothing and re-solved the permutation `order_report` had just
solved (same seed), so a guarded MEDIUM write now does 4 dense solves, not 5.

```text
dense solve, d_ff = 4096          0.28 s    -> 27 s projected at d_ff = 18944 (x d^3)
guarded write, accumulate mode    4 solves  -> ~109 s per write at 18944, before canaries
guarded write, woodbury mode      0.008 s (1 live edit) / 0.92 s (1000 live edits) at 4096
                                  -> ~0.2 s / ~20 s at 18944 (x d^2), plus a one-time
                                  C0^-1 of ~90 s (2.9 GB float64)
```

Reading: in `accumulate` mode the N = 1000 regime alone is ~30 h and the 2000 isolated
N = 1 writes ~60 h of solves on this CPU, past the 24 h veto before a single canary
pass. `woodbury` brings the solve side of the whole grid to a few hours. If point 3 is
adopted, the proposal is to pin `mode="woodbury"` and to re-run this script on the
study machine (with `--canary_seconds` from one measured canary pass) before pinning
the guard schedule.
