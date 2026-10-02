# Transactional delta store: slice-1 pre-registration

Status: **proposed 2026-10-02, committed before the first run.** Instrument:
`experiments/sandbox/live_learning` (the mesa). Position and audit:
`docs/LIVING_MODEL_POSITIONING.md`.

## 1. Question

Can a live model state be **transactional**? Every fact enters as a signed delta; every
fact can be revoked by subtracting it; the state after a revoke is a pure function of
the remaining delta set (hash identity), while old recall and general capability are
preserved. This slice measures the guarantees G1/G2/G3/G5 of Appendix A at
small-model scale; it is not the MoE slice and not the controller.

## 2. Design (pinned)

- Base frozen. **Each fact produces one LoRA delta trained from the same base**
  (independent training, deterministic seeds), stored as its `(A, B)` pair with an id.
- Live state: `S = base + sum_i B_i A_i`, materialised in id order into a fixed-capacity
  adapter (active prefix only; the tail zeroed). `revoke(id)`: drop the delta, rebuild
  in id order; `state_hash` hashes the base hash plus the active tensors in id order.
- `promote(fact)`: train, write, commit `{id, hash, provenance}`; a failing canary
  check rolls the write back (atomicity).
- Provenance v0: fact -> delta map plus per-answer attribution with the leave-one-out
  probe (does the fact's probe answer change when the delta is suspended).

## 3. Metrics and thresholds (soft, pinned before the run)

| guarantee | operationalisation | slice-1 target |
| :-- | :-- | :-- |
| G2 identity | hash(revoke(i) then rebuild) == hash(independently built set minus i) | exact (bitwise on tensors) |
| G2 behaviour | forgotten-fact recall | <= 0.1 |
| G3 isolation | retained-fact recall drop; capability delta; canary KLD | <= 10 %; <= 5 %; reported |
| G5 provenance | attribution accuracy over probes | >= 0.9 |
| cost | teach / revoke / answer latency | revoke < teach |

## 4. Arms

(a) memory-only (existing mesa), (b) transactional deltas with revoke, (c) deltas
without revoke (interference control), (d) bounded context (existing). Same stream,
same seeds, same model.

## 5. Falsification

- If revoke does not restore the independent-build hash, the implementation is wrong
  and the design changes before any claim.
- If transactional deltas cannot match the memory arm's recall at equal or better
  unlearn properties on the mesa stream, the slice answers "no" and records it.
- If G3 cannot be held (retained drop or canary KLD over target), the delta surface is
  too coarse and the next step is the controller, not more deltas.

## 6. Pinned decisions (from Section 9 of the positioning note)

- delta form: independent per-fact LoRA from the base, summed in id order;
- revoke tolerance: tensor hash must be bitwise; model behaviour in bf16 is reported
  with measured deltas (no behavioural bitwise claim);
- external anchor: the mesa stream plus a TOFU-class unlearn anchor (bundle check in
  the run commit); sleep-time/LaMP ids verified at the same time;
- controller: out of scope for slice-1; VLM mirror: triggered only if slice-1 passes.

## 7. Deliverables

The ledger code, JSON results under `results/live_learning/`, and a guarantee table
(G1/G2/G3/G5 with measured values) appended to the positioning note. Failed cells are
reported as failed, not removed.

## First smoke (2026-10-02, code `experiments/sandbox/live_learning/ledger.py`)

Three facts, Qwen2.5-0.5B, R = 16, 16 steps, lr 3e-4:

- **G2 identity holds for the last-added revoke: the state hash returns exactly to the
  recorded pre-add hash** (bitwise over the tensor set).
- Cost: add 3.6 s/fact, revoke 0.01 s.
- Behavioural recall is 0/3 with this delta configuration: the transactional mechanics
  are demonstrated, the delta efficacy is the open item - consistent with the mesa's
  LoRA trade-off (few updates on a 0.5B do not memorise). Next: a small delta-config
  sweep and the 1.5B rerun before any behavioural claim; the identity result stands on
  its own.

## Second smoke (2026-10-02, Qwen2.5-1.5B): the sum does not compose

R = 16, 16 steps, lr 3e-4; materialisation verified against the zeroed base (the same
probe answers "Arel" with the delta and a generic greeting without it).

| after adding | p1 | p2 | p3 |
| :-- | :-- | :-- | :-- |
| p1 | yes | no | no |
| p1+p2 | **no** | yes | no |
| p1+p2+p3 | no | no | **no** |

A single independently trained delta recalls its fact; adding the next breaks the
previous one, and the third breaks all three: catastrophic interference at the merge
level. The transactional mechanics (G2 identity, 0.01 s revoke) stand; the delta form
as specified does not compose. This is the slice's live question (G3 isolation).
Candidate routes, to be pinned before trying any of them: (a) train each delta against
the current merged state with a norm budget; (b) shrink per-delta magnitude
(lr/rank/steps sweep); (c) orthogonalise/constrain the update subspace (merging
literature: TIES/DARE-style); (d) stop summing into one adapter and route facts to
separate experts - the MoE organ. No route is tried until it is written here first.

## Route pinned (2026-10-02, before the third smoke)

The second smoke showed summation is the failure. Route pinned, in order:

1. **(MR) merged, as-is** - the control (current behaviour).
2. **(SW) small-magnitude control** - per-delta lr/rank/steps shrunk so the sum stays
   weak; cheap, tries to keep one adapter.
3. **(RE) routed experts** - PRIMARY: each fact keeps its own delta (expert); a
   parameter-free router (TF-IDF over the fact keys, same family as the repo's
   analytic routers) selects ONE expert per query; the merged sum is never used at
   inference. Rationale: G3 failure came from composition, routing removes composition
   entirely, and a parameter-free router keeps G1/G2/G5 clean. Subspace constraints
   are deferred until RE is measured.

Readings: recall under MR/SW/RE on the same 3 facts; G2 identity must hold under RE
(revoking an expert removes it from the router's set; state hash unchanged in
meaning); router accuracy reported per probe.

## Third smoke (2026-10-02, 1.5B): routed experts work

Same three deltas as the second smoke, answered by the parameter-free router (TF-IDF
over the fact keys, one expert per query, the merged sum never used):

| arm | p1 | p2 | p3 | after revoking p3 |
| :-- | :-- | :-- | :-- | :-- |
| merged (SW-0, as-is) | no | no | no | - |
| **routed (RE, primary)** | **yes** | **yes** | **yes** | p1 yes, p2 yes, p3 no |

G2 identity still exact (bitwise pre-add hash), revoke 0.01 s, add ~3.5 s/fact; the
router picked the right expert for all three probes without any trained parameters.
The G3 failure was composition, and routing removes composition: this is the first
working transactional-fact configuration. SW (small-magnitude single adapter) is now
deprioritised; the next steps are capacity (more facts), router accuracy at scale, and
the expert organ formalised (the MoE slice) with the same ledger semantics.

## Capacity experiment (pinned 2026-10-02, before the run)

The third smoke validated routed experts at 3 facts. Capacity question: does the
same configuration hold at 10 and 30 facts? Pinned before running:

- arms: routed (RE) vs merged (MR control), same deltas;
- readings: recall per fact (all / after revoking the last addition), router accuracy
  (routed recall implies the right expert), G2 identity at the last revoke, add and
  revoke latency, adapter capacity CAP = 64 (rank 16 per fact), model 1.5B;
- expectation: merged collapses immediately; routed holds while the router separates
  the keys; failures are reported per fact, not aggregated away.

## Capacity results (2026-10-02, 1.5B)

`--facts 10` and `--facts 30` (the stream holds 18 facts; the second run clips to all
of them), CAP 64, rank 16/fact:

| run | merged recall | routed recall | after revoking the last | G2 identity | add | revoke |
| :-- | --: | --: | :-- | :-- | --: | --: |
| 10 facts | 0/10 | **10/10** | 9 retained, revoked gone | exact | 3.28 s | 0.03 s |
| 18 facts | 0/18 | **18/18** | 17 retained, revoked gone | exact | 3.33 s | 0.07 s |

Router accuracy is 100% in both runs (implied by routed recall, one expert per query,
parameter-free). The slice's primary configuration therefore holds through the whole
stream: transactional add/revoke, exact state identity, full recall, sub-100 ms revoke.
Open next: G5 provenance (leave-one-out), the G3 table (canary KLD, retained drop),
capacity beyond the stream (synthetic key space), then the expert organ/MoE slice and
the controller.

## G3/G5 experiment (pinned 2026-10-02, before the run)

All 18 facts, 1.5B, CAP 64, deltas as before.

- **G5 provenance (leave-one-out):** per fact, the routed answer with the full set
  must contain the fact and the routed answer with that fact suspended must not;
  accuracy = attributed / n. Reported per fact.
- **G3 canary KLD:** next-token distributions on the four capability questions, base
  (adapter zeroed) vs full set; mean symmetric KL reported.
- **G3 retained effect:** suspending the last-added fact must leave every other fact
  recalled; reported as retained n/n.

## G3/G5 results (2026-10-02, 1.5B, all 18 facts)

- **G5 provenance (leave-one-out): 1.00** - every fact's routed answer changes when
  its delta is suspended, and none of the 18 fails the with/without pattern.
- **G3 retained (suspend the last): 17/17** - the revoked fact is gone, nothing else
  moves.
- **G3 canary KLD (base vs the full set all-active): 33.5** - very large. This is the
  worst-case state: at query time the routed path activates ONE expert, so the
  deployed damage is the per-expert KLD, not the union. The union number is still the
  honest reading of "what the full knowledge state does to unrelated behaviour", and
  it says the magnitude of the deltas (lr 3e-4, 16 steps) is too coarse to leave the
  general distribution alone when everything is active.
- Next measurement, pinned before running: per-expert canary KLD under routed
  inference (suspend all but the chosen expert), plus a magnitude sweep if the
  per-expert number is also large.

## Magnitude cliff (2026-10-02, 1.5B, 18 facts)

| lr / steps | provenance | retained | per-expert KLD mean / max | all-active KLD |
| :-- | --: | --: | --: | --: |
| 1e-4 / 8 | 0.06 | 1/17 | 0.18 / 0.54 | 25.7 |
| 3e-4 / 16 | 1.00 | 17/17 | 10.96 / 22.08 | 32.8 |

The facts learn only at the strong setting, and the strong setting distorts general
behaviour; the weak setting leaves behaviour alone and learns nothing. A sharp
magnitude cliff: per-fact LoRA deltas as shaped here cannot hold recall and a small
footprint at once. (The weak deltas' KLD 0.18 is mostly bf16 logit noise, visible
because base and active forwards are separate.)

Pinned before trying any fix, in order:

1. **KL-anchored delta training** - add a penalty on the canary prompts' next-token
   distribution during per-fact training (the foot-print is trained against, not
   hoped for); primary next attempt.
2. Mid-magnitude sweep (lr 2e-4 / 12 steps) to map the cliff.
3. Subspace constraints (orthogonalise the delta against canary directions or the
   other experts' directions) - deferred until 1 and 2 are read.

No other route is tried before being written here.
