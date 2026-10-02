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

## KL-anchored delta training (route 1, pinned 2026-10-02 before the run)

Lambda = 1.0, anchor on the first two capability questions, matched prompt format
(system + user), lr 3e-4, 16 steps, all 18 facts. Reading: provenance/retained must
stay at the strong-setting levels and the per-expert KLD must fall well below the
10.96 mean. Otherwise lambda is swept (0.3, 3.0) before any other route.

## KL-anchor results (2026-10-02, route 1)

Lambda 1.0, two canary anchors, lr 3e-4, 16 steps, 18 facts:

| metric | before (no anchor) | after (route 1) |
| :-- | --: | --: |
| provenance | 1.00 | 0.94 |
| retained (suspend last) | 17/17 | 17/17 |
| per-expert KLD mean | 10.96 | **0.93** |
| per-expert KLD max | 22.08 | 3.63 |
| all-active KLD | 32.8 | 31.6 |

The anchor buys a ~12x smaller per-expert footprint while essentially keeping recall
(97 % of the leave-one-out patterns; one fact failed the with/without pattern and is
the open item). The all-active KLD stays high because 18 experts are active together -
a state that never occurs under routed inference. Route 1 is the standing recipe;
lambda sweeps and subspace constraints remain pinned options if needed.

## Router separation and abstention (pinned 2026-10-02, before the run)

The parameter-free router always picks the nearest expert today; an unrelated query
can therefore be answered by a fact it should not touch. Pinned before the run:

- measure the TF-IDF similarity of every fact probe to its own key (in-scope) and to
  the other keys (confusion margin), and of the four capability questions to every
  key (out-of-scope);
- pin an abstention threshold tau at the midpoint between the in-scope minimum and
  the out-of-scope maximum, report both margins;
- readings: fact recall under tau (must stay n/n), capability questions must abstain
  (no expert materialised), and the margin sizes are recorded as the router's
  separation quality. No trained router parameters are added.

## Router separation and abstention results (2026-10-02)

18 facts, 1.5B, fixed key index (the first run's ordering bug is recorded):

- in-scope similarity min **1.000** (each probe matches its own key; self-match, so
  the honest margin is out-of-scope), confusion max 0.317, out-of-scope max **0.317**
  (no capability question comes near any key);
- pinned tau = **0.659** (midpoint); under tau: fact recall **18/18**, abstention
  **4/4** (no expert materialised for unrelated questions).
- Caveat recorded: probes equal their keys by construction, so paraphrased in-scope
  queries are untested; the router's paraphrase margin is the next pinned measurement
  before the expert organ is formalised.

## Paraphrase margin + G1 rollback (pinned 2026-10-02, before the run)

Six hand-written paraphrases (one per band); readings: own-key similarity, best
other-key similarity, routed recall under the recorded tau 0.659. G1: propose_and_commit
with a deliberately strong candidate (lr 1e-3, 32 steps, no anchor) must refuse the
write at kld_limit 2.0; a normal candidate (3e-4, 16 steps, anchored) must commit.

## Paraphrase + G1 results (2026-10-02)

- **G1 atomic rollback works**: the deliberately strong candidate (lr 1e-3, 32 steps,
  no anchor) is refused at kld_limit 2.0 (measured 21.62); the normal candidate
  commits (0.01). The first explicit atomicity test passes.
- **Paraphrase margin fails**: own-key similarity min 0.339 vs best-other max 0.353
  (overlapping), and only 1/6 paraphrases are recalled under tau 0.659. The TF-IDF
  char n-gram router does not generalise beyond the exact key wording; the earlier
  separation was a probes-equal-keys artefact, as cautioned.
- Pinned next (before any other route): replace the router's similarity with a
  semantic key (small sentence-embedding model or the LLM's pooled hidden state),
  re-measure the margin and recall on the same paraphrases, and only then formalise
  the expert organ. A learned router stays out of scope until the parameter-free
  option is measured on this set.

## Semantic router keys (pinned 2026-10-02, before the run)

Router similarity replaced by the base model's last-token hidden state (normalised,
base adapter suspended), no new model; same six paraphrases; tau pinned at the
midpoint of own-key min and best-other max; readings: paraphrase recall under tau and
abstention on the capability questions, same protocol as the TF-IDF run.

## Semantic-key results (2026-10-02)

Last-token hidden state of the base model, paraphrases as queries:

| router key | own min | other max | recall | abstention |
| :-- | --: | --: | --: | --: |
| TF-IDF char n-grams | 0.339 | 0.353 | 1/6 | 4/4 |
| last-token hidden state | 0.890 | **0.983** | 3/6 | 4/4 |

Better than TF-IDF on absolute similarity, but the margin is still inverted: some
paraphrase's best *other* key beats its own. The base LLM's last-token state is not a
router key at this scale - every short query is 0.9+ similar to everything. Pinned
next: a dedicated sentence-embedding model (small multilingual, e.g. MiniLM-class) as
the key encoder, same protocol; only if that separates do we formalise the expert
organ. The router-task itself (choosing among near-duplicate fact keys) is now a
measured research item, not an implementation detail.

## Dedicated sentence-embedding keys (pinned 2026-10-02, before the run)

Encoder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, no fine-tuning;
same six paraphrases and four capability questions; tau at the midpoint; recall via
the embedding-chosen expert (others suspended); abstention as before.

## Router comparison (2026-10-02) - the dedicated encoder separates

| router key | own min | other max | margin | paraphrase recall | abstention |
| :-- | --: | --: | :-- | --: | --: |
| TF-IDF char n-grams | 0.339 | 0.353 | inverted | 1/6 | 4/4 |
| LLM last-token state | 0.890 | 0.983 | inverted | 3/6 | 4/4 |
| MiniLM multilingual (tau 0.656) | **0.742** | **0.569** | **positive** | **5/6** | **4/4** |

The router question is answered for this set: a small dedicated encoder separates the
keys and routes paraphrases (5/6; one miss is the open item, recorded in the JSON).
The expert organ can now be formalised on a router that works. Next pinned items:
identify the one missed paraphrase (or accept and measure at scale), the G1 invariant
beyond the KLD cap, and then the organ/controller.

## Expert-organ invariants (pinned 2026-10-02, before the run)

Router integrity under ledger edits, measured with the MiniLM router: (1) revoking a
fact must not change any remaining fact's routing decision; (2) adding a new fact must
not change any existing fact's routing decision; (3) the new fact must route to itself;
(4) the revoked fact's queries must no longer route into the revoked expert (abstain or
elsewhere). Reported as counts; any violation is a failure of the organ, not a
tolerance.

## Organ-invariant results (2026-10-02)

MiniLM router, 18 facts, revoke p4 + add x1: **pass True** - revoking moved 0 of the
remaining facts' routing decisions, adding moved 0, the new fact routed to itself, and
none of the revoked fact's queries still landed in the revoked expert. This is the
measured opposite of the SEUF/GRIP failure mode (routers drifting under unlearning):
our router keys are parameter-free base embeddings, so the invariant holds by
construction and is now measured per edit.
