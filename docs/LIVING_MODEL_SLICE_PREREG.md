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

## Controller v0 (pinned 2026-10-02, before the run)

Promotion rule: when a fact's query count reaches 2 (recurrence), train a candidate
expert and commit it only if its canary KLD is within 2.0; otherwise refuse. Stream:
6 facts queried 3x, 12 queried once. Readings: promoted set vs the recurring set,
routed recall of the promoted facts, expert count vs always-promote (18) and
never-promote (0), refusal count. The controller is a rule, no trained parameters;
a learned policy is out of scope until this one is measured.

## Controller v0 results (2026-10-02)

Promotion rule (recurrence >= 2, KLD cap 2.0), stream of 6 recurring + 12 one-shot
facts: promoted exactly the recurring six (policy match True), zero refusals, routed
recall 6/6, experts 6 vs always-promote 18 (12 experts saved) and never-promote 0.
The rule-based controller is the standing v0; a learned policy remains out of scope
until the rule's failure modes (recurrence threshold, budget pressure, drift) are
measured.

## Slice-1 closeout (2026-10-02): both single misses diagnosed, no mechanism failure

- **Provenance 0.94 (c6)** is a measurement artefact: c6's answer token "30" is
  non-discriminative - the probe returns "30" even with the c6 expert suspended (in
  `ledger_klanchor_15b.json`, with=True and without=True), so leave-one-out cannot
  attribute it. Fix: discriminative answer tokens in the stream (a data change, no
  code change); recorded, not silently patched.
- **Paraphrase 5/6 (p1)** is not routing: the MiniLM router chose the correct expert
  for all six paraphrases (`chosen == fid`, verified). The miss is answer robustness -
  the p1 expert was trained on the exact probe wording and does not answer "Arel" to
  "Adım neydi?". Fix: train deltas with a paraphrase pair, or accept and measure at
  scale.

With that, slice-1 is closed: no outstanding mechanism failure; the guarantee table
(Appendix B of the positioning note) stands, and the two open items are recorded as
refinements (discriminative stream answers; paraphrase-robust delta training).

## Paraphrase augmentation vs held-out set B (pinned 2026-10-02, before the run)

Each of the six facts keeps its exact pair and gains one paraphrase pair (set A,
already seen in the earlier router runs); the test set is six NEW paraphrases (set B,
never trained). Readings: router choice == fact (routing accuracy) and answer recall
on B. This separates routing generalisation (MiniLM) from answer generalisation
(delta training), the two halves of the p1 miss.

## Paraphrase-augmentation results (2026-10-02)

Held-out set B, never trained: **routing 6/6, recall 6/6**. One paraphrase pair per
fact in training is enough for the delta to answer unseen wordings, and the MiniLM
router generalises on its own (it was already 6/6). The two halves of the p1 miss are
therefore both closed: routing generalises by design, answer generalises with one
augmentation pair. The remaining open items are the c6 discriminative-token data fix,
the learned controller and the VLM mirror.

## G1 on failures (pinned 2026-10-02, before the run)

Atomicity is not only the policy refusal: a training exception and a bad-input
ValueError/IndexError must both leave the state hash and the delta set untouched. The
ordering bug found in propose_and_commit (the delta was written before the key was
resolved) is fixed first. Readings: raised exception type, hash unchanged and delta
set unchanged for each injected failure.

## G1 failure-atomicity results (2026-10-02)

Injected training exception ("boom") and bad input (empty pairs -> IndexError): after
both, the state hash and the delta set are unchanged (**pass True**); the fix moved
the key resolution before the first write in propose_and_commit. G1 now covers the
policy refusal (KLD cap), a training failure and a bad-input failure - all three leave
no trace.

## Stream v2 c6 + learned-controller design (pinned 2026-10-02, before the run)

**c6 v2**: the calibration fact becomes a discriminative code ("Kalibrasyon kodu
7310'dur." / "Kalibrasyon kodu nedir?" / "7310"), leaving the other 17 facts and all
protocols unchanged. Reading: with the c6 expert suspended the answer must not contain
7310; with it, it must - i.e., the provenance artefact closes. v2 is a separate stream
version; v1 results stay on record.

**Learned controller (design pinned, implementation next)**: a contextual bandit over
{promote, defer} per fact, features = recurrence count, router margin (own sim minus
best-other sim), candidate KLD (measured without committing), expert budget used;
reward = +1 later-probe recall, cost = a pinned expert-budget price. Training signal
from the same stream protocol; readings vs the rule v0: expert count at equal recall,
refusals, and the budget curve. Implementation is deliberately deferred: it needs a
probe placement protocol that v0 did not require, which is its own pinned decision.

## c6 v2 results (2026-10-02)

With the c6 expert: "7310"; without it: a generic explanation, no code - **attributed
True**. The last measured artefact (the non-discriminative "30") is closed by the v2
stream; v1 stays on record. The learned-controller design is pinned above and is the
remaining implementation item, followed by the VLM mirror.

## Controller v1: learned promotion policy (pinned 2026-10-02, before the run)

Missing decision from the earlier pin - probe placement - is resolved as: **the served
query is the probe**. The controller observes the outcome (token hit) of every query it
serves, at zero extra generation cost; no separate probing schedule. Reward is
immediate (no future discount; stated as the limitation): +1 for a token hit on the
served path, and a one-time cost of 0.3 when a promotion commits. A KLD refusal locks
the fact to the memory path (no repeated retraining attempts); a committed promotion
switches the fact to its expert for the rest of the stream.

Memory tier (E): MiniLM retrieval over the fact keys (query == key by construction, so
this isolates the generation tier) + note injection into the base prompt; served with
all deltas zeroed. Expert tier: the fact's own delta materialised alone.

Two arms, same stream as v0 (6 facts x3, 12 facts x1), same model and session:
- rule: promote on the second query (v0), deferred facts served by memory;
- bandit: epsilon-greedy (eps=0.2, alpha=0.5, seed 0, ties -> defer), context =
  (last served outcome, min(count, 2)), actions {promote, defer}.

Readings per arm: experts, refusals, final readout recall over all 18 facts (path and
token hit per fact), Q table. Predictions: (1) Q(defer) > Q(promote) in the memory-ok
context and the reverse in the memory-fail context; (2) bandit promotions concentrate
on facts whose memory tier fails (plus exploration commits, counted); (3) bandit final
recall >= rule's at comparable or lower expert count.

## Controller v1 results + v1.1 UCB (diagnosis 2026-10-02, v1.1 pinned before its run)

v1 as pinned **failed two of its three predictions**: the bandit chose promote 0 times in
30 decisions (with eps=0.2, P = 0.9^30 by pure luck), so experts 0 and final recall
15/18 vs the rule's 16/18; Q shows a zero-value defer in the memory-fail context and no
promote sample at all. Diagnosis: (a) the failure context can never learn - after a
failure Q(defer) = 0, Q(promote) = 0 never sampled, tie -> defer, i.e. the known-bad arm
ties the unknown arm; (b) no forced exploration, plus seed luck. The rule's six experts
also incidentally covered p3, the one recurring fact whose memory tier answered wrongly
("Kahve içmek için bir çarşaf...") - visible only because this run measured the memory
tier for all facts for the first time (rule final readout fails w3 and c2 in memory).

v1.1 fix: **UCB1** over the same contexts, actions and reward (bonus
sqrt(ln(N+1)/n_a), C=1.0, unsampled action first, tie -> defer, same seed and stream).
Predictions: (1) promote is sampled once in every context, so Q(promote) is measured
exactly where the failure context needs it; (2) the learned policy promotes in the
(fail, 2) context (p3) and defers otherwise, so expert count = 1 + the forced
first-sample commits; (3) final recall >= rule's 16 at fewer experts.

## Controller v1.2: repair bandit, decision after the serve (pinned 2026-10-02, before the run)

v1.1 UCB **over-corrected**: 17 experts, recall 17/18. The trace shows the mechanism:
the defer arm in the (none,1) context was sampled twice (one hit, one miss - p3's
deterministic memory failure), so Q(defer) = 0.25 with n = 2, while promote was
sampled repeatedly at a constant 0.7; the bonus could not revive defer and 15 of 18
facts were promoted at their first query, before memory even served them. Diagnosis:
when the controller decides *before* serving, the promote counterfactual ("what would
memory have done?") is never observed, and one unlucky defer sample poisons the arm.
The decision point, not the exploration rule, is the flaw.

v1.2: **serve first, repair after failure.** Every unpromoted fact is served by the
memory tier; a query with a token hit needs no decision; only on a miss does the
controller decide {promote, defer} (same UCB1: C=1.0, alpha=0.5, tie -> defer, context
= min(count,2); the pinned reward is unchanged: +1 token hit on the served path minus
0.3 when a promotion commits - for defer the served path is the observed miss, reward
0). The reward myopia still stands as the known limitation.

Predictions: promoted = {p3, w3, c2} (the deterministic memory failures), experts 3,
recall 18/18 (a repaired fact is served by its expert at the final readout); Q shows
promote > defer in the failure contexts. This would beat the rule (16/18 at 6 experts)
and both pre-serve variants on the expert/recall trade-off.

## Controller v1.x results (2026-10-02)

| arm | experts | final recall |
| --- | --- | --- |
| rule (recurrence >= 2) | 6 | 16/18 |
| v1 eps bandit | 0 | 15/18 |
| v1.1 UCB, decide-before-serve | 17 | 17/18 |
| v1.2 repair (serve first, UCB) | **3** | **18/18** |

v1.2 promoted exactly {p3, w3, c2} - the deterministic memory failures - and reaches
full recall at half the rule's expert count. Four failure events occurred (p3 twice,
w3, c2): the count-1 context split its first samples (one defer, two promotes) and
p3's second failure promoted in the count-2 context; Q(promote) = 0.525 / 0.35 vs
Q(defer) = 0.0 in both contexts. The lesson of the arc is that the **decision point**
mattered more than the exploration rule: before serving, the promote counterfactual is
unobserved and one unlucky defer sample starves the arm (v1.1); after serving, the
outcome *is* the counterfactual and one sample per action suffices. Remaining known
limitation: the reward is myopic; a future-aware repair needs a recurrence signal (or
the final readout counted as future value) - pinned as the next controller item.

## Cross-session E+Delta hybrid on the event stream (pinned 2026-10-02, before the run)

Protocol: `stream.build_events(seed=0)` drives the hybrid. Memory tier: one note per
fact (teach text) plus MiniLM retrieval over the fact probes, tau = 0.5; a retrieved
note is injected into the base prompt, no note above tau -> base answer. Expert tier:
the adopted v1.2 repair policy - a fact-tagged probe served by memory that misses
triggers the UCB repair decision (same constants: C=1.0, alpha=0.5, cost 0.3, context
min(count,2)); a committed fact is served by its delta. Notes are pre-loaded for both
sessions (the harness knows the taught facts, as in the controller runs); fillers are
no-ops. Update events rewrite the note and, if the fact has an expert, revoke + retrain
the delta (replace transaction; a refusal falls back to memory). Unlearn drops the note
and revokes any expert. Session 2 replays probes only, controller online.

Readings: per-note recall in s1 and s2, the s1-immediate vs s2 retention table, update
probe (sutlu present, sutsu z absent, retrained?), unlearn probe (Zeytin absent) and
neighbour (Arel present), capability, promotions/refusals/cost, retrieval decisions.

Predictions: (1) w3 and c2 miss at their immediate probes and repair (the first miss,
p3's, defers by the UCB tie rule), so experts are {w3, c2} plus p3 if its second miss
precedes the update or falls in s2; (2) s2 recall covers every non-unlearned fact
(repairs included) - the session boundary changes nothing for memory-served facts and
the ledger keeps the experts; (3) the update is honoured with the old token absent and
the unlearned token is absent in s2; (4) capability 4/4 in both sessions.

## Cross-session hybrid results, part 1 (2026-10-02)

Final state: experts {c2, p3, w3}; s1-imm 17/18 (p3's deterministic miss, the first
repair sample deferred by the UCB tie), s1-del 5/6 (p3 again), s1-upd 1/1, unlearn
1/1 and neighbour 1/1, s1-intf 2/2, **s2 17/17** (retention: every fact true/true
except p3 false/true - memory failed at s1, repaired at s2). The p3 sequence shows the
tie rule spending two samples (ctx c1, then ctx c2) before promoting: a measured
exploration cost.

Two findings against the predictions:
1. **Capability 2/4, both sessions** (pinned 4/4). cap4 ("Bir yılda kaç ay vardır?")
   failed because retrieval injected w2's note at sim 0.63 >= tau 0.5 - a false
   positive. cap3 ("Suyun kimyasal formülü nedir?") failed at the base tier itself
   ("CnH2n+2"; retrieval sim 0.25, no note) - a base/prompt weakness, not the ledger.
2. **Post-update promotion trap**: the update rewrote only the note, so a fact promoted
   *after* the update would be trained from the stale static FACT record. Here the
   frozen s2 protocol expects the stale token ("sütsüz") and the stale expert *passed* -
   the protocol masked the bug by construction. The update itself passed in-session
   (s1-upd 1/1); the replace transaction was not exercised (p3 was memory-only at
   update time).

## Cross-session hybrid v2 (pinned 2026-10-02, before the run)

Three fixes after part 1: (a) tau 0.65 (excludes w2's 0.63 false positive); (b)
repairs train from the *current* note and current answer token (an answers map updated
by update events), not the stale FACT record; (c) fact probes are evaluated against
the current token in both sessions (the frozen s2 expects pre-update tokens by
construction). Predictions: experts {w3, c2} - p3's updated note answers its probe, so
the update replaces the need for an expert; s2 17/17 with p3 evaluated against "sütlü";
capability >= 3/4 (cap3's base miss may persist); the update retrain path is still not
exercised (p3 stays memory-only at update time).

## Cross-session hybrid v2 results (2026-10-02)

Predictions confirmed: experts {w3, c2}; the update replaced the need for p3's expert
(its updated note answers the probe - s2 path memory, "sütlü" present, evaluated
against the current token, retention [false, true]); s2 17/17; capability 3/4 - cap4
now passes (tau 0.65 stops w2's 0.626 false positive, base answers "Yılın 12 ayı
vardır"), cap3 remains the base-tier miss ("CnH2n+2" with no note, sim 0.25). Cost 0.6
(two promotions). The update retrain path remains unexercised (p3 was memory-only at
update time) - noted, not fixed: it is the same commit transaction as a normal
promotion.

Cross-session summary: within-session repairs {w3, c2}; p3's correction lives in the
E-tier note; the ledger keeps the two experts across the session boundary; unlearn and
update both honoured; capability stays base-bound (3/4) with retrieval gated by tau.

## VLM mirror feasibility v0 (pinned 2026-10-02, before the run)

Model: Qwen/Qwen3-VL-2B-Instruct (bf16; 4.3 GB; the newest small multilingual VLM
that fits the shared 8 GB GPU). Images: two procedurally generated panels (white
background, three shapes each; panel A green circle / red square / blue triangle,
panel B blue circle / green square / red triangle) - nothing downloaded. Fact:
"Kalibrasyon düğmesinin kodu Tira'dır." (nonce code), probe "Kalibrasyon düğmesinin
kodu nedir?" expecting "Tira".

Protocol: (1) base vision sanity on A ("Görseldeki dairenin rengi nedir?" -> yeşil);
(2) base probe on A -> must NOT contain "tira"; (3) train one LoRA delta (r=16,
q_proj/v_proj, lr 3e-4, 12 steps, pairs: teach echo + probe->Tira) from the frozen
base - **no KL anchor in this v0** (stated deviation from the standing recipe; the
question is only whether a delta can teach an image-grounded fact at all; anchor drift
is measured instead); (4) probe A -> "tira" present; (5) probe B (same question, panel
B) -> leakage check (the pair text alone might leak across images); (6) revoke (zero
the LoRA) -> probe A back to no "tira"; (7) canaries (text-only capability questions)
before and after training -> drift report.

Predictions: (1)+(2) pass (a 2B VLM sees colors and cannot know a nonce); (4) pass iff
a 2B VLM absorbs a fact from 12 LoRA steps; (6) pass (zeroing is exact); (5) and (7)
are measured, no prediction - first VLM-side readings of grounding and anchor drift.
Failure modes to record: OOM on the shared GPU, a generation that ignores the image,
or a delta that changes the answer without the image.

## VLM mirror feasibility v0 results (2026-10-02)

Runs on the shared 8 GB, no OOM. Checks: sanity sees green true; base probe does not
contain the nonce true; after 12 LoRA steps probe A = "Tira" true; revoke (zeroing)
returns to the base answer true; **image_leak true** - the same probe on panel B also
answers "Tira": the v0 delta is text-memorising, not image-conditioned. Canary drift
without the KL anchor: "Beş kere altı kaç eder?" went from a correct answer to
**"Tira"** - catastrophic interference on an unrelated question - and the capital
canary drifted İstanbul -> Ankara. First VLM-side readings: the frozen base sees and
does not know; a delta teaches in 12 steps and zeroing is exact; the KL anchor is
necessary on the VLM side too; image-conditioning must come from the router (same
question, different images, different experts), not from a single text pair.

## VLM mirror v1 (pinned 2026-10-02, before the run)

Two facts with the *same probe text* on different panels: A -> "Tira", B -> "Vok".
Each delta is trained with the standing KL anchor (lambda 1.0 on the two text-only
canary prompts, base logits from the zeroed model; the check canary is a KL prompt -
as in the text recipe). Serving is image-keyed: DINOv2 (frozen, cached) embeds the
panels; the query image routes to the nearest stored key with tau 0.9 and abstains
below it. Protocol: base unknowns; train both deltas; serve A and B (route + code);
cross-image check (delta_A alone with panel B - the v0 leak, recorded); canary after
training with delta_A active (the 5x6 answer must not be "Tira"); revoke B (drop its
delta), query B -> must abstain to base (no Vok, and no Tira from A), query A still
"Tira".

Predictions: anchor holds (no "Tira" on 5x6); route 2/2; served codes 2/2; delta_A
alone still leaks "Tira" on B; after revoke B the tau gate abstains -> no Vok and no
Tira; A unaffected.

## VLM mirror v1 diagnosis sweep (pinned 2026-10-02, before the run)

v1 failed unexpectedly: after training, every served answer (both facts, restored
deltas, post-revoke) looked like the base. Two candidate causes: KL dominance (lambda
1.0 swamping the pair loss) or a restore bug (v0 only verified zeroing, never
restoring a saved delta). Sweep: train fact A at lambda 0.1 and 1.0 (lambda 0 known
from v0: learns, damages); for each arm record the in-memory probe (still-trained
params), the 5x6 canary (anchor), the LoRA tensor norm, then save -> zero -> restore
(report the exact max tensor diff) -> the restored probe. Diagnostics only; the v2
protocol is pinned after the readings.

## VLM mirror v1 diagnosis results (2026-10-02)

The sweep found **lora_norm = 0.0 for both lambdas** - training had not moved a single
LoRA parameter, and the restore diff was trivially zero. Root cause: `set_lora(None)`
zeroes *both* LoRA factors before training; with A = 0 and B = 0 the pair loss has zero
gradient through both (dL/dB needs A x, dL/dA needs B), a dead stationary point. v0
worked because peft's default init has A random and B zero. Fix: `reset_lora` - B zero,
A re-initialised kaiming - at the start of every delta training; zeroing both remains
the revoke/serve operation. The v1 failure was a harness bug, not KL dominance. The
fixed sweep and the v2 protocol are pinned next.

## VLM mirror v2 (pinned 2026-10-02, before the run)

Fixed sweep results: with `reset_lora` every lambda (0.0 / 0.1 / 1.0) teaches, in-memory
and restored, restore tensor diff exactly 0; the canary was clean for all arms at this
seed (v0's "Tira" canary was therefore seed/trajectory-specific - the KL anchor is
adopted as the standing recipe, not as a measured necessity in this run). v2 protocol:
panels A (circle/square/triangle row) and B' (a structurally distinct row:
triangle/circle/square) with the *same probe text*; codes A -> "Tira", B' -> "Vok";
each delta trained with the KL anchor (lambda 1.0, 12 steps, reset start); image-keyed
routing by DINOv2 (tau 0.9, abstain below the threshold); checks: base unknowns; route
2/2 (the full sim matrix is recorded); served codes 2/2; canary after delta_A; leak of
delta_A on B' (recorded); revoke B' -> the route must abstain (cross sim < tau) and the
answer must contain neither Vok nor Tira; A still serves "Tira".

## VLM mirror v2 results (2026-10-02)

Base unknown both true; route 2/2 by argmax; **served 2/2** ("Tira", "Vok") - the frozen
base gains two image-keyed experts through 12-step KL-anchored deltas, each served by
its own delta; canary clean; after revoking B', A still serves "Tira" and the B' answer
returns to base. Leak recorded: delta_A alone still answers "Tira" on B' (text
memorisation inside an expert); the router is what disambiguates. **Abstention after
revoke failed**: DINOv2 CLS keys of the two panels are 0.985 similar (own 1.0, margin
0.015), so the tau 0.9 gate cannot tell them apart - routing only works by argmax on
known panels. The mirror is green on teach/serve/revoke/A-isolation but pending on
*refusal*: the key needs visual separation (more distinct panels, or DINOv2 mean-patch
/ CLIP keys) plus a margin criterion, not just a single tau. Pinned next: a key
comparison (DINOv2 CLS vs mean-patch vs CLIP on the panels, margins recorded), then
the revoke-abstain check with the margin gate.

## VLM key comparison (pinned 2026-10-02, before the run)

Pure measurement, no VLM: embed panels A, B' (v2), and three distractors - D1 (the old
colour-swap panel), D2 (a single gray square), D3 (two shapes on a diagonal) - with
four key types: DINOv2 CLS, DINOv2 mean-patch, CLIP (laion ViT-B/32) image embedding,
and the concatenation DINOv2 CLS + CLIP (both normalised, renormalised). Readings: the
A-B' cross similarity and the maximum distractor similarity to either key, per type.
The v3 protocol (key type and tau by a pinned rule) is pinned after reading this.

## VLM key comparison results + v3 protocol (pinned 2026-10-02)

Margin = 1 - worst off-key similarity (the worst is the colour-swap distractor d1):
DINOv2 CLS 0.0155 (cross A-B' 0.9845), DINOv2 mean 0.0078 (cross 0.9857, d1 0.9922),
**CLIP 0.0362** (cross 0.9376, d1 0.9638), concat 0.0265. CLIP wins; all margins are
small because the panels are near-identical scenes (same layout, swapped colours), so
the exact-query regime (own similarity 1.0) is the one that matters here.

v3 protocol: CLIP image keys (pooled vision output, projected, normalised); tau = 0.97
(pinned rule: midway between the worst off-key 0.9638 and 1.0, rounded to 0.97);
serving goes through route -> materialise (no delta below tau); otherwise the same v2
protocol (KL-anchored deltas, 12 steps). Checks: base unknowns; route 2/2; served 2/2;
canary clean; **distractor refusal** - D1 (colour-swap), D2 (gray square), D3
(diagonal): all must abstain; **revoke B'**: route B' -> None (0.9376 < 0.97), serve ->
base, no Vok and no Tira; A still "Tira". Prediction: all pass; the tightest margin is
D1 at 0.0062, flagged as fragile (exact-panel regime; unseen-photo generalisation is
out of scope).

## VLM mirror v3 results (2026-10-02)

**All seven checks pass**: base unknowns true; route 2/2 (own sims 1.0, cross 0.938);
served "Tira"/"Vok"; canary clean; distractor refusal true (d1 0.964, d2 0.522, d3
0.806 - all below tau 0.97); revoke B' abstains (0.938 < 0.97 -> base, no Vok and no
Tira); A still "Tira". The VLM mirror now stands: a frozen 2B VLM acquires two
image-keyed facts as KL-anchored deltas; the CLIP-key router serves each fact, abstains
on distractors, and abstains after a revoke; zeroing is exact. Flagged: d1's margin is
0.006 (exact-panel regime); unseen-photo generalisation is out of scope - keys must be
recalibrated (or made more distinctive) before real photos.

## VLM store port: G1/G2/G4 battery (pinned 2026-10-02, before the run)

`vlm_ledger.py`: a VlmDeltaStore mirroring the text store - per-fact deltas over the
frozen Qwen3-VL-2B, CLIP keys, `state_hash` over the active delta tensors, training via
vlm_mirror2.train_delta_kl (KL anchor), `propose_and_commit` measuring the candidate's
canary Jeffreys KLD against the zeroed base (cap 2.0 default) and refusing over it,
`serve` via route -> activate (tau 0.97), `save`/`load_state` for durability. Battery:
add A, add B; hash deterministic; serves 2/2; **G1 refusal** - a text-only adversarial
candidate trained on the canary question itself ("Beş kere altı kaç eder?" -> "Tira",
lambda 0, 20 steps) must exceed the cap and be refused with the hash unchanged;
**G2 identity** - revoke B restores the recorded pre-B hash bitwise, revoke A restores
the empty hash; **G4 durable** - save the state, reload it as a new session, hash
equal, serves 2/2. Predictions: both adds commit with small KLDs; the adversarial
candidate is refused; all identity/durability checks true; the reloaded session serves
2/2. The controller port is the next increment after this.

## VLM store port results (2026-10-02)

**All eight checks pass.** The adds commit with canary KLD 0.098 (A) and 0.191 (B) - the
anchored candidates barely move the base; the adversarial text-only candidate (the
canary question itself answered "Tira", lambda 0, 20 steps) measures KLD **23.928** and
is refused with the hash unchanged (G1 works with a real margin); revoke B restores the
pre-B hash bitwise and revoke A the empty hash (G2); save -> reload reproduces the hash
exactly and the reloaded session serves both facts (G4 + cross-session). The VLM side
now mirrors the text store's transaction core: CLIP-keyed per-fact deltas, KLD-capped
commits, exact revoke, durable state, routed serving. Remaining VLM item: the repair
controller over an interaction stream.

## VLM repair controller over a stream (pinned 2026-10-02, before the run)

The adopted v1.2 policy on the VLM store: serve through the CLIP router (base when no
expert or below tau); a fact-tagged miss triggers the UCB repair decision (C=1.0,
alpha=0.5, tie -> defer, context (last outcome, min(count,2)), cost 0.3, commit only
through propose_and_commit's KLD cap). Stream: A and B' three queries each, C (a
column panel) and D (two stacked circles) one query each; codes "Zun"/"Mek". Readings:
the per-decision trace, experts and their KLDs, route choices, the final readout over
all four panels, the canary with no delta active, the state hashes, and the CLIP sim
matrix of the four panels. Predictions: the first miss (A q1) defers, A repairs at q3;
B', C, D repair at their first miss; experts 4; final readout 4/4; route choices
correct for promoted facts; every committed KLD under the cap; canary clean.

## VLM repair controller results (2026-10-02)

**All seven checks pass** and the trace matches the pinned predictions: A q1/q2 defer
(the first samples of their contexts), A q3 promotes; B' promotes at its first miss by
the unsampled-promote rule; C and D promote greedily once Q(promote) > Q(defer);
experts 4 with committed KLDs 0.098 / 0.191 / 0.167 / 0.020 - all far under the 2.0
cap; final readout 4/4 through the CLIP router ("Tira"/"Vok"/"Zun"/"Mek"), route
choices correct, canary clean with no delta active, state hash changed. The CLIP sim
matrix of the four panels has a maximum off-key similarity of 0.938 (A-B'), below tau
0.97. The v1.2 policy transfers to the VLM side: serve -> observed miss -> UCB repair,
with the same exploration cost (A's two deferred samples) and the same result (full
recall, minimal experts).

## VLM update/unlearn battery (pinned 2026-10-02, before the run)

Mirror of the text session semantics on the VLM store. Sequence: (1) add A ("Tira");
(2) **update A** - revoke the old expert, propose_and_commit the new pair ("Bora", same
panel; the KLD cap still gates it); (3) add B' ("Vok"); (4) isolation probes; (5)
**unlearn B'** - revoke. Readings: hashes h0/h1 (after add A)/h2 (after the update,
state {A_new})/h3 (after B')/h4 (after unlearn, must equal h2 bitwise), the update
record (old revoked, new committed, KLD), and the probe responses - after the update A
must answer "Bora" and not "Tira"; B' stays "Vok"; after unlearn B' must not answer
"Vok" while A still answers "Bora"; canary clean. Predictions: the update commits under
the cap and serves the new code; isolation holds; h4 == h2; unlearn is silent; all
checks true.

## VLM update/unlearn results (2026-10-02)

**All nine checks pass.** The update (revoke + retrain "Bora") commits with KLD 0.065
and serves the new code while "Tira" is gone; B' stays isolated ("Vok"); unlearn B' is
silent (base answers, no "Vok") while A still serves "Bora"; the hash after unlearn
equals the hash recorded after the update bitwise (h4 == h2); canary clean; all three
commits under the cap (0.098 / 0.065 / 0.132; ~6 s per training). The VLM side now
covers add/update/unlearn/serve/route/abstain/revoke with the transaction guarantees.

## VLM scene variants: key robustness + expert generalisation (pinned 2026-10-03)

The "different photo of the same scene" proxy for real photos. Scenes: P1 (row
circle/square/triangle), variants of P1 - P1j (all shapes jittered 3-5 px), P1s
(shapes scaled to ~70%), P1bg (same shapes on a light gray background) - and P2 (row
triangle/circle/square). Readings: CLIP sims P1 vs each variant and vs P2; the pinned
tau rule: tau_v = midpoint between the maximum cross-scene similarity (P1-P2) and the
minimum own-variant similarity; if min own-variant <= max cross, the variant set has no
separating tau (reported, and the run keeps tau 0.97). VLM: one expert trained on P1
("Tira"); serving through the router with tau_v: P1 must serve "Tira"; each variant
records whether it routes to P1 and whether the expert answers "Tira" (generalisation
is measured, not predicted); P2 must abstain; canary clean. Predictions: jitter/scale/
bg sims stay high enough for a separating tau_v; routing works for P1 and P2 abstains;
the expert's answer generalisation to variants is the open reading.

## VLM scene variants results (2026-10-03)

All six checks pass. CLIP sims: P1j 0.9697, P1bg 0.9534, P1s 0.9453, cross-scene P2
0.9398; the pinned rule separates (min own 0.9453 > max cross 0.9398) with tau_v =
0.9425 - a 0.0055 margin, flagged. Serving: P1 -> "Tira"; all three variants route to
the P1 expert and the served answer is "Tira"; P2 abstains; canary clean. **Caveat,
recorded:** the variant answers cannot be attributed to visual generalisation because
the v2 leak already showed the expert answers "Tira" to the probe text on a different
panel - the answers are consistent with text memorisation; the load-bearing mechanism
here is the router (variants match the key, cross-scene does not). A visual
generalisation claim needs an expert trained against contrast pairs (same question,
other scenes -> different/refusing answers), pinned as the next step.

## VLM contrast-pair expert: visual grounding (pinned 2026-10-03)

The v2 leak (the delta answers "Tira" to the probe text on any panel) means the expert
is text-memorising, so the variant answers could not be attributed to vision. Fix to be
tested: **contrast pairs** - train the P1 expert on (P1 teach), (P1 probe -> "Tira"),
and negatives (P2 probe -> "Bilmiyorum.", P3 probe -> "Bilmiyorum.", P3 = the column
scene), same KL anchor and the same commit path (KLD cap via propose_and_commit with a
custom trainer). Readings: direct expert probes (delta active, no routing) on P1
(expect "Tira"), P2 and P3 (expect no "Tira" - the leak test), the three P1 variants
jitter/scale/bg (expect "Tira" - now attributable to vision because the same probe text
yields different answers by image), the routed serve (P1 -> Tira, P2 abstains), the
committed KLD and the canary. Predictions: P1 Tira; P2/P3 no Tira; variants Tira;
routed serve correct; KLD under cap; canary clean. If the variants refuse, that is a
generalisation failure of the pixel-level delta and is recorded as such.

## VLM contrast-pair results (2026-10-03)

**All nine checks pass.** The contrast-trained expert (KLD 0.018, 12 s) answers "Tira"
on P1 and its jitter/scale/background variants, and **"Bilmiyorum." on P2 and P3 with
the same probe text** - the v2 text leak is closed and the variant answers are now
attributable to vision (same question, different answer by image). Routed serving: P1
-> "Tira", P2 abstains (sim 0.94 < tau); canary clean; under cap. This is the first
image-conditioned VLM delta: negative scenes in training turn a text-memorising expert
into a visually gated one, and it still generalises to unseen variants of its scene.
