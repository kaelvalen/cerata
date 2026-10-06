# The living model: transactional, auditable learning during interaction - positioning

Status: **brainstorm output, 2026-10-02; step-0 originality audit (Appendix A) and the
measured program so far (Appendix B, 2026-10-03).** This note seeds the program that
`experiments/sandbox/live_learning/` (VLM scripts under `vlm/`) measures.

## 1. The claim

> A deployed model should be a **transactional state**: every interaction can write to
> it, every write is auditable, and any write can be revoked so that the state returns
> to exactly where it was. Memory, weights and routing are one ledger.

One sentence to defend: **ACID for model memory** - atomic, consistent, isolated,
durable edits during interaction, fast enough to happen inside a conversation, and
provable enough to survive an audit. The acronym is a metaphor for the guarantee set
below, not DB-ACID: our "isolation" is an interference budget (locality), not
concurrent-transaction isolation, and G1-G5 are defined in Section 3, not imported
from databases.

## 2. Why this is not already in the literature (first sweep)

Three lines exist and each owns a piece; none owns the state.

| line | exemplars (audited: Appendix A) | what it has | what it lacks |
| :-- | :-- | :-- | :-- |
| memory / retrieval | MemGPT/Letta, LongMemEval-scale systems, sleep-time consolidation | instant writes, cross-session persistence | weights are untouched; no interference budget; unlearning = delete a row |
| parameter adaptation | LoRA personalization (PAC-Bayes Meta-LoRA), LoRA memory laws, self-edits (SEAL-like), fast weights (Titans-like) | durable, sometimes fast adaptation | no exact revoke, no provenance, no measured isolation |
| MoE / continual | MoSEs sub-experts, upcycling lines | capacity across domains, routing | router games; unlearning concentrates on experts |
| unlearning | SEUF (ACL 2025), GRIP (2026), FIT to Forget (2026), CURaTE (2026), separable-expert proxies (2026) | sequential deletion, router constraints, delete-by-artifact | "forget quality" replaces state identity; no hash-level reversibility; no three-timescale ledger |
| this program | - | the combination below | - |

The absent thing is a **system property**, not a block: *revoke(id), and the state
hash equals the pre-promote hash* - plus a per-answer provenance trail and a measured
interference budget, across within-conversation, cross-session and task-level time
scales.

The per-fact LoRA + retrieval-router **architecture is prior art** (MELO, GRACE, WISE,
LoraRetriever, T-Patcher - Appendix A). This program does not claim it; it claims the
transaction ledger, provenance, and the measured guarantee set on that architecture,
benchmarked against those systems on the same facts (Section 10).

## 3. The system

```text
base model (frozen, hash H0)
  + episodic store E      instant writes, ids, deletable rows
  + delta store D         signed low-rank updates, each with its inverse
  + expert organ M        deltas live in an expert bank; the router is an invariant
  + controller C          what to promote, where, how much (rules -> learned)
  + ledger L              append-only: {id, kind, provenance, hash, parent state}
                          state = H(H0, L);  snapshot / rollback / audit / gc
```

Interface: `observe / recall / promote / revoke / snapshot / audit / gc`.

Guarantees to claim and measure (the actual contribution):

- **G1 atomic** - a write applies fully or not at all; a guard violation rolls back.
- **G2 reversible** - `revoke(id)` restores the pre-`promote(id)` state hash
  (bitwise on the arithmetic path; a stated tolerance for SGD deltas).
- **G3 isolated** - a promotion moves old recall by <= epsilon and the router
  distribution on a canary set by <= delta (the SEUF/GRIP failure mode, as an
  invariant).
- **G4 durable** - the same hash across sessions; the ledger replays to the same state.
- **G5 auditable** - per answer: memory / which delta / base, and per fact: where it
  lives, and one command to remove it and prove it.

## 4. Borrowed and new

Borrowed: signed adapters and task arithmetic, MoE routing, episodic memory,
unlearning objectives, the sandbox bench. New: the transaction/ledger abstraction for
a deployed model, the state-identity guarantee (G2), the router-integrity invariant
(G3 measured), provenance as a first-class output (G5), and one controller across the
three time scales. The delta is the guarantee set and its measurement, not a new
block. This is the LLM continuation of the CERATA guards (locality, reversibility,
order invariance, purity) - the same values, a harder substrate.

## 5. Feasibility (8 GB)

Prototype on Qwen2.5-0.5B/1.5B with LoRA deltas; the mesa already runs context /
memory / LoRA mechanisms end to end on this card. Exact arithmetic lives on the
linear paths (heads, fast-weight state, low-rank deltas); full-transformer bitwise
revocation is explicitly out of scope and must be stated as a limit. The GPU is
shared with the Counterpart project, so runs are small-model first.

## 6. The smallest slice (2 weeks)

1. Ledger + transactional `promote`/`revoke` over the mesa: one LoRA delta per fact,
   its inverse stored; `revoke` subtracts and then verifies.
2. A measured "revoke = exact" table: state hash, canary KL, and behavior deltas,
   bitwise where true, tolerance where SGD.
3. Router-integrity and interference-budget measurements on canary sets.
4. One external anchor (LongMemEval/TOFU-class) next to the mesa metrics.
5. Deliverables: the guarantee table, the positioning note with verified references,
   and the falsifiable predictions below.

Then, in order: controller v1 (rules) -> v2 (learned), the expert organ, the VLM
mirror, and the confirmatory scale study (Section 10).

## 7. Falsifiable predictions (revised 2026-10-03)

- Revocation itself is not the discriminator: a row delete is trivially exact with the
  weights untouched. The measured claim is the **delta's existence proof** - at least
  one task family where per-fact deltas beat retrieval at equal cost:
  image-conditioned recognition (the VLM contrast result is the first), behaviour or
  format changes that text cannot store, and paraphrase cases retrieval misses.
- Memory-only passes exact revocation trivially; the delta layer must pass
  **behavioural** return (canary KL and retain-set accuracy after revoke) and pay for
  it in interference - the budget is the measurement.
- Without a router invariant, revoke is a router trick (SEUF/GRIP predict this); with
  it, locality and abstention are measurable at scale (the confirmatory study).
- A controller over promotion beats always-promote (interference) and never-promote
  (no durability) on the three-timescale metric set, at N where it is statistically
  separable.
- If the router and the guarantees do not survive N = 1000, that is the paper: the
  scaling failure is the result.

## 8. Risks

- **Scale**: exactness shrinks as the adapted surface grows; claim only what is
  measured.
- **Grand-system vaporware**: build the smallest slice that demonstrates the
  guarantees; the framing carries the ambition, not the size.
- **Router games**: any MoE must treat routing as an invariant with a canary, or G2/G3
  become unfalsifiable.
- **Judge noise**: capability/quality probes need a second scorer before any claim.

## 9. Open decisions (pin before the slice's pre-registration)

- the delta form on the arithmetic path (closed-form statistics vs trained low-rank);
- the revoke tolerance for SGD deltas and how it is reported;
- which external anchor is the pair for the mesa;
- the controller's first signal set (recurrence, confidence, interference risk);
- the VLM mirror's trigger (after the LLM slice passes, not before).

## 10. The confirmatory study (2026-10-03)

The slice record is an exploratory sequence; the single confirmatory run it points to
is pinned in `docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`: N in {50, 200, 1000} facts
(nonce generator + CounterFact/zsRE subsets) with fresh held-out paraphrases and
distractors; arms memory-only RAG, shared-LoRA sequential, summed deltas, GRACE, MELO,
WISE, and this system; metrics with bootstrap CIs (efficacy, paraphrase
generalisation, locality, router precision/recall/abstention, behavioural return after
revoke, time, storage); primary claim - the router and the guarantees survive N = 1000
- and the delta>memory existence proof on a task family retrieval cannot serve
(image-conditioned recognition first).

## Appendix A: originality audit (step 0), 2026-10-02

Verified this pass against the sources: Titans (2501.00663, NeurIPS 2025), SEAL
(2506.10943), SEUF (ACL 2025 long), GRIP (2601.16905), FIT to Forget (2601.21682),
CURaTE (2604.14644, ACL Findings 2026), Separable Expert Architecture (2604.21571),
MoSEs (2511.06237), Meta-LoRA personalization (2608.12389), How LoRA Remembers
(2605.30260), Storage Is Not Strategy (2609.37858), ALTER (2603.01792), MCU MLLM
unlearning (2608.04548). Still open: sleep-time compute reference, LaMP, LongMemEval /
LoCoMo ids, MUSE, TOFU / WMDP (bundle check in the slice's prereg).

### A.1 Guarantee matrix

G1 atomicity/rollback, G2 state-identity revocation, G3 routing/interference
invariant measured, G4 durable cross-session state identity, G5 per-answer provenance.
(partial = only under a restricted setting or only measured, not guaranteed)

| work | G1 | G2 | G3 | G4 | G5 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| MemGPT/Letta-class memory | - | partial (row delete, weights untouched) | - | yes | partial (retrieval) |
| LoRA personalization (Meta-LoRA) | - | - | - | yes | - |
| SEAL (self-edits, RL) | - | - | - | yes | - |
| Titans (test-time neural memory) | - | - | - | partial (weight decay forgetting) | - |
| MoSEs (MoE + PEFT continual) | - | - | partial (router stability) | yes | - |
| SEUF (MoE unlearning) | - | - | partial (anchor loss on the router) | yes | - |
| GRIP (MoE router constraints) | - | - | yes (routing stability measured) | yes | - |
| FIT to Forget (continual unlearning) | - | - | partial (utility metrics) | yes | - |
| CURaTE (refusal gate) | - | partial (no weights changed) | - | yes | - |
| Separable experts (deletable proxies) | - | partial (artifact deletion; behavioral return, KL ~ 0.21 nats, no bitwise identity) | partial (cross-user contamination) | yes | - |
| MELO (neuron-indexed dynamic LoRA, AAAI 2024) | - | partial (block deletion = artifact deletion) | partial (locality reported) | yes | - |
| GRACE (key-value adaptors + deferral radius) | - | partial (artifact deletion) | partial (deferral radius ~ our tau) | yes | - |
| WISE (side memory + router) | - | partial (artifact deletion) | partial (router measured) | yes | - |
| LoraRetriever / T-Patcher | - | partial | - | yes | - |
| **this program** | yes | partial (artifact deletion; behavioural return measured) | yes (invariant, not a fix) | yes (hash) | yes (per answer, per fact) |

### A.2 The sharpened gap (revised 2026-10-03)

The earlier version of this section drew the line at "artifact deletion vs exact
revocation inside a shared, adapting state". That line does not survive: the current
system is itself artifact deletion - the base is frozen, and because independently
trained deltas interfere when summed, each query activates exactly one delta, so
revoking a fact removes an adapter. Either the program adds a genuinely shared,
exactly-revocable layer (closed-form statistics on the linear paths), or the claim is
the honest, narrower one:

**artifact deletion + provenance + a measured router invariant + measured behavioural
return**, with a delta>memory existence proof on a task family retrieval cannot serve
(image-conditioned recognition first). The closest systems (MELO, GRACE, WISE) share
the architecture; none reports the transaction ledger, the provenance trail, or the
guarantee table, and none is benchmarked on the same facts at N = 1000. The claim is
therefore not "a better memory" and not "exact revocation in a shared state"; it is
**a state with an audited edit protocol**, and the protocol is the contribution.

### A.3 Consequences for the slice

- The mesa's metric set grows: state-hash identity (G2), canary router KL (G3),
  provenance accuracy (G5), beside the existing recall/latency/interference columns;
  SEUF/GRIP-style routing stability and FIT-style Forget/Retain metrics go in for
  comparability.
- The exactness frontier must be published, not hidden: bitwise where the arithmetic
  path applies (closed-form contributions, signed deltas on linear paths), stated
  tolerance where SGD deltas are revoked; no full-transformer bitwise claim.
- Positioning sentence for the paper: *we do not claim a better memory; we claim a
  model state whose edits are transactional and auditable, and we measure the
  guarantees the literature currently assumes.*
- If the slice cannot separate G2/G5 from the closest three on the mesa and one
  external anchor, the thesis is falsified and the mesa record says so (Section 7).

## Appendix B: measured guarantees (2026-10-03)

System: frozen Qwen2.5-1.5B (text) and Qwen3-VL-2B (vision) + per-fact LoRA deltas
from the same frozen base + a parameter-free semantic router (MiniLM for text, CLIP
for images) + the transaction ledger. Instrument:
`experiments/sandbox/live_learning/` (VLM scripts under `vlm/`); every number is from
a pinned run (`docs/LIVING_MODEL_SLICE_PREREG.md`; the index at its end maps themes to
commits).

### B.0 How to read this

Hash identity is a bookkeeping invariant of the design, not a measured property: the
state hash is a pure function of the delta set, so "revoke restores the hash" cannot
fail unless there is a bug. The load-bearing measurements are behavioural - canary KL
and retain accuracy after revoke, router precision/recall/abstention, and the
delta>memory existence proof. Rows below are annotated; the confirmatory study
(`docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`) turns them into CIs at N up to 1000.

### B.1 Text side

| guarantee | measurement | result |
| :-- | :-- | :-- |
| G1 atomic | a strong candidate (KLD 21.62) refused at cap 2.0; an injected training exception and bad input leave the state hash and delta set unchanged | pass |
| G2 state identity | revoke restores the recorded pre-add hash bitwise | bookkeeping (by construction) |
| G2 behaviour | after revoke the revoked token is gone, neighbours/canaries intact (17/17, 18/18 runs; the update retrain leaves no residue) | pass |
| G3 isolation | per-expert canary KLD 0.93 with KL-anchored deltas (was 10.96); paraphrase margin positive only with MiniLM (own 0.742 vs other 0.569); held-out paraphrase augmentation routing 6/6, recall 6/6; abstention 4/4 | bounded, pass |
| G4 durability | the state hash is a pure function of the delta set; cross-session replay reproduces it | exact |
| G5 provenance | leave-one-out attribution 1.00 with the discriminative v2 token | pass |
| organ invariant | revoke/add move 0 remaining routing decisions (the SEUF/GRIP failure mode measured as absent) | pass |
| composition | independently trained deltas summed destroy each other (0/18); the same deltas survive routed (18/18) - the MoE is a measured requirement | measured |
| controller | rule 6 experts / 16-18; eps bandit 0 / 15-18; UCB decide-before-serve 17 / 17-18; **repair (serve-first, UCB on misses) 3 experts / 18-18**; cross-session hybrid v2: 2 experts, s2 17/17, update and unlearn honoured | pass |
| cost | add ~3-5 s/fact; revoke < 0.1 s; routed answers ~1 s | reported |

### B.2 VLM side (shared 8 GB)

| guarantee | measurement | result |
| :-- | :-- | :-- |
| mirror | the base sees and does not know a nonce; a 12-step KL-anchored delta teaches an image-keyed fact; zeroing revokes exactly | pass |
| router | CLIP keys beat DINOv2 CLS/mean/concat on margin; v3 all seven checks: route 2/2, served 2/2, distractor refusal, revoke abstention | pass |
| G1 atomic | KLD-capped commits: adds 0.098 / 0.191; an adversarial text-only candidate at 23.928 refused with the hash unchanged | pass |
| G2 state identity | revoke restores the recorded pre-add hash bitwise in a multi-fact state | bookkeeping (see B.0) |
| G4 durability | save -> reload reproduces the hash; the reloaded session serves both facts | exact |
| update/unlearn | update (revoke + retrain) commits (0.065) and serves the new code; unlearn is silent; the hash after unlearn equals the post-update hash | pass |
| controller | serve -> miss -> UCB repair: experts 4, final readout 4/4, route choices correct | pass |
| variants | jitter/scale/background variants route and serve; cross-scene abstains (tau_v 0.9425; margin 0.0055, flagged) | pass |
| contrast | with negative scenes the expert is image-conditioned: "Tira" on P1 and its variants, "Bilmiyorum." on P2/P3 with the same probe text; KLD 0.018 | pass |

### B.3 Open items

- Text: future-aware repair reward (recurrence signal); capability cap3 is base-bound.
- VLM: real-photo key calibration (synthetic panel margins are 0.0055-0.06); the
  pixel-level expert generalises to synthetic variants but has not seen real scenes.

### B.4 Review round (2026-10-04): what the N=1000 table does not yet prove

- The delta > RAG gap is measured against plain-prompt RAG (0.834 at N=1000) and now
  also against strong-RAG arms: the explicit-instruction arm reaches 0.837 at N=1000
  (plain 0.834; ours 0.999, CI-separated), Q/A formatting and few-shot hurt (0.71 and
  0.52 at N=200). The gate passed - the editing claim stands at N=1000.
- The router is effectively a **subject-string dictionary** (substring gate + semantic
  pick); route 0.997 / abstention 1.00 are near-tautological, and aliases, typos and
  pronouns are untested. The CounterFact paraphrase score (~0.32) is a **router
  failure**, not delta generalisation.
- The GRACE/WISE/MELO columns are **approximations** (logit bias / side-memory rule /
  uncapped self), not official implementations; EasyEdit baselines are pinned.
- The prereg's own thresholds (canary KL <= 0.05, retain +/-1, provenance >= 0.95,
  router precision >= 0.95, revoke n >= 100) were **not reported** in the N=1000 table.
- Paper identity is unresolved (editing vs systems/guarantee); see
  `docs/LIVING_MODEL_REVIEW_2026-10-04.md` for the pinned order and the gate.
