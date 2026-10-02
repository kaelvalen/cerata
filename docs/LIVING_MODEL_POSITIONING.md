# The living model: transactional, auditable learning during interaction - positioning

Status: **brainstorm output, 2026-10-02; step-0 originality audit appended (Appendix
A).** This note seeds the program that `experiments/sandbox/live_learning/` currently
measures; it is not a pre-registration yet.

## 1. The claim

> A deployed model should be a **transactional state**: every interaction can write to
> it, every write is auditable, and any write can be revoked so that the state returns
> to exactly where it was. Memory, weights and routing are one ledger.

One sentence to defend: **ACID for model memory** - atomic, consistent, isolated,
durable edits during interaction, fast enough to happen inside a conversation, and
provable enough to survive an audit.

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
mirror.

## 7. Falsifiable predictions

- Memory-only baselines cannot pass G2 with weights untouched; a delta layer can, and
  the measured gap is the contribution, not the accuracy.
- Without G3's router invariant, revoke either fails or is a router trick (SEUF/GRIP
  predict this); the invariant makes the difference measurable.
- A controller over promotion beats both "always promote" (interference) and "never
  promote" (no durability) on the three-timescale metric set.
- If none of the above separates from the baselines, the thesis is wrong and the
  mesa results say so.

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
| **this program** | yes | yes (bitwise on the arithmetic path, tolerance reported) | yes (invariant, not a fix) | yes (hash) | yes (per answer, per fact) |

### A.2 The sharpened gap

No published system provides **G2 together with G5 under a measured G3**, across
within-conversation, cross-session and task-level time scales in one live state. The
closest three, precisely: **GRIP** treats router integrity as a *fix* for
parameter-based unlearning and measures routing stability, but claims no state
identity and no provenance; **Separable Expert** makes deletion deterministic by
*architecture* (user data never in shared weights) and verifies a behavioral return
to baseline, but that is artifact deletion, not exact revocation inside a shared,
adapting state; **SEAL/Titans** adapt the model's own state at test time without any
revocation or audit. The claim is therefore not "a better memory"; it is **a state
with guarantees** - and the guarantees are the contribution.

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

## Appendix B: slice-1 measured guarantees (2026-10-02)

System: frozen Qwen2.5-1.5B + per-fact LoRA experts (independent, from the base) +
a parameter-free MiniLM router + the transaction ledger. Instrument:
`experiments/sandbox/live_learning/`; every number below is from a pinned run
(`docs/LIVING_MODEL_SLICE_PREREG.md`).

| guarantee | measurement | result |
| :-- | :-- | :-- |
| G1 atomic | strong candidate (KLD 21.62) refused at cap 2.0; normal candidate (0.01) commits | pass |
| G2 state identity | revoke(last) restores the recorded pre-add hash, bitwise over the tensor set | exact |
| G2 behaviour | revoked fact gone; every other fact intact (17/17, 18/18 runs) | pass |
| G3 isolation | per-expert canary KLD 0.93 with the KL-anchored deltas (was 10.96) | bounded |
| G3 routing | paraphrase margin positive only with a dedicated encoder: own 0.742 vs other 0.569; recall 5/6; abstention 4/4 | pass (one miss) |
| G4 durability | state is a pure function of the delta set; same hash across sessions | exact |
| G5 provenance | leave-one-out attribution 0.94 (one miss) | pass (one miss) |
| cost | add ~3-5 s/fact; revoke < 0.1 s; routed answers ~1 s | reported |

The composition result that motivates the organ: independently trained deltas summed
into one adapter destroy each other (0/18); the same deltas survive when routed (18/18).
The MoE is therefore a measured requirement of this state, not a design preference.
