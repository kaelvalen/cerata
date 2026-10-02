# The living model: transactional, auditable learning during interaction - positioning

Status: **brainstorm output, 2026-10-02; not a pre-registration.** This note seeds the
program that `experiments/sandbox/live_learning/` currently measures. Exact references
are to be verified in the originality audit (step 0); the neighbor list below is from
a first sweep, not a finished review.

## 1. The claim

> A deployed model should be a **transactional state**: every interaction can write to
> it, every write is auditable, and any write can be revoked so that the state returns
> to exactly where it was. Memory, weights and routing are one ledger.

One sentence to defend: **ACID for model memory** - atomic, consistent, isolated,
durable edits during interaction, fast enough to happen inside a conversation, and
provable enough to survive an audit.

## 2. Why this is not already in the literature (first sweep)

Three lines exist and each owns a piece; none owns the state.

| line | exemplars (verify in audit) | what it has | what it lacks |
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
