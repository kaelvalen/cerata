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
