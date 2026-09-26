# Documentation map

**The name.** The project and package were renamed from PAL-MoE / `pal_moe` to CERATA /
`cerata` on 2026-09-26 (`POSITIONING.md` section 5). Documents that are still open -
the architecture, the positioning, the unrun pre-registrations, the indices - use the
new names. Records of studies that already ran (their pre-registrations, results,
Stage 1, the contracts, `v1/`) are left as written: their `pal_moe/...` paths name the
code at the commits they cite (`git show <commit>:pal_moe/...`), and the current file
is the same path under `cerata/` (v1 modules under `cerata/legacy/`).

Pre-registrations and results stay at their paths: the results documents cite them by
path and commit hash, and `cerata/experts/policies.py` gates `by_confusion` on the
literal path `docs/P2_BOUND_PREREG.md`. Do not move them.

## Start here

| Document | What it is |
| :-- | :-- |
| [`V3_ARCHITECTURE.md`](V3_ARCHITECTURE.md) | The current architecture: one fixed address space, three time scales, the API, the four guards, and a table tracing every design decision to a result |
| [`DIAGNOSIS_SYNTHESIS.md`](DIAGNOSIS_SYNTHESIS.md) | What Stage 1 and the first two follow-up studies jointly established, frozen before the next pre-registration |
| [`STAGE1_RESULTS.md`](STAGE1_RESULTS.md) | Every Stage 1 result (E0, S2-S11), the consolidated findings and what the evidence does not say |
| [`POSITIONING.md`](POSITIONING.md) | What the literature already owns, what is defensible, and the two-paper plan |
| [`PTM_CIL_PREREG.md`](PTM_CIL_PREREG.md) | Paper A: does any expert bank add anything over an analytic router on the seven standard benchmarks? **Proposed, not run** |
| [`V3_LLM_PREREG.md`](V3_LLM_PREREG.md) | The first LM study (single-fact learning on a frozen 7B). **Proposed, not run** |

## Rules and contracts

| Document | What it is |
| :-- | :-- |
| [`STAGE1_PLAN.md`](STAGE1_PLAN.md) | The stage order, the two experiment rules and the per-stage write-ups |
| [`MEASUREMENT_CONTRACT.md`](MEASUREMENT_CONTRACT.md) | S0: what every run must report (`cerata/eval/schema.py`) |
| [`ARCHITECTURE_CONTRACT.md`](ARCHITECTURE_CONTRACT.md) | S1: the four interfaces and their registries (`cerata/arch/`) |
| [`RESULTS_INVENTORY.md`](RESULTS_INVENTORY.md) | What every `results/` directory contains and its status (up to AC3) |
| [`RESEARCH_MAP.md`](RESEARCH_MAP.md) | The literature line behind each component, and the closest prior work to the current results |

## The study chain after Stage 1

In the order they were run. Each has a pre-registration committed before its runner
(E-TID and E-TID2 are the recorded exceptions) and a results document; the runners are
indexed in [`../experiments/README.md`](../experiments/README.md).

| Study | Pre-registration | Results | Outcome |
| :-- | :-- | :-- | :-- |
| Router ranking | [prereg](ROUTER_RANKING_PREREG.md) | [results](ROUTER_RANKING_RESULTS.md) | A supervised `z -> T` gate ranks worse than prototypes (refuted, opposite direction) |
| Representation x routing | [prereg](REPRESENTATION_ROUTING_PREREG.md) | [results](REPRESENTATION_ROUTING_RESULTS.md) | Neither factor helps alone; the interaction is real but too small |
| Decision routing | [prereg](DECISION_ROUTING_PREREG.md) | [results](DECISION_ROUTING_RESULTS.md) | Comparative supervision worse in every seed, not significant after correction |
| Aggregation | [prereg](AGGREGATION_PREREG.md) | [results](AGGREGATION_RESULTS.md) | Uniform top-3 mixing is significantly worse than winner-take-all |
| Expert formulation (E1) | [prereg](EXPERT_FORMULATION_PREREG.md) | [results](EXPERT_FORMULATION_RESULTS.md) | A shared decision space is neutral to mildly harmful |
| Evidence experts (E2) | [prereg](E2_EVIDENCE_PREREG.md) | [results](E2_EVIDENCE_RESULTS.md) | Not executed: the first veto failed and the protocol was infeasible |
| Coupling | [prereg](COUPLING_PREREG.md) | [results](COUPLING_RESULTS.md) | Re-aligning accumulated evidence projections interferes destructively |
| Interference | [prereg](INTERFERENCE_PREREG.md) | [results](INTERFERENCE_RESULTS.md) | Mechanistic support for new-vs-old projection interference (not causal) |
| Intervention | [prereg](INTERVENTION_PREREG.md) | [results](INTERVENTION_RESULTS.md) | Cutting the non-owner path recovers routing coverage, not accuracy |
| Owner side | [prereg](OWNER_SIDE_PREREG.md) | [results](OWNER_SIDE_RESULTS.md) | The accuracy collapse also has an owner-side causal component |
| AC1 address freeze | [prereg](AC1_ADDRESS_FREEZE_PREREG.md) | [results](AC1_ADDRESS_FREEZE_RESULTS.md) | Freezing the learned shared query collapses routing (-28.7 pp) |
| AC3 address space | [prereg](AC3_ADDRESS_SPACE_PREREG.md) | [results](AC3_ADDRESS_SPACE_RESULTS.md) | Fixed parameter-free retrieval is a drop-in for the learned address (+0.048 pp, TOST-equivalent) |
| E-TID | none (exploratory) | [results](E_TID_RESULTS.md) | The routing tax is not the Bayes error of task identity given `z` |
| E-TID2 | none (exploratory) | [results](E_TID2_RESULTS.md) | A continual ridge router adds +4.1 / +6.1 pp; the bank then adds +0.62 / -0.03 pp |
| P2-BOUND | [prereg](P2_BOUND_PREREG.md) | [results](P2_BOUND_RESULTS.md) | The experts rescue ~30 % of what they could; boundaries move mass, not conversion |

## The v1 record

[`v1/`](v1/README.md) holds the v1 design, its benchmark tables (moved from the
repository README), the v1 protocol and design facts (`v1/BENCHMARK.md`), the v1 paper
plan, the superseded v2 specification, the pre-v3 code review and the Turkish
presentation notes.
