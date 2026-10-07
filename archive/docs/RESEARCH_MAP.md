# Research Map - CERATA (formerly PAL-MoE) components and their literature lineage

Which research line each part of the codebase comes from, what is
implemented vs adapted vs absent, and where the evidence lives. Sections 1-4
map the v1 components (now in `cerata/legacy/`); sections 5-6 map the prior
work closest to the Stage 1 results and to the v3 paths. Companion to
`v1/BENCHMARK.md` (protocol/design facts) and `v1/EXPERIMENT_PLAN.md` (paper plan).
Bibliographic details must be verified against the original papers before
they enter a bibliography (see the checklist in `v1/EXPERIMENT_PLAN.md` section 7).

## 1. Continual-learning foundations

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Regularization | EWC (Kirkpatrick et al., 2017) | `cerata/legacy/baselines/ewc.py` (multi-task + online Fisher) | v1 MNIST/CIFAR tables |
| Regularization | SI (Zenke et al., 2017) | not implemented | - |
| Distillation | LwF (Li & Hoiem, 2017) | `--lambda_lwf` in `legacy/adaptation/ttt.py` (opt-in) | design facts 11-13 |
| Gradient projection | GEM / A-GEM (Lopez-Paz 2017; Chaudhry 2019) | `cerata/legacy/baselines/agem.py` | v1 tables |
| Replay | ER (Chaudhry 2019 / Rolnick 2019) | `cerata/legacy/baselines/replay.py` + `legacy/baselines/buffer.py` | v1 tables |
| Replay + distillation | DER / DER++ (Buzzega et al., 2020) | `cerata/legacy/baselines/der.py` (`DERPP`) | v1 tables |
| Asymmetric replay | ER-ACE (Caccia et al., 2022) | `cerata/legacy/baselines/der.py` (`ERACE`) | v1 tables |
| Interference-based selection | MIR (Aljundi et al., 2019) | `cerata/legacy/baselines/mir.py` | wave-2 runs |
| Prototypes / NCM | iCaRL (Rebuffi et al., 2017) | `cerata/legacy/baselines/icarl.py`; prototype anchors `v_p`, `o_p` in `legacy/memory/prototype_memory.py` | v1 tables, E8 |
| Latent replay | Latent Replay (Pellegrini et al., 2020) | `cerata/legacy/baselines/latent_replay.py`; latent exemplars `x_p`; design fact 19 | E5, E4 |
| Coreset selection | k-center / uncertainty | `--proto_selection`, `--proto_candidate_pool` | toolbox tests |

## 2. Capacity expansion and parameter isolation

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Progressive networks | PNN (Rusu et al., 2016) | function-preserving clone in `cerata/legacy/models/expert.py` (`clone_function_preserving`, `widen`) | tests; `--expansion_action widen` |
| Expert selection | Expert Gate (Aljundi et al., 2017) | `cerata/legacy/trigger/` + `cerata/legacy/builder/expert_builder.py` (validation gate) | E7 (gated vs forced) |
| Parameter isolation | PackNet (Mallya & Lazebnik, 2018), HAT (Serra et al., 2018) | expert freezing + exact routing lock (`freeze_historical_experts`, zero-weight-decay router group) | design fact 15 |
| Merging | model soup / TIES / task arithmetic | `cerata/legacy/merge.py`, `experiments/stage1/infra/merge_experts.py` | toolbox tests |

## 3. Mixture-of-Experts in continual learning

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Fixed sparse MoE | Switch-style load balancing | `experiments/stage1/infra/run_benchmark.py::_run_standard_moe` ("Standard MoE") | v1 baselines |
| MoE theory in CL | theory of MoE in continual learning (2024) | motivation for the allocation/reuse question (H5); no theory implementation | E7 result (no reuse) |
| Adaptive expert expansion | adaptive / Incremental MoE (2025), MoE-Adapters++ (2025) | dynamic allocation via trigger + gate; direct comparison is future work | E7; `v1/EXPERIMENT_PLAN.md §7` |
| Routing stability | router anchoring / distillation | `--router_anchor_steps`, `--router_anchor_margin`, prototype-owner distillation | design facts 11, 15; E5 |

## 4. Evaluation and protocol

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Benchmark framework | Mammoth (Boschini et al., 2022) | protocol inspiration; all methods re-implemented in one codebase | `experiments/stage1/infra/run_benchmark.py` |
| Equal-byte fairness | equal-byte non-inferiority protocol (2026) | `memory_bytes`/`state_bytes`/`stored_bytes` in every result; `--buffer_size`/`--icarl_k`; E4 sweep | `results/equalbyte*` |
| Online / task-free CL | MOSE (Yan et al., CVPR 2024) | `cerata/eval/task_free.py`, `--task_free_eval`, `--trigger energy` | domain-shift pilot |
| Foundation backbones | frozen ImageNet ResNet-18 / ViT-B/16 | `cerata/legacy/models/encoder.py`, `--encoder_weights imagenet`, feature cache | ViT promotion (fact 18) |

## 5. Pretrained-model continual learning (closest to Stage 1 and E-TID2)

The Stage 1 headline - a training-free or closed-form readout on frozen pretrained
features beats the trained expert bank - is a known result in this line. The
programme's contribution is the measured decomposition (selection vs capacity vs
isolation, P2-BOUND), not the readout.

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| NCM on frozen pretrained features | Janson et al., "A Simple Baseline that Questions the Use of Pretrained-Models in Continual Learning" (NeurIPS 2022 workshop) | `L0_ncm` in `cerata/experts/ladder.py` | S2: 70.34 vs v1 59.30 |
| Prototype classifier + first-session adaptation | SimpleCIL / APER (Zhou et al., "Revisiting Class-Incremental Learning with Pre-Trained Models", IJCV 2024) | `L0_ncm`; no first-session adaptation arm | S2 |
| Random projection + continual Gram-matrix ridge | RanPAC (McDonnell et al., NeurIPS 2023): 92.2 % on CIFAR-100 with PETL | `core/random_features.py` + `LinearStats(feature_map=...)` / `Cerata(random_features=M)`, with the penalty pinned after task 0 (RanPAC re-selects it per task) | built; `PTM_CIL_PREREG.md` (not run) |
| Analytic (closed-form) CL: recursive least squares equals joint training | ACIL (Zhuang et al., NeurIPS 2022), DS-AL (AAAI 2024), G-ACIL / GACL (2024) | the MEDIUM path's continual ridge is this statement; E-TID2's G3 guard is its check | E-TID2 G3 |
| Analytic readout with feature adaptation | AnaCP (NeurIPS 2025 spotlight) | not implemented; the strongest analytic rival for a readout arm | - |
| Exact, gradient-free continual unlearning | ACU, "Analytic Continual Unlearning" (Zhuang group, 2025) | `forget` as a downdate of the statistics is this idea; our addition is measuring it as a guard | vision guards |
| Second-order class statistics on frozen features | FeCAM (NeurIPS 2023), LayUP (2023/24) | not implemented | - |
| CIL = within-task x task-id prediction; task-id ~ OOD detection | Kim et al., "A Theoretical Study on Solving Continual Learning" (NeurIPS 2022) | the routing-tax framing of Stage 1; P2 measures what a bank does given a task-id predictor | Stage 1, P2-BOUND |
| Per-task adapters + prototype complement | EASE (Zhou et al., "Expandable Subspace Ensemble", CVPR 2024) | the L3 bank: rank-8 residual adapters per task; an external bank in `PTM_CIL_PREREG.md` | S2, S11, P2-BOUND |
| Task adapters + merging + training-free retrieval | MOS, "Model Surgery" (Sun et al., AAAI 2025) | an external bank in `PTM_CIL_PREREG.md` | - |
| Mixture of task-specific experts with expert filtering | MoTE (Li et al., Knowledge-Based Systems 2025); Adaptive Expert Forest (2026); Bi-Level Routing MoE (2026); SEMA, self-expansion with mixture of adapters (CVPR 2025) | the MoE-for-PTM-CIL line the P2 decomposition is aimed at; MoTE is an external bank | - |
| Slow learner + classifier alignment | SLCA / SLCA++ (ICCV 2023 / 2024) | not implemented | - |
| Prompt selection by query-key matching | L2P (Wang et al., CVPR 2022), DualPrompt (ECCV 2022), CODA-Prompt (Smith et al., CVPR 2023) | not implemented; their key matching is the same selection problem as the router | - |

**What is and is not new here.** Continual closed-form ridge (ACIL, RanPAC), exact
closed-form forgetting (ACU) and task-specific expert banks (EASE, MOS, MoTE) are all
published. Not published, to the extent of the 2026-09-26 search: a per-sample
decomposition of what *any* bank adds once routing is analytic (rescue `rho`,
breakage `beta`, mass `m`), applied across published banks and benchmarks. That is
`PTM_CIL_PREREG.md`. Known protocol gap: every vision number in this repository so far
uses an ImageNet-1K ViT-B/16; the literature uses ImageNet-21K weights.

## 6. Knowledge editing (the v3 LM paths)

| Literature line | Representative work | In this repo | Evidence |
| :-- | :-- | :-- | :-- |
| Locate-and-edit, rank-one | ROME (Meng et al., NeurIPS 2022) | CounterFact loader and metrics (`cerata/eval/editing.py`) | loaders only |
| Mass editing, closed form with a key-covariance prior | MEMIT (Meng et al., ICLR 2023) | `edit/down_proj.py` (`DownProjEdit`): the MEMIT closed form, stated as such, plus an exact downdate | tiny-model tests only |
| Null-space constrained editing | AlphaEdit (Fang et al., ICLR 2025) | not implemented; the fix to try if the MEDIUM path loses specificity at N = 1000 | - |
| Memory-based editing with a scope decision | SERAC (Mitchell et al., ICML 2022) | the FAST path: retrieval over frozen keys, the `tau` threshold as the scope decision | tiny-model tests only |
| Discrete key-value adaptors, deferral radius | GRACE (Hartvigsen et al., NeurIPS 2023) | the FAST path's closest relative: a codebook at one layer, a radius that decides retrieval | - |
| Side memory with routing, lifelong editing | WISE (Wang et al., NeurIPS 2024); MEMOIR (NeurIPS 2025, +0.13 over WISE at 1000 edits); LEMoE (2024) | the same FAST / MEDIUM split, without the guards | - |
| Sequential editing at scale | UltraEdit (2025), "Lifelong Knowledge Editing requires Better Regularization" (2025) | the MEDIUM path's regime at N = 1000 | - |
| Undoing edits | SoLA, reversible lifelong editing by semantic-routing LoRA (2026); OneEdit rollback (2024); "On Reversibility ... in Parametric Knowledge Editing" (2026: reversal is behavioural, not parametric, for existing methods); reversing in-context edits (NAACL 2025) | `forget`: a parametric downdate, fp-close to the never-edited delta and bitwise when the last edit goes - the measurable difference from this line | tiny-model tests only |
| Learned editors, zsRE evaluation | MEND (Mitchell et al., ICLR 2022) | the zsRE split used by `V3_LLM_PREREG.md` | loaders only |
| Multi-hop edit evaluation | MQuAKE (Zhong et al., EMNLP 2023) | MQuAKE-CF-3k loader | loaders only |
| Anisotropy of contextual representations | Ethayarajh (EMNLP-IJCNLP 2019) | the reason a raw-cosine `tau` on LM hidden states needs calibration (`V3_LLM_PREREG.md`, amendment 2) | - |

**Not claimed as novel for v3:** the closed-form edit (MEMIT), retrieval-based
editing (SERAC / GRACE / WISE / MEMOIR), closed-form continual ridge (ACIL / RanPAC),
exact analytic unlearning (ACU), undoable edits as such (SoLA, OneEdit). **Candidate
contribution, not yet shown:** the guard contract as a system property - every
`write` / `forget` measured for locality, reversibility and order invariance, with a
zero-parameter router - at 7B scale (`V3_LLM_PREREG.md`).

Entries in sections 5 and 6 were located by a web search on 2026-09-26 (arXiv itself
was not reachable from the review container); verify authors, venues and numbers
against the papers before they enter a bibliography. `archive/docs/POSITIONING.md` has the
links.

## 7. What was claimed for v1 (and what is not)

**Not claimed as novel:** prototype/NCM memory, replay, distillation, gating,
function-preserving expansion, routing locks - all individually established.

**Claimed (to be defended by the experiments):**

1. the combination of *dynamically expanded experts + compact latent replay +
   prototype-anchored routing* under a **class-incremental, fixed-memory**
   protocol;
2. **function-space anchoring** of routing and expert behaviour through
   `v_p/r_p/o_p`, including the exact routing lock correction (fact 15);
3. an **equal-byte evaluation protocol** and the resulting trade-off
   (accuracy vs forgetting per stored byte).

**Explicitly refuted by the current evidence (do not claim):**

- "dynamic allocation reuses experts" (H5): gated allocation equals forced
  expansion on CIFAR-100 (20/20 experts);
- "the OOD negative-boundary loss is the key mechanism" in the current recipe:
  it is neutral once router distillation is present (E5, revising fact 1);
- "zero replay" as memory-free: it is *zero raw replay*.

## 8. Where new code should go

| If you add... | Put it in... | Wire it via |
| :-- | :-- | :-- |
| a router | `cerata/router/` | `Cerata(router=...)`; it must pass the purity guard (0 trainable parameters) |
| a retrieval index (ANN) | `cerata/address/` | behind the `ExactCosineIndex` calls; the exact index stays the reference |
| a closed-form edit path | `cerata/edit/` | additive statistics with an exact downdate, float64 |
| a consolidation policy | `cerata/experts/policies.py` | `consolidate(policy, prereg=...)`; gate it on its pre-registration |
| a backbone or LM adapter | `cerata/core/` | content-hashed, frozen |
| a metric or statistic | `cerata/eval/` | the measurement contract (`eval/schema.py`) |
| an experiment | `docs/<NAME>_PREREG.md`, then `experiments/<name>.py` | commit both before the run; results stay in untracked `results/` |
| anything v1 | nowhere: `cerata/legacy/` is frozen bitwise | - |
