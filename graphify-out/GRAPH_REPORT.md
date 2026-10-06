# Graph Report - cerata  (2026-10-06)

## Corpus Check
- 283 files · ~299,617 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 2, .pt 2, .cff 1)

## Summary
- 3628 nodes · 8919 edges · 148 communities (106 shown, 42 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 250 edges (avg confidence: 0.88)
- Token cost: 807,078 input · 0 output

## Community Hubs (Navigation)
- v1 Tests & Prototype Memory
- S1 Experts, Routers & Buffers
- VLM Live-Learning Sandbox
- S1 Backbones & Protocols
- Stage1 Benchmark & Calibration
- Continual Evaluator Metrics
- v1 Benchmark Methodology
- Decision Routing & RR Factorial
- PTM-CIL Random Projection Eval
- Living Model Prereg & Metrics
- Run-Record Schema
- CerataLM & Down-Proj Editing
- Text Live-Learning Harness
- Text Ledger & Bandit
- Router Eval & Checkpoint Diagnosis
- S4/S5 Datasets & Protocols
- Confirm Pilot & External Data
- Split-CIFAR Data
- Positioning & Research Map
- v3 SLOW Path Experts
- Cerata API Facade
- PTM-CIL Benchmarks
- PAL-MoE v2 Spec
- E2 Evidence & Intervention
- Paper Reporting
- Stage1 Ladder Results
- FAST Memory & Address Index
- MEDIUM Linear Stats
- S2 Ladder Hooks
- v1 Attention Router
- E0 Runner
- Guarded Editor
- Router Ranking Runner
- S9 Robustness
- Cached Backbone
- LLM Editing Eval (CounterFact)
- Energy Trigger & Validation Gate
- Architecture & Measurement Contracts
- Cosine Readout
- Ladder Model
- MEDIUM Random Features
- Core Hashing & Backbones
- Aggregation & Task-Free Metrics
- Text Context Mechanisms
- AC1 Address Freeze Runner
- HF Causal LM Wrapper
- E2 & AC1 Preregs
- S10 Scaling
- Feature Cache
- S11 Confirmatory Stats
- Owner-Side & P2 Bound Preregs
- Diagnosis & Decision Routing Docs
- S9 Corruptions
- External Export Tests
- Router Diagnostics & Geometry
- iCaRL Baseline
- Router Ranking & Rep-Routing Preregs
- Identity & Legacy Experts
- Legacy & Prototype Routers
- DER++ Baseline
- AC3 Address Space & README
- Interference & Intervention Preregs
- External Datasets Common
- Coupling Ablation & v3 LM Cost
- E0 Representation Ceiling
- AC3 Address Space Runner
- S8 Budget
- PTM Data Preparation
- Domain Shift Data
- S6 Order
- Config Loading
- P2 Bound & Task Constructions
- PTM-CIL Prereg & Paper A
- External Export Summaries
- v3 Anchor Tests
- Expert Formulation Runner
- S5b Domains
- Prototype Anchor Matrices
- Split-CIFAR-100
- E0 Cosine Head
- S6b Difficulty
- Paired Statistics
- OOD Boundary Loss (TTT)
- EASE Export
- A-GEM Baseline
- Replay Baseline
- Prototype Merge & Prune
- Generative Memory (VAE)
- Legacy Expert Merging
- Recipe: paper_wave1b
- Recipe: paper_wave1c
- CI Pipeline & Contributing
- Literature Update 2026-09
- E-TID Ceiling
- Arch Registry
- Frozen Module Backbone
- Split Folder Data
- E-TID2 Ridge Router
- Feature Tensor Dataset
- Recipe: paper_wave2
- Corruption Kernels
- Multi-Task Loss Weighting
- Latent Replay Generator
- Raw Anchor Refresh
- Task Router Protocol
- Recipe: reservoir_check
- Recipe: paper_wave1d
- AUROC Utilities
- Latency Measurement
- Timm Shim
- Recipe: cifar100_full
- Recipe: gate_ablation
- Recipe: resnet18_frozen
- Recipe: resnet18_multiseed
- Recipe: cifar10_frozen
- Recipe: memory_pareto
- Recipe: mnist_domainshift
- Recipe: multiseed_cifar
- Recipe: paper_all
- Recipe: paper_all_resume
- Recipe: readout_ablation
- Recipe: vit_multiseed
- Recipe: vit_quick
- E0 Accept Head
- Shifted CIFAR-100
- Equal-Byte Memory Accounting
- pal_moe Package Init
- Recipe: paper_status
- Class-Incremental Protocol
- Anchor Refresh
- Capacity Sweep
- Package Metadata

## God Nodes (most connected - your core abstractions)
1. `PrototypeMemory` - 113 edges
2. `SharedEncoder` - 99 edges
3. `MLPExpert` - 92 edges
4. `DynamicMoE` - 87 edges
5. `DynamicRouter` - 74 edges
6. `DeltaStore` - 52 edges
7. `LinearStats` - 48 edges
8. `ExpertBuilder` - 48 edges
9. `Cerata` - 46 edges
10. `digest()` - 44 edges

## Surprising Connections (you probably didn't know these)
- `FastMemory` --implements--> `FAST Path: append-only KV memory`  [INFERRED]
  cerata/memory/kv_store.py → docs/V3_ARCHITECTURE.md
- `Lesson: Call Existing Entry Point, Don't Re-derive Loop` --references--> `run_level()`  [EXTRACTED]
  docs/STAGE1_PLAN.md → experiments/stage1/s2_ladder.py
- `test_mask_unseen_hides_future_classes()` --calls--> `mask_unseen()`  [EXTRACTED]
  tests/test_arch.py → cerata/arch/readouts.py
- `lm()` --uses--> `HFCausalLM`  [INFERRED]
  tests/test_v3_lm.py → cerata/core/hf_lm.py
- `pinned_readouts()` --uses--> `RandomProjection`  [INFERRED]
  experiments/ptm/ptm_headroom.py → cerata/core/random_features.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Tests of the pointwise winner-take-all routing invariant** — docs_diagnosis_synthesis_pointwise_wta_invariant, docs_decision_routing_prereg_comparative_supervision, docs_aggregation_prereg_wta_vs_uniform_top3, docs_expert_formulation_prereg_e1_shared_decision_space, docs_e2_evidence_prereg_evidence_producing_experts [EXTRACTED 1.00]
- **v2 address programme: coupling -> AC1 -> AC3 -> stable keys** — docs_coupling_results_destructive_realignment, docs_ac1_address_freeze_results_address_not_decomposable, docs_ac3_address_space_results_fixed_address_drop_in, docs_ac1_address_freeze_results_stable_keys_plastic_reader [EXTRACTED 1.00]
- **v3 three time scales on a fixed address space** — readme_fast_memory_path, readme_medium_closed_form_edit, readme_slow_consolidation, readme_four_guards, readme_cerata_api_facade [EXTRACTED 1.00]
- **Non-owner interference: localisation -> mechanism -> intervention** — docs_coupling_results, docs_interference_prereg_owner_nonowner_decomposition, docs_interference_prereg_expert_count_ladder, docs_intervention_prereg_owner_only_arm, docs_intervention_results_routing_accuracy_dissociation [EXTRACTED 1.00]
- **Living model guarantee set G1-G5** — docs_living_model_positioning_g1_atomic, docs_living_model_positioning_g2_reversible, docs_living_model_positioning_g3_isolated, docs_living_model_positioning_g4_durable, docs_living_model_positioning_g5_auditable [EXTRACTED 1.00]
- **Confirmatory-study editing baselines** — docs_living_model_positioning_melo, docs_living_model_positioning_grace, docs_living_model_positioning_wise, docs_living_model_confirmatory_prereg_rag_arm, docs_living_model_review_2026_10_04_easyedit [EXTRACTED 1.00]
- **v3 Guard Contract on every write/forget/consolidate** — docs_v3_architecture_guard_locality, docs_v3_architecture_guard_reversibility, docs_v3_architecture_guard_order_invariance, docs_v3_architecture_guard_router_purity, docs_positioning_auditable_learning_contract [EXTRACTED 1.00]
- **Three time scales over one fixed address space** — docs_v3_architecture_fixed_address_space, docs_v3_architecture_fast_path, docs_v3_architecture_medium_path, docs_v3_architecture_slow_path [EXTRACTED 1.00]
- **P2 per-sample decomposition of what an expert bank adds** — docs_p2_bound_prereg_p2_decomposition_identity, docs_p2_bound_prereg_rescuable_mass, docs_p2_bound_prereg_rescue_rate, docs_p2_bound_prereg_break_rate, docs_p2_bound_prereg_p2_contrast [EXTRACTED 1.00]
- **Stage 1 Complexity Ladder Levels** — docs_stage1_plan_l0_ncm, docs_stage1_plan_l1_ridge, docs_stage1_plan_l2a_shared_joint, docs_stage1_plan_l2b_shared_seq, docs_stage1_plan_l3_per_task, docs_stage1_plan_l4_oracle [EXTRACTED 1.00]
- **Capacity vs Routing Decomposition Evidence Chain** — docs_stage1_results_s8_budget, docs_stage1_results_s6b_designed_difficulty, docs_stage1_results_s9_robustness, docs_stage1_results_s10_scalability, docs_stage1_results_s11_confirmatory, docs_stage1_results_capacity_routing_decomposition [EXTRACTED 1.00]
- **Isolation Realization Metrics** — docs_stage1_plan_r_iso, docs_stage1_plan_r_iso_ncm, docs_stage1_plan_routing_tax, docs_stage1_results_coverage_at_m [INFERRED 0.85]
- **Zero-raw-replay routing stabilization mechanisms** — docs_v1_benchmark_router_anchor_distillation, docs_v1_benchmark_ood_negative_boundary_loss, docs_v1_benchmark_prototype_anchored_inference_routing, docs_v1_benchmark_historical_routing_lock, docs_v1_benchmark_prototype_memory [EXTRACTED 1.00]
- **Equal-byte memory claim evidence chain** — docs_v1_experiment_plan_h2_equal_byte_latent_replay, docs_v1_experiment_plan_e4_equal_byte_pareto, docs_v1_benchmark_storage_semantics_fact19, docs_v1_benchmark_raw_equal_byte_sweep, docs_v1_experiment_plan_ao1_item_vs_byte_budget [INFERRED 0.85]
- **Measurement artifacts corrected in v1 tables** — docs_v1_benchmark_batchnorm_eval_fix, docs_v1_benchmark_historical_routing_lock, docs_v1_benchmark_storage_semantics_fact19, docs_v1_benchmark_owner_aware_merging [INFERRED 0.75]
- **PAL-MoE v2 three-term core objective (L_task + L_gate + L_func) over frozen gate stack** — docs_v1_palmoe_v2_spec_l_gate, docs_v1_palmoe_v2_spec_l_func, docs_v1_palmoe_v2_spec_distributionmemory, docs_v1_palmoe_v2_spec_residualgaterouter, docs_v1_palmoe_v2_spec_growingcosineclassifier [EXTRACTED 1.00]
- **v2 allocation: coverage filter -> logistic gate fit + top-1 calibration -> counterfactual probe** — docs_v1_palmoe_v2_spec_expertcoveragestats, docs_v1_palmoe_v2_spec_candidate_filter, docs_v1_palmoe_v2_spec_logistic_gate_fit, docs_v1_palmoe_v2_spec_gate_calibration_top1, docs_v1_palmoe_v2_spec_counterfactual_probe [EXTRACTED 1.00]
- **PAL-MoE v1 three jointly designed mechanisms** — docs_v1_sunum_dynamic_expert_allocation, docs_v1_sunum_prototype_memory, docs_v1_sunum_prototype_anchored_routing, docs_v1_sunum_palmoe_v1 [EXTRACTED 1.00]

## Communities (148 total, 42 thin omitted)

### Community 0 - "v1 Tests & Prototype Memory"
Cohesion: 0.04
Nodes (73): PrototypeMemory, SharedEncoder, MLPExpert, DynamicMoE, DynamicRouter, test_adapter_experts_freeze_the_base_pathway(), test_add_expert_null_space_basis(), test_candidate_training_does_not_inflate_usage() (+65 more)

### Community 1 - "S1 Experts, Routers & Buffers"
Cohesion: 0.04
Nodes (9): expert_param_count(), SampleBuffer, task_class_counts(), LatentReplayTrainer, MIR, Prototype, EMAEncoder, DistanceRouter (+1 more)

### Community 2 - "VLM Live-Learning Sandbox"
Cohesion: 0.06
Nodes (41): main(), main(), triples_for(), main(), ask(), cur_logits(), emb_clip(), emb_dino() (+33 more)

### Community 3 - "S1 Backbones & Protocols"
Cohesion: 0.04
Nodes (34): build_projected_backbone(), optional_backbone(), wrap_encoder(), assert_shapes(), Backbone, ClassificationExpert, identity_check(), is_classification_expert() (+26 more)

### Community 4 - "Stage1 Benchmark & Calibration"
Cohesion: 0.04
Nodes (27): calibrate_expert_temperatures(), fit_temperature(), BiasCorrectionHead, NCMHead, record_from_runner_result(), EWC, NaiveFineTuning, _active_params_per_sample() (+19 more)

### Community 5 - "Continual Evaluator Metrics"
Cohesion: 0.04
Nodes (22): ContinualEvaluator, ContinualTrainer, ExpertBuilder, build_prototype_memory(), QuantitativeTrigger, inspect_routing_distribution(), main(), run_diagnostic_experiment() (+14 more)

### Community 6 - "v1 Benchmark Methodology"
Cohesion: 0.05
Nodes (52): Attention-Style Router, PAL-MoE Benchmark Methodology, Replay Buffer Sampling (recency vs reservoir), Checkpoint Forgetting Diagnostic (diagnose_checkpoint.py), CI Verification (lint, pytest, benchmark smoke), Class-Incremental Evaluation Protocol (Avg Acc, Forgetting, BWT), DER++ baseline, Measured Design Facts (1-22) (+44 more)

### Community 7 - "Decision Routing & RR Factorial"
Cohesion: 0.05
Nodes (34): bank(), bank_hash(), build_hypotheses(), _contract(), evaluate(), gate_hash(), guards(), main() (+26 more)

### Community 8 - "PTM-CIL Random Projection Eval"
Cohesion: 0.05
Nodes (33): RandomProjection, task_increments(), decompose(), ExpertDump, p2_decomposition(), route_by_owner(), route_by_task_sum(), route_top2_task_sum() (+25 more)

### Community 9 - "Living Model Prereg & Metrics"
Cohesion: 0.06
Nodes (62): Living Model Confirmatory Pre-registration, Base-referenced No-leak Metric, Behavioural Return Metric (return_match), Capacity Guard (CAP=64, PilotStore), CounterFact, Router v2 (Entity-aware), Memory-only RAG Arm, Strong-RAG Gate (+54 more)

### Community 10 - "Run-Record Schema"
Cohesion: 0.06
Nodes (34): aggregate_runs(), _bootstrap_ci(), _clean(), _deep_defaults(), _default_run_id(), _get(), load_run_record(), load_run_records() (+26 more)

### Community 11 - "CerataLM & Down-Proj Editing"
Cohesion: 0.07
Nodes (18): GuardConfig, CerataLM, LMPrediction, DownProjEdit, estimate_key_covariance(), KeyPrior, KeyValueContribution, synthetic_prior() (+10 more)

### Community 12 - "Text Live-Learning Harness"
Cohesion: 0.07
Nodes (12): propose_and_commit (KLD cap), chat(), load_model(), make_lora(), main(), main(), main(), score() (+4 more)

### Community 13 - "Text Ledger & Bandit"
Cohesion: 0.08
Nodes (19): expert_serve(), final_readout(), hit(), main(), memory_serve(), Retriever, run_learned(), run_repair() (+11 more)

### Community 14 - "Router Eval & Checkpoint Diagnosis"
Cohesion: 0.06
Nodes (27): main(), evaluate(), main(), RouterV1, main(), main(), main(), main() (+19 more)

### Community 15 - "S4/S5 Datasets & Protocols"
Cohesion: 0.06
Nodes (31): build_run_record(), satisfied_blocks(), _contract(), aggregate(), build_tasks(), _contract(), ensure_cache(), main() (+23 more)

### Community 16 - "Confirm Pilot & External Data"
Cohesion: 0.08
Nodes (28): download(), generate(), main(), make_subject(), make_word(), pairs(), base_answer(), ci() (+20 more)

### Community 17 - "Split-CIFAR Data"
Cohesion: 0.07
Nodes (25): get_split_cifar10_tasks(), SplitCIFAR10Task, get_split_mnist_tasks(), SplitMNISTTask, build_cached_encoder(), build_encoder(), build_moe(), build_router() (+17 more)

### Community 18 - "Positioning & Research Map"
Cohesion: 0.07
Nodes (49): The Mirage of Model Editing (arXiv:2502.11177), by_arrival Consolidation Policy, POSITIONING.md, Auditable Learning Contract, Paper B: Auditable Post-deployment Learning, Parametric Exact Edit Removal on an LM, API Arm (rp readout through Cerata), Documentation Map (+41 more)

### Community 19 - "v3 SLOW Path Experts"
Cohesion: 0.09
Nodes (29): Batch, by_arrival(), by_confusion(), check_gate(), confusion_matrix(), cross_fitted_confusion(), PolicyGateError, assert_pure() (+21 more)

### Community 20 - "Cerata API Facade"
Cohesion: 0.08
Nodes (10): Cerata, ConsolidationReport, Example, GuardViolation, Prediction, ReversibilityError, StateHash, PrototypeTaskRouter (+2 more)

### Community 21 - "PTM-CIL Benchmarks"
Cohesion: 0.07
Nodes (25): class_order(), load_raw_cache(), split_tasks(), stream_test_order(), train_model(), aggregate(), analytic_arms(), api_arm() (+17 more)

### Community 22 - "PAL-MoE v2 Spec"
Cohesion: 0.06
Nodes (44): v1/BENCHMARK.md, Expansion trigger (supervised composite score or label-free energy novelty), Function-preserving expert clone from parent, Inference-time prototype anchoring (distance-ratio confidence), Router distillation to prototype owners, v1 training losses (task CE, router-stability KL, expert-stability MSE, negative boundary, encoder EMA), Validation gate for candidate experts, AllocationController (REUSE vs EXPAND) (+36 more)

### Community 23 - "E2 Evidence & Intervention"
Cohesion: 0.07
Nodes (19): E2Model, guards(), main(), save(), normalize(), run_e2(), s11_reference(), audit() (+11 more)

### Community 24 - "Paper Reporting"
Cohesion: 0.10
Nodes (22): _agg(), aggregate_dir(), equal_byte_section(), fmt_bytes(), fmt_pct(), growth_section(), _iter_result_files(), latency_section() (+14 more)

### Community 25 - "Stage1 Ladder Results"
Cohesion: 0.05
Nodes (40): S2 Complexity Ladder (L0-L4), Corrected FWT Definition (Representation Ridge Probe), Hard-Routing Outcome Taxonomy (A/B/C), Expert Bank Buys Isolation, Not Capacity, L0_ncm Nearest-Class-Mean Readout, L1_ridge Closed-Form Ridge Readout, L2a_shared_joint Shared Adapter (Joint Control), L2b_shared_seq Shared Sequential Adapter (+32 more)

### Community 26 - "FAST Memory & Address Index"
Cohesion: 0.07
Nodes (5): ExactCosineIndex, SearchResult, FastMemory, MemoryHit, test_fast_memory_delete_restores_state_digest()

### Community 27 - "MEDIUM Linear Stats"
Cohesion: 0.09
Nodes (3): Contribution, LinearStats, RidgeClassRouter

### Community 28 - "S2 Ladder Hooks"
Cohesion: 0.09
Nodes (22): _NullHandle, trainable_hook(), build_readout(), iter_batches(), load_tasks(), set_seed(), _entropy(), forward_transfer() (+14 more)

### Community 29 - "v1 Attention Router"
Cohesion: 0.08
Nodes (4): AttentionRouter, _project_to_null_space(), test_attention_router_routing_expansion_and_lock(), test_null_space_anchor_makes_new_row_orthogonal()

### Community 31 - "Guarded Editor"
Cohesion: 0.13
Nodes (3): GuardedEditor, _max_abs(), EditRecord

### Community 32 - "Router Ranking Runner"
Cohesion: 0.10
Nodes (14): bank_hash(), build_hypotheses(), _contract(), evaluate_arm(), fit_gate(), guards(), LearnedGateRouter, main() (+6 more)

### Community 33 - "S9 Robustness"
Cohesion: 0.10
Nodes (19): cache_path(), load_shift_cache(), aggregate(), _concat(), _contract(), corruption_conditions(), default_grid(), evaluate() (+11 more)

### Community 34 - "Cached Backbone"
Cohesion: 0.08
Nodes (7): CachedBackbone, ProjectedBackbone, RandomProjectionBackbone, _register_shared_encoder_archs(), _shared_encoder(), SharedEncoderBackbone, __init__()

### Community 35 - "LLM Editing Eval (CounterFact)"
Cohesion: 0.14
Nodes (18): counterfact_scores(), EditCase, load_canary(), load_counterfact(), load_mquake(), load_zsre(), locality(), multihop_accuracy() (+10 more)

### Community 36 - "Energy Trigger & Validation Gate"
Cohesion: 0.09
Nodes (10): ValidationGateResult, energy(), EnergyTrigger, _RunningStats, AlwaysTrigger, TriggerEvaluationResult, test_always_trigger_fires_on_new_task(), test_rejected_expansion_owner_is_newest_expert() (+2 more)

### Community 37 - "Architecture & Measurement Contracts"
Cohesion: 0.10
Nodes (29): Architecture Contract (S1), Backbone interface (encode(x) -> z), ClassificationExpert interface (Z -> Y), legacy v1 only, E0/L2 ladder: L2a joint 76.62, L2b sequential 55.67, L3 per-task 70.56, L4 oracle 97.64, Readout interface (fit / predict; ncm, ridge, ...), RepresentationExpert interface (Z -> Z), identity at init, Router interface (route, top_k), S1 guard: max |delta metric| = 0 on fixed smoke (+21 more)

### Community 38 - "Cosine Readout"
Cohesion: 0.09
Nodes (5): CosineReadout, MLPReadout, NCMReadout, test_ncm_matches_a_hand_computed_reference(), test_ncm_running_mean_is_exact_when_fit_in_two_halves()

### Community 40 - "MEDIUM Random Features"
Cohesion: 0.14
Nodes (14): _largest_within(), one_hot(), select_ridge(), canary_set(), run_cell(), _data(), _fitted(), test_flat_curve_takes_the_largest_c_and_flags_an_edge() (+6 more)

### Community 41 - "Core Hashing & Backbones"
Cohesion: 0.11
Nodes (8): FrozenFeatureBackbone, digest(), file_digest(), module_digest(), tensor_digest(), _update(), decoder_layers(), CerataLM Facade

### Community 42 - "Aggregation & Task-Free Metrics"
Cohesion: 0.10
Nodes (12): StreamingEvaluator, arm_metrics(), bank_hash(), build_hypotheses(), _contract(), guards(), main(), save() (+4 more)

### Community 43 - "Text Context Mechanisms"
Cohesion: 0.09
Nodes (3): ContextMechanism, LoRAMechanism, MemoryMechanism

### Community 44 - "AC1 Address Freeze Runner"
Cohesion: 0.10
Nodes (14): check_anchors(), check_guard(), final_drift(), harness_revision(), main(), manipulation_check(), run_cell(), stats() (+6 more)

### Community 45 - "HF Causal LM Wrapper"
Cohesion: 0.13
Nodes (3): down_projection(), HFCausalLM, hook()

### Community 47 - "E2 & AC1 Preregs"
Cohesion: 0.11
Nodes (29): AC1 Address Freeze Pre-registration, Amendment 1: anchor band |delta| <= 1.5e-3 (substrate boundary effect), C0+P_frozen: shared query P trained on task 0 only, then frozen, Passive key/query drift probe (manipulation check), v2 invariant: stable address + plastic value + immutable consolidation + associative memory, AC1 Address Freeze Results, Finding: freezing P collapses routing (-0.1507 C@3, -28.7 pp); address not decomposable into frozen halves, Restated v2 invariant: stable KEYS + plastic READER + immutable consolidation + associative memory (+21 more)

### Community 48 - "S10 Scaling"
Cohesion: 0.11
Nodes (16): run_e0(), aggregate(), build_tasks(), default_grid(), evaluate(), _key(), load_source(), main() (+8 more)

### Community 49 - "Feature Cache"
Cohesion: 0.13
Nodes (10): build_feature_cache(), CachedFeatureEncoder, CachedTask, _collate_features(), _encode_split(), FeatureCache, load_feature_cache(), save_feature_cache() (+2 more)

### Community 50 - "S11 Confirmatory Stats"
Cohesion: 0.13
Nodes (16): holm(), evaluate(), _acc(), _base_tasks(), build_hypotheses(), _cells_index(), _contract(), default_grid() (+8 more)

### Community 51 - "Owner-Side & P2 Bound Preregs"
Cohesion: 0.12
Nodes (24): Owner-Side Residual Pre-registration, Bitwise Anchor Invariance Guard, C0 Arm (old projections frozen), C1 Arm (owner + non-owner evidence gradients), E2 Accuracy Collapse, Passive Evidence Drift Probe, OWNER-ONLY Arm (non-owner paths cut), Owner-Side Residual Results (+16 more)

### Community 52 - "Diagnosis & Decision Routing Docs"
Cohesion: 0.13
Nodes (23): AC3 Address Space Pre-registration, Aggregation (WTA vs Top-3) Pre-registration, Winner-take-all vs uniform top-3 mixture over identical candidate set, Aggregation Results, Finding: uniform top-3 mixture dilutes specialization (-7.3 / -4.2 pp), Decision Routing Pre-registration, ceiling@3 = coverage@3 x conditional_oracle@3, Comparative supervision: owner-vs-all hinge loss (gamma 0.2, mean over negatives) (+15 more)

### Community 53 - "S9 Corruptions"
Cohesion: 0.13
Nodes (13): build_backbone(), clean_guard(), condition_shifts(), corruption_shift(), CorruptionShift, _dataset_cls(), extract_all(), extract_test() (+5 more)

### Community 54 - "External Export Tests"
Cohesion: 0.09
Nodes (8): BenchmarkSpec, StubNet, test_data_manager_holds_pinned_order_and_split(), test_forced_expert_rejects_a_mismatched_classifier(), test_forced_expert_uses_only_its_subspace(), test_micro_batches_reproduce_the_full_batch_step(), __init__(), tiny()

### Community 55 - "Router Diagnostics & Geometry"
Cohesion: 0.14
Nodes (7): print_router_diagnostics(), router_diagnostics(), geometry_report(), nearest_other_margin(), silhouette_score(), BenchmarkResult, test_geometry_report_separates_clusters()

### Community 56 - "iCaRL Baseline"
Cohesion: 0.11
Nodes (4): ICaRL, ICaRLWrapper, _build_base_encoder(), test_icarl_herding_matches_bruteforce()

### Community 58 - "Router Ranking & Rep-Routing Preregs"
Cohesion: 0.16
Nodes (21): Representation x Routing Objective Pre-registration, Candidate-specific Expert-adapted Ranking Space, Globally Consistent Supervised Routing Objective, Representation x Routing Objective Results, Coverage / Conditional-Oracle Dissociation, Cross-expert Separability as the Routing Constraint, Router Ranking Study Pre-registration, Ceiling@m = Coverage@m x Conditional Oracle Identity (+13 more)

### Community 59 - "Identity & Legacy Experts"
Cohesion: 0.12
Nodes (5): IdentityExpert, LegacyClassificationExpert, ResidualAdapter, ResidualMLPExpert, test_residual_adapter_zero_rank_is_exactly_the_identity()

### Community 60 - "Legacy & Prototype Routers"
Cohesion: 0.14
Nodes (3): LegacyRouter, PrototypeRouter, test_prototype_router_candidate_set_and_distribution()

### Community 61 - "DER++ Baseline"
Cohesion: 0.12
Nodes (3): DERPP, ERACE, test_derpp_update_buffer_chunked_logits_are_consistent()

### Community 62 - "AC3 Address Space & README"
Cohesion: 0.14
Nodes (22): fixed_proto address: parameter-free cosine retrieval over frozen class prototypes, AC3 Address Space Results, Finding: fixed retrieval address is a drop-in (+0.048 pp, TOST-equivalent, 0 router params), Fixed address trades top-1 decodability for top-3 coverage (net zero), Prototype router (training-free absolute reference, C@3 0.9494 / 0.8915), Routing tax (L4 - L3), property of task geometry (coherent vs dispersed), E-TID2 Ridge Router Results, Finding: bank adds +0.62 / -0.03 pp over ridge alone (inside 1 pp SESOI) (+14 more)

### Community 63 - "Interference & Intervention Preregs"
Cohesion: 0.19
Nodes (19): C0 Arm (freeze old projections), C1 Arm (train old projections), Interference Mechanism Pre-registration, C@3 Routing Coverage, Pinned E2 Contract, E2Model, Expert-Count Ladder (T in 2..20), Interference Asymmetry (non-owner/owner mass) (+11 more)

### Community 64 - "External Datasets Common"
Cohesion: 0.14
Nodes (9): eval_transform(), image_datasets(), label_positions(), pilot_data_manager(), _raw(), stream_test_loader(), cache_path(), extract() (+1 more)

### Community 66 - "Coupling Ablation & v3 LM Cost"
Cohesion: 0.14
Nodes (12): main(), save(), run(), vetoes(), main(), run_with_argv(), guarded_write_seconds(), contrib() (+4 more)

### Community 67 - "E0 Representation Ceiling"
Cohesion: 0.16
Nodes (11): emit_contracts(), iter_batches(), load_tasks(), main(), routing_recall(), run_class_incremental(), run_joint(), run_ncm() (+3 more)

### Community 68 - "AC3 Address Space Runner"
Cohesion: 0.14
Nodes (11): bilinear_scores(), build_router(), evaluate_with(), harness_revision(), main(), prototype_scores(), run_cell(), stats() (+3 more)

### Community 69 - "S8 Budget"
Cohesion: 0.17
Nodes (14): aggregate(), acc(), first(), pick(), st(), _contract(), default_grid(), _forgetting() (+6 more)

### Community 70 - "PTM Data Preparation"
Cohesion: 0.19
Nodes (9): count_split(), digests(), download(), find_archive(), main(), split_root(), unpack(), verify() (+1 more)

### Community 71 - "Domain Shift Data"
Cohesion: 0.15
Nodes (8): apply_phase_shift(), permutation_transform(), _apply(), _rebuild_loader(), rotation_transform(), ShiftedTask, _TransformedDataset, test_domain_shift_phase_stream()

### Community 72 - "S6 Order"
Cohesion: 0.16
Nodes (9): aggregate(), main(), save(), order_confusion(), pool_tasks(), _print(), repartition(), run_configuration() (+1 more)

### Community 73 - "Config Loading"
Cohesion: 0.18
Nodes (10): apply_config(), ConfigError, _describe(), load_config(), _range_ok(), _type_ok(), Configs Index, Feature-cache note (design fact 19): replay baselines store cached features (+2 more)

### Community 74 - "P2 Bound & Task Constructions"
Cohesion: 0.20
Nodes (9): args_data_dir(), build_construction(), superclass_of(), evaluate_arm(), feature_bytes(), git_rev(), main(), part_b_cell() (+1 more)

### Community 75 - "PTM-CIL Prereg & Paper A"
Cohesion: 0.20
Nodes (17): ImageNet-1K vs ImageNet-21K Backbone Protocol Gap, Paper A: When Does an Expert Bank Earn Its Keep?, PTM_CIL_PREREG.md, Analytic Readouts (ncm, ridge, rp), A3.2 Expert-subspace Ridge Readout (pal_l3-ridgewp), ExpertDump Format, H2: Banks Help More on High-shift Benchmarks, A3.3 Representation Headroom Scan (+9 more)

### Community 76 - "External Export Summaries"
Cohesion: 0.22
Nodes (9): dump_name(), dump_summary(), official_final_accuracy(), use_repo(), write_json(), check(), load_config(), main() (+1 more)

### Community 77 - "v3 Anchor Tests"
Cohesion: 0.18
Nodes (11): needs(), _out(), test_anchor_ac3_fixed_proto_cell(), test_anchor_etid2_cell(), test_anchor_etid2_report_recomputes_exactly(), test_anchor_s11_e0_cell_bitwise(), test_anchor_s11_e0_stored_means(), test_anchor_s11_hypotheses_recompute_exactly() (+3 more)

### Community 79 - "Expert Formulation Runner"
Cohesion: 0.22
Nodes (10): build_hypotheses(), _contract(), guards(), main(), save(), make_projection(), module_hash(), _print() (+2 more)

### Community 80 - "S5b Domains"
Cohesion: 0.18
Nodes (8): aggregate(), build_domain_cache(), combine_stream(), _contract(), main(), save(), _print(), run_cell()

### Community 81 - "Prototype Anchor Matrices"
Cohesion: 0.16
Nodes (6): build(), build(), build(), build(), build(), cifar_augment()

### Community 82 - "Split-CIFAR-100"
Cohesion: 0.16
Nodes (5): get_split_cifar100_tasks(), SplitCIFAR100Task, build_tasks(), main(), test_split_cifar100_loader()

### Community 84 - "S6b Difficulty"
Cohesion: 0.23
Nodes (8): separability(), aggregate(), _contract(), _diff(), main(), save(), _print(), run_configuration()

### Community 85 - "Paired Statistics"
Cohesion: 0.20
Nodes (5): paired_stats(), signed_rank_statistic(), tost(), t_sf(), westfall_young()

### Community 86 - "OOD Boundary Loss (TTT)"
Cohesion: 0.15
Nodes (3): energy_boundary_loss(), TestTimeAdapter, test_energy_boundary_loss_pushes_prototypes_down()

### Community 87 - "EASE Export"
Cohesion: 0.15
Nodes (7): make_dump(), set_random(), task_of_class(), accumulating_init_train(), forced_predictions(), port_error(), run_one()

### Community 92 - "Legacy Expert Merging"
Cohesion: 0.33
Nodes (5): model_soup(), task_arithmetic(), _tensor_names(), ties_merge(), test_model_merging_utilities()

### Community 93 - "Recipe: paper_wave1b"
Cohesion: 0.29
Nodes (10): LD_LIBRARY_PATH, PYTHONPATH, run_ablation_cell(), run_c10(), run_c100(), run_drift(), run_growth(), paper_wave1b.sh script (+2 more)

### Community 94 - "Recipe: paper_wave1c"
Cohesion: 0.29
Nodes (10): LD_LIBRARY_PATH, PYTHONPATH, run_ablation_cell(), run_c10(), run_c100(), run_drift(), run_growth(), paper_wave1c.sh script (+2 more)

### Community 95 - "CI Pipeline & Contributing"
Cohesion: 0.22
Nodes (5): CI/CD Pipeline (ci.yml), CI Benchmark Reproducibility Smoke job, CI Build & Package job, CI Lint & Format job (ruff, black), CI Test & Coverage job (pytest, py3.10-3.12)

### Community 97 - "Literature Update 2026-09"
Cohesion: 0.31
Nodes (10): Literature Update 2026-09-28, BetaEdit / CP-MoE / MePo / AlphaEdit repro / SinglePrompt, Adapt before Continual Learning (arXiv:2506.03956), PTM_CIL Amendment 3 (A3.1-A3.3), EASE, MOS, ObjectNet License Blocking Condition, RanPAC (arXiv:2307.02251) (+2 more)

### Community 98 - "E-TID Ceiling"
Cohesion: 0.33
Nodes (7): class_to_task_scores(), coverage(), fit(), knn_scores(), main(), run(), stack()

### Community 101 - "Split Folder Data"
Cohesion: 0.25
Nodes (4): FolderTask, get_split_folder_tasks(), test_split_folder_tasks(), test_split_folder_tasks_from_images()

### Community 103 - "E-TID2 Ridge Router"
Cohesion: 0.36
Nodes (5): decode(), fit_ridge(), main(), run_cell(), part_a_cell()

### Community 105 - "Recipe: paper_wave2"
Cohesion: 0.36
Nodes (7): LD_LIBRARY_PATH, PYTHONPATH, run_raw_c10(), run_raw_c100(), paper_wave2.sh script, SSL_CERT_FILE, step()

### Community 106 - "Corruption Kernels"
Cohesion: 0.50
Nodes (4): _apply_corruption(), _disk_kernel(), _to_pil(), _to_tensor()

### Community 114 - "Recipe: reservoir_check"
Cohesion: 0.40
Nodes (5): LD_LIBRARY_PATH, PYTHONPATH, paper_reservoir_check.sh script, SSL_CERT_FILE, step()

### Community 115 - "Recipe: paper_wave1d"
Cohesion: 0.40
Nodes (5): LD_LIBRARY_PATH, PYTHONPATH, paper_wave1d.sh script, SSL_CERT_FILE, step()

### Community 121 - "Recipe: cifar100_full"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_full.sh script, SSL_CERT_FILE

### Community 122 - "Recipe: gate_ablation"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_gate_ablation.sh script, SSL_CERT_FILE

### Community 123 - "Recipe: resnet18_frozen"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_resnet18_frozen.sh script, SSL_CERT_FILE

### Community 124 - "Recipe: resnet18_multiseed"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_resnet18_multiseed.sh script, SSL_CERT_FILE

### Community 125 - "Recipe: cifar10_frozen"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar10_resnet18_frozen.sh script, SSL_CERT_FILE

### Community 126 - "Recipe: memory_pareto"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, memory_pareto.sh script, SSL_CERT_FILE

### Community 127 - "Recipe: mnist_domainshift"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, mnist_domainshift_shared.sh script, SSL_CERT_FILE

### Community 128 - "Recipe: multiseed_cifar"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, multiseed_cifar.sh script, SSL_CERT_FILE

### Community 129 - "Recipe: paper_all"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, paper_all.sh script, SSL_CERT_FILE

### Community 130 - "Recipe: paper_all_resume"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, paper_all_resume.sh script, SSL_CERT_FILE

### Community 131 - "Recipe: readout_ablation"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, readout_ablation.sh script, SSL_CERT_FILE

### Community 132 - "Recipe: vit_multiseed"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, vit_cifar_multiseed.sh script, SSL_CERT_FILE

### Community 133 - "Recipe: vit_quick"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, vit_cifar_quick.sh script, SSL_CERT_FILE

## Ambiguous Edges - Review These
- `Function-space merge with accept/reject` → `Same-owner-only prototype merging`  [AMBIGUOUS]
  docs/v1/PALMOE_V2_SPEC.md · relation: semantically_similar_to

## Knowledge Gaps
- **127 isolated node(s):** `cifar100_full.sh script`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `SSL_CERT_FILE`, `cifar100_gate_ablation.sh script` (+122 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1269 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **42 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Function-space merge with accept/reject` and `Same-owner-only prototype merging`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `PrototypeMemory` connect `v1 Tests & Prototype Memory` to `S1 Experts, Routers & Buffers`, `Energy Trigger & Validation Gate`, `Continual Evaluator Metrics`, `pal_moe Package Init`, `FAISS ANN Index`, `Memory Footprint Estimate`, `Expert Builder`, `Prototype Memory Stability`, `Raw Anchor Refresh`, `Prototype Anchor Matrices`, `ANN Query`, `Split-CIFAR Data`, `Latent Replay Generator`, `v3 Anchor Tests`, `Prototype Registration`, `Prototype Merge & Prune`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PrototypeMemory` (e.g. with `ContinualEvaluator` and `ContinualTrainer`) actually correct?**
  _`PrototypeMemory` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cifar100_full.sh script`, `PYTHONPATH`, `LD_LIBRARY_PATH` to the rest of the system?**
  _127 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `v1 Tests & Prototype Memory` be split into smaller, more focused modules?**
  _Cohesion score 0.036819172113289764 - nodes in this community are weakly interconnected._
- **Why does `DynamicMoE` connect `v1 Tests & Prototype Memory` to `S1 Experts, Routers & Buffers`, `Stage1 Benchmark & Calibration`, `Continual Evaluator Metrics`, `Energy Trigger & Validation Gate`, `v3 Anchor Tests`, `Split-CIFAR Data`, `Prototype Anchor Matrices`, `OOD Boundary Loss (TTT)`, `v1 MoE Model`?**
  _High betweenness centrality (0.036) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `SharedEncoder` (e.g. with `_register_shared_encoder_archs()` and `SharedEncoderBackbone`) actually correct?**
  _`SharedEncoder` has 5 INFERRED edges - model-reasoned connections that need verification._