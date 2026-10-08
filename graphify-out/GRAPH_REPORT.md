# Graph Report - cerata  (2026-10-08)

## Corpus Check
- 289 files · ~303,783 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 2, .pt 2, .cff 1)

## Summary
- 3462 nodes · 8435 edges · 165 communities (122 shown, 43 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 231 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Legacy PAL-MoE v1 Core
- Delta Store & Ledger Evaluation
- Stage-1 Experiment Harness
- Evaluation Heads & Baselines
- PTM-CIL Evaluation & Export
- Replay Method Implementations
- Prototype Memory & Model Building
- Frozen Backbone & Key Editing
- Legacy Training Utilities
- Component Interface Contracts
- Readout Pinning Experiments
- Guarded Editor API
- Architecture Protocol Definitions
- Readout Implementations
- Policy Gates & Consolidation
- Expert Bank Experiments
- Task Routers & Purity Checks
- Ledger Architecture Concepts
- Result Aggregation Tooling
- Run Records & Statistics
- E2 Evidence Experiments
- Memory & Address Indexing
- Legacy Attention Router
- Causal LM Core & Tokenizer
- Trigger & Validation Gates
- Backbone Implementations
- Expert Ladder Model
- Editing Evaluation Datasets
- Feature Caching Pipeline
- Anchor Checks & Staging
- Eval Grid Shared Helpers
- Projected Backbones
- Delta Contributions & Stats
- Sandbox Retriever Serving
- CIFAR/MNIST Task Splits
- CerataLM Guard API
- Distribution Shift Benchmarks
- NCM Evaluation Experiments
- Adapted Space Routers
- Gate Fitting Experiments
- Sandbox LoRA Experiments
- RAG Answering Sandbox
- E0 Runner & Adapters
- Expert Variant Implementations
- Legacy Router Implementations
- Replay Buffer Methods
- Living Model Slice Concepts
- Class-Incremental Runner Scripts
- Router Scoring Experiments
- Shared Aggregation Utilities
- Latent Replay Generator
- Sandbox Memory Mechanisms
- Sandbox CLI Runners
- Router Objective Findings
- Benchmark Statistics
- Corruption Aggregation Utilities
- Sandbox Data Generation
- Sandbox Embed & Teach
- Legacy Diagnostics & Tests
- Shifted Task Datasets
- VLM Delta Store Sandbox
- v3 Anchor Tests
- Dump & Export Tooling
- Bank Hypothesis Experiments
- Bank Cell Aggregation
- Entropy & MI Diagnostics
- Config Loading & Validation
- External Export Tests
- Correlation & Order Analysis
- ICaRL Wrapper Implementation
- Router v2/v3 Sandbox
- Projection & Hash Experiments
- Pilot LoRA Variants
- Delta Training Sandbox
- Dataset Download & Verify
- Rejector Bank Implementation
- EWC Implementation
- Ridge Router Study Concepts
- Task Data Utilities
- Dump Construction Helpers
- Streaming Evaluator
- LoRA Mechanism Sandbox
- Expert Bank Literature Concepts
- Owner-Side Interference Concepts
- Living Model Documentation
- Forward Transfer Probes
- Geometry Report & Silhouette
- AGEM Implementation
- Replay Trainer Implementation
- Sandbox Embedders
- Benchmark Protocol Concepts
- Rejector AUROC Analysis
- Latent Replay Trainer
- Stage-1 Follow-up Documents
- Paper Run Queue & Audits
- PAL-MoE v2 Spec Concepts
- paper_wave1b.sh Runner
- paper_wave1c.sh Runner
- Ablation Runner
- Conditional VAE Implementation
- Model Merging Utilities
- EMA Encoder Implementation
- Decision Routing Study Concepts
- Cited Literature & Amendments
- E-TID Ceiling Analysis
- Learned Gate Router
- Shift Cache Loading
- Distance Router Implementation
- P2-Bound & Consolidation Concepts
- E-TID2 Ridge Router Experiment
- Registry Implementation
- Frozen Module Backbone
- Folder Task Dataset
- Router Diagnostics Panel
- ICaRL Memory Accounting
- PAL-MoE v1 Documentation
- paper_wave2.sh Runner
- Corruption Transform Utilities
- Temperature Calibration
- Router Evaluation Sandbox
- Interference Study Concepts
- Uncertainty Weighting
- Living Model Confirmatory Concepts
- paper_reservoir_check.sh Runner
- paper_wave1d.sh Runner
- Forward Timing Utilities
- Task Router Protocol
- Micro-batch Tests
- Expert Formulation Concepts
- Code Review & Refactor Backlog
- Timm Shim
- cifar100_full.sh Runner
- cifar100_gate_ablation.sh Runner
- cifar100_resnet18_frozen.sh Runner
- cifar100_resnet18_multiseed.sh Runner
- cifar10_resnet18_frozen.sh Runner
- memory_pareto.sh Runner
- mnist_domainshift_shared.sh Runner
- multiseed_cifar.sh Runner
- paper_all.sh Runner
- paper_all_resume.sh Runner
- readout_ablation.sh Runner
- vit_cifar_multiseed.sh Runner
- vit_cifar_quick.sh Runner
- AcceptHead Implementation
- VLM Ledger Script
- CI/CD Pipeline
- Address Freeze Concepts
- ShiftedCIFAR100 Dataset
- ForceToken Processor
- paper_status.sh Runner
- Project Metadata

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
- `Summed Deltas Do Not Compose` --semantically_similar_to--> `Interference Asymmetry (nonowner/owner mass)`  [INFERRED] [semantically similar]
  docs/LIVING_MODEL_SLICE_PREREG.md → archive/docs/INTERFERENCE_PREREG.md
- `CERATA Guards (locality, reversibility, order invariance, purity)` --semantically_similar_to--> `ACID Guarantee Set (G1-G5)`  [INFERRED] [semantically similar]
  archive/docs/V3_ARCHITECTURE.md → docs/LIVING_MODEL_POSITIONING.md
- `Parameter-Free Retrieval Address` --semantically_similar_to--> `Router-Integrity Invariant (G3)`  [INFERRED] [semantically similar]
  archive/docs/V3_ARCHITECTURE.md → docs/LIVING_MODEL_POSITIONING.md
- `Parameter-Free Retrieval Address` --semantically_similar_to--> `Transaction Ledger`  [INFERRED] [semantically similar]
  archive/docs/V3_ARCHITECTURE.md → docs/LIVING_MODEL_POSITIONING.md
- `dump_summary()` --uses--> `ExpertDump`  [INFERRED]
  archive/experiments/ptm/external/common.py → cerata/eval/decomposition.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Routing-Failure Diagnostic Chain** — archive_docs_diagnosis_synthesis_routing_tax, archive_docs_diagnosis_synthesis_router_ranking_study, archive_docs_diagnosis_synthesis_representation_routing, archive_docs_diagnosis_synthesis_cross_expert_separability, archive_docs_diagnosis_synthesis_winner_take_all [INFERRED 0.85]
- **PAL-MoE v1 Mechanism Stack** — archive_docs_v1_benchmark_ood_negative_boundary_loss, archive_docs_v1_benchmark_router_anchor_distillation, archive_docs_v1_benchmark_prototype_anchored_routing, archive_docs_v1_benchmark_historical_routing_lock, archive_docs_v1_benchmark_feature_cache [INFERRED 0.85]
- **PAL-MoE v2 Architecture Components** — archive_docs_v1_palmoe_v2_spec_residual_adapter, archive_docs_v1_palmoe_v2_spec_residual_gate_router, archive_docs_v1_palmoe_v2_spec_distribution_memory, archive_docs_v1_palmoe_v2_spec_allocation_controller, archive_docs_v1_palmoe_v2_spec_global_cosine_classifier [INFERRED 0.85]
- **E2 collapse gradient-path decomposition arms (C1 / OWNER-ONLY / C0)** — archive_docs_coupling_prereg_c1_arm, archive_docs_intervention_prereg_owner_only_arm, archive_docs_coupling_prereg_c0_arm, archive_docs_owner_side_prereg_two_component_decomposition [EXTRACTED 1.00]
- **v3 contract: three time scales and the four guards** — readme_fast_path, readme_medium_path, readme_slow_path, readme_four_guards [EXTRACTED 1.00]
- **live-learning transactional stack (ledger, controller, VLM store, keys)** — experiments_sandbox_live_learning_state_ledger_core, experiments_sandbox_live_learning_state_repair_bandit, experiments_sandbox_live_learning_state_vlm_deltastore, experiments_sandbox_live_learning_state_clip_keys [INFERRED 0.75]
- **Routing-Invariant Follow-Up Chain** — archive_docs_router_ranking_prereg_document, archive_docs_representation_routing_prereg_document, archive_docs_decision_routing_prereg_document, archive_docs_aggregation_prereg_document, archive_docs_expert_formulation_prereg_document [EXTRACTED 1.00]
- **Address / Value-Path Diagnosis Chain** — archive_docs_interference_prereg_document, archive_docs_intervention_results_document, archive_docs_ac1_address_freeze_results_document, archive_docs_e2_evidence_results_document [EXTRACTED 1.00]
- **Living-Model Guarantee Framework** — docs_living_model_positioning_transaction_ledger, docs_living_model_positioning_acid_guarantee_set, docs_living_model_positioning_state_identity_revocation, docs_living_model_positioning_router_integrity_invariant, docs_living_model_positioning_provenance [EXTRACTED 1.00]

## Communities (165 total, 43 thin omitted)

### Community 0 - "Legacy PAL-MoE v1 Core"
Cohesion: 0.04
Nodes (66): ContinualTrainer, ExpertBuilder, SharedEncoder, MLPExpert, DynamicMoE, DynamicRouter, QuantitativeTrigger, test_adapter_experts_freeze_the_base_pathway() (+58 more)

### Community 1 - "Delta Store & Ledger Evaluation"
Cohesion: 0.04
Nodes (42): audit(), check_invariance(), headline(), main(), run_cell(), _step(), check_anchors(), final_drift() (+34 more)

### Community 2 - "Stage-1 Experiment Harness"
Cohesion: 0.04
Nodes (62): superclass_groups(), _contract(), run_cell(), aggregate(), _contract(), _git_commit(), main(), measure_latency() (+54 more)

### Community 3 - "Evaluation Heads & Baselines"
Cohesion: 0.04
Nodes (26): _active_params_per_sample(), _banner(), _git_commit(), _load_pretrained_encoder(), _method_features(), _pretrain_cache_path(), _pretrain_encoder(), _record_baseline_result() (+18 more)

### Community 4 - "PTM-CIL Evaluation & Export"
Cohesion: 0.04
Nodes (40): RandomProjection, class_order(), load_raw_cache(), split_tasks(), stream_test_order(), task_increments(), decompose(), ExpertDump (+32 more)

### Community 5 - "Replay Method Implementations"
Cohesion: 0.06
Nodes (5): TestTimeAdapter, SampleBuffer, task_class_counts(), MIR, test_sample_buffer_reservoir_balances_tasks()

### Community 6 - "Prototype Memory & Model Building"
Cohesion: 0.05
Nodes (30): build_model(), _detect_encoder_arch(), infer_config(), main(), main(), build_cached_encoder(), build_encoder(), build_moe() (+22 more)

### Community 7 - "Frozen Backbone & Key Editing"
Cohesion: 0.05
Nodes (22): cache_path(), extract(), main(), guarded_write_seconds(), main(), solve_seconds(), sync(), synthetic_prior() (+14 more)

### Community 8 - "Legacy Training Utilities"
Cohesion: 0.07
Nodes (8): cifar_augment(), contrib(), build(), build(), build(), build(), build(), build()

### Community 9 - "Component Interface Contracts"
Cohesion: 0.05
Nodes (58): AC1: address-freeze completion, C0 arm (current W trainable, older W frozen), C0+P_frozen arm, query drift probe, shared query P, stable address + plastic value + immutable consolidation + associative memory, AC3: address-space ablation, address/value coupling in E2 (+50 more)

### Community 10 - "Readout Pinning Experiments"
Cohesion: 0.08
Nodes (26): Benchmark Reproducibility Smoke job, aggregate(), main(), pinned_readouts(), run_cell(), train_joint(), evaluate_arm(), feature_bytes() (+18 more)

### Community 11 - "Guarded Editor API"
Cohesion: 0.10
Nodes (9): GuardedEditor, _max_abs(), ConsolidationReport, EditRecord, Example, GuardViolation, Prediction, ReversibilityError (+1 more)

### Community 12 - "Architecture Protocol Definitions"
Cohesion: 0.06
Nodes (20): assert_shapes(), Backbone, ClassificationExpert, identity_check(), is_classification_expert(), is_representation_expert(), Readout, RepresentationExpert (+12 more)

### Community 13 - "Readout Implementations"
Cohesion: 0.07
Nodes (11): CosineReadout, LinearReadout, LogisticReadout, MLPReadout, NCMReadout, _NullHandle, RidgeReadout, trainable_hook() (+3 more)

### Community 14 - "Policy Gates & Consolidation"
Cohesion: 0.10
Nodes (28): Batch, by_arrival(), by_confusion(), check_gate(), confusion_matrix(), cross_fitted_confusion(), PolicyGateError, _batches() (+20 more)

### Community 15 - "Expert Bank Experiments"
Cohesion: 0.08
Nodes (30): bank(), bank_hash(), build_hypotheses(), _contract(), evaluate(), gate_hash(), guards(), main() (+22 more)

### Community 16 - "Task Routers & Purity Checks"
Cohesion: 0.07
Nodes (6): Cerata, assert_pure(), RouterPurityError, trainable_count(), PrototypeTaskRouter, RidgeClassRouter

### Community 17 - "Ledger Architecture Concepts"
Cohesion: 0.07
Nodes (43): class-level routing decision, continual ridge router, E-TID: offline task-ID ceiling, lin_class joint linear probe, break rate beta, by_confusion consolidation policy, P2-BOUND study, rescuable mass m (+35 more)

### Community 18 - "Result Aggregation Tooling"
Cohesion: 0.10
Nodes (23): _agg(), aggregate_dir(), equal_byte_section(), fmt_bytes(), fmt_pct(), growth_section(), _iter_result_files(), latency_section() (+15 more)

### Community 19 - "Run Records & Statistics"
Cohesion: 0.08
Nodes (26): aggregate_runs(), _bootstrap_ci(), _get(), paired_delta(), _std(), validate_run_record(), _deep_merge(), _record() (+18 more)

### Community 20 - "E2 Evidence Experiments"
Cohesion: 0.08
Nodes (18): main(), save(), run(), vetoes(), E2Model, guards(), main(), save() (+10 more)

### Community 21 - "Memory & Address Indexing"
Cohesion: 0.08
Nodes (5): ExactCosineIndex, SearchResult, FastMemory, MemoryHit, test_fast_memory_delete_restores_state_digest()

### Community 22 - "Legacy Attention Router"
Cohesion: 0.08
Nodes (4): AttentionRouter, _project_to_null_space(), test_attention_router_routing_expansion_and_lock(), test_null_space_anchor_makes_new_row_orthogonal()

### Community 23 - "Causal LM Core & Tokenizer"
Cohesion: 0.10
Nodes (5): down_projection(), HFCausalLM, hook(), CharTokenizer, lm()

### Community 24 - "Trigger & Validation Gates"
Cohesion: 0.09
Nodes (10): ValidationGateResult, energy(), EnergyTrigger, _RunningStats, AlwaysTrigger, TriggerEvaluationResult, test_always_trigger_fires_on_new_task(), test_rejected_expansion_owner_is_newest_expert() (+2 more)

### Community 25 - "Backbone Implementations"
Cohesion: 0.08
Nodes (9): CachedBackbone, RandomProjectionBackbone, _register_shared_encoder_archs(), _shared_encoder(), SharedEncoderBackbone, wrap_encoder(), __init__(), test_cached_encoder_wraps_to_the_identity_backbone() (+1 more)

### Community 26 - "Expert Ladder Model"
Cohesion: 0.12
Nodes (3): mask_unseen(), _entropy(), LadderModel

### Community 27 - "Editing Evaluation Datasets"
Cohesion: 0.14
Nodes (18): counterfact_scores(), EditCase, load_canary(), load_counterfact(), load_mquake(), load_zsre(), locality(), multihop_accuracy() (+10 more)

### Community 28 - "Feature Caching Pipeline"
Cohesion: 0.10
Nodes (9): build_feature_cache(), CachedFeatureEncoder, CachedTask, _collate_features(), _encode_split(), FeatureCache, FeatureTensorDataset, load_feature_cache() (+1 more)

### Community 29 - "Anchor Checks & Staging"
Cohesion: 0.10
Nodes (14): check_anchors(), check_guard(), final_drift(), harness_revision(), main(), manipulation_check(), run_cell(), stats() (+6 more)

### Community 30 - "Eval Grid Shared Helpers"
Cohesion: 0.11
Nodes (17): aggregate(), build_tasks(), _contract(), default_grid(), evaluate(), _key(), load_source(), main() (+9 more)

### Community 31 - "Projected Backbones"
Cohesion: 0.11
Nodes (11): main(), main(), build_projected_backbone(), optional_backbone(), ProjectedBackbone, expert_param_count(), build_backbone(), build_expert() (+3 more)

### Community 33 - "Sandbox Retriever Serving"
Cohesion: 0.14
Nodes (15): expert_serve(), final_readout(), hit(), main(), memory_serve(), Retriever, run_learned(), run_repair() (+7 more)

### Community 34 - "CIFAR/MNIST Task Splits"
Cohesion: 0.11
Nodes (12): load_tasks(), build_tasks(), build_tasks(), ensure_cache(), get_split_cifar100_tasks(), SplitCIFAR100Task, get_split_cifar10_tasks(), SplitCIFAR10Task (+4 more)

### Community 35 - "CerataLM Guard API"
Cohesion: 0.13
Nodes (9): GuardConfig, CerataLM, LMPrediction, _prior(), test_hooks_expose_keys_and_down_proj_io(), test_lm_fast_write_retrieve_forget_is_bitwise(), test_lm_forget_is_downdate_and_forget_all_removes_the_hook(), test_lm_medium_edit_changes_target_and_forgets_exactly() (+1 more)

### Community 36 - "Distribution Shift Benchmarks"
Cohesion: 0.12
Nodes (14): build_backbone(), cache_path(), clean_guard(), condition_shifts(), corruption_shift(), CorruptionShift, _dataset_cls(), extract_all() (+6 more)

### Community 37 - "NCM Evaluation Experiments"
Cohesion: 0.13
Nodes (14): aggregate(), analytic_arms(), api_arm(), external_dumps(), git_rev(), main(), NCM, own_bank_dumps() (+6 more)

### Community 38 - "Adapted Space Routers"
Cohesion: 0.14
Nodes (4): adapted_scores(), AdaptedSpaceRouter, GateRouter, RoutingLoss

### Community 39 - "Gate Fitting Experiments"
Cohesion: 0.13
Nodes (13): bank_hash(), build_hypotheses(), _contract(), evaluate_arm(), fit_gate(), guards(), main(), save() (+5 more)

### Community 40 - "Sandbox LoRA Experiments"
Cohesion: 0.12
Nodes (3): train_sequential(), load_model(), make_lora()

### Community 42 - "RAG Answering Sandbox"
Cohesion: 0.22
Nodes (18): base_answer(), ci(), eval_ours(), grace_answer(), hit(), main(), rag_system(), run_grace() (+10 more)

### Community 44 - "Expert Variant Implementations"
Cohesion: 0.12
Nodes (5): IdentityExpert, LegacyClassificationExpert, ResidualAdapter, ResidualMLPExpert, test_residual_adapter_zero_rank_is_exactly_the_identity()

### Community 45 - "Legacy Router Implementations"
Cohesion: 0.14
Nodes (3): LegacyRouter, PrototypeRouter, test_prototype_router_candidate_set_and_distribution()

### Community 46 - "Replay Buffer Methods"
Cohesion: 0.12
Nodes (3): DERPP, ERACE, test_derpp_update_buffer_chunked_logits_are_consistent()

### Community 47 - "Living Model Slice Concepts"
Cohesion: 0.12
Nodes (22): Living Model Review 2026-10-04, Adaptive Chameleon or Stubborn Sloth (Xie et al., ICLR 2024), EasyEdit, Multi-fact Delta Capacity Curve, Paper Identity: Editing vs Systems, Router is a Dictionary (finding 2), Transactional Delta Store Slice-1 Pre-registration, CLIP Image Keys (+14 more)

### Community 48 - "Class-Incremental Runner Scripts"
Cohesion: 0.16
Nodes (11): emit_contracts(), iter_batches(), load_tasks(), main(), routing_recall(), run_class_incremental(), run_joint(), run_ncm() (+3 more)

### Community 49 - "Router Scoring Experiments"
Cohesion: 0.14
Nodes (11): bilinear_scores(), build_router(), evaluate_with(), harness_revision(), main(), prototype_scores(), run_cell(), stats() (+3 more)

### Community 50 - "Shared Aggregation Utilities"
Cohesion: 0.14
Nodes (14): aggregate(), acc(), first(), pick(), st(), default_grid(), _key(), main() (+6 more)

### Community 51 - "Latent Replay Generator"
Cohesion: 0.10
Nodes (4): energy_boundary_loss(), LatentReplayGenerator, test_energy_boundary_loss_pushes_prototypes_down(), test_latent_generative_replay_gaussian_and_vae()

### Community 52 - "Sandbox Memory Mechanisms"
Cohesion: 0.11
Nodes (3): chat(), ContextMechanism, MemoryMechanism

### Community 53 - "Sandbox CLI Runners"
Cohesion: 0.17
Nodes (9): main(), ask(), set_lora(), main(), serve(), main(), answer(), main() (+1 more)

### Community 54 - "Router Objective Findings"
Cohesion: 0.13
Nodes (13): Decision Routing Pre-registration, Pointwise Supervision, Candidate-Specific Adapted Routing Space, Representation x Routing Pre-registration, Representation x Routing Objective Factorial, Representation x Routing Results, Router Ranking Pre-registration, Prototype Ranking R0 (+5 more)

### Community 55 - "Benchmark Statistics"
Cohesion: 0.15
Nodes (9): git_rev(), main(), BenchmarkResult, holm(), paired_stats(), signed_rank_statistic(), tost(), t_sf() (+1 more)

### Community 56 - "Corruption Aggregation Utilities"
Cohesion: 0.12
Nodes (11): aggregate(), corruption_conditions(), default_grid(), evaluate(), _key(), main(), save(), _print() (+3 more)

### Community 57 - "Sandbox Data Generation"
Cohesion: 0.17
Nodes (7): main(), download(), generate(), main(), make_subject(), make_word(), pairs()

### Community 58 - "Sandbox Embed & Teach"
Cohesion: 0.19
Nodes (10): embed(), load(), load_embedder(), teach_text(), main(), route(), lora_named(), main() (+2 more)

### Community 59 - "Legacy Diagnostics & Tests"
Cohesion: 0.13
Nodes (8): inspect_routing_distribution(), main(), run_diagnostic_experiment(), set_seed(), build_prototype_memory(), build_single_head(), test_factory_builds_consistent_models(), test_param_reporting_fields()

### Community 60 - "Shifted Task Datasets"
Cohesion: 0.15
Nodes (8): apply_phase_shift(), permutation_transform(), _apply(), _rebuild_loader(), rotation_transform(), ShiftedTask, _TransformedDataset, test_domain_shift_phase_stream()

### Community 61 - "VLM Delta Store Sandbox"
Cohesion: 0.15
Nodes (4): main(), triples_for(), main(), VlmDeltaStore

### Community 62 - "v3 Anchor Tests"
Cohesion: 0.18
Nodes (12): _build_v1(), needs(), _out(), test_anchor_ac3_fixed_proto_cell(), test_anchor_etid2_cell(), test_anchor_etid2_report_recomputes_exactly(), test_anchor_s11_e0_cell_bitwise(), test_anchor_s11_e0_stored_means() (+4 more)

### Community 63 - "Dump & Export Tooling"
Cohesion: 0.22
Nodes (9): dump_name(), dump_summary(), official_final_accuracy(), use_repo(), write_json(), check(), load_config(), main() (+1 more)

### Community 64 - "Bank Hypothesis Experiments"
Cohesion: 0.18
Nodes (10): arm_metrics(), bank_hash(), build_hypotheses(), _contract(), guards(), main(), save(), _print() (+2 more)

### Community 65 - "Bank Cell Aggregation"
Cohesion: 0.15
Nodes (11): _acc(), build_hypotheses(), _cells_index(), default_grid(), _key(), main(), save(), order_variance() (+3 more)

### Community 66 - "Entropy & MI Diagnostics"
Cohesion: 0.13
Nodes (10): _entropy(), expert_entropy_mi(), _normalized_mi(), aggregate(), build_domain_cache(), combine_stream(), main(), save() (+2 more)

### Community 67 - "Config Loading & Validation"
Cohesion: 0.18
Nodes (10): apply_config(), ConfigError, _describe(), load_config(), _range_ok(), _type_ok(), Configs Index, Feature-cache note (design fact 19): replay baselines store cached features (+2 more)

### Community 68 - "External Export Tests"
Cohesion: 0.12
Nodes (5): BenchmarkSpec, StubNet, test_forced_expert_rejects_a_mismatched_classifier(), test_forced_expert_uses_only_its_subspace(), tiny()

### Community 70 - "Correlation & Order Analysis"
Cohesion: 0.13
Nodes (9): aggregate(), main(), save(), order_confusion(), pool_tasks(), _print(), repartition(), run_configuration() (+1 more)

### Community 72 - "Router v2/v3 Sandbox"
Cohesion: 0.16
Nodes (7): _norm(), _osa(), Router, RouterV2, main(), perturb_subject(), perturbed_probe()

### Community 73 - "Projection & Hash Experiments"
Cohesion: 0.22
Nodes (10): build_hypotheses(), _contract(), guards(), main(), save(), make_projection(), module_hash(), _print() (+2 more)

### Community 74 - "Pilot LoRA Variants"
Cohesion: 0.23
Nodes (8): main(), eval_variant(), extract_delta(), main(), make_variant_lora(), set_lora_params(), train_variant(), PilotStore

### Community 75 - "Delta Training Sandbox"
Cohesion: 0.27
Nodes (10): cur_logits(), panel(), prep(), reset_lora(), scene(), to_device(), train_delta(), train_delta_contrast() (+2 more)

### Community 76 - "Dataset Download & Verify"
Cohesion: 0.27
Nodes (8): count_split(), digests(), download(), find_archive(), main(), split_root(), unpack(), verify()

### Community 79 - "Ridge Router Study Concepts"
Cohesion: 0.21
Nodes (9): E-TID2: continual class-level ridge router over the L3 bank - results, L3 Per-Task Expert Bank, P1 Contrast (ridge_routed - proto), RidgeReadout(768, 100) Router, SESOI (1 pp smallest effect of interest), Router Ranking Study - results, R2 Learned Gate (z -> T supervised), lock_historical_routing(t) (+1 more)

### Community 80 - "Task Data Utilities"
Cohesion: 0.21
Nodes (6): label_positions(), pilot_data_manager(), _raw(), stream_test_loader(), eval_transform(), image_datasets()

### Community 81 - "Dump Construction Helpers"
Cohesion: 0.15
Nodes (7): make_dump(), set_random(), task_of_class(), accumulating_init_train(), forced_predictions(), port_error(), run_one()

### Community 84 - "Expert Bank Literature Concepts"
Cohesion: 0.19
Nodes (9): Shared Evidence Pathway (W -> 128 -> g), PTM-CIL Pre-registration, EASE Expert Bank, ExpertDump (.npz export contract), PTM-CIL Benchmarks Study, Research Map, MEMIT Closed-Form Mass Editing, RanPAC (Random Projection + Continual Gram Ridge) (+1 more)

### Community 85 - "Owner-Side Interference Concepts"
Cohesion: 0.23
Nodes (9): Interference mechanism: the expert-count ladder - results, Asymmetry (non-owner mass / owner mass), Expert-Count Ladder, Owner vs Non-Owner Evidence Mass, Owner-side residual: two-component decomposition of the E2 collapse - results, C0 Reference Arm (no update), Non-Owner Component (OWNER-ONLY - C1), Owner-Side Component (C0 - OWNER-ONLY) (+1 more)

### Community 86 - "Living Model Documentation"
Cohesion: 0.22
Nodes (6): V3 Architecture, Experiments Index (archive), Living Model Positioning, The Living Model (transactional state), Provenance (per-answer, per-fact), Experiments README

### Community 87 - "Forward Transfer Probes"
Cohesion: 0.18
Nodes (7): build_readout(), iter_batches(), forward_transfer(), probe_accuracy(), test_readouts_are_interchangeable_behind_the_same_call(), test_readouts_learn_a_linearly_separable_problem(), test_unknown_name_fails_loudly_with_the_available_names()

### Community 88 - "Geometry Report & Silhouette"
Cohesion: 0.26
Nodes (4): geometry_report(), nearest_other_margin(), silhouette_score(), test_geometry_report_separates_clusters()

### Community 91 - "Sandbox Embedders"
Cohesion: 0.24
Nodes (7): emb_clip(), emb_dino(), load_clip(), load_dino(), pairs_for(), main(), main()

### Community 92 - "Benchmark Protocol Concepts"
Cohesion: 0.24
Nodes (4): PAL-MoE Benchmark Methodology, Equal-Byte Protocol, Memory Budget Accounting, E5 Component Ablation

### Community 95 - "Stage-1 Follow-up Documents"
Cohesion: 0.31
Nodes (8): Aggregation Pre-registration, Aggregation Results, Decision Routing Results, E2 Evidence Results, Expert Formulation Pre-registration, E2 Evidence-Producing Experts (deferred), Results Inventory, Stage-1 Follow-Up Chain

### Community 96 - "Paper Run Queue & Audits"
Cohesion: 0.22
Nodes (10): PAL-MoE Paper Experiment Plan, E8 Anchor Refresh, Apples-to-Oranges Audit (AO1-AO10), E4 Equal-Byte Pareto, E7 Expert Growth and Reuse, H5 Refuted (no expert reuse), Routing Retention RR_t, H5 Refuted (gated = forced, 20/20 experts) (+2 more)

### Community 97 - "PAL-MoE v2 Spec Concepts"
Cohesion: 0.29
Nodes (7): PAL-MoE v2 - Technical Specification, AllocationController, E0 Representation Ceiling and Adapter Headroom, Growing Cosine Classifier, v2 Core Losses L_task + L_gate + L_func, p_anchor (stored old-model output), ResidualAdapter

### Community 98 - "paper_wave1b.sh Runner"
Cohesion: 0.29
Nodes (10): LD_LIBRARY_PATH, PYTHONPATH, run_ablation_cell(), run_c10(), run_c100(), run_drift(), run_growth(), paper_wave1b.sh script (+2 more)

### Community 99 - "paper_wave1c.sh Runner"
Cohesion: 0.29
Nodes (10): LD_LIBRARY_PATH, PYTHONPATH, run_ablation_cell(), run_c10(), run_c100(), run_drift(), run_growth(), paper_wave1c.sh script (+2 more)

### Community 100 - "Ablation Runner"
Cohesion: 0.31
Nodes (6): make_tasks(), pretrain_encoder(), run_all_ablations(), run_single_config(), random_create(), set_seed()

### Community 102 - "Model Merging Utilities"
Cohesion: 0.33
Nodes (5): model_soup(), task_arithmetic(), _tensor_names(), ties_merge(), test_model_merging_utilities()

### Community 104 - "Decision Routing Study Concepts"
Cohesion: 0.36
Nodes (7): Diagnosis Synthesis (Stage 1 -> Router Ranking -> Representation x Routing), DECISION_ROUTING_PREREG, Decision Structure Hypothesis, Representation x Routing Objective, Router Ranking Study, Routing Tax, Stage 1

### Community 105 - "Cited Literature & Amendments"
Cohesion: 0.27
Nodes (9): Literature update 2026-09-28, Adapt before Continual Learning (arXiv:2506.03956), The Mirage of Model Editing (arXiv:2502.11177), MOS Baseline, PTM_CIL_PREREG Amendment 3 (adopted 2026-09-28), RanPAC (arXiv:2307.02251), SinglePrompt (arXiv:2604.04420), TOSCA (arXiv:2502.14762) (+1 more)

### Community 106 - "E-TID Ceiling Analysis"
Cohesion: 0.33
Nodes (7): class_to_task_scores(), coverage(), fit(), knn_scores(), main(), run(), stack()

### Community 108 - "Shift Cache Loading"
Cohesion: 0.24
Nodes (6): load_shift_cache(), _concat(), load_condition(), load_train_condition(), _regroup(), verify_regroup()

### Community 110 - "Distance Router Implementation"
Cohesion: 0.22
Nodes (3): DistanceRouter, test_prune_keeps_historical_routing_lock(), test_router_learnable_temperature()

### Community 111 - "P2-Bound & Consolidation Concepts"
Cohesion: 0.33
Nodes (6): P2 Contrast (bank value over ridge readout), E2 Deferred (evidence-producing expert), P2-BOUND: why the expert bank adds < 1 pp over a ridge router - results, Confusion-Aligned Consolidation (A1), Rescue Rate rho, Superclass Grouping (A2)

### Community 112 - "E-TID2 Ridge Router Experiment"
Cohesion: 0.36
Nodes (5): decode(), fit_ridge(), main(), run_cell(), part_a_cell()

### Community 115 - "Folder Task Dataset"
Cohesion: 0.25
Nodes (4): FolderTask, get_split_folder_tasks(), test_split_folder_tasks(), test_split_folder_tasks_from_images()

### Community 118 - "ICaRL Memory Accounting"
Cohesion: 0.22
Nodes (3): ICaRL, test_icarl_herding_matches_bruteforce(), test_stored_memory_accounting()

### Community 119 - "PAL-MoE v1 Documentation"
Cohesion: 0.36
Nodes (8): PAL-MoE talk script (gncl), Catastrophic Forgetting, DistributionMemory, PAL-MoE v1: the published record, Latent Replay (128-d vectors, not raw images), PAL-MoE (dynamic MoE for class-incremental CL), PAL-MoE sunum notlari (Turkish presentation notes), Prototype Memory (v_p, r_p, o_p)

### Community 120 - "paper_wave2.sh Runner"
Cohesion: 0.36
Nodes (7): LD_LIBRARY_PATH, PYTHONPATH, run_raw_c10(), run_raw_c100(), paper_wave2.sh script, SSL_CERT_FILE, step()

### Community 122 - "Corruption Transform Utilities"
Cohesion: 0.50
Nodes (4): _apply_corruption(), _disk_kernel(), _to_pil(), _to_tensor()

### Community 124 - "Router Evaluation Sandbox"
Cohesion: 0.32
Nodes (3): evaluate(), main(), RouterV1

### Community 126 - "Interference Study Concepts"
Cohesion: 0.33
Nodes (5): Interference Pre-registration, Expert-Count Ladder (T in 2,4,8,12,20), Interference Asymmetry (nonowner/owner mass), Intervention Results, Rewrite Mass (mean ||delta_W_j||)

### Community 130 - "Living Model Confirmatory Concepts"
Cohesion: 0.43
Nodes (6): Living Model Confirmatory Pre-registration, Entity-Aware Router v2, Fuzzy Entity-Match Router v3, LoRA+ Recipe, Multi-Fact Delta Capacity (grouped delta), Strong-RAG Arms (instruct / qa / fewshot)

### Community 131 - "paper_reservoir_check.sh Runner"
Cohesion: 0.40
Nodes (5): LD_LIBRARY_PATH, PYTHONPATH, paper_reservoir_check.sh script, SSL_CERT_FILE, step()

### Community 132 - "paper_wave1d.sh Runner"
Cohesion: 0.40
Nodes (5): LD_LIBRARY_PATH, PYTHONPATH, paper_wave1d.sh script, SSL_CERT_FILE, step()

### Community 138 - "Expert Formulation Concepts"
Cohesion: 0.60
Nodes (3): Expert Formulation E1 (shared decision space) - results, E0 Anchor (S11 L3 / num_tasks), E1 Shared Decision Space

### Community 139 - "Code Review & Refactor Backlog"
Cohesion: 0.50
Nodes (5): Code Review - PAL-MoE codebase, ContinualTrainer.train_task God Class, Layer Dependency Direction, Refactor Backlog, ResidualGateRouter

### Community 141 - "cifar100_full.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_full.sh script, SSL_CERT_FILE

### Community 142 - "cifar100_gate_ablation.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_gate_ablation.sh script, SSL_CERT_FILE

### Community 143 - "cifar100_resnet18_frozen.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_resnet18_frozen.sh script, SSL_CERT_FILE

### Community 144 - "cifar100_resnet18_multiseed.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_resnet18_multiseed.sh script, SSL_CERT_FILE

### Community 145 - "cifar10_resnet18_frozen.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar10_resnet18_frozen.sh script, SSL_CERT_FILE

### Community 146 - "memory_pareto.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, memory_pareto.sh script, SSL_CERT_FILE

### Community 147 - "mnist_domainshift_shared.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, mnist_domainshift_shared.sh script, SSL_CERT_FILE

### Community 148 - "multiseed_cifar.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, multiseed_cifar.sh script, SSL_CERT_FILE

### Community 149 - "paper_all.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, paper_all.sh script, SSL_CERT_FILE

### Community 150 - "paper_all_resume.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, paper_all_resume.sh script, SSL_CERT_FILE

### Community 151 - "readout_ablation.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, readout_ablation.sh script, SSL_CERT_FILE

### Community 152 - "vit_cifar_multiseed.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, vit_cifar_multiseed.sh script, SSL_CERT_FILE

### Community 153 - "vit_cifar_quick.sh Runner"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, vit_cifar_quick.sh script, SSL_CERT_FILE

### Community 157 - "CI/CD Pipeline"
Cohesion: 0.50
Nodes (4): Build & Package job, CI/CD Pipeline, Lint & Format job, Test & Coverage job

## Knowledge Gaps
- **119 isolated node(s):** `cifar100_full.sh script`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `SSL_CERT_FILE`, `cifar100_gate_ablation.sh script` (+114 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1273 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **43 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Transactional Delta Store Slice-1 Pre-registration` connect `Living Model Slice Concepts` to `Delta Store & Ledger Evaluation`, `Delta Training Sandbox`, `VLM Ledger Script`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PrototypeMemory` (e.g. with `ContinualEvaluator` and `ContinualTrainer`) actually correct?**
  _`PrototypeMemory` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cifar100_full.sh script`, `PYTHONPATH`, `LD_LIBRARY_PATH` to the rest of the system?**
  _119 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Legacy PAL-MoE v1 Core` be split into smaller, more focused modules?**
  _Cohesion score 0.03983840197508697 - nodes in this community are weakly interconnected._
- **Why does `DynamicRouter` connect `Legacy PAL-MoE v1 Core` to `Expert Merge & Prune`, `Evaluation Heads & Baselines`, `Ablation Runner`, `Replay Method Implementations`, `Prototype Memory & Model Building`, `Gate Fitting Experiments`, `Learned Gate Router`, `Distance Router Implementation`, `Legacy Attention Router`, `Trigger & Validation Gates`, `v3 Anchor Tests`?**
  _High betweenness centrality (0.031) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `SharedEncoder` (e.g. with `run_single_config()` and `_register_shared_encoder_archs()`) actually correct?**
  _`SharedEncoder` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Should `Delta Store & Ledger Evaluation` be split into smaller, more focused modules?**
  _Cohesion score 0.0412448443944507 - nodes in this community are weakly interconnected._