# Graph Report - cerata  (2026-10-06)

## Corpus Check
- 284 files · ~301,702 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 2, .pt 2, .cff 1)

## Summary
- 3662 nodes · 8979 edges · 171 communities (120 shown, 51 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 250 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ce18b760`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- test_v1.py
- run_benchmark.py
- vlm_core.py
- arch/__init__.py
- calibrate_expert_temperatures
- run_benchmark
- Measured Design Facts (1-22)
- rr_factorial.py
- ExpertDump
- Living Model Positioning
- test_schema.py
- DownProjEdit
- pathlib
- DeltaStore
- json
- build_run_record
- pilot.py
- diagnose_checkpoint.py
- Research Map: Components and Literature Lineage
- torch
- Cerata
- test_ptm_cil.py
- PAL-MoE v2 (budget-conditioned modular CL)
- e2_evidence.py
- os
- s9_robustness.py
- FastMemory
- LinearStats
- s2_ladder.py
- AttentionRouter
- protocols.py
- GuardedEditor
- facts.py
- s10_scaling.py
- arch/backbones.py
- editing.py
- test_rejected_expansion_owner_is_newest_expert
- PAL-MoE Measurement Contract (S0)
- mask_unseen
- LadderModel
- rr_ranking.py
- digest
- StreamingEvaluator
- LoRAMechanism
- ac1_address_freeze.py
- HFCausalLM
- Tensor
- Coupling Ablation Results
- RidgeClassRouter
- feature_cache.py
- s11_confirmatory.py
- Owner-Side Residual Pre-registration
- Stage 1 Plan and Results
- s9_corruptions.py
- test_external_export.py
- eval/__init__.py
- ICaRLWrapper
- Documentation Map
- ResidualAdapter
- PrototypeRouter
- DERPP
- CERATA README
- Interference Mechanism Pre-registration
- extract_ptm_features.py
- SampleBuffer
- common.py
- E0Runner
- e0_representation_ceiling.py
- ac3_address_space.py
- prepare_ptm_data.py
- domain_shift.py
- policies.py
- config.py
- p2_bound.py
- PTM_CIL_PREREG.md
- s8_budget.py
- test_v3_anchors.py
- ._validate_candidate_impl
- run_one
- aggregation.py
- expert_entropy_mi
- get_split_cifar100_tasks
- schema.py
- s6b_difficulty.py
- eval/stats.py
- LatentReplayGenerator
- expert_formulation.py
- AGEM
- ReplayTrainer
- Prototype
- .fit
- copy
- paper_wave1b.sh
- paper_wave1c.sh
- CONTRIBUTING.md
- Literature Update 2026-09-28
- build_projected_backbone
- Registry
- FrozenModuleBackbone
- s4_datasets.py
- RejectorBank
- FeatureTensorDataset
- paper_wave2.sh
- s5b_domains.py
- UncertaintyWeighter
- aggregate
- Stage 1 Results
- .rejector_inputs
- TaskRouter
- paper_reservoir_check.sh
- paper_wave1d.sh
- PAL-MoE v1: the published record
- e_tid_ceiling.py
- LearnedGateRouter
- transfer_suite
- cifar100_full.sh
- cifar100_gate_ablation.sh
- cifar100_resnet18_frozen.sh
- cifar100_resnet18_multiseed.sh
- cifar10_resnet18_frozen.sh
- memory_pareto.sh
- mnist_domainshift_shared.sh
- multiseed_cifar.sh
- paper_all.sh
- paper_all_resume.sh
- readout_ablation.sh
- vit_cifar_multiseed.sh
- vit_cifar_quick.sh
- NCMReadout
- 11. S8 - what a budget buys, and in which regime
- Equal-byte comparison: raw pipeline vs feature-cache protocol
- _mixed_memory_package
- .__call__
- paper_status.sh
- Class-incremental protocol (no task id at test)
- Anchor refresh after calibration
- Capacity sweep (max_experts): accuracy flat, forgetting drops
- cerata
- 14. S11 - the confirmatory stage
- e_tid2_ridge_router.py
- _time_forward
- 12. S9 - does the decomposition survive a distribution shift?
- order_confusion
- AcceptHead
- tost
- 13. S10 - scalability: the extra capacity is stranded behind the router
- ShiftedCIFAR100
- 10. S6b - designed difficulty: the routing tax *is* geometry-dependent
- 7. S5 - the protocol axis
- 8. S5b - Domain-IL: is the routing tax a Class-IL artifact?
- README.md
- test_icarl_herding_matches_bruteforce

## God Nodes (most connected - your core abstractions)
1. `PrototypeMemory` - 113 edges
2. `SharedEncoder` - 99 edges
3. `MLPExpert` - 92 edges
4. `DynamicMoE` - 87 edges
5. `DynamicRouter` - 74 edges
6. `DeltaStore` - 52 edges
7. `LinearStats` - 48 edges
8. `ExpertBuilder` - 48 edges
9. `Documentation Map` - 47 edges
10. `Cerata` - 46 edges

## Surprising Connections (you probably didn't know these)
- `17. Measurement bugs this programme found` --references--> `mask_unseen()`  [INFERRED]
  docs/STAGE1_RESULTS.md → cerata/arch/readouts.py
- `7.3 A bug this stage found` --references--> `run_level()`  [INFERRED]
  docs/STAGE1_PLAN.md → experiments/stage1/ladder/s2_ladder.py
- `FastMemory` --implements--> `FAST Path: append-only KV memory`  [INFERRED]
  cerata/memory/kv_store.py → docs/V3_ARCHITECTURE.md
- `2. Measurement (S0) and architecture (S1) in one line each` --references--> `RepresentationExpert`  [INFERRED]
  docs/STAGE1_PLAN.md → cerata/arch/protocols.py
- `2. Measurement (S0) and architecture (S1) in one line each` --references--> `ClassificationExpert`  [INFERRED]
  docs/STAGE1_PLAN.md → cerata/arch/protocols.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Living model guarantee set G1-G5** — docs_living_model_positioning_g1_atomic, docs_living_model_positioning_g2_reversible, docs_living_model_positioning_g3_isolated, docs_living_model_positioning_g4_durable, docs_living_model_positioning_g5_auditable [EXTRACTED 1.00]
- **Confirmatory-study editing baselines** — docs_living_model_positioning_melo, docs_living_model_positioning_grace, docs_living_model_positioning_wise, docs_living_model_confirmatory_prereg_rag_arm, docs_living_model_review_2026_10_04_easyedit [EXTRACTED 1.00]
- **Non-owner interference: localisation -> mechanism -> intervention** — docs_coupling_results, docs_interference_prereg_owner_nonowner_decomposition, docs_interference_prereg_expert_count_ladder, docs_intervention_prereg_owner_only_arm, docs_intervention_results_routing_accuracy_dissociation [EXTRACTED 1.00]
- **P2 per-sample decomposition of what an expert bank adds** — docs_p2_bound_prereg_p2_decomposition_identity, docs_p2_bound_prereg_rescuable_mass, docs_p2_bound_prereg_rescue_rate, docs_p2_bound_prereg_break_rate, docs_p2_bound_prereg_p2_contrast [EXTRACTED 1.00]
- **PAL-MoE v1 three jointly designed mechanisms** — docs_v1_sunum_dynamic_expert_allocation, docs_v1_sunum_prototype_memory, docs_v1_sunum_prototype_anchored_routing, docs_v1_sunum_palmoe_v1 [EXTRACTED 1.00]
- **v2 allocation: coverage filter -> logistic gate fit + top-1 calibration -> counterfactual probe** — docs_v1_palmoe_v2_spec_expertcoveragestats, docs_v1_palmoe_v2_spec_candidate_filter, docs_v1_palmoe_v2_spec_logistic_gate_fit, docs_v1_palmoe_v2_spec_gate_calibration_top1, docs_v1_palmoe_v2_spec_counterfactual_probe [EXTRACTED 1.00]
- **PAL-MoE v2 three-term core objective (L_task + L_gate + L_func) over frozen gate stack** — docs_v1_palmoe_v2_spec_l_gate, docs_v1_palmoe_v2_spec_l_func, docs_v1_palmoe_v2_spec_distributionmemory, docs_v1_palmoe_v2_spec_residualgaterouter, docs_v1_palmoe_v2_spec_growingcosineclassifier [EXTRACTED 1.00]
- **v2 address programme: coupling -> AC1 -> AC3 -> stable keys** — docs_coupling_results_destructive_realignment, docs_ac1_address_freeze_results_address_not_decomposable, docs_ac3_address_space_results_fixed_address_drop_in, docs_ac1_address_freeze_results_stable_keys_plastic_reader [EXTRACTED 1.00]
- **v3 Guard Contract on every write/forget/consolidate** — docs_v3_architecture_guard_locality, docs_v3_architecture_guard_reversibility, docs_v3_architecture_guard_order_invariance, docs_v3_architecture_guard_router_purity, docs_positioning_auditable_learning_contract [EXTRACTED 1.00]
- **Three time scales over one fixed address space** — docs_v3_architecture_fixed_address_space, docs_v3_architecture_fast_path, docs_v3_architecture_medium_path, docs_v3_architecture_slow_path [EXTRACTED 1.00]
- **v3 three time scales on a fixed address space** — readme_fast_memory_path, readme_medium_closed_form_edit, readme_slow_consolidation, readme_four_guards, readme_cerata_api_facade [EXTRACTED 1.00]
- **Tests of the pointwise winner-take-all routing invariant** — docs_diagnosis_synthesis_pointwise_wta_invariant, docs_decision_routing_prereg_comparative_supervision, docs_aggregation_prereg_wta_vs_uniform_top3, docs_expert_formulation_prereg_e1_shared_decision_space, docs_e2_evidence_prereg_evidence_producing_experts [EXTRACTED 1.00]
- **Zero-raw-replay routing stabilization mechanisms** — docs_v1_benchmark_router_anchor_distillation, docs_v1_benchmark_ood_negative_boundary_loss, docs_v1_benchmark_prototype_anchored_inference_routing, docs_v1_benchmark_historical_routing_lock, docs_v1_benchmark_prototype_memory [EXTRACTED 1.00]
- **Measurement artifacts corrected in v1 tables** — docs_v1_benchmark_batchnorm_eval_fix, docs_v1_benchmark_historical_routing_lock, docs_v1_benchmark_storage_semantics_fact19, docs_v1_benchmark_owner_aware_merging [INFERRED 0.75]
- **Equal-byte memory claim evidence chain** — docs_v1_experiment_plan_h2_equal_byte_latent_replay, docs_v1_experiment_plan_e4_equal_byte_pareto, docs_v1_benchmark_storage_semantics_fact19, docs_v1_benchmark_raw_equal_byte_sweep, docs_v1_experiment_plan_ao1_item_vs_byte_budget [INFERRED 0.85]

## Communities (171 total, 51 thin omitted)

### Community 0 - "test_v1.py"
Cohesion: 0.04
Nodes (86): ContinualTrainer, ExpertBuilder, PrototypeMemory, SharedEncoder, MLPExpert, DynamicMoE, DynamicRouter, QuantitativeTrigger (+78 more)

### Community 1 - "run_benchmark.py"
Cohesion: 0.05
Nodes (7): expert_param_count(), TestTimeAdapter, ICaRL, MIR, NaiveFineTuning, build_prototype_memory(), EMAEncoder

### Community 2 - "vlm_core.py"
Cohesion: 0.06
Nodes (41): main(), main(), triples_for(), main(), ask(), cur_logits(), emb_clip(), emb_dino() (+33 more)

### Community 3 - "arch/__init__.py"
Cohesion: 0.08
Nodes (24): identity_check(), is_classification_expert(), is_representation_expert(), build_backbone(), build_expert(), build_readout(), build_router(), describe() (+16 more)

### Community 5 - "run_benchmark"
Cohesion: 0.04
Nodes (26): load_feature_cache(), BiasCorrectionHead, NCMHead, ContinualEvaluator, record_from_runner_result(), _active_params_per_sample(), _banner(), _git_commit() (+18 more)

### Community 6 - "Measured Design Facts (1-22)"
Cohesion: 0.05
Nodes (47): Attention-Style Router, Replay Buffer Sampling (recency vs reservoir), Checkpoint Forgetting Diagnostic (diagnose_checkpoint.py), CI Verification (lint, pytest, benchmark smoke), Class-Incremental Evaluation Protocol (Avg Acc, Forgetting, BWT), DER++ baseline, Measured Design Facts (1-22), Experience Replay (ER) baseline (+39 more)

### Community 7 - "rr_factorial.py"
Cohesion: 0.05
Nodes (34): bank(), bank_hash(), build_hypotheses(), _contract(), evaluate(), gate_hash(), guards(), main() (+26 more)

### Community 8 - "ExpertDump"
Cohesion: 0.11
Nodes (17): decompose(), ExpertDump, p2_decomposition(), route_by_owner(), route_by_task_sum(), route_top2_task_sum(), _task_mass(), _dump() (+9 more)

### Community 9 - "Living Model Positioning"
Cohesion: 0.06
Nodes (63): Living Model Confirmatory Pre-registration, Base-referenced No-leak Metric, Behavioural Return Metric (return_match), Capacity Guard (CAP=64, PilotStore), CounterFact, Router v2 (Entity-aware), Memory-only RAG Arm, Strong-RAG Gate (+55 more)

### Community 10 - "test_schema.py"
Cohesion: 0.09
Nodes (26): aggregate_runs(), _bootstrap_ci(), _get(), paired_delta(), upgrade_v0_result(), validate_run_record(), _deep_merge(), _record() (+18 more)

### Community 11 - "DownProjEdit"
Cohesion: 0.13
Nodes (9): DownProjEdit, KeyPrior, KeyValueContribution, guarded_write_seconds(), main(), solve_seconds(), sync(), synthetic_prior() (+1 more)

### Community 12 - "pathlib"
Cohesion: 0.06
Nodes (13): propose_and_commit (KLD cap), chat(), load_model(), make_lora(), main(), main(), main(), main() (+5 more)

### Community 13 - "DeltaStore"
Cohesion: 0.08
Nodes (19): expert_serve(), final_readout(), hit(), main(), memory_serve(), Retriever, run_learned(), run_repair() (+11 more)

### Community 14 - "json"
Cohesion: 0.07
Nodes (29): v3 Restructure Anchors, main(), main(), main(), main(), main(), main(), main() (+21 more)

### Community 15 - "build_run_record"
Cohesion: 0.27
Nodes (9): build_run_record(), satisfied_blocks(), _contract(), _contract(), _contract(), _contract(), _contract(), _contract() (+1 more)

### Community 16 - "pilot.py"
Cohesion: 0.11
Nodes (23): pairs(), base_answer(), ci(), eval_ours(), ForceToken, grace_answer(), hit(), main() (+15 more)

### Community 17 - "diagnose_checkpoint.py"
Cohesion: 0.06
Nodes (30): get_split_mnist_tasks(), build_cached_encoder(), build_encoder(), build_moe(), build_router(), build_single_head(), load_checkpoint(), save_checkpoint() (+22 more)

### Community 18 - "Research Map: Components and Literature Lineage"
Cohesion: 0.08
Nodes (44): AC1 Address Freeze Results, POSITIONING.md, Auditable Learning Contract, Paper B: Auditable Post-deployment Learning, Parametric Exact Edit Removal on an LM, Research Map: Components and Literature Lineage, ACIL (Zhuang et al., NeurIPS 2022), ACU: Analytic Continual Unlearning (2025) (+36 more)

### Community 19 - "torch"
Cohesion: 0.10
Nodes (38): CI Benchmark Reproducibility Smoke job, GuardConfig, Batch, load_tasks(), estimate_key_covariance(), data(), main(), _batches() (+30 more)

### Community 21 - "test_ptm_cil.py"
Cohesion: 0.04
Nodes (43): iter_batches(), set_seed(), RandomProjection, class_order(), load_raw_cache(), split_tasks(), stream_test_order(), task_increments() (+35 more)

### Community 22 - "PAL-MoE v2 (budget-conditioned modular CL)"
Cohesion: 0.06
Nodes (45): v1/BENCHMARK.md, Expansion trigger (supervised composite score or label-free energy novelty), Function-preserving expert clone from parent, Inference-time prototype anchoring (distance-ratio confidence), Router distillation to prototype owners, v1 training losses (task CE, router-stability KL, expert-stability MSE, negative boundary, encoder EMA), Validation gate for candidate experts, v1/PALMOE_V2_SPEC.md (+37 more)

### Community 23 - "e2_evidence.py"
Cohesion: 0.05
Nodes (24): E2Model, guards(), main(), save(), normalize(), run_e0(), run_e2(), s11_reference() (+16 more)

### Community 24 - "os"
Cohesion: 0.09
Nodes (22): _agg(), aggregate_dir(), equal_byte_section(), fmt_bytes(), fmt_pct(), growth_section(), _iter_result_files(), latency_section() (+14 more)

### Community 25 - "s9_robustness.py"
Cohesion: 0.10
Nodes (19): cache_path(), load_shift_cache(), aggregate(), _concat(), _contract(), corruption_conditions(), default_grid(), evaluate() (+11 more)

### Community 26 - "FastMemory"
Cohesion: 0.07
Nodes (4): ExactCosineIndex, SearchResult, FastMemory, MemoryHit

### Community 27 - "LinearStats"
Cohesion: 0.07
Nodes (13): Contribution, _largest_within(), LinearStats, one_hot(), select_ridge(), canary_set(), run_cell(), _data() (+5 more)

### Community 28 - "s2_ladder.py"
Cohesion: 0.19
Nodes (12): forward_transfer(), probe_accuracy(), LevelSpec, aggregate(), _contract(), _git_commit(), main(), measure_latency() (+4 more)

### Community 29 - "AttentionRouter"
Cohesion: 0.07
Nodes (5): AttentionRouter, DistanceRouter, _project_to_null_space(), test_attention_router_routing_expansion_and_lock(), test_null_space_anchor_makes_new_row_orthogonal()

### Community 30 - "protocols.py"
Cohesion: 0.10
Nodes (7): assert_shapes(), Backbone, ClassificationExpert, Readout, RepresentationExpert, Router, 2. Measurement (S0) and architecture (S1) in one line each

### Community 31 - "GuardedEditor"
Cohesion: 0.07
Nodes (11): GuardedEditor, _max_abs(), CerataLM, LMPrediction, ConsolidationReport, EditRecord, Example, GuardViolation (+3 more)

### Community 32 - "facts.py"
Cohesion: 0.12
Nodes (11): download(), main(), generate(), main(), make_subject(), make_word(), evaluate(), main() (+3 more)

### Community 33 - "s10_scaling.py"
Cohesion: 0.12
Nodes (15): aggregate(), build_tasks(), default_grid(), evaluate(), _key(), load_source(), main(), save() (+7 more)

### Community 34 - "arch/backbones.py"
Cohesion: 0.07
Nodes (10): CachedBackbone, RandomProjectionBackbone, _register_shared_encoder_archs(), _shared_encoder(), SharedEncoderBackbone, wrap_encoder(), __init__(), CachedFeatureEncoder (+2 more)

### Community 35 - "editing.py"
Cohesion: 0.14
Nodes (18): counterfact_scores(), EditCase, load_canary(), load_counterfact(), load_mquake(), load_zsre(), locality(), multihop_accuracy() (+10 more)

### Community 36 - "test_rejected_expansion_owner_is_newest_expert"
Cohesion: 0.11
Nodes (8): energy(), EnergyTrigger, _RunningStats, AlwaysTrigger, TriggerEvaluationResult, test_always_trigger_fires_on_new_task(), test_rejected_expansion_owner_is_newest_expert(), evaluate()

### Community 37 - "PAL-MoE Measurement Contract (S0)"
Cohesion: 0.11
Nodes (25): Architecture Contract (S1), Backbone interface (encode(x) -> z), ClassificationExpert interface (Z -> Y), legacy v1 only, E0/L2 ladder: L2a joint 76.62, L2b sequential 55.67, L3 per-task 70.56, L4 oracle 97.64, Readout interface (fit / predict; ncm, ridge, ...), RepresentationExpert interface (Z -> Z), identity at init, Router interface (route, top_k), S1 guard: max |delta metric| = 0 on fixed smoke (+17 more)

### Community 38 - "mask_unseen"
Cohesion: 0.09
Nodes (8): CosineReadout, LinearReadout, LogisticReadout, mask_unseen(), MLPReadout, _NullHandle, RidgeReadout, trainable_hook()

### Community 40 - "rr_ranking.py"
Cohesion: 0.13
Nodes (13): bank_hash(), build_hypotheses(), _contract(), evaluate_arm(), fit_gate(), guards(), main(), save() (+5 more)

### Community 41 - "digest"
Cohesion: 0.11
Nodes (7): FrozenFeatureBackbone, digest(), file_digest(), module_digest(), tensor_digest(), _update(), decoder_layers()

### Community 43 - "LoRAMechanism"
Cohesion: 0.09
Nodes (3): ContextMechanism, LoRAMechanism, MemoryMechanism

### Community 44 - "ac1_address_freeze.py"
Cohesion: 0.10
Nodes (14): check_anchors(), check_guard(), final_drift(), harness_revision(), main(), manipulation_check(), run_cell(), stats() (+6 more)

### Community 45 - "HFCausalLM"
Cohesion: 0.13
Nodes (3): down_projection(), HFCausalLM, hook()

### Community 46 - "Tensor"
Cohesion: 0.06
Nodes (8): build(), build(), build(), build(), build(), build(), cifar_augment(), contrib()

### Community 47 - "Coupling Ablation Results"
Cohesion: 0.11
Nodes (28): AC1 Address Freeze Pre-registration, Amendment 1: anchor band |delta| <= 1.5e-3 (substrate boundary effect), C0+P_frozen: shared query P trained on task 0 only, then frozen, Passive key/query drift probe (manipulation check), v2 invariant: stable address + plastic value + immutable consolidation + associative memory, Finding: freezing P collapses routing (-0.1507 C@3, -28.7 pp); address not decomposable into frozen halves, Restated v2 invariant: stable KEYS + plastic READER + immutable consolidation + associative memory, Bilinear learned address s_j = <normalize(Pz), normalize(W_j E_j z)> (+20 more)

### Community 48 - "RidgeClassRouter"
Cohesion: 0.13
Nodes (5): assert_pure(), RouterPurityError, trainable_count(), RidgeClassRouter, test_router_purity_rejects_a_learned_router()

### Community 49 - "feature_cache.py"
Cohesion: 0.18
Nodes (6): build_feature_cache(), CachedTask, _collate_features(), _encode_split(), FeatureCache, save_feature_cache()

### Community 50 - "s11_confirmatory.py"
Cohesion: 0.13
Nodes (16): holm(), evaluate(), _acc(), _base_tasks(), build_hypotheses(), _cells_index(), _contract(), default_grid() (+8 more)

### Community 51 - "Owner-Side Residual Pre-registration"
Cohesion: 0.27
Nodes (10): Owner-Side Residual Pre-registration, Bitwise Anchor Invariance Guard, C0 Arm (old projections frozen), C1 Arm (owner + non-owner evidence gradients), E2 Accuracy Collapse, Passive Evidence Drift Probe, OWNER-ONLY Arm (non-owner paths cut), Owner-Side Residual Results (+2 more)

### Community 52 - "Stage 1 Plan and Results"
Cohesion: 0.09
Nodes (23): 1. Stage order, 3.1 Findings, 3.2 Two bugs the ladder found, 3. S2 results - the complexity ladder, 4. The FWT definition was wrong, 5.1 Findings, 5.2 Verdict against the interpretation scheme, 5.3 A measurement bug S3 found (+15 more)

### Community 53 - "s9_corruptions.py"
Cohesion: 0.13
Nodes (13): build_backbone(), clean_guard(), condition_shifts(), corruption_shift(), CorruptionShift, _dataset_cls(), extract_all(), extract_test() (+5 more)

### Community 54 - "test_external_export.py"
Cohesion: 0.09
Nodes (8): BenchmarkSpec, StubNet, test_data_manager_holds_pinned_order_and_split(), test_forced_expert_rejects_a_mismatched_classifier(), test_forced_expert_uses_only_its_subspace(), test_micro_batches_reproduce_the_full_batch_step(), __init__(), tiny()

### Community 55 - "eval/__init__.py"
Cohesion: 0.14
Nodes (7): print_router_diagnostics(), router_diagnostics(), geometry_report(), nearest_other_margin(), silhouette_score(), BenchmarkResult, test_geometry_report_separates_clusters()

### Community 58 - "Documentation Map"
Cohesion: 0.12
Nodes (34): Aggregation (WTA vs Top-3) Pre-registration, Winner-take-all vs uniform top-3 mixture over identical candidate set, Aggregation Results, Finding: uniform top-3 mixture dilutes specialization (-7.3 / -4.2 pp), Decision Routing Pre-registration, ceiling@3 = coverage@3 x conditional_oracle@3, Comparative supervision: owner-vs-all hinge loss (gamma 0.2, mean over negatives), Pointwise gate (cross-entropy on stored prototypes) (+26 more)

### Community 59 - "ResidualAdapter"
Cohesion: 0.12
Nodes (5): IdentityExpert, LegacyClassificationExpert, ResidualAdapter, ResidualMLPExpert, test_residual_adapter_zero_rank_is_exactly_the_identity()

### Community 60 - "PrototypeRouter"
Cohesion: 0.14
Nodes (3): LegacyRouter, PrototypeRouter, test_prototype_router_candidate_set_and_distribution()

### Community 61 - "DERPP"
Cohesion: 0.12
Nodes (3): DERPP, ERACE, test_derpp_update_buffer_chunked_logits_are_consistent()

### Community 62 - "CERATA README"
Cohesion: 0.11
Nodes (25): AC3 Address Space Pre-registration, fixed_proto address: parameter-free cosine retrieval over frozen class prototypes, AC3 Address Space Results, Finding: fixed retrieval address is a drop-in (+0.048 pp, TOST-equivalent, 0 router params), Fixed address trades top-1 decodability for top-3 coverage (net zero), Prototype router (training-free absolute reference, C@3 0.9494 / 0.8915), Routing tax (L4 - L3), property of task geometry (coherent vs dispersed), E-TID2 Ridge Router Results (+17 more)

### Community 63 - "Interference Mechanism Pre-registration"
Cohesion: 0.19
Nodes (19): C0 Arm (freeze old projections), C1 Arm (train old projections), Interference Mechanism Pre-registration, C@3 Routing Coverage, Pinned E2 Contract, E2Model, Expert-Count Ladder (T in 2..20), Interference Asymmetry (non-owner/owner mass) (+11 more)

### Community 64 - "extract_ptm_features.py"
Cohesion: 0.14
Nodes (9): eval_transform(), image_datasets(), label_positions(), pilot_data_manager(), _raw(), stream_test_loader(), cache_path(), extract() (+1 more)

### Community 65 - "SampleBuffer"
Cohesion: 0.05
Nodes (7): SampleBuffer, task_class_counts(), EWC, LatentReplayTrainer, test_ewc_online_keeps_single_fisher(), test_sample_buffer_reservoir_balances_tasks(), test_stored_memory_accounting()

### Community 66 - "common.py"
Cohesion: 0.17
Nodes (10): dump_name(), dump_summary(), official_final_accuracy(), task_of_class(), use_repo(), write_json(), check(), load_config() (+2 more)

### Community 68 - "e0_representation_ceiling.py"
Cohesion: 0.16
Nodes (11): emit_contracts(), iter_batches(), load_tasks(), main(), routing_recall(), run_class_incremental(), run_joint(), run_ncm() (+3 more)

### Community 69 - "ac3_address_space.py"
Cohesion: 0.14
Nodes (11): bilinear_scores(), build_router(), evaluate_with(), harness_revision(), main(), prototype_scores(), run_cell(), stats() (+3 more)

### Community 70 - "prepare_ptm_data.py"
Cohesion: 0.19
Nodes (9): count_split(), digests(), download(), find_archive(), main(), split_root(), unpack(), verify() (+1 more)

### Community 71 - "domain_shift.py"
Cohesion: 0.15
Nodes (8): apply_phase_shift(), permutation_transform(), _apply(), _rebuild_loader(), rotation_transform(), ShiftedTask, _TransformedDataset, test_domain_shift_phase_stream()

### Community 72 - "policies.py"
Cohesion: 0.16
Nodes (7): by_arrival(), by_confusion(), check_gate(), confusion_matrix(), cross_fitted_confusion(), PolicyGateError, test_by_arrival_is_first_introduction()

### Community 73 - "config.py"
Cohesion: 0.18
Nodes (10): apply_config(), ConfigError, _describe(), load_config(), _range_ok(), _type_ok(), Configs Index, Feature-cache note (design fact 19): replay baselines store cached features (+2 more)

### Community 74 - "p2_bound.py"
Cohesion: 0.18
Nodes (10): args_data_dir(), build_construction(), superclass_of(), evaluate_arm(), feature_bytes(), git_rev(), main(), part_a_cell() (+2 more)

### Community 75 - "PTM_CIL_PREREG.md"
Cohesion: 0.09
Nodes (33): P2-BOUND Pre-registration, Break Rate beta, by_arrival Consolidation Policy, by_confusion Consolidation Policy, by_superclass Grouping (A2), 5-fold Cross-fitted Confusion Matrix, P2 Contrast (expert bank over ridge router), Rescuable Mass m (+25 more)

### Community 76 - "s8_budget.py"
Cohesion: 0.17
Nodes (13): aggregate(), acc(), first(), pick(), st(), _contract(), default_grid(), _forgetting() (+5 more)

### Community 77 - "test_v3_anchors.py"
Cohesion: 0.23
Nodes (8): needs(), test_anchor_ac3_fixed_proto_cell(), test_anchor_etid2_cell(), test_anchor_etid2_report_recomputes_exactly(), test_anchor_s11_e0_cell_bitwise(), test_anchor_s11_e0_stored_means(), test_anchor_s11_hypotheses_recompute_exactly(), test_pal_moe_compat_package_aliases_not_copies()

### Community 79 - "run_one"
Cohesion: 0.12
Nodes (7): make_dump(), set_random(), TimmShim, accumulating_init_train(), forced_predictions(), port_error(), run_one()

### Community 80 - "aggregation.py"
Cohesion: 0.20
Nodes (10): arm_metrics(), bank_hash(), build_hypotheses(), _contract(), guards(), main(), save(), _print() (+2 more)

### Community 81 - "expert_entropy_mi"
Cohesion: 0.14
Nodes (10): aggregate(), _entropy(), expert_entropy_mi(), main(), save(), _normalized_mi(), _print(), run_cell() (+2 more)

### Community 82 - "get_split_cifar100_tasks"
Cohesion: 0.12
Nodes (7): get_split_cifar100_tasks(), SplitCIFAR100Task, get_split_cifar10_tasks(), SplitCIFAR10Task, SplitMNISTTask, test_split_cifar100_loader(), test_split_cifar10_tasks()

### Community 83 - "schema.py"
Cohesion: 0.16
Nodes (8): _clean(), _deep_defaults(), _default_run_id(), load_run_record(), load_run_records(), _sibling_meta(), _std(), upgrade_v0_payload()

### Community 84 - "s6b_difficulty.py"
Cohesion: 0.23
Nodes (7): separability(), aggregate(), _diff(), main(), save(), _print(), run_configuration()

### Community 85 - "eval/stats.py"
Cohesion: 0.31
Nodes (3): paired_stats(), signed_rank_statistic(), westfall_young()

### Community 86 - "LatentReplayGenerator"
Cohesion: 0.09
Nodes (4): energy_boundary_loss(), LatentReplayGenerator, test_energy_boundary_loss_pushes_prototypes_down(), test_latent_generative_replay_gaussian_and_vae()

### Community 87 - "expert_formulation.py"
Cohesion: 0.22
Nodes (10): build_hypotheses(), _contract(), guards(), main(), save(), make_projection(), module_hash(), _print() (+2 more)

### Community 92 - "copy"
Cohesion: 0.33
Nodes (5): model_soup(), task_arithmetic(), _tensor_names(), ties_merge(), test_model_merging_utilities()

### Community 93 - "paper_wave1b.sh"
Cohesion: 0.29
Nodes (10): LD_LIBRARY_PATH, PYTHONPATH, run_ablation_cell(), run_c10(), run_c100(), run_drift(), run_growth(), paper_wave1b.sh script (+2 more)

### Community 94 - "paper_wave1c.sh"
Cohesion: 0.29
Nodes (10): LD_LIBRARY_PATH, PYTHONPATH, run_ablation_cell(), run_c10(), run_c100(), run_drift(), run_growth(), paper_wave1c.sh script (+2 more)

### Community 95 - "CONTRIBUTING.md"
Cohesion: 0.25
Nodes (4): CI/CD Pipeline (ci.yml), CI Build & Package job, CI Lint & Format job (ruff, black), CI Test & Coverage job (pytest, py3.10-3.12)

### Community 97 - "Literature Update 2026-09-28"
Cohesion: 0.27
Nodes (11): Literature Update 2026-09-28, BetaEdit / CP-MoE / MePo / AlphaEdit repro / SinglePrompt, Adapt before Continual Learning (arXiv:2506.03956), PTM_CIL Amendment 3 (A3.1-A3.3), EASE, The Mirage of Model Editing (arXiv:2502.11177), MOS, ObjectNet License Blocking Condition (+3 more)

### Community 98 - "build_projected_backbone"
Cohesion: 0.16
Nodes (4): build_projected_backbone(), optional_backbone(), ProjectedBackbone, main()

### Community 101 - "s4_datasets.py"
Cohesion: 0.13
Nodes (11): FolderTask, get_split_folder_tasks(), aggregate(), build_tasks(), ensure_cache(), main(), _save(), _print_matrix() (+3 more)

### Community 105 - "paper_wave2.sh"
Cohesion: 0.36
Nodes (7): LD_LIBRARY_PATH, PYTHONPATH, run_raw_c10(), run_raw_c100(), paper_wave2.sh script, SSL_CERT_FILE, step()

### Community 106 - "s5b_domains.py"
Cohesion: 0.19
Nodes (7): aggregate(), build_domain_cache(), combine_stream(), main(), save(), _print(), run_cell()

### Community 109 - "aggregate"
Cohesion: 0.15
Nodes (8): aggregate(), main(), save(), pool_tasks(), _print(), repartition(), run_configuration(), s5_protocols_entropy()

### Community 110 - "Stage 1 Results"
Cohesion: 0.15
Nodes (13): 15. Consolidated findings, 16. Pre-registered hypotheses and their verdicts, 17. Measurement bugs this programme found, 18. What this evidence does NOT say, 19. Open items and the stage order, 1. Why the programme was reset, 20. Artefacts, 2. E0 - the ceiling, before any new architecture (+5 more)

### Community 114 - "paper_reservoir_check.sh"
Cohesion: 0.40
Nodes (5): LD_LIBRARY_PATH, PYTHONPATH, paper_reservoir_check.sh script, SSL_CERT_FILE, step()

### Community 115 - "paper_wave1d.sh"
Cohesion: 0.40
Nodes (5): LD_LIBRARY_PATH, PYTHONPATH, paper_wave1d.sh script, SSL_CERT_FILE, step()

### Community 116 - "PAL-MoE v1: the published record"
Cohesion: 0.22
Nodes (10): PAL-MoE Benchmark Methodology, Research Toolkit (opt-in knobs), Shared Generalist Expert, CODE_REVIEW.md, PAL-MoE talk script (gncl), Catastrophic forgetting, CL approach families: replay, regularization, parameter isolation, Class-Shared Domain Shift (rotating MNIST) (+2 more)

### Community 117 - "e_tid_ceiling.py"
Cohesion: 0.33
Nodes (7): class_to_task_scores(), coverage(), fit(), knn_scores(), main(), run(), stack()

### Community 120 - "transfer_suite"
Cohesion: 0.24
Nodes (7): run_cell(), few_shot_accuracy(), main(), _print_summary(), probe_accuracy(), run_condition(), transfer_suite()

### Community 121 - "cifar100_full.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_full.sh script, SSL_CERT_FILE

### Community 122 - "cifar100_gate_ablation.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_gate_ablation.sh script, SSL_CERT_FILE

### Community 123 - "cifar100_resnet18_frozen.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_resnet18_frozen.sh script, SSL_CERT_FILE

### Community 124 - "cifar100_resnet18_multiseed.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar100_resnet18_multiseed.sh script, SSL_CERT_FILE

### Community 125 - "cifar10_resnet18_frozen.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, cifar10_resnet18_frozen.sh script, SSL_CERT_FILE

### Community 126 - "memory_pareto.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, memory_pareto.sh script, SSL_CERT_FILE

### Community 127 - "mnist_domainshift_shared.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, mnist_domainshift_shared.sh script, SSL_CERT_FILE

### Community 128 - "multiseed_cifar.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, multiseed_cifar.sh script, SSL_CERT_FILE

### Community 129 - "paper_all.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, paper_all.sh script, SSL_CERT_FILE

### Community 130 - "paper_all_resume.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, paper_all_resume.sh script, SSL_CERT_FILE

### Community 131 - "readout_ablation.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, readout_ablation.sh script, SSL_CERT_FILE

### Community 132 - "vit_cifar_multiseed.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, vit_cifar_multiseed.sh script, SSL_CERT_FILE

### Community 133 - "vit_cifar_quick.sh"
Cohesion: 0.40
Nodes (4): LD_LIBRARY_PATH, PYTHONPATH, vit_cifar_quick.sh script, SSL_CERT_FILE

### Community 134 - "NCMReadout"
Cohesion: 0.25
Nodes (3): NCMReadout, test_ncm_matches_a_hand_computed_reference(), test_ncm_running_mean_is_exact_when_fit_in_two_halves()

### Community 136 - "11. S8 - what a budget buys, and in which regime"
Cohesion: 0.25
Nodes (8): 11.1 The parameter axis: the tax does not close, 11.2 The memory axis: the only axis that buys anything, 11.3 The active axis: a pure cost, 11.4 The baseline question: the closed-form readout dominates, 11.5 Forgetting is a floor effect here, and that is a finding about the setup, 11.6 The reading, per regime, 11.7 What S8 does not say, 11. S8 - what a budget buys, and in which regime

### Community 141 - ".__call__"
Cohesion: 0.50
Nodes (4): _apply_corruption(), _disk_kernel(), _to_pil(), _to_tensor()

### Community 148 - "14. S11 - the confirmatory stage"
Cohesion: 0.29
Nodes (7): 14.1 Design, 14.2 The primary results, 14.3 H4 and H5: capacity against resolution, on the same seeds, 14.4 The secondary arms, 14.5 The six verdicts, 14.6 What S11 does not establish, 14. S11 - the confirmatory stage

### Community 149 - "e_tid2_ridge_router.py"
Cohesion: 0.48
Nodes (4): decode(), fit_ridge(), main(), run_cell()

### Community 151 - "12. S9 - does the decomposition survive a distribution shift?"
Cohesion: 0.40
Nodes (5): 12.1 Corruption: the tax grows, the realized value collapses, 12.2 Spurious cue: a learned shortcut, with a regime-dependent landing site, 12.3 Consistency checks, 12.4 What S9 adds to the story, 12. S9 - does the decomposition survive a distribution shift?

### Community 152 - "order_confusion"
Cohesion: 0.40
Nodes (4): 9.1 Findings, 9.2 What this does not settle, 9. S6 - order sensitivity, order_confusion()

### Community 155 - "13. S10 - scalability: the extra capacity is stranded behind the router"
Cohesion: 0.50
Nodes (4): 13.1 The scaling tables, 13.2 Findings, 13.3 The three-factor answer, stated honestly, 13. S10 - scalability: the extra capacity is stranded behind the router

### Community 158 - "10. S6b - designed difficulty: the routing tax *is* geometry-dependent"
Cohesion: 0.67
Nodes (3): 10.1 Findings, 10.2 Consequence for S8, 10. S6b - designed difficulty: the routing tax *is* geometry-dependent

### Community 159 - "7. S5 - the protocol axis"
Cohesion: 0.67
Nodes (3): 7.1 Findings, 7.2 What this changes, 7. S5 - the protocol axis

### Community 160 - "8. S5b - Domain-IL: is the routing tax a Class-IL artifact?"
Cohesion: 0.67
Nodes (3): 8.1 Findings, 8.2 Bugs this stage found, 8. S5b - Domain-IL: is the routing tax a Class-IL artifact?

## Ambiguous Edges - Review These
- `Function-space merge with accept/reject` → `Same-owner-only prototype merging`  [AMBIGUOUS]
  docs/v1/PALMOE_V2_SPEC.md · relation: semantically_similar_to

## Knowledge Gaps
- **171 isolated node(s):** `cifar100_full.sh script`, `PYTHONPATH`, `LD_LIBRARY_PATH`, `SSL_CERT_FILE`, `cifar100_gate_ablation.sh script` (+166 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1314 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **51 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Function-space merge with accept/reject` and `Same-owner-only prototype merging`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `Documentation Map` connect `Documentation Map` to `Literature Update 2026-09-28`, `PAL-MoE Measurement Contract (S0)`, `policies.py`, `Living Model Positioning`, `PTM_CIL_PREREG.md`, `Coupling Ablation Results`, `Research Map: Components and Literature Lineage`, `Owner-Side Residual Pre-registration`, `PAL-MoE v1: the published record`, `CERATA README`, `Interference Mechanism Pre-registration`?**
  _High betweenness centrality (0.084) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `PrototypeMemory` (e.g. with `ContinualEvaluator` and `ContinualTrainer`) actually correct?**
  _`PrototypeMemory` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `cifar100_full.sh script`, `PYTHONPATH`, `LD_LIBRARY_PATH` to the rest of the system?**
  _171 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_v1.py` be split into smaller, more focused modules?**
  _Cohesion score 0.038763796909492276 - nodes in this community are weakly interconnected._
- **Why does `PrototypeMemory` connect `test_v1.py` to `run_benchmark.py`, `.sync_on_prune`, `test_rejected_expansion_owner_is_newest_expert`, `run_benchmark`, `_mixed_memory_package`, `test_v3_anchors.py`, `Tensor`, `._validate_candidate_impl`, `diagnose_checkpoint.py`, `LatentReplayGenerator`, `Prototype`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `SharedEncoder` (e.g. with `_register_shared_encoder_archs()` and `SharedEncoderBackbone`) actually correct?**
  _`SharedEncoder` has 5 INFERRED edges - model-reasoned connections that need verification._