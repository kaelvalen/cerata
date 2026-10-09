# Graph Report - cerata  (2026-10-09)

## Corpus Check
- 151 files · ~122,213 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 6 file(s) not represented in the graph (top: (none) 2, .pt 2, .cff 1)

## Summary
- 2245 nodes · 5291 edges · 110 communities (63 shown, 47 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 133 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `894ad16e`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DynamicMoE
- json
- constructions.py
- ContinualEvaluator
- ptm_benchmarks.py
- typing
- test_v1.py
- facade.py
- Tensor
- ContinualTrainer
- GuardedEditor
- protocols.py
- Tensor
- torch
- test_arch.py
- Cerata
- live-learning sandbox
- numpy
- test_schema.py
- dataclasses
- ExactCosineIndex
- AttentionRouter
- HFCausalLM
- expert_trigger.py
- arch/backbones.py
- LadderModel
- editing.py
- feature_cache.py
- test_ptm_cil.py
- arch/__init__.py
- LinearStats
- ledger_bandit.py
- domain_shift.py
- CerataLM
- decompose
- router/__init__.py
- decomposition.py
- run.py
- pilot.py
- FastMemory
- ResidualAdapter
- PrototypeRouter
- DERPP
- Transactional Delta Store Slice-1 Pre-registration
- test_ncm_and_bias_correction_heads
- v3 contract: learning after deployment is an API call on a frozen base
- Living Model Review 2026-10-04
- energy_boundary_loss
- LoRAMechanism
- pathlib
- test_rejected_expansion_owner_is_newest_expert
- eval/__init__.py
- Living model - frozen confirmatory summary (2026-10-09)
- Living model - two-paper plan (2026-10-09, user decision)
- build_single_head
- tost
- test_v3_anchors.py
- test_runner_baseline_loop_forwards_per_task_kwargs
- test_prepare_ptm_data_reunpacks_a_replaced_archive
- apply_config
- test_external_export.py
- MLPExpert
- ICaRL
- facts.py
- panel
- EWC
- StreamingEvaluator
- Living Model Positioning
- ladder.py
- geometry_report
- AGEM
- ReplayTrainer
- LatentReplayTrainer
- .fit
- merge.py
- SharedEncoder
- .pretrain_contrastive
- Registry
- FrozenModuleBackbone
- router_diagnostics
- calibrate_expert_temperatures
- Living Model Confirmatory Pre-registration
- TaskRouter
- RidgeClassRouter
- CI/CD Pipeline
- cerata

## God Nodes (most connected - your core abstractions)
1. `PrototypeMemory` - 107 edges
2. `SharedEncoder` - 90 edges
3. `MLPExpert` - 85 edges
4. `DynamicMoE` - 79 edges
5. `DynamicRouter` - 64 edges
6. `DeltaStore` - 52 edges
7. `LinearStats` - 41 edges
8. `ExpertBuilder` - 41 edges
9. `digest()` - 40 edges
10. `Cerata` - 38 edges

## Surprising Connections (you probably didn't know these)
- `test_anchor_ac3_fixed_proto_cell()` --calls--> `_max_abs()`  [INFERRED]
  tests/test_v3_anchors.py → cerata/api/guards.py
- `test_mask_unseen_hides_future_classes()` --calls--> `mask_unseen()`  [EXTRACTED]
  tests/test_arch.py → cerata/arch/readouts.py
- `test_random_backbone_is_seeded_and_frozen()` --calls--> `build_backbone()`  [EXTRACTED]
  tests/test_arch.py → cerata/arch/registry.py
- `test_der_erace_agem_smoke()` --uses--> `AGEM`  [INFERRED]
  tests/test_v1.py → cerata/legacy/baselines/agem.py
- `test_der_erace_agem_smoke()` --uses--> `DERPP`  [INFERRED]
  tests/test_v1.py → cerata/legacy/baselines/der.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Living-Model Guarantee Framework** — docs_living_model_positioning_transaction_ledger, docs_living_model_positioning_acid_guarantee_set, docs_living_model_positioning_state_identity_revocation, docs_living_model_positioning_router_integrity_invariant, docs_living_model_positioning_provenance [EXTRACTED 1.00]
- **v3 contract: three time scales and the four guards** — readme_fast_path, readme_medium_path, readme_slow_path, readme_four_guards [EXTRACTED 1.00]
- **live-learning transactional stack (ledger, controller, VLM store, keys)** — experiments_sandbox_live_learning_state_ledger_core, experiments_sandbox_live_learning_state_repair_bandit, experiments_sandbox_live_learning_state_vlm_deltastore, experiments_sandbox_live_learning_state_clip_keys [INFERRED 0.75]

## Communities (110 total, 47 thin omitted)

### Community 0 - "DynamicMoE"
Cohesion: 0.05
Nodes (23): TestTimeAdapter, DynamicMoE, DynamicRouter, test_add_expert_null_space_basis(), test_capacity_control_remaps_tracked_expert(), test_dynamic_router_and_entropy(), test_energy_trigger_flags_out_of_distribution_batches(), test_feature_cache_matches_raw_encoder() (+15 more)

### Community 1 - "json"
Cohesion: 0.06
Nodes (16): propose_and_commit (KLD cap), make_lora(), main(), main(), main(), main(), DeltaStore, main() (+8 more)

### Community 2 - "constructions.py"
Cohesion: 0.21
Nodes (4): args_data_dir(), build_construction(), separability(), superclass_of()

### Community 4 - "ptm_benchmarks.py"
Cohesion: 0.11
Nodes (13): class_order(), image_datasets(), load_raw_cache(), split_tasks(), stream_test_order(), task_increments(), test_stream_loader_matches_the_runners_label_order(), test_class_order_mirrors_pilot() (+5 more)

### Community 5 - "typing"
Cohesion: 0.06
Nodes (7): expert_param_count(), SampleBuffer, task_class_counts(), MIR, NaiveFineTuning, test_mir_selects_interfered_samples(), test_sample_buffer_reservoir_balances_tasks()

### Community 6 - "test_v1.py"
Cohesion: 0.06
Nodes (26): UncertaintyWeighter, build_moe(), LatentReplayGenerator, Prototype, PrototypeMemory, load_checkpoint(), save_checkpoint(), test_diagnose_detects_resnet_and_encoder_meta() (+18 more)

### Community 7 - "facade.py"
Cohesion: 0.12
Nodes (7): FrozenFeatureBackbone, digest(), file_digest(), module_digest(), tensor_digest(), _update(), decoder_layers()

### Community 8 - "Tensor"
Cohesion: 0.06
Nodes (6): build(), build(), build(), build(), build(), build()

### Community 9 - "ContinualTrainer"
Cohesion: 0.09
Nodes (13): ContinualTrainer, QuantitativeTrigger, test_distill_skips_out_of_range_owners(), test_loss_weighting_uncertainty_trains(), test_lwf_and_ema_terms_run(), test_optimizer_state_is_carried_across_rebuilds(), test_periodic_stability_and_ood_knobs(), test_quantitative_trigger() (+5 more)

### Community 11 - "GuardedEditor"
Cohesion: 0.15
Nodes (3): GuardedEditor, _max_abs(), EditRecord

### Community 12 - "protocols.py"
Cohesion: 0.08
Nodes (9): assert_shapes(), Backbone, ClassificationExpert, is_classification_expert(), is_representation_expert(), Readout, RepresentationExpert, Router (+1 more)

### Community 13 - "Tensor"
Cohesion: 0.11
Nodes (3): MLPReadout, RidgeReadout, test_ridge_is_closed_form_and_exactly_incremental()

### Community 14 - "torch"
Cohesion: 0.12
Nodes (30): Batch, one_hot(), by_arrival(), by_confusion(), check_gate(), confusion_matrix(), cross_fitted_confusion(), PolicyGateError (+22 more)

### Community 15 - "test_arch.py"
Cohesion: 0.10
Nodes (16): identity_check(), NCMReadout, describe(), build_legacy_router(), test_backbones_satisfy_the_contract(), test_duplicate_registration_is_an_error_unless_intentional(), test_expert_trains_away_from_the_identity(), test_experts_satisfy_the_contract_and_start_as_the_identity() (+8 more)

### Community 17 - "live-learning sandbox"
Cohesion: 0.22
Nodes (13): living model program, LIVING_MODEL_CONFIRMATORY_PREREG.md, LIVING_MODEL_POSITIONING.md, LIVING_MODEL_REVIEW_2026-10-04.md, LIVING_MODEL_SLICE_PREREG.md, one jointly-trained delta carries 256 facts, CLIP keys, cross-session hybrid (ledger_session) (+5 more)

### Community 18 - "numpy"
Cohesion: 0.14
Nodes (13): main(), adapt_counterfact(), build_raw_alignment(), canary_kld(), ci(), easyedit_summary(), gen(), gen_raw() (+5 more)

### Community 19 - "test_schema.py"
Cohesion: 0.06
Nodes (37): aggregate_runs(), _bootstrap_ci(), build_run_record(), _clean(), _deep_defaults(), _default_run_id(), _get(), load_run_record() (+29 more)

### Community 20 - "dataclasses"
Cohesion: 0.16
Nodes (6): ConsolidationReport, Example, GuardViolation, Prediction, ReversibilityError, StateHash

### Community 22 - "AttentionRouter"
Cohesion: 0.07
Nodes (5): AttentionRouter, DistanceRouter, _project_to_null_space(), test_attention_router_routing_expansion_and_lock(), test_null_space_anchor_makes_new_row_orthogonal()

### Community 23 - "HFCausalLM"
Cohesion: 0.10
Nodes (5): down_projection(), HFCausalLM, hook(), CharTokenizer, lm()

### Community 24 - "expert_trigger.py"
Cohesion: 0.13
Nodes (7): energy(), EnergyTrigger, _RunningStats, AlwaysTrigger, TriggerEvaluationResult, test_always_trigger_fires_on_new_task(), evaluate()

### Community 25 - "arch/backbones.py"
Cohesion: 0.07
Nodes (11): CachedBackbone, optional_backbone(), ProjectedBackbone, RandomProjectionBackbone, _register_shared_encoder_archs(), _shared_encoder(), SharedEncoderBackbone, wrap_encoder() (+3 more)

### Community 26 - "LadderModel"
Cohesion: 0.12
Nodes (3): mask_unseen(), _entropy(), LadderModel

### Community 27 - "editing.py"
Cohesion: 0.14
Nodes (18): counterfact_scores(), EditCase, load_canary(), load_counterfact(), load_mquake(), load_zsre(), locality(), multihop_accuracy() (+10 more)

### Community 28 - "feature_cache.py"
Cohesion: 0.10
Nodes (10): build_feature_cache(), CachedFeatureEncoder, CachedTask, _collate_features(), _encode_split(), FeatureCache, FeatureTensorDataset, load_feature_cache() (+2 more)

### Community 30 - "test_ptm_cil.py"
Cohesion: 0.13
Nodes (9): RandomProjection, _batches(), test_palmoe_with_random_features_passes_every_guard(), test_random_projection_is_a_seeded_fixed_buffer(), test_sanity_veto_covers_an_unreferenced_backbone_through_a_referenced_one(), test_select_ridge_is_deterministic_and_scale_free(), test_stats_with_feature_map_continual_equals_one_shot_and_forgets(), test_stats_without_feature_map_keep_their_digest_and_layout() (+1 more)

### Community 31 - "arch/__init__.py"
Cohesion: 0.12
Nodes (9): build_projected_backbone(), CosineReadout, LinearReadout, LogisticReadout, _NullHandle, trainable_hook(), build_backbone(), build_expert() (+1 more)

### Community 32 - "LinearStats"
Cohesion: 0.08
Nodes (10): Contribution, _largest_within(), LinearStats, select_ridge(), _data(), _fitted(), test_flat_curve_takes_the_largest_c_and_flags_an_edge(), test_power_of_two_rescaling_returns_the_same_model() (+2 more)

### Community 33 - "ledger_bandit.py"
Cohesion: 0.12
Nodes (19): expert_serve(), final_readout(), hit(), main(), memory_serve(), Retriever, run_learned(), run_repair() (+11 more)

### Community 34 - "domain_shift.py"
Cohesion: 0.06
Nodes (20): apply_phase_shift(), permutation_transform(), _apply(), _rebuild_loader(), rotation_transform(), ShiftedTask, _TransformedDataset, get_split_cifar100_tasks() (+12 more)

### Community 35 - "CerataLM"
Cohesion: 0.07
Nodes (15): GuardConfig, CerataLM, LMPrediction, DownProjEdit, estimate_key_covariance(), KeyPrior, KeyValueContribution, _prior() (+7 more)

### Community 37 - "decompose"
Cohesion: 0.20
Nodes (11): decompose(), ExpertDump, _dump(), test_a3_1_expert_score_round_trips(), test_a3_1_task_sum_relaxes_the_premise_but_keeps_the_identity(), test_a3_1_top2_tie_breaks_with_the_expert_score_within_the_margin(), dump(), test_a3_1_top2_without_expert_scores_is_refused() (+3 more)

### Community 38 - "router/__init__.py"
Cohesion: 0.20
Nodes (4): assert_pure(), RouterPurityError, trainable_count(), test_router_purity_rejects_a_learned_router()

### Community 39 - "decomposition.py"
Cohesion: 0.21
Nodes (6): p2_decomposition(), route_by_owner(), route_by_task_sum(), route_top2_task_sum(), _task_mass(), test_a3_1_task_sum_rules_by_mass_not_by_the_peak()

### Community 42 - "pilot.py"
Cohesion: 0.10
Nodes (32): main(), main(), pairs(), eval_variant(), extract_delta(), main(), make_variant_lora(), set_lora_params() (+24 more)

### Community 44 - "ResidualAdapter"
Cohesion: 0.12
Nodes (5): IdentityExpert, LegacyClassificationExpert, ResidualAdapter, ResidualMLPExpert, test_residual_adapter_zero_rank_is_exactly_the_identity()

### Community 45 - "PrototypeRouter"
Cohesion: 0.14
Nodes (3): LegacyRouter, PrototypeRouter, test_prototype_router_candidate_set_and_distribution()

### Community 46 - "DERPP"
Cohesion: 0.12
Nodes (3): DERPP, ERACE, test_derpp_update_buffer_chunked_logits_are_consistent()

### Community 47 - "Transactional Delta Store Slice-1 Pre-registration"
Cohesion: 0.20
Nodes (15): Multi-fact Delta Capacity Curve, Transactional Delta Store Slice-1 Pre-registration, CLIP Image Keys, Summed Deltas Do Not Compose, Contrast-pair Expert (visual grounding), Cross-session E+Delta Hybrid, KL-anchored Delta Training, Magnitude Cliff (+7 more)

### Community 48 - "test_ncm_and_bias_correction_heads"
Cohesion: 0.20
Nodes (4): BiasCorrectionHead, NCMHead, build_cached_encoder(), test_ncm_and_bias_correction_heads()

### Community 49 - "v3 contract: learning after deployment is an API call on a frozen base"
Cohesion: 0.22
Nodes (11): MiniLM multilingual router, CERATA, Cerata API facade, CerataLM facade, EditRecord, FAST path (key-value memory), Four guards (locality, reversibility, order invariance, zero-parameter router), MEDIUM path (closed-form edit from float64 sufficient statistics) (+3 more)

### Community 50 - "Living Model Review 2026-10-04"
Cohesion: 0.22
Nodes (8): Living Model Review 2026-10-04, Adaptive Chameleon or Stubborn Sloth (Xie et al., ICLR 2024), EasyEdit, Paper Identity: Editing vs Systems, Router is a Dictionary (finding 2), Experiments README, Live Learning Sandbox README, Living-model program: transactional, auditable learning during interaction

### Community 52 - "LoRAMechanism"
Cohesion: 0.08
Nodes (4): chat(), ContextMechanism, LoRAMechanism, MemoryMechanism

### Community 53 - "pathlib"
Cohesion: 0.07
Nodes (39): main(), main(), triples_for(), main(), ask(), cur_logits(), emb_clip(), emb_dino() (+31 more)

### Community 54 - "test_rejected_expansion_owner_is_newest_expert"
Cohesion: 0.29
Nodes (3): ValidationGateResult, test_rejected_expansion_owner_is_newest_expert(), validate_candidate()

### Community 55 - "eval/__init__.py"
Cohesion: 0.22
Nodes (5): BenchmarkResult, holm(), paired_stats(), signed_rank_statistic(), westfall_young()

### Community 56 - "Living model - frozen confirmatory summary (2026-10-09)"
Cohesion: 0.29
Nodes (6): Carried caveats, Frozen table (N=1000), Gates and baselines, Living model - frozen confirmatory summary (2026-10-09), Open decision, The claim

### Community 58 - "Living model - two-paper plan (2026-10-09, user decision)"
Cohesion: 0.40
Nodes (4): Living model - two-paper plan (2026-10-09, user decision), Paper 1 - drawn now, from the frozen evidence, Paper 2 - close the gaps (later), Pointers

### Community 59 - "build_single_head"
Cohesion: 0.13
Nodes (7): build_encoder(), build_prototype_memory(), build_router(), build_single_head(), test_factory_builds_consistent_models(), test_param_reporting_fields(), test_runner_baseline_helpers()

### Community 62 - "test_v3_anchors.py"
Cohesion: 0.18
Nodes (12): _build_v1(), needs(), _out(), test_anchor_ac3_fixed_proto_cell(), test_anchor_etid2_cell(), test_anchor_etid2_report_recomputes_exactly(), test_anchor_s11_e0_cell_bitwise(), test_anchor_s11_e0_stored_means() (+4 more)

### Community 67 - "apply_config"
Cohesion: 0.24
Nodes (7): apply_config(), ConfigError, _describe(), load_config(), _range_ok(), _type_ok(), test_config_validation_and_precedence()

### Community 68 - "test_external_export.py"
Cohesion: 0.09
Nodes (7): BenchmarkSpec, StubNet, test_forced_expert_rejects_a_mismatched_classifier(), test_forced_expert_uses_only_its_subspace(), test_micro_batches_reproduce_the_full_batch_step(), __init__(), tiny()

### Community 69 - "MLPExpert"
Cohesion: 0.06
Nodes (16): ExpertBuilder, MLPExpert, test_adapter_experts_freeze_the_base_pathway(), test_candidate_training_does_not_inflate_usage(), test_capacity_control_through_expert_builder(), test_expert_width_growth_is_function_preserving(), test_function_preserving_expert_expansion(), test_inference_paths_restore_training_mode() (+8 more)

### Community 71 - "ICaRL"
Cohesion: 0.12
Nodes (3): ICaRL, ICaRLWrapper, test_icarl_herding_matches_bruteforce()

### Community 72 - "facts.py"
Cohesion: 0.08
Nodes (16): download(), main(), generate(), main(), make_subject(), make_word(), _norm(), _osa() (+8 more)

### Community 86 - "Living Model Positioning"
Cohesion: 0.38
Nodes (3): Living Model Positioning, The Living Model (transactional state), Provenance (per-answer, per-fact)

### Community 87 - "ladder.py"
Cohesion: 0.13
Nodes (12): build_readout(), iter_batches(), load_tasks(), set_seed(), evaluate(), forward_transfer(), probe_accuracy(), LevelSpec (+4 more)

### Community 88 - "geometry_report"
Cohesion: 0.26
Nodes (4): geometry_report(), nearest_other_margin(), silhouette_score(), test_geometry_report_separates_clusters()

### Community 102 - "merge.py"
Cohesion: 0.23
Nodes (5): model_soup(), task_arithmetic(), _tensor_names(), ties_merge(), test_model_merging_utilities()

### Community 103 - "SharedEncoder"
Cohesion: 0.06
Nodes (17): EMAEncoder, SharedEncoder, test_contrastive_pretraining(), test_conv_encoder_channels_and_feature_dim(), test_der_erace_agem_smoke(), test_encoder_in_channels_override(), test_encoder_stability_loss_gradient_flow(), test_frozen_encoder_stays_in_eval_mode() (+9 more)

### Community 123 - "calibrate_expert_temperatures"
Cohesion: 0.25
Nodes (3): calibrate_expert_temperatures(), fit_temperature(), test_expert_temperature_calibration_reduces_nll()

### Community 130 - "Living Model Confirmatory Pre-registration"
Cohesion: 0.43
Nodes (6): Living Model Confirmatory Pre-registration, Entity-Aware Router v2, Fuzzy Entity-Match Router v3, LoRA+ Recipe, Multi-Fact Delta Capacity (grouped delta), Strong-RAG Arms (instruct / qa / fewshot)

### Community 157 - "CI/CD Pipeline"
Cohesion: 0.40
Nodes (5): Benchmark Reproducibility Smoke job, Build & Package job, CI/CD Pipeline, Lint & Format job, Test & Coverage job

## Knowledge Gaps
- **30 isolated node(s):** `Fact`, `cerata`, `The claim`, `Frozen table (N=1000)`, `Gates and baselines` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 867 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **47 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PrototypeMemory` connect `test_v1.py` to `DynamicMoE`, `ContinualEvaluator`, `MLPExpert`, `typing`, `SharedEncoder`, `Tensor`, `ContinualTrainer`, `test_ncm_and_bias_correction_heads`, `test_rejected_expansion_owner_is_newest_expert`, `build_single_head`, `test_v3_anchors.py`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `PrototypeMemory` (e.g. with `ContinualEvaluator` and `ContinualTrainer`) actually correct?**
  _`PrototypeMemory` has 6 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Fact`, `cerata`, `The claim` to the rest of the system?**
  _30 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `DynamicMoE` be split into smaller, more focused modules?**
  _Cohesion score 0.050505050505050504 - nodes in this community are weakly interconnected._
- **Why does `SharedEncoder` connect `SharedEncoder` to `DynamicMoE`, `typing`, `._build_resnet`, `test_v1.py`, `MLPExpert`, `ContinualTrainer`, `calibrate_expert_temperatures`, `.pretrain_contrastive`, `test_arch.py`, `test_rejected_expansion_owner_is_newest_expert`, `arch/backbones.py`, `build_single_head`, `feature_cache.py`, `test_v3_anchors.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `SharedEncoder` (e.g. with `_register_shared_encoder_archs()` and `SharedEncoderBackbone`) actually correct?**
  _`SharedEncoder` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `json` be split into smaller, more focused modules?**
  _Cohesion score 0.06442058496853018 - nodes in this community are weakly interconnected._