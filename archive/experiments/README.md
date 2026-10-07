# Experiments index

The runners orchestrate `cerata/`; they do not contain method logic. Every study
runner names its pre-registration in its docstring, was committed before it was run
(E-TID and E-TID2 are the recorded exceptions), and writes its JSON under `results/`
(untracked: a fresh clone re-runs the runner).

```bash
export PYTHONPATH=.
python experiments/<runner>.py --help
```

## Sandbox (`experiments/sandbox/`)

The October living-model program (transactional, auditable learning during
interaction) lives in [`sandbox/live_learning/`](sandbox/live_learning/README.md)
with its own README, STATE and `text/` / `confirm/` / `vlm/` subdirectories; docs
`docs/LIVING_MODEL_*`.

## v3 (the current architecture, `archive/docs/V3_ARCHITECTURE.md`)

| Script | Purpose |
| :-- | :-- |
| `v3/v3_api_smoke.py` | `write -> predict -> forget -> predict` through `Cerata`, every guard; `--synthetic` needs no cache (runs in CI) |
| `v3/v3_anchors.py` | Re-runs the stored S11 E0, AC3 and E-TID2 cells and reports every delta (V3 section 7) |
| `v3/v3_etid2_api.py` | E-TID2 through the v3 API; called by `v3/v3_anchors.py` |
| `v3/v3_lm_cost.py` | Times a guarded MEDIUM write's solve work (accumulate vs woodbury) and projects it to a 7B `d_ff` |

## PTM-CIL (`archive/docs/PTM_CIL_PREREG.md`, proposed, not run)

| Script | Purpose |
| :-- | :-- |
| `ptm/prepare_ptm_data.py` | Download (Google Drive via gdown; ObjectNet by hand from OneDrive), unpack into `data/ptm/`, verify class lists and counts, write `data/ptm/MANIFEST.json` with archive sha256 |
| `ptm/extract_ptm_features.py` | Frozen timm ViT-B/16 (in21k / in21k_ft_in1k) features for the seven benchmarks, original labels (needs the `ptm` extra) |
| `ptm/ptm_cil.py` | NCM / ridge / RanPAC readouts, our bank and external `ExpertDump`s routed and decomposed (including A3.1's alternative rules and the A3.2 `pal_l3-ridgewp` dump, amendment 3), the guarded API arm; `--synthetic` runs in CI |
| `ptm/ptm_headroom.py` | A3.3: the representation headroom scan - the joint arm (`L2a_shared_joint`) against frozen, `ridge`/`rp` readouts, the 10 pp labels; `--synthetic` covered by the test suite |
| `ptm/external/ease_export.py` | EASE (official code, published configs) on the pinned splits as an `ExpertDump`; `--check` is amendment 2's fidelity veto against the official logs (needs a clone of the EASE repository and timm) |

## Diagnostic studies after Stage 1 (in the order they were run)

Each row is a pre-registration / results pair in `docs/`. `DIAGNOSIS_SYNTHESIS.md`
explains why the chain turned from the router's ranking to its decision rule, then
to the address, then to task identity.

| Script | Study | Docs |
| :-- | :-- | :-- |
| `stage1/diagnostics/rr_ranking.py` | Router ranking: a supervised `z -> T` gate against prototype ranking | `ROUTER_RANKING_*` |
| `stage1/diagnostics/rr_factorial.py` | Representation x routing-objective factorial | `REPRESENTATION_ROUTING_*` |
| `stage1/diagnostics/decision_routing.py` | Comparative vs pointwise supervision of the routing scores | `DECISION_ROUTING_*` |
| `stage1/diagnostics/aggregation.py` | Winner-take-all vs uniform top-3 aggregation | `AGGREGATION_*` |
| `stage1/diagnostics/expert_formulation.py` | E1: one shared decision space for every expert | `EXPERT_FORMULATION_*` |
| `stage1/diagnostics/e2_evidence.py` | E2: evidence-producing experts | `E2_EVIDENCE_*` |
| `stage1/diagnostics/coupling_ablation.py` | Accumulated evidence re-alignment (C0 vs C1) | `COUPLING_*` |
| `stage1/diagnostics/interference.py` | New prototypes against old projections (passive instrumentation) | `INTERFERENCE_*` |
| `stage1/diagnostics/intervention.py` | Cutting the non-owner evidence path | `INTERVENTION_*` |
| `stage1/diagnostics/owner_side.py` | Owner-side residual of the E2 collapse | `OWNER_SIDE_*` |
| `stage1/diagnostics/ac1_address_freeze.py` | Freezing the shared query in the routing path | `AC1_ADDRESS_FREEZE_*` |
| `stage1/diagnostics/ac3_address_space.py` | Fixed retrieval address vs the learned evidence address | `AC3_ADDRESS_SPACE_*` |
| `stage1/diagnostics/e_tid_ceiling.py` | Offline task-ID ceiling on the frozen space (exploratory) | `E_TID_RESULTS.md` |
| `stage1/diagnostics/e_tid2_ridge_router.py` | Continual class-level ridge router over the L3 bank (exploratory) | `E_TID2_RESULTS.md` |
| `stage1/diagnostics/p2_bound.py` | Why the bank adds < 1 pp over ridge: decomposition + consolidation policy | `P2_BOUND_*` |

## Stage 1 (`archive/docs/STAGE1_PLAN.md`, `archive/docs/STAGE1_RESULTS.md`)

`stage1/infra/run_all.py` is the single entry point: `--list` shows what is done and what is left,
`--dry-run` prints the exact commands.

| Script | Stage |
| :-- | :-- |
| `stage1/diagnostics/e0_representation_ceiling.py` | E0: representation ceiling and adapter headroom on cached features |
| `stage1/ladder/s2_ladder.py` | S2: the complexity ladder under one fixed recipe (the shared ladder code now lives in `cerata/experts/ladder.py`) |
| `stage1/ladder/s3_run.py`, `stage1/ladder/s3_backbones.py`, `stage1/ladder/s3_report.py`, `stage1/ladder/s3_check_vit_features.py` | S3: backbone generalization (driver, per-backbone cache, table, ViT-feature regression check) |
| `stage1/ladder/s4_datasets.py` | S4: dataset generalization |
| `stage1/ladder/s5_protocols.py`, `stage1/ladder/s5b_domains.py` | S5: Class-IL vs Task-IL; S5b: Domain-IL |
| `stage1/ladder/s6_order.py`, `stage1/ladder/s6b_difficulty.py` | S6: order sensitivity; S6b: the designed `coherent` / `dispersed` contrast |
| `stage1/ladder/s7_transfer.py` | S7: representation transfer at every checkpoint |
| `stage1/ladder/s8_budget.py`, `stage1/ladder/s8_report.py` | S8: what a capacity / memory budget buys, per routing regime |
| `stage1/ladder/s9_corruptions.py`, `stage1/ladder/s9_robustness.py` | S9: controlled shifts and whether the decomposition survives them |
| `stage1/ladder/s10_scaling.py` | S10: bank capacity vs candidate count |
| `stage1/ladder/s11_confirmatory.py` | S11: the pre-registered confirmatory tests (paired stats now in `cerata/eval/stats.py`) |

## v1 benchmark (the published v1 record, `archive/docs/v1/BENCHMARK.md`, `archive/docs/v1/README.md`)

| Script | Purpose | Typical use |
| :-- | :-- | :-- |
| `stage1/infra/run_benchmark.py` | The v1 benchmark: 14+ method ids, all knobs, single seed | `--config configs/...` |
| `stage1/infra/run_benchmark_multi.py` | Multi-seed driver (mean ± std), `--aggregate_only` to rebuild an aggregate | `--seeds "42 1 2" --config ...` |
| `stage1/infra/run_ablation.py` | Controlled grid: loss components, init, gate, top-k, encoder | `--configs "OOD"` |

Tools:

| Script | Purpose |
| :-- | :-- |
| `stage1/infra/paper_report.py` | Scan `results/` into `results/paper_report.md` + Pareto/growth/latency figures |
| `stage1/infra/plot_results.py` | Standalone figure generation for single/multi-seed JSONs |
| `stage1/infra/measure_latency.py` | Per-sample forward latency (batch 1/128) for the runner geometries |
| `stage1/infra/diagnose_checkpoint.py` | Per-task expert-accuracy / routing-share diagnosis (router vs expert bottleneck) |
| `stage1/diagnostics/debug_routing_asymmetry.py` | Early routing-funnel dump; its `results/routing_asymmetry_debug.json` is cited in `cerata/legacy/adaptation/ttt.py` |
| `stage1/infra/merge_experts.py` | Merge trained experts into one serving head (soup / TIES / task arithmetic) |
| `stage1/infra/repair_missing_rows.py` | Merge a single-method re-run into existing per-seed JSONs (dry run by default) |
| `stage1/infra/prepare_tiny_imagenet.py` | Flatten the official Tiny-ImageNet train layout into an ImageFolder tree (symlinks) |

Recipes (`recipes/`, the v1 paper queue):

| Script | Purpose |
| :-- | :-- |
| `paper_all.sh`, `paper_all_resume.sh` | Full paper queue (wave 1 then wave 2), one detached command |
| `paper_wave1b.sh`, `paper_wave1c.sh`, `paper_wave1d.sh` | Consolidation waves (equal-byte, growth, ablation, drift, capacity) |
| `paper_wave2.sh` | Raw equal-byte, MIR, Tiny-ImageNet, slow regenerations, latency, domain shift |
| `paper_reservoir_check.sh` | Equal-byte fairness check with reservoir-sampled replay baselines |
| `paper_status.sh` | Progress / FAILED / summary check for the running queue |
| `multiseed_cifar.sh`, `cifar100_resnet18_multiseed.sh`, `vit_cifar_multiseed.sh` | 3-seed error bars per benchmark |
| `cifar100_full.sh`, `cifar100_resnet18_frozen.sh`, `cifar10_resnet18_frozen.sh`, `vit_cifar_quick.sh` | Single-benchmark runs |
| `cifar100_gate_ablation.sh`, `readout_ablation.sh` | Focused ablations |
| `memory_pareto.sh`, `mnist_domainshift_shared.sh` | Memory-budget sweep and domain-shift pilot |

`stage1/infra/run_benchmark.py` writes `benchmark_results_seed<s>.json` (metrics, byte accounting,
routing diagnostics) and `benchmark_meta_seed<s>.json` (seed, git hash, args, duration)
into the output directory.
