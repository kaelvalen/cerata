# CERATA

**Closed-form, Exactly Reversible, Auditable learning after deployment.** Named after
the *tabula cerata*, the Roman wax tablet: written on, and wiped clean without a trace.

This project was called **PAL-MoE** (Prototype-Anchored Lifelong Mixture of Experts)
until 2026-09-26. It started as a dynamic Mixture-of-Experts for class-incremental
continual learning (v1: one expert per task behind a prototype-anchored router, latent
replay instead of raw images). A pre-registered measurement programme then took that
design apart - including the finding that the expert bank adds nothing once routing
is analytic - and the code, and now the name, follow what the measurements support:

> **v3: learning after deployment is an API call on a frozen base.** One fixed,
> never-trained address space and three time scales - a FAST key-value memory (one
> item, O(1) write, exact delete), a MEDIUM closed-form edit from float64 sufficient
> statistics (one batch, order-invariant, subtractable) and a SLOW consolidation into
> frozen representation experts. Four guards (locality, reversibility, order
> invariance, a zero-parameter router) run on every `write`, `forget` and
> `consolidate`.

Architecture: [`docs/V3_ARCHITECTURE.md`](docs/V3_ARCHITECTURE.md). Every document:
[`docs/README.md`](docs/README.md). The v1 design and its benchmark tables:
[`docs/v1/README.md`](docs/v1/README.md).

## Where the evidence stands (2026-09-26)

On frozen ImageNet ViT-B/16 features, CIFAR-100 in 20 tasks, unless noted:

| Finding | Number | Source |
| :-- | :-- | :-- |
| A training-free prototype readout (NCM) beats v1 | 70.34 vs 59.30, 0 parameters, 307 KB | [`STAGE1_RESULTS.md`](docs/STAGE1_RESULTS.md) |
| A closed-form ridge readout is the strongest single model | wins on six backbones and four datasets | [`STAGE1_RESULTS.md`](docs/STAGE1_RESULTS.md) |
| The binding constraint is selection, not expert capacity | oracle routing +27 pp over the routed bank | [`STAGE1_RESULTS.md`](docs/STAGE1_RESULTS.md) |
| A learned routing address collapses when frozen; fixed retrieval does not | -28.7 pp (AC1); +0.048 pp, TOST-equivalent, 0 router parameters (AC3) | [AC1](docs/AC1_ADDRESS_FREEZE_RESULTS.md), [AC3](docs/AC3_ADDRESS_SPACE_RESULTS.md) |
| A continual class-level ridge router fixes most of the routing tax | +4.10 / +6.08 pp (`coherent` / `dispersed`), 6/6 seeds | [`E_TID2_RESULTS.md`](docs/E_TID2_RESULTS.md) |
| **Once routing is fixed, the expert bank is redundant** | +0.62 / -0.03 pp over ridge alone, inside the 1 pp SESOI | [`E_TID2_RESULTS.md`](docs/E_TID2_RESULTS.md) |
| Why: experts convert little of what they own | they rescue ~30 % of rescuable samples and break ~3 % of correct ones; re-grouping moves mass, not conversion | [`P2_BOUND_RESULTS.md`](docs/P2_BOUND_RESULTS.md) |

All of these use torchvision's ImageNet-1K ViT-B/16; the pre-trained-model CIL
literature uses ImageNet-21K weights and seven standard benchmarks, so none of the
numbers above is comparable to published tables yet.

What does **not** exist yet: any LM result. The 7B backend and the editing harness are
built and tested on a tiny random model only; the study is
[`docs/V3_LLM_PREREG.md`](docs/V3_LLM_PREREG.md) (proposed, not run). The SLOW path
has no positive result behind it on the vision side (the last two rows).

**Next.** [`docs/POSITIONING.md`](docs/POSITIONING.md) sets out what the literature
already owns, what is defensible, and the plan: first a pre-registered study of when
any expert bank - ours, EASE, MOS, MoTE - adds anything over an analytic router on the
standard benchmarks ([`docs/PTM_CIL_PREREG.md`](docs/PTM_CIL_PREREG.md); code built,
not run), then the LM study.

## Quick start

```bash
uv sync --extra dev            # or: pip install -e ".[dev]"; extras: "lm" (LM backend), "ptm" (timm)
```

```python
from cerata.api import Cerata, Batch, Example

model = Cerata(dim=768, num_classes=100, router="ridge_class", canary=canary_feats)
rec = model.write(Batch(z, y, task=0))    # MEDIUM: closed-form edit -> EditRecord
rec = model.write(Example(z1, 7))         # FAST: one memory row -> EditRecord
model.forget(rec.id)                      # exact delete / statistics downdate
rep = model.consolidate("by_arrival")     # SLOW: frozen representation experts
pred = model.predict(z_test)              # labels, logits, routed experts, per-sample source
st = model.state()                        # base hash, ordered edit log, digest
```

Every `EditRecord` carries the locality, reversibility, order and purity reports of
that call. The LM facade (`cerata.api.lm.CerataLM`) has the same calls:
`write("a sentence")` is a FAST memory row retrieved into the context,
`write(Batch(prompts, targets))` a MEDIUM down-projection edit.

```bash
export PYTHONPATH=.
python experiments/v3_api_smoke.py --synthetic   # write -> predict -> forget, every guard
python -m pytest                                 # the whole suite
python experiments/run_all.py --list             # Stage 1: what is done, what is left
```

Every runner, with its pre-registration: [`experiments/README.md`](experiments/README.md).
Study outputs go to `results/`, which is not tracked: a fresh clone re-runs the runners.

## Project structure

```text
cerata/
├── cerata/
│   ├── api/         # Cerata / CerataLM facades, GuardedEditor (the four guards), records
│   ├── core/        # frozen backbones, HF causal-LM adapter, features, constructions, hashing
│   ├── address/     # parameter-free retrieval over frozen keys (ExactCosineIndex)
│   ├── router/      # prototype, ridge_class, the router-purity guard
│   ├── memory/      # FastMemory (fast path)
│   ├── edit/        # float64 closed-form edits (LinearStats, DownProjEdit)
│   ├── experts/     # the Stage 1 ladder bank, consolidation policies
│   ├── readout/     # the S1 readout registry
│   ├── eval/        # metrics, measurement contract (schema), paired stats, editing harness
│   ├── arch/        # S1 contract: protocols and registries
│   ├── data/        # split MNIST / CIFAR / CIFAR-100 / folder, feature cache
│   ├── legacy/      # v1, frozen bitwise (models, memory, trainer, baselines, builder, trigger)
│   └── ...
├── pal_moe/         # compatibility only: the old name and the v1 import paths, same module objects
├── experiments/     # thin runners: v3, the study chain, Stage 1, the v1 benchmark
├── configs/         # validated JSON configs for the v1 benchmark
├── docs/            # architecture, pre-registrations, results, contracts; v1/ is the v1 record
└── tests/           # v1, S0/S1 contracts, v3 API and guards, v1 checkpoint shims, LM backend
```

Development workflow and the rules the codebase depends on:
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If you use **CERATA** (or its v1, PAL-MoE) in your research, please cite:

```bibtex
@software{cerata2026,
  author = {Hakbilen, Mehmet Arda},
  title = {CERATA: Closed-form, Exactly Reversible, Auditable Learning after Deployment},
  note = {formerly PAL-MoE: Prototype-Anchored Lifelong Mixture of Experts},
  url = {https://github.com/kaelvalen/cerata},
  version = {0.2.0},
  year = {2026}
}
```

Old code keeps working: `import pal_moe` (with a `DeprecationWarning`) resolves every
old import path, and old checkpoints unpickle, to the same `cerata` objects.

## License

This project is licensed under the [MIT License](LICENSE).
