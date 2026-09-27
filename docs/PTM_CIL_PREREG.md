# PTM-CIL: when does an expert bank earn its keep over an analytic readout? - pre-registration

Status: **proposed, not run.** The runner (`experiments/ptm_cil.py`), the feature
extractor (`experiments/extract_ptm_features.py`), the benchmark protocol
(`cerata/data/ptm_benchmarks.py`) and the decomposition
(`cerata/eval/decomposition.py`) are committed with this document and pass a synthetic
smoke (`tests/test_ptm_cil.py`); no benchmark feature has been extracted and no number
below exists. The owner approves or amends this document, in a dated amendment, before
the first real cell runs.

## 1. The question

E-TID2 and P2-BOUND found, on CIFAR-100 with an ImageNet-1K ViT-B/16, that once routing
is done by a continual ridge readout, this repository's expert bank adds +0.62 / -0.03
pp, because the experts rescue ~30 % of the samples they could and break ~3 % of those
the readout had right. Two things limit that result: it is one bank (ours), and one
benchmark on a backbone the literature does not use.

> On the seven benchmarks and the ViT-B/16 backbones the pre-trained-model CIL
> literature uses, does **any** expert bank - ours or a published one (EASE, MOS, MoTE)
> - add accuracy once samples are routed by an analytic readout (NCM, ridge, RanPAC's
> random-feature ridge)? Where it does not, is the limit the rescue rate `rho`, the
> breakage `beta`, or the rescuable mass `m`?

Relation to prior work. Kim et al. (NeurIPS 2022) decompose CIL into within-task and
task-id prediction and show both are necessary and sufficient; this study measures,
per sample, what a bank does **conditional on** a given task-id predictor: the P2
identity (`cerata/eval/decomposition.py`). The readouts are the published frozen-feature
baselines: SimpleCIL (Zhou et al., IJCV 2024), ACIL-style closed-form ridge, and RanPAC
without PETL (McDonnell et al., NeurIPS 2023). Nothing here claims a new readout.

## 2. Design

**Backbones.** timm ViT-B/16, frozen, `num_classes=0` (768-d pooled output):

```text
in21k_1k   vit_base_patch16_224.augreg_in21k_ft_in1k    primary (timm 0.6's vit_base_patch16_224)
in21k      vit_base_patch16_224.augreg_in21k             secondary
```

**Benchmarks and protocol** (`cerata/data/ptm_benchmarks.py`, mirroring LAMDA-PILOT):
the processed splits linked from the RevisitingCIL repository; class order
`np.random.seed(1993); permutation(C)`, identity for VTAB; the evaluation transform
`Resize(256, bicubic) -> CenterCrop(224) -> ToTensor()` without normalisation, for train
and test features alike. Task splits, pinned here:

```text
cifar100        100 classes   B0 Inc10   10 tasks
cub             200           B0 Inc20   10
imagenet_r      200           B0 Inc20   10
imagenet_a      200           B0 Inc20   10
objectnet       200           B0 Inc20   10
omnibenchmark   300           B0 Inc30   10
vtab             50           B0 Inc10    5
```

Every method in this study is run by us on exactly these splits; no number is copied
from a paper into a comparison.

**Seeds.** 0, 1, 2, 3, 4, 5 (six, so an exact paired permutation test can reach
p = 0.031). The seed sets the random projection `W`, the ridge-selection split, and the
bank's initialisation. The class order is fixed (1993) for comparability with the
literature.

**Readouts** (all continual, closed-form or training-free):

```text
ncm     cosine to class means (SimpleCIL)
ridge   ridge on the raw 768-d features, one float64 accumulator
rp      ridge on relu(z @ W), W ~ N(0, 1) [768, 10000] (RanPAC without PETL)
```

The ridge penalties are selected **once, on task 0's training data** (RanPAC's rule:
80/20 seeded split, MSE, grid `10^-8 .. 10^8`) and pinned for the stream. RanPAC
re-selects per task; pinning is what keeps the readout order-invariant and exactly
removable, and the deviation is stated, not hidden.

**Banks.**

```text
pal_l3   this repository's L3 bank: one rank-8 residual adapter per task, shared readout,
         the S11 recipe (10 epochs, lr 1e-3, batch 128, lambda_func 1.0)
ease     EASE (Zhou et al., CVPR 2024), official code, its published hyper-parameters
mos      MOS (Sun et al., AAAI 2025), official code
mote     MoTE (Li et al., Knowledge-Based Systems 2025), official code
```

An external bank enters as an `ExpertDump` (section 8): its prediction under every
forced expert/task, per test sample, plus its own native prediction.

**Routing.** For each (bank, readout): route every test sample to the task that owns
the readout's argmax class; the system prediction is the bank's prediction under that
task.

**The API arm** (`--api`). The `rp` readout also runs through `Cerata` (one guarded
`write` per task, `random_features=10000`), and the last task is forgotten at the end.
It records the four guard reports and the wall time per write.

## 3. Endpoints

**Primary.** `P2(bank, rp) = acc(system) - acc(rp readout)` per benchmark, for each
bank, mean over six seeds, paired by seed. SESOI: 1 pp.

**Secondary.**

- `m`, `rho`, `beta`, `P2_max` for every (bank, readout, benchmark);
- final and average-incremental accuracy of each readout;
- each external bank's native accuracy against the `rp` readout alone (does the
  published bank beat RanPAC without PETL under identical splits and backbone?);
- the API arm's guard reports, time per write and stored bytes.

## 4. Outcome table (fixed in advance)

Read in order; the first matching row is the reading.

| result | reading |
| :-- | :-- |
| any veto in section 5 fails | an implementation failure; nothing below is read |
| for every bank, `P2(bank, rp) < +1 pp` on at least 6 of 7 benchmarks (in21k_1k) | **banks are redundant given an analytic router**, beyond our own bank and beyond CIFAR-100. Report `rho` / `beta` per bank |
| some bank has `P2(bank, rp) >= +1 pp` on at least 4 of 7 benchmarks, positive in 6/6 seeds each | that bank earns its keep over an analytic router; report where (`m`, `rho`) and which benchmarks |
| neither: `P2 >= +1 pp` on 1-3 benchmarks | benchmark-dependent value; read against H2 |
| `rho < 0.5` for every bank on every benchmark | the value-path limit (P2-BOUND) generalises across banks |

**Hypotheses stated before the data.**

- H1 (from P2-BOUND): `P2(pal_l3, rp) < +1 pp` on cifar100.
- H2: where banks help, they help more on the high-shift benchmarks (imagenet_a,
  objectnet, vtab) than on cifar100 and imagenet_r - isolation buys most where the
  frozen representation is worst. Checked by the sign of the mean P2 difference
  between the two groups; descriptive, not tested.

## 5. Vetoes

```text
identity         max |m*rho + P(!r,!tau)*rho' - P(r)*beta - P2| <= 1e-12, every cell
premise          r_not_tau = 0 (a right class implies its owner task), every cell
dump order       every ExpertDump's y equals the cache's task-concatenated test labels
api              rp through Cerata: argmax identical, max |d logit| <= 1e-8 vs direct
guards           every write passes reversibility, order invariance and purity; the
                 final forget passes
ridge            penalties selected on task 0 only (recorded per cell)
sanity           ncm final accuracy within 2 pp of the SimpleCIL number the owner copies
                 into amendment 1, from the APER paper, for the same backbone and split,
                 before the run; outside it, the feature pipeline is wrong and the
                 study stops
feasibility      extraction + all cells projected under 48 h on the RTX 5060 (8 GB)
```

The runner enforces the last two: `--simplecil_reference` supplies amendment 1's
numbers (a benchmark without one fails the sanity veto), and the run stops as soon as
the per-cell time projects the grid past 48 h. A failed veto writes the record, marked
`veto_failed`, and prints no result.

## 6. Statistics

Paired over seeds (`cerata/eval/stats.py`): exact permutation test over the 2^6 sign
flips, per (bank, benchmark); Holm across the seven benchmarks within a bank; TOST at
+/-1 pp for the redundancy reading. Secondary endpoints are reported with seed-level
intervals and read together, not tested.

## 7. What this licenses, and what it does not

- **Licenses:** a statement about whether expert banks add accuracy over analytic
  routers on the standard PTM-CIL benchmarks, with the per-sample decomposition of why;
  whether the published banks beat RanPAC without PETL under identical splits.
- **Does not license:** any claim about PETL / first-session adaptation (not run), other
  backbones, other protocols (task-free, blurry, domain-IL), or LMs.

## 8. Exporting an external bank

An `ExpertDump` (`cerata/eval/decomposition.py`) is an `.npz` named
`{method}__{benchmark}__{backbone}__seed{seed}.npz` with

```text
y              [N]     test labels in the stream's class indexing (class_order position)
task_of_class  [C]     owning task of every class
expert_pred    [N, T]  the bank's predicted label when forced to use expert / task t
native_pred    [N]     the bank's own prediction with its own routing
meta           JSON: method, repository commit, config, seed
```

Rows must follow `cerata.data.ptm_benchmarks.stream_test_order(...)`: task by task,
original ImageFolder order within a task. "Forced to expert t" means, per method: EASE -
the classifier on the features of adapter t only; MOS - the adapter retrieval replaced
by adapter t; MoTE - the expert filter replaced by expert t alone. Each exporter is a
small patch to the method's evaluation loop, committed under `experiments/external/`
before its bank is run.

## Amendment 1 (2026-09-27, before any benchmark cell was run)

Written after the data were prepared and the features extracted, before any readout,
bank or decomposition was computed on a real benchmark (one attempted run stopped while
loading the cache, before any cell; fixed in kaelvalen/cerata#5).

**1. Data integrity.** All six archives match the md5 sums the maintainers published in
RevisitingCIL issue #5 (cub.zip, imagenet-r.zip, ina.zip, omnibenchmark.zip, vtab.zip,
objnet.tgz), and every benchmark's class and image counts equal the reference table
(`experiments/prepare_ptm_data.py`, `data/ptm/MANIFEST.json`). CIFAR-100 is checked by
torchvision's own md5.

**2. The sanity references, from the primary source.** The SimpleCIL numbers come from
the RevisitingCIL repository's own run logs (github.com/zhoudw-zdw/RevisitingCIL, commit
`a2e6b71`, `logs/simplecil/<dataset>/exp_1993_pretrained_vit_b16_224_in21k.log`, the
last entry of each "CNN top1 curve"), not from a paper table read second-hand:

```text
cifar100__in21k        0.8128        cub__in21k            0.8677
imagenet_r__in21k      0.5455        imagenet_a__in21k     0.4885
objectnet__in21k       0.5358        omnibenchmark__in21k  0.7315
vtab__in21k            0.8437
```

- **Why the final accuracy transfers across splits.** SimpleCIL's final classifier is the
  set of all class means, whatever order and grouping they arrived in, so its last-step
  accuracy does not depend on the task split (the logs use B5/B10/B30 splits; this study
  pins B10/B20/B30). The average incremental accuracy does depend on it and is not used.
- **Which backbone.** The logs exist for `pretrained_vit_b16_224_in21k` only (timm 0.6's
  `vit_base_patch16_224_in21k`, this study's `in21k`); no published log exists for
  `in21k_1k`. The sanity veto is therefore checked on the `in21k` cells. An `in21k_1k`
  cell passes when the same benchmark's `in21k` check passed (same pipeline, other
  weights), and fails when no referenced backbone of that benchmark was run
  (`experiments/ptm_cil.py::sanity_veto`). This makes the secondary backbone
  mandatory for every benchmark in the primary run.
- **Rejected.** A set of numbers offered in review (attributed to the APER paper;
  CIFAR-100 76.21, CUB 61.31, ImageNet-R and ImageNet-A both 61.35, VTAB also 61.35) was
  not used: three benchmarks share one value, CUB contradicts the logs by 25 pp, and the
  cited sources were other papers. The logs above replace them.
- The ±2 pp band is unchanged. It also absorbs the timm version difference (0.6.12 in
  the logs, 1.0.30 here).

**3. Run order.** Extract `in21k` features for all seven benchmarks, then run the grid with
`--backbones in21k_1k,in21k` and
`--simplecil_reference results/ptm_cil/simplecil_reference.json` holding the seven
numbers above (as fractions).
