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

## Amendment 2 (2026-09-27, proposed, before any external bank was run)

Written with the EASE exporter (`experiments/external/ease_export.py`, `common.py`),
before it was run on any real benchmark (it has run only on synthetic images with random
weights, `tests/test_external_export.py` and a CPU smoke).

**1. Source.** EASE runs from its official repository (github.com/sun-hailong/CVPR24-Ease,
commit `6d67508`), whose `Learner` is unchanged; LAMDA-PILOT's copy is identical in
`models/ease.py` and `backbone/vit_ease.py`. The hyper-parameters are the repository's
per-dataset, per-backbone configs (`exps/ease_{cifar,cub,ina,inr,obj,omni,vtab}[_in21k].json`):
`vit_base_patch16_224_ease` for `in21k_1k`, `vit_base_patch16_224_in21k_ease` for `in21k`.

**2. What the exporter replaces, and nothing else.**

- The split: the pinned one (section 2). Three official configs use another split
  (CIFAR-100 B5 Inc5, CUB B10 Inc10, ObjectNet B10 Inc10); their other hyper-parameters are
  used as published on the pinned split.
- The class order: 1993 (identity for VTAB, which the official config also leaves
  unshuffled). The official trainer seeds the class order with the run seed; here the run
  seed (0-5) seeds only `torch`, as the official `_set_random` does.
- The data: our verified splits, loaded into the official `DataManager` class with the
  official transforms.
- The weights: the official code asks timm 0.6.12 for `vit_base_patch16_224` and
  `vit_base_patch16_224_in21k`; timm 1.x resolves the first to a different checkpoint
  (`augreg2`). The names are pinned to this study's two tags. **Port veto:** with its
  fresh (zero-output) adapter, the EASE backbone must equal the stock timm model to
  `1e-4` on random images before training (measured on the smoke: 2.8e-6); otherwise
  the run stops.
- `--micro_batch`, if the RTX 5060 cannot hold a published batch: the batch's gradient is
  accumulated over chunks, so the optimiser sees the published batch size. Recorded in
  each dump's meta.

**3. Forced expert.** `expert_pred[:, t]` is the cosine classifier on adapter t's
subspace alone: adapter t's [CLS] feature against every class's prototype in subspace t
(EASE's own synthesised prototypes for the classes older than t), argmax over all
classes. `native_pred` is EASE's `forward(test=True)`. A second dump, EASE's native
logits restricted to task t's classes (`ease-wp`), goes to `exploratory/`; the runner does
not read it and it enters no endpoint.

**4. Fidelity veto (new, section 5).** For the four benchmarks whose official split is
the pinned one, the mean over the six seeds of EASE's native final accuracy must be within
2 pp of the official log's final accuracy (`logs/ease/<dataset>/0/<inc>/_1993_*.log`,
last "CNN top1 curve" entry), per backbone:

```text
                 in21k_1k   in21k
imagenet_a        0.5853    0.5504
imagenet_r        0.7733    0.7617
omnibenchmark     0.7397    0.7480
vtab              0.9336    0.9355
```

(`ease_export.py --check`.) A failure on any of these excludes the EASE bank from the
outcome table: it is reported as a failed port, not as a result. The other three
benchmarks have no comparable log; their native accuracy is reported against the
`rp` readout only.

**5. Budget, fixed now.** The 48 h feasibility veto covers the readout grid, not
the external banks. EASE trains one adapter per task through the whole ViT. If the first
EASE run projects the full EASE grid (7 benchmarks x 2 backbones x 6 seeds) past 7 days on
the RTX 5060, EASE runs on `in21k_1k` only; if that still projects past 7 days, on seeds
0-2, with the per-benchmark test reported as descriptive (three seeds cannot reach
p < 0.05 in an exact sign-flip test).

## Amendment 3 (proposed 2026-09-28; adopted 2026-09-28)

Adopted by the owner, unchanged, before any benchmark cell was run (amendment 1's
precondition still holds: nothing has been evaluated on a real benchmark); the
implementation was directed in the same session. Sections 1-4 and the primary outcome
table are unchanged. No number from the source brief that failed the re-check is used
(notably "ridge 89.1 vs expert 87.3", which is not a number that exists in this
repository).

Implementation record (2026-09-28, before the first real cell): A3.1 landed in
`cerata/eval/decomposition.py` and `experiments/ptm_cil.py`; A3.2 is the second own-bank
dump `pal_l3-ridgewp` in the same runner; A3.3 is `experiments/ptm_headroom.py`. No real
cell has run.

**1. The question this adds.** E-TID2 and P2-BOUND measured, in one frozen regime, that the
class-level ridge router fixes most of the routing tax and the bank then adds < 1 pp,
because the experts rescue ~30 % of the samples they could. Three explanations remain, one
test per explanation, all secondary to the primary endpoint (P2 of section 3) and read
together with it:

```text
A3.1  the decision rule wastes coverage (peak class vs summed posterior)
A3.2  the expert's readout, not its representation, loses the rescue
A3.3  the frozen representation itself has no headroom on this benchmark
```

The brief's own H2 sweeps 5-100 invented groups. This amendment does not invent clusterings
(P2-BOUND's `by_confusion` was hindsight-offline and cost 19.8x the stored bytes), so it
tests the decision rule at the only grouping the routing already has: the task partition.
Grouping by superclass or confusion is the *consolidation* question, and P2-BOUND Part B
owns it.

**2. A3.1 - group-summed decision rule.** Evaluated on the final logits, exactly where the
primary decomposition is. Per (readout, bank), three rules:

```text
owner_class     the primary rule, unchanged: owner of the argmax class
owner_task_sum  p = softmax(seen-class logits), temperature pinned at 1.0;
                task score = sum of p over the task's classes; route to the argmax task
own_bank_top2   our bank only, exploratory: if the top two task scores are within 0.1 of
                the total mass, both experts are evaluated and the prediction is the one
                with the higher expert-space max class score; external banks have no
                comparable score in the dump and do not enter this arm
```

`temperature = 1.0` and the `0.1` margin are stated modeling choices, pinned here, not
tuned after the first cell. The `owner_task_sum` rule can route away from the owner of the
argmax class, so `r_not_tau > 0` is possible there; the identity still holds algebraically
and the premise veto (`r_not_tau = 0`) applies to `owner_class` only. For each rule:
`tau`, group-decision C@1 (the task of the true class), system accuracy, and `m / rho /
beta / P2`, paired by seed and read against the primary row.

Reading, fixed before the data:

- if `owner_task_sum` raises `tau` by >= 2 pp at equal or better system accuracy, keep it
  as a reported alternative rule and re-read P2 under it;
- if every bank's P2 stays inside +/-1 pp under every rule, the redundancy reading
  generalizes over the decision rule; this is the brief's H2 falsified at the rule level
  and is reported as such;
- if `own_bank_top2` lifts our bank's P2 above +1 pp in 6/6 seeds, the expert's value is in
  resolving ambiguous coarse decisions, not in its representation; otherwise the arm
  retires.

**3. A3.2 - expert-subspace closed-form readout (our bank only).** For `pal_l3`, the
forced-expert prediction is recomputed with a float64 ridge fitted **in the expert's own
adapted representation** (`model.apply_experts(z, t)`), not with the native cosine
prototype: accumulate the sufficient statistics per task when the task arrives, select the
penalty with the section 2 rule (80/20 split, same grid) on that expert's own training
data, record it per cell, and argmax over all classes with the seen mask. Everything else
(`task_of_class`, `native_pred`, the dump format) is unchanged; the result is a second dump,
`pal_l3-ridgewp`, so the primary dump keeps the frozen definition of "forced expert t".

This is E-TID2's exploratory `ridge_masked` moved inside the expert: there the shared
readout was masked to the routed task's classes; here the readout is fitted in the expert's
adapted space. One variable: the readout inside the fixed expert placement.

Reading, fixed before the data (per benchmark, primary backbone, paired over seeds):

- `rho` improves by >= 0.20 (about 30 % -> >= 50 %) on >= 4 of 7 benchmarks: the value-path
  limit is partly a readout limit; the next expert formulation should carry its own
  closed-form readout;
- `rho` improves by < 0.05 everywhere: the adapted representation itself is the limit;
  readout swaps inside the expert do not fix it and the A3.3 explanation takes over;
- in between: benchmark-dependent, reported per benchmark, no pooled claim.

**4. A3.3 - representation headroom scan (descriptive).** For each benchmark on the primary
backbone: train the ladder's joint arm (`L2a_shared_joint`, `joint=True`: all tasks at once,
offline - an upper bound, not a stream) and run the `rp` and `ridge` readouts (section 2
procedure) on the adapted features. Report `rp(joint) - rp(frozen)` final accuracy.

This is not a PETL experiment: no sequential adaptation, no pretraining data, no per-task
order; PETL and anything adaptive on ObjectNet stay out of scope (section 7) - the ObjectNet
license needs checking first (`LITERATURE_UPDATE_2026-09-28.md` section 2). This is the only
training arm this amendment adds; it runs on `in21k_1k`, seeds 0-2, and only after A3.1 and
A3.2. If its projection would break section 5's 48 h veto, it runs on seed 0 as a
single-seed diagnostic, or is dropped with the drop recorded - never the other way around.

Reading, thresholds fixed before the data: gap >= 10 pp marks the benchmark
**representation-limited** (routers and banks inside the frozen regime cannot close it; the
PETL line owns it); gap < 10 pp marks it **readout-limited** (the frozen-regime results are
the whole story). The labels go into `PTM_CIL_RESULTS.md` as labels, not as hypotheses with
a p-value.

**5. Code, guards and order.** Every new arm is implemented in `experiments/ptm_cil.py`
(or a small `experiments/ptm_headroom.py`) and committed before the first real cell; the
section 5 identity and dump-order vetoes apply to every rule and arm, with the A3.1 premise
note above; the API arm is unchanged. No hyperparameter is tuned after the first cell. A
failed new arm is reported as failed, not removed.

**6. What this does not license.** No PETL or first-session-adaptation claim (section 7
stands). No claim that summed-posterior routing is better in general. No external bank is
re-exported for A3.2. No benchmark, backbone or protocol is added. Nothing here changes the
primary outcome table's section 4 readings.

## Amendment 4 (2026-09-29, owner-directed port of the v3-restructure amendment 3)

The section 2 penalty rule - a **fixed** grid `10^-8 .. 10^8`, ties to the smallest
lambda - is retired for every future run, including A3.2's per-expert penalties and
A3.3's readouts. A veto check on a headroom ceiling cell found it is not scale-free: for
the 10,000-d random-projection readout the Gram diagonal is O(1e6), every grid point was
negligible (flat held-out curve, 100 % train accuracy) and the tie rule slid to the
smallest lambda; the frozen baselines were silently under-regularised and `G` inflated.
The v3 restructure fixed this on its branch (`cf1b721`); this amendment ports the same
rule into `cerata.edit.stats.select_ridge`:

- the features entering the choice are normalised to unit mean squared norm
  (`s = sqrt(mean ||phi(z)||^2)` on the pre-stream data);
- `lambda = c * trace(A_feat)/d` with `c` from `{1e-6, ..., 1e1}`;
- ties go to the **largest** `c` whose held-out MSE is within 0.1 % of the best
  (1-SE style), not to the smallest lambda;
- an edge choice extends the grid once by three decades; if it stays an edge the cell is
  flagged `converged = false` and its reading is withheld;
- `LinearStats` gains `bias_ridge`; the selector returns both penalties of the raw
  feature scale, so a fit on rescaled features is the same model.

`tests/test_ridge_select.py` pins the choice invariance (power-of-two, x100, edge
extension), the largest-c tie rule and the regression-target path. Any cell computed
before this amendment under the old rule - the CIFAR-100 pilot, the synthetic smoke - is
a pilot artifact, not a section 4/5 reading; a screen that used it is restarted.
