# Positioning: what CERATA can claim in the literature, and the plan to get there

Written 2026-09-26 after a literature search (links in section 6). It is a plan, not a
result: nothing below has been run on a real benchmark or model.

## 1. The mistake to stop making

Presenting PAL-MoE as "a Mixture of Experts that learns, with advantages" contradicts
the repository's own strongest results. Once routing is done by a continual ridge
readout, the bank adds +0.62 / -0.03 pp (E-TID2), because its experts rescue ~30 % of
the samples they could and break ~3 % of those the readout had right (P2-BOUND). A
reviewer who opens the repository finds that first.

And every v3 component has an owner in the literature:

| v3 component | already published |
| :-- | :-- |
| continual ridge = joint ridge (MEDIUM path, E-TID2 G3) | ACIL (NeurIPS 2022), DS-AL (AAAI 2024), G-ACIL |
| random-feature ridge on a frozen ViT | RanPAC (NeurIPS 2023) |
| exact closed-form forgetting | ACU, Analytic Continual Unlearning (2025) |
| "selection is the binding constraint" | Kim et al., NeurIPS 2022: CIL = within-task x task-id prediction |
| per-task adapters with expert selection | EASE (CVPR 2024), MOS (AAAI 2025), MoTE (KBS 2025), Adaptive Expert Forest (2026) |
| key-value memory at a layer with a retrieval radius (FAST path) | GRACE (NeurIPS 2023) |
| side memory with routing for lifelong LM editing | WISE (NeurIPS 2024), MEMOIR (NeurIPS 2025) |
| undoable edits | SoLA (2026), OneEdit (2024) |

Protocol problems a reviewer sees at once:

- **backbone:** every vision number here uses torchvision's ImageNet-1K ViT-B/16; the
  PTM-CIL literature uses ImageNet-21K weights (RanPAC reports 92.2 % on CIFAR-100
  against our 76.77 % ridge - the comparison is not like for like, and looks bad);
- **benchmarks:** CIFAR-100 and our own `coherent` / `dispersed` constructions, not the
  standard seven (CIFAR-100, CUB, ImageNet-R, ImageNet-A, ObjectNet, OmniBenchmark,
  VTAB);
- **rivals:** none of RanPAC, ACIL, EASE, MOS, MoTE (vision) or GRACE, WISE, MEMOIR,
  AlphaEdit (LM) was run;
- **LM:** no result exists, and the pre-registered grid was infeasible as specified
  (`V3_LLM_PREREG.md`, amendment 2 and its measurement note).

## 2. What is defensible

1. **A per-sample decomposition of what an expert bank adds given a task-id
   predictor**, pre-registered, with a measured answer (P2-BOUND): rescue rate, breakage,
   rescuable mass. Kim et al. give the necessary-and-sufficient framing; the
   decomposition of a bank's contribution conditional on routing, applied across
   published banks, is not in the searched literature.
2. **An auditable learning contract.** Every `write` / `forget` returns measured
   locality, reversibility and order-invariance reports and a zero-parameter router
   check, across a fast memory, a closed-form edit and a slow bank, for vision and LM
   behind one API. The pieces exist (ACU, SoLA, GRACE); the contract and its measured
   cost do not.
3. **Parametric, exact edit removal on an LM.** The 2026 reversibility study finds
   reversal in existing editors is behavioural, not parametric. The MEDIUM path's
   downdate is parametric: fp-close to the never-edited delta, bitwise when the last
   edit goes. Only worth claiming with the LM result.

## 3. Paper A (do first): "When does an expert bank earn its keep?"

Pre-registration: [`PTM_CIL_PREREG.md`](PTM_CIL_PREREG.md). Code: built and smoke-tested
(`experiments/ptm/ptm_cil.py`, `experiments/ptm/extract_ptm_features.py`,
`cerata/eval/decomposition.py`, `cerata/data/ptm_benchmarks.py`). What remains is on
the GPU machine, in this order:

1. `python experiments/ptm/prepare_ptm_data.py`: downloads the processed splits linked
   from the RevisitingCIL README (ObjectNet by hand from OneDrive), unpacks them into
   `data/ptm/`, verifies the class lists and counts, checks each archive's md5 against
   the sums published in RevisitingCIL issue #5, and records md5, sha256 and counts in
   `data/ptm/MANIFEST.json`. CIFAR-100 downloads itself.
2. `uv sync --extra dev --extra ptm`, then per benchmark and backbone:
   `python experiments/ptm/extract_ptm_features.py --benchmark <b> --backbone in21k_1k --device cuda`.
3. Copy SimpleCIL's published accuracy for each benchmark into a dated amendment 1 of
   the pre-registration (the sanity veto), and approve or edit the rest of it.
4. Export the external banks: clone EASE, MOS (LAMDA-PILOT) and MoTE, run each with its
   published config on the same splits, and add a small exporter per method under
   `experiments/ptm/external/` that writes an `ExpertDump` (section 8 of the
   pre-registration). Commit the exporters before running them.
5. `python experiments/ptm/ptm_cil.py --benchmarks cifar100,cub,imagenet_r,imagenet_a,objectnet,omnibenchmark,vtab --backbones in21k_1k --external_dir results/ptm_cil/external --simplecil_reference results/ptm_cil/simplecil_reference.json --api --device cuda`
   (the reference file holds amendment 1's numbers as `{"<benchmark>__<backbone>": accuracy}`).
6. Write `PTM_CIL_RESULTS.md` against the outcome table, whatever row lands.

Venues: TMLR, CoLLAs, the CLVision workshop; an analysis track at a main conference if
the result is clean across the seven benchmarks. In this paper the MoE is the object
of study, not the claim.

## 4. Paper B (after A): "Auditable post-deployment learning"

Prerequisites, in order:

1. Adopt, edit or reject amendment 2 of `V3_LLM_PREREG.md`; with the measurements
   there, pin `mode="woodbury"` and re-run `experiments/v3/v3_lm_cost.py` on the study
   machine with a measured canary pass.
2. Run V3-LLM-1.
3. Add GRACE, WISE, MEMOIR and AlphaEdit through EasyEdit on the same CounterFact / zsRE
   cases and N, plus two metrics the others do not report: parametric distance after
   `forget`, and guard cost per call.
4. The vision side of B reuses paper A's `rp` readout through `Cerata` (its API arm
   already records guard reports and write time).

## 5. The name

"PAL-MoE" stated the one thing the evidence does not support, so the project became
**CERATA** (Closed-form, Exactly Reversible, Auditable learning after deployment) on
2026-09-26, after checking for collisions: TABULA was rejected (TabuLa and TabuLa-8B
are tabular-data LLM papers; `tabula`, `tabula-py`, `tabulaml` are taken on PyPI).
The v1 design keeps its name, PAL-MoE, as a historical record (`docs/v1/`); the
`pal_moe` package keeps old imports and checkpoints working. The GitHub repository
itself is renamed in its settings by the owner (GitHub redirects the old URL).

## 6. Sources (searched 2026-09-26)

- RanPAC: https://arxiv.org/abs/2307.02251 - code https://github.com/RanPAC/RanPAC
- DS-AL: https://ojs.aaai.org/index.php/AAAI/article/view/29670 - G-ACIL: https://arxiv.org/abs/2403.15706
- ACU: https://arxiv.org/abs/2505.12239
- AnaCP: https://openreview.net/forum?id=qQbvLU34F1
- Kim et al. 2022: https://arxiv.org/abs/2211.02633
- SimpleCIL / APER: https://arxiv.org/abs/2303.07338 - code and datasets https://github.com/zhoudw-zdw/RevisitingCIL
- LAMDA-PILOT (protocol reference): https://github.com/sun-hailong/LAMDA-PILOT
- EASE: https://arxiv.org/abs/2403.12030 - MOS: https://arxiv.org/abs/2412.09441 - MoTE: https://arxiv.org/abs/2506.11038
- Adaptive Expert Forest: https://arxiv.org/abs/2602.20911 - Bi-Level Routing MoE: https://arxiv.org/abs/2602.03473
- FeCAM: https://papers.neurips.cc/paper_files/paper/2023/file/15294ba2dcfb4521274f7aa1c26f4dd4-Paper-Conference.pdf - LayUP: https://arxiv.org/abs/2312.08888 - SLCA++: https://arxiv.org/abs/2408.08295
- GRACE: https://openreview.net/forum?id=xupL1Q0ft- - WISE: https://arxiv.org/abs/2405.14768 - MEMOIR: https://arxiv.org/abs/2506.07899
- AlphaEdit: https://arxiv.org/abs/2410.02355 - UltraEdit: https://arxiv.org/abs/2505.14679 - EasyEdit: https://github.com/zjunlp/EasyEdit
- SoLA: https://arxiv.org/abs/2603.11239 - On Reversibility: https://doi.org/10.3390/app16136567 - OneEdit: https://arxiv.org/abs/2409.07497 - reversing in-context edits: https://arxiv.org/abs/2410.12586
