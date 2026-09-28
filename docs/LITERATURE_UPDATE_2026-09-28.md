# Literature update 2026-09-28: the hard-benchmark and LLM/VLM brief, checked against primary sources

Status: **a dated source check, not a result and not a plan change by itself.** It records
what was verified against the primary page, what failed verification, and what each verified
item changes for the two open pre-registrations. It follows the rule that rejected the
"APER" numbers in `PTM_CIL_PREREG.md` amendment 1: a number enters a pre-registration or a
results document only after the primary source is read.

## 0. Provenance

An external research brief (received 2026-09-28, 213 lines, 44 sources, in Turkish)
proposed: lock the router to a closed-form class-level rule, reduce experts to plastic
adapters where a representation gap is measured (ImageNet-A, ObjectNet, DomainNet), move
LLM single-fact learning to retrieval/KV memory, and keep parametric edit for batches; it
added a tear-out/bolt-on table and an H1-H10 roadmap. Its diagnosis matches this
repository's published results (Stage 1, E-TID2, P2-BOUND, AC1/AC3) and the two-paper plan
in `POSITIONING.md`; the sections below check the claims that are *new* to this repository.
`PTM_CIL_PREREG.md` amendment 3 (adopted 2026-09-28) is the one design change that
follows.

## 1. Verified against the primary source (checked 2026-09-28)

| # | Claim in the brief | What the primary page shows | What it changes here |
| :-- | :-- | :-- | :-- |
| 1 | TOSCA's re-run puts RanPAC and MOS neck-and-neck on ImageNet-A (54.76 vs 54.60) | arXiv:2502.14762v2 (TMLR): RanPAC 54.76 ± 1.2, MOS 54.60 ± 0.8, EASE 54.99 ± 1.0, SimpleCIL 48.77 ± 0.1 in the same column block of the comparison table | The strongest outside prior for the primary question ("banks ≈ analytic readout on IN-A") now has a primary source. Still not copied: the comparison here runs on our own protocol (prereg section 2). Do not quote a metric name from the paired columns without reading the PDF header - the pairs in that table do not all report the same statistic |
| 2 | "Adapt before CL" raises RanPAC on ObjectNet from 63.79 to 64.92 | arXiv:2506.03956v4 (AAAI 2026), Table 7: RanPAC +1.13; but MOS 62.75 → 60.06 (−2.69), FeCAM +2.19, L2P +2.58, DualPrompt +3.20 | The adaptation phase is not uniformly positive. This is the evidence behind A3.3's decision to keep PETL out of scope, not a method to adopt; the brief's blanket "PETL closes the gap" needs the per-method caveat |
| 3 | RanPAC's own paper has no ObjectNet number; its "OB" column is OmniBenchmark | arXiv:2307.02251v2 HTML: "ObjectNet" does not occur; "OmniBenchmark" does; the OB column (78.8 for the joint linear probe; 68.0 in the ResNet-50 appendix) is at OmniBenchmark level, not ObjectNet level | A RanPAC-ObjectNet number is either re-run by us or is a third party's (ACL Table 7 has one). Our own ObjectNet arm runs the readouts itself, so this only constrains literature comparisons |
| 4 | The 2026 items exist and say what is claimed | BetaEdit (arXiv:2605.09285, preprint, no venue), CP-MoE (arXiv:2605.20247, accepted CoLLAs 2026), MePo (arXiv:2602.07940, preprint), AlphaEdit reproducibility (arXiv:2606.26783, preprint), SinglePrompt (arXiv:2604.04420, CVPR Findings 2026) | Citable only at the venue shown; none supplies a number to a pre-registration |

## 2. Not verified; not usable here

- **"The frozen ridge beats the expert within-task, 89.1 vs 87.3."** No such number exists
  in this repository. 89.1 is a parameter count (89.1 K, `STAGE1_RESULTS.md`); the measured
  values are E-TID2's router C@1 0.8616 / 0.9681 with P2 +0.62 / −0.03 pp, and P2-BOUND's
  rescue rho ≈ 0.30. The brief's number is a conflation and must not enter any document.
- **"RanPAC 62.4 on IN-A in its own protocol."** The v2 HTML contains 62.6 (a ResNet-50
  table), not 62.4. Read the PDF table before quoting.
- **MOS's own table numbers** (IN-A 59.12, ObjectNet 63.62, ...), **SD-LoRA's** (55.96,
  72.82, ...) and the brief's other table values: not checked here; the same rule applies
  before any use. One cross-check passed incidentally: the brief's EASE IN-A 55.04 equals
  the official-log number pinned in `PTM_CIL_PREREG.md` amendment 2 (`in21k` 0.5504).
- **UltraEdit's scale claim**: "1M edits" in v1 versus "over 2M" in v3/OpenReview, and its
  TMLR 2026 status. This repository cites arXiv:2505.14679; treat the count as
  version-dependent. The brief itself flags this.
- **"Sparse memory finetuning was desk-rejected at ICLR 2026"** (OpenReview link in the
  brief): reported, not verified.
- **ObjectNet license**: the brief says the RanDumb authors replaced ObjectNet because its
  license forbids model training. Not checked. This is a blocking condition for any
  PETL/adaptation experiment on ObjectNet; A3.3's scope note keeps PETL out partly for
  this reason.
- The brief's H5 (Si-Blurry/online) and H6-H10 (LLM/VLM) arms are not checked here at all;
  they are pointers for later.

## 3. What follows (pointers, not decisions)

- `PTM_CIL_PREREG.md`, **amendment 3 (adopted 2026-09-28)**: A3.1 group-summed decision
  rule; A3.2 expert-subspace closed-form readout; A3.3 representation headroom scan. PETL
  itself stays out of scope (section 7 of the prereg).
- `V3_LLM_PREREG.md`: the brief's WILD point (**Mirage**, arXiv:2502.11177: single-edit
  success 96.8 % → 38.5 % without teacher forcing) is not adopted here. It bears directly
  on that document's primary endpoints (its section 3), so it should be decided in a
  dated amendment of that document before its development smoke, together with the
  already-proposed amendment 2 (`mode="woodbury"`, retrieval calibration). Not drafted
  now.
- VLM and online/blurry lines (CoIN arXiv:2403.08350; MCITlib arXiv:2508.07307;
  MoE-Adapters4CL arXiv:2403.11549; MePo; SinglePrompt): recorded as possible later
  lines, no action. A blurry/online protocol is outside `PTM_CIL_PREREG.md` section 7.

## 4. Sources checked

- TOSCA - https://arxiv.org/abs/2502.14762 (v2, TMLR)
- Adapt before Continual Learning (ACL) - https://arxiv.org/abs/2506.03956 (v4, AAAI 2026)
- RanPAC - https://arxiv.org/abs/2307.02251 (v2)
- BetaEdit - https://arxiv.org/abs/2605.09285
- CP-MoE - https://arxiv.org/abs/2605.20247 (CoLLAs 2026)
- MePo - https://arxiv.org/abs/2602.07940
- Reproducibility study of AlphaEdit - https://arxiv.org/abs/2606.26783
- SinglePrompt - https://arxiv.org/abs/2604.04420 (CVPR Findings 2026)
- The Mirage of Model Editing - https://arxiv.org/abs/2502.11177
