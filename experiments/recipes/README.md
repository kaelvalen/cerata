# Recipes

Shell recipes (the v1/pal_moe benchmark and paper queue); outputs land under `results/`.

| script | what |
| :-- | :-- |
| `cifar100_full.sh` | Full 20-task Split-CIFAR-100 recipe: frozen encoder + feature cache, 5 epochs/task, router-owner distillation, all comparison methods. |
| `cifar100_gate_ablation.sh` | Relative-vs-absolute validation gate ablation on 20-task CIFAR-100. Single-seed evidence (results/cifar100_relgate vs results/cifar100_big_frozen) |
| `cifar100_resnet18_frozen.sh` | Strong-backbone long-horizon recipe: frozen ImageNet ResNet-18 + feature cache on 20-task Split-CIFAR-100 (relative validation gate). |
| `cifar100_resnet18_multiseed.sh` | 3-seed error bars for the strong-backbone 20-task Split-CIFAR-100 recipe (frozen ImageNet ResNet-18, feature cache, relative validation gate). |
| `cifar10_resnet18_frozen.sh` | Strong-representation recipe: ImageNet ResNet-18 backbone (frozen) + feature cache on Split-CIFAR-10, no pretraining needed. |
| `memory_pareto.sh` | Memory-budget Pareto curve for PAL-MoE v2 (CIFAR-100 + frozen ViT-B/16). Pure (zero-raw-replay) and hybrid at three latent-memory budgets; the |
| `mnist_domainshift_shared.sh` | Class-shared domain-shift stress test: the same 10 classes under rotating input phases; compares the pure recipe with the stabilized generalist expert. |
| `multiseed_cifar.sh` | 3-seed error bars for the CIFAR tables (run overnight / on a free GPU). Note: the multi-seed driver forwards --pretrain_cache, so seed 42 reuses the |
| `paper_all.sh` | Full paper queue: wave 1b (corrected equal-byte + consolidation) followed by wave 2 (repair, raw-pipeline equal-byte, MIR, Tiny-ImageNet, slow |
| `paper_all_resume.sh` | Resume the full paper queue after a restart: wave 1d (E9 seeds 1/2 + E3-fast) followed by wave 2 (repair, raw equal-byte, MIR, Tiny-ImageNet, slow |
| `paper_reservoir_check.sh` | Fairness check: replay baselines with reservoir sampling at an equal byte budget. The AO10 appendix showed that the published 'recency' default |
| `paper_status.sh` | Quick status of the paper run queue (read-only).  |
| `paper_wave1b.sh` | Paper wave 1b: the corrected equal-byte sweep plus the remaining wave-1 consolidation runs. |
| `paper_wave1c.sh` | Paper wave 1c: resume after the seed-argument fix.  |
| `paper_wave1d.sh` | Paper wave 1d: resume after the server restart.  |
| `paper_wave2.sh` | Paper wave 2: repair, the raw-pipeline equal-byte sweep (the honest H2 test), external baselines, the serious benchmark and the slow regenerations. |
| `readout_ablation.sh` | Read-out ablation on the frozen ViT-B/16 stack: learned MoE head (moe) vs nearest-class-mean over stored latents (ncm) vs bias-corrected logits (bias), |
| `vit_cifar_multiseed.sh` | 3-seed error bars for the ViT-B/16 strong-backbone stack (PAL-MoE v2). Seeds 1/2 reuse the seed-invariant feature cache written by seed 42 |
| `vit_cifar_quick.sh` | PAL-MoE v2 quick validation: CIFAR-10 + frozen ImageNet ViT-B/16, 15 epochs/task, one expert per task (5), refresh anchors after a 10-epoch calibration. |
