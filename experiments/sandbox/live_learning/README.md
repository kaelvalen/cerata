# live learning sandbox (fast & dirty - NOT a pre-registered study)

The program this bench measures: [`docs/LIVING_MODEL_POSITIONING.md`](../../../docs/LIVING_MODEL_POSITIONING.md)
(transactional, auditable learning during interaction; ACID for model memory).

The question: can a small local model **learn during interaction** - quickly, durably
across sessions, cumulatively, and **with exact unlearning** - in a way we can measure?

Three mechanisms behind one dialogue loop, same model, same stream:

| mechanism | on teach | on query |
| :-- | :-- | :-- |
| `context` | (transcript only) | full recent transcript in the prompt |
| `memory` | fact stored in a text store | TF-IDF retrieval, notes injected into the prompt |
| `lora` | a few LoRA steps on the teaching sentence | weights only, no transcript |

The stream is general: personal facts, world facts, small procedures, one update, one
unlearn, plus capability probes. Metrics: immediate recall, delayed recall (after
fillers), cross-session recall, interference (do new facts break old ones), unlearn
accuracy (the forgotten token must not appear) and teaching latency.

Success signal for "promote to a lab-mode study": adaptation under ~5 s/turn, delayed
recall >= ~70 %, cross-session persistence, old-fact damage < ~5 %, exact unlearn with
capability intact. Soft gates only - this directory is for quick answers.

```bash
export LD_LIBRARY_PATH=/nix/store/38v10xhwhypb747h3z4c2i0a19hkiwx2-nvidia-x11-615.71.09/lib
export TRITON_LIBCUDA_PATH=$LD_LIBRARY_PATH   # NixOS: triton wants /sbin/ldconfig
.venv/bin/python experiments/sandbox/live_learning/run.py --mechanism memory \
    --model Qwen/Qwen2.5-0.5B-Instruct
```

Outputs go to `results/live_learning/`.

## First sweep (2026-10-02)

GPU shared with the Counterpart project, so most runs used Qwen2.5-0.5B; the 1.5B
numbers are from the windows where ~4.4 GB were free. Columns: immediate, delayed
(after 8 fillers), cross-session, capability; `teach` = seconds per teach.

| mechanism | model | imm | del | s2 | cap | teach |
| :-- | :-- | --: | --: | --: | --: | --: |
| context (12-turn window) | 1.5B | 1.00 | 0.17 | 0.00 | 1.00 | 0.0 s |
| memory (TF-IDF retrieval) | 1.5B | 0.94 | 0.83 | 0.88 | 0.75 | 0.0 s |
| lora (chat pairs, no replay) | 1.5B | 0.50 | 0.00 | 0.35 | 0.00 | 1.7 s |
| context | 0.5B | 0.78 | 0.17 | 0.00 | 0.25 | 0.0 s |
| memory | 0.5B | 0.83 | 0.83 | 0.94 | 0.50 | 0.0 s |
| lora (replay, lr 1e-4) | 0.5B | 0.00 | 0.33 | 0.00 | 0.25 | 1.9 s |
| lora (replay, lr 3e-4) | 0.5B | 0.33 | 0.50 | 0.06 | 0.00 | 3.3 s |

First read: the bundled context window does what it says (immediate perfect, gone by
the delayed probes); episodic memory is the only mechanism that carries everything
across the delay and the session boundary at this scale; LoRA trades memorisation
against capability - at a usable learning rate it forgets facts and damages the base
(function words and fact tokens leak into unrelated answers), and replay plus three
anchors does not fix it. The open knob is the consolidation side: when to promote a
memory into the weights, with what regularisation. Next: re-run the same table on
1.5B when the GPU is free, then a replay/regularisation sweep for LoRA.
