# CERATA

Transactional knowledge updates for frozen language models. The name comes from the
*tabula cerata*, the Roman wax tablet, which could be written on and wiped clean.

Cerata keeps a pre-trained language model frozen and stores each fact learned after
deployment as a separate entry: a note (the fact's statement, served in the system prompt
with an instruction to follow it) and, if the note alone does not produce the answer, a
low-rank adapter. An adapter is trained with a KL penalty toward the base model and is
accepted only if its divergence on 40 held-out canary prompts is below a threshold. A
router without trainable parameters (a match of the complete stored subject, then a
comparison of the query's relation with the stored relation) applies at most one entry
to a query and otherwise answers with the unmodified base model. Removing a fact deletes
its entry, and every answer reports which entry produced it.

The paper describing the system and its evaluation is in [`paper/`](paper/)
(`paper/build/main.pdf`).

## Current results

Qwen2.5-1.5B-Instruct, 1000 facts per dataset, every query served through the router,
adapter and note served together unless stated. Source:
`docs/LIVING_MODEL_CONFIRMATORY_PREREG.md`, entries 64 to 77. The default router
compares relation texts with a relation encoder (all-mpnet-base-v2 trained offline on
7500 triplets from facts disjoint from the dev split and the test stores, then frozen).

| Reading | CounterFact, router v4 (MiniLM, tau 0.631) | CounterFact, default (tau 0.750) | zsRE, default |
| :-- | :-- | :-- | :-- |
| Edit prompts answered | 0.996 | 0.996 | 0.989 |
| Paraphrases answered | 0.803 | 0.891 | 0.982 |
| Paraphrases answered, adapter only | 0.669 | 0.730 | 0.967 |
| Other relation of a stored subject: output contains a stored answer | 0.237 | 0.043 | not measured |
| Same relation, other subject (neighborhood): leakage | 0.016 | 0.017 | not measured |
| Synthetic-name distractors and NQ questions routed | 0 | 0 | 0 |
| Removed facts returning to the base answer (100 removed) | 1.00 | 1.00 | 0.84 (1.00 removing by content) |
| Facts that need an adapter (tiered store) | 0.162 | 0.162 | 0.206 |

Most of the trained encoder's gain is on the 23 CounterFact relations it was trained
on: on the 11 held-out relations, paraphrase routing rises only from 0.63 to 0.75 and
leakage is not reduced. Training the encoder on templates for 1785 Wikidata relations as
well does not change this: the held-out relations are mostly close neighbors of trained
ones (country of origin and location of formation, original language and language of
the work), and their boundary has to be learned from examples of that pair. With
general-purpose encoders, the routing threshold trades
paraphrase accuracy for leakage, and adapter training with subject-local anchors, notes
instead of adapters, a contrast against the store's keys and a check by the base model
did not improve on that trade-off. The divergence check uses 40 held-out canary prompts
with a threshold of 0.968; the two training prompts understate an adapter's effect about
seventeen-fold.

## Repository layout

```text
cerata/
├── cerata/                         # library code from the earlier programs (API, routers, edits, eval)
├── experiments/sandbox/live_learning/
│   ├── text/                       # exploratory text runs: store, router, controller, sessions
│   ├── confirm/                    # confirmatory harness: data adapters, pilot, routed and locality evals
│   ├── vlm/                        # vision-language runs (Qwen3-VL-2B)
│   └── STATE.md                    # current state and next steps
├── docs/                           # positioning, pre-registration records, review notes
├── paper/                          # LaTeX source, tables, figures, numbers generator
└── tests/
```

Every experiment was registered in `docs/LIVING_MODEL_SLICE_PREREG.md` (exploratory
phase) or `docs/LIVING_MODEL_CONFIRMATORY_PREREG.md` (confirmatory phase) before it was
run. Result files are written to `results/`, which is not tracked.

## Running

```bash
uv sync --extra dev --extra lm          # or: pip install -e ".[dev,lm]"
python -m pytest                        # test suite

# re-evaluate a saved store with router v4 and the three serving arms (requires the
# saved checkpoint); the locality and threshold analyses read its output
.venv/bin/python experiments/sandbox/live_learning/confirm/v4_eval.py \
  --ckpt results/live_learning/confirm/cf_n1000.ckpt \
  --facts results/live_learning/confirm/external_counterfact_n1000.json \
  --out results/live_learning/confirm/cf_n1000_v4_eval.json
.venv/bin/python experiments/sandbox/live_learning/confirm/serving_locality.py
.venv/bin/python experiments/sandbox/live_learning/confirm/relevance_eval.py
.venv/bin/python experiments/sandbox/live_learning/confirm/train_relation_encoder.py
.venv/bin/python experiments/sandbox/live_learning/confirm/router_v7_select.py --scorers relenc \
  --out results/live_learning/confirm/router_v8_select.json --prereg SS76
.venv/bin/python experiments/sandbox/live_learning/confirm/v7_eval.py --dataset counterfact \
  --selection results/live_learning/confirm/router_v8_select.json --suffix v8 --prereg SS76
.venv/bin/python paper/make_numbers.py   # regenerate paper/numbers.tex

# build the paper
cd paper && mkdir -p build && nix run nixpkgs#tectonic -- main.tex -o build
```

Environment notes for the GPU runs (NixOS, shared 8 GB GPU) are in
`experiments/sandbox/live_learning/README.md`.

## Earlier work

The project started as PAL-MoE, a mixture-of-experts model for class-incremental
learning, followed by a closed-form vision study (CERATA v3). That material was removed
from the working tree in commit `894ad16`; it is available in the history at
`894ad16^` (for example `git show 894ad16^:archive/docs/README.md`). The `cerata/`
package still contains the library code from those programs, and `import pal_moe` is no
longer supported.

## Citation

```bibtex
@software{cerata2026,
  author = {Hakbilen, Mehmet Arda},
  title  = {CERATA: Transactional Knowledge Updates for Frozen Language Models},
  url    = {https://github.com/kaelvalen/cerata},
  year   = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
