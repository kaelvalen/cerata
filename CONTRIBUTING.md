# Contributing

## Setup

```bash
uv sync --extra dev                 # or: pip install -e ".[dev]"
uv sync --extra dev --extra lm      # + the LM backend (transformers, accelerate, bitsandbytes)
uv sync --extra dev --extra ptm     # + timm, for PTM-CIL feature extraction
```

`pyproject.toml` is the only dependency source; `uv.lock` pins it. Python >= 3.10.

## Checks (what CI runs)

```bash
.venv/bin/python -m pytest                          # the whole suite
.venv/bin/python experiments/v3_api_smoke.py --synthetic
.venv/bin/python experiments/ptm_cil.py --synthetic --seeds 0,1 --api
ruff check . && black --check .
```

- Tests that need untracked data skip with a reason (`pytest -rs` lists them): the
  stored `results/` JSONs for the anchor tests, `data/editing/` for the editing-data
  loaders, `transformers` for the LM backend tests.
- **Check formatting under Python 3.12.** `black==26.5.1` formats long `assert`
  messages differently depending on the interpreter it runs on, not only on
  `target-version`, and CI runs 3.12:

  ```bash
  uv venv --python 3.12 /tmp/black312
  uv pip install --python /tmp/black312/bin/python "ruff==0.16.8" "black==26.5.1"
  /tmp/black312/bin/ruff check . && /tmp/black312/bin/black --check .
  ```

## Rules the codebase depends on

1. **A refactor that changes a number is a bug, not a finding.** Code that stored
   results depend on (`cerata/experts/ladder.py`, `cerata/core/`, `cerata/eval/stats.py`,
   the routers) must reproduce them: run `experiments/v3_anchors.py` on the machine
   that has `results/` and report the deltas (`docs/V3_ARCHITECTURE.md` section 7).
2. **`cerata/legacy/` is frozen bitwise.** It is the v1 (PAL-MoE) record; the old
   import paths (`pal_moe.models`, `pal_moe.adaptation`, ...) are `sys.modules` aliases
   onto it, registered by the `pal_moe` compatibility package, so old checkpoints still
   unpickle. New code never imports `pal_moe`. `tests/fixtures/*_stage1.pt` prove that; regenerate
   them only with `tests/fixtures/make_v1_fixtures.py` against the stage1-final tree.
3. **One variable per experiment; a failing arm is reported, not tuned**
   (`docs/STAGE1_PLAN.md` section 1).
4. **Pre-register, then commit the runner, then run.** The pre-registration fixes the
   question, arms, vetoes and outcome table; anything decided after data exists is a
   dated amendment that says so. Results documents cite the pre-registration by path
   and commit, so pre-registrations are never moved or rewritten.
5. **Guards are measurements.** A guard failure stops a run and is reported as an
   implementation failure, not as a scientific outcome.

## Commits

Conventional prefixes with a scope, as in the history: `feat(v3): ...`,
`fix(tests): ...`, `docs(p2-bound): ...`, `results(ac3): ...`, `refactor(...)`,
`build(ci): ...`, `chore: ...`, `style: ...`. A `results(...)` commit records a run
that already happened; its runner and pre-registration are earlier commits.
