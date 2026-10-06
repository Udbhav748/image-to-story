# Contributing

## Setup

```bash
python -m pip install -e ".[dev]"
python scripts/download_models.py
python -m spacy download en_core_web_sm
make check     # lint + typecheck + tests
```

## Rules

These are the rules that keep one canonical architecture instead of two
overlapping ones.

**One definition per concept.** If a data contract exists, it exists once.
`tests/regression/test_architecture_invariants.py` enumerates the contracts that
must have a single definition site and fails if a second appears. If you need a
different shape, that is a new contract with a new name, not a variant of an
existing one.

**No aliases to paper over duplicates.** `from x import Y as PipelineY` in place of
`from x import Y` hides the problem, it does not solve it. Fix the import.

**One entry point.** Production runs through `python -m image_story` or the
`image-story` console script. Do not add a `main_*.py` at the repository root.
The Challenge V1 entry point under `challenges/` is the only exception and it
stays there.

**Tests by responsibility.**

| Directory | Contains |
|---|---|
| `tests/unit/` | Pure logic, no model inference. Fast. |
| `tests/integration/` | The pipeline with stubbed or real models, runtime proofs. |
| `tests/regression/` | Pinned prior behaviour, architecture invariants, isolation checks. |
| `tests/fixtures/` | Golden cases and shared fixtures. |

Do not delete a test because a path changed. Move it and update the import.

**Challenge V1 is frozen.** It is the baseline the rest of the work is measured
against. Do not "improve" it, do not import from it, do not let it import the
production package. `tests/regression/test_challenge_v1_isolation.py` enforces
the isolation; its own suite runs via `make test-challenge`.

**V3 lands incrementally.** Before writing a V3 subsystem, add its interface and
a failing test. Do not build a parallel pipeline. Do not add a second retrieval
path, a second context builder or a second evaluation path; extend the existing
one. Update `docs/architecture/roadmap.md` when a phase is complete.

**Keep it runnable.** If you change the pipeline, `python -m image_story
benchmarks/challenge-8-images --mode fast` must still work. Fast mode is the
smoke test because it needs the fewest models.

## Style

`ruff check src tests scripts` and `black` at line length 100. The codebase is
not fully reformatted; match the file you are editing rather than reformatting
unrelated lines.

Docstrings explain *why*, especially where behaviour looks surprising. Comments
that restate the code are noise. Keep the module docstring current when you move
a responsibility between packages; several of them were rewritten during the
restructure precisely because they described the old layout.

## Before opening a PR

```bash
make check                 # lint, typecheck, full suite
python -m image_story benchmarks/challenge-8-images --mode fast
```

Then say what changed, which contracts were touched, and what a reviewer should
look at. If you changed a mode preset, say so: `tests/unit/test_config_defaults.py`
pins all of them and a diff there means the numbers in `docs/experiments/` are now
stale.