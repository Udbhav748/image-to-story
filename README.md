# Image -> Story

Turn images into stories that can be checked against what is actually visible.

This repository holds three things, and it matters which is which:

| | What it is | Where it lives | Status |
|---|---|---|---|
| **Challenge V1** | The original challenge submission. Florence-2 perception, a deterministic context builder, and a grounding metric. | `challenges/image-story-v1/` | Historical baseline. Preserved and runnable. Not imported by the production package. |
| **V2 (production)** | The current system. Evidence-grounded perception, hierarchical memory, retrieval, creative planning, claim verification, multi-metric evaluation. | `src/image_story/` | **Canonical implementation.** This is what you run. |
| **V3** | Causal reconstruction, narrative possibility space, multiple candidates, bidirectional verification, multi-objective selection. | Not implemented. | Research direction only. Described in [`docs/architecture/roadmap.md`](docs/architecture/roadmap.md). No code exists for it. |

Nothing in V3 exists yet. Nothing in this README should be read as a claim that it does.

## What the production system does

```
IMAGE(S)
  -> INGESTION / SESSION      validate, hash, deduplicate, order, cache
  -> VISUAL PERCEPTION        Florence-2, optionally GroundingDINO and OCR
  -> VISUAL EVIDENCE          EvidenceRecord with provenance and confidence
  -> WORLD MODEL              entities, recurrences, disappearances, open loops
  -> HIERARCHICAL MEMORY      scenes, entity memory, state transitions, narrative elements
  -> RETRIEVAL                flat or scene-aware (hierarchical) evidence retrieval
  -> CREATIVE PLANNING        7-beat story plan, characters, conflict, humor, surprise
  -> STORY GENERATION         Qwen2.5-0.5B-Instruct, length-controlled
  -> VERIFICATION             claim extraction, claim/evidence verification
  -> EVALUATION               grounding, continuity, narrative quality, repetition
  -> ARTIFACTS                story + evidence + metrics + manifest
```

Every fact the story can use is traceable to an `EvidenceRecord`, and every claim
in the story is checked against those records. That traceability is the point of
the system; it is also the part most likely to be extended in V3.

See [`docs/architecture/current.md`](docs/architecture/current.md) for the
implemented architecture and [`docs/decisions/collaboration-contract.md`](docs/decisions/collaboration-contract.md)
for the shared contracts a second developer codes against.

## Install

```bash
git clone https://github.com/Udbhav748/image-to-story
cd image-to-story
python -m pip install -e .
python scripts/download_models.py     # ~2GB; needed once
python -m spacy download en_core_web_sm   # for claim extraction
```

The pipeline runs offline by default (`HF_HUB_OFFLINE=1`).

## Run

One entry point for the production system:

```bash
python -m image_story benchmarks/challenge-8-images
# or, after installing:
image-story benchmarks/challenge-8-images
```

```bash
# Modes
python -m image_story IMAGES --mode fast       # no grounding, no FAISS, no verification
python -m image_story IMAGES --mode standard   # default
python -m image_story IMAGES --mode full       # adds OCR

# One story across an ordered image sequence
python -m image_story IMAGES --multi

# Hierarchical collection memory: scenes, entity memory, transitions, narrative
python -m image_story IMAGES --collection

# Named experiment with a written summary
python scripts/run_experiment.py IMAGES --name v2-standard \\
    --summary artifacts/experiments/v2-standard.json

# Aggregate past runs into a comparison table
python scripts/build_report.py artifacts/experiments
```

`make help` lists the same operations as make targets.

## Repository layout

```
src/image_story/       canonical production package
challenges/            Challenge V1, isolated and independently runnable
benchmarks/            input image sets
tests/                 unit / integration / regression / fixtures
configs/               PipelineConfig presets (fast, standard=v2, full, baseline)
scripts/               thin wrappers around the package
docs/                  architecture, experiments, decisions, research
artifacts/             run output (git-ignored)
```

### Package layout

| Package | Responsibility |
|---|---|
| `domain/` | Every data contract and vocabulary. Depends on nothing above it. |
| `config/` | Settings and the mode presets consumed by `PipelineConfig.from_mode`. |
| `ingestion/` | Validate, hash, deduplicate, cache, order images; collection jobs. |
| `vision/` | Florence-2, GroundingDINO, OCR. Perception only. |
| `evidence/` | The evidence text projection, provenance records, confidence banding. |
| `world/` | Entity tracking and `WorldState` construction across frames. |
| `memory/` | Embeddings, FAISS, scenes, entity memory, transitions, narrative memory. |
| `retrieval/` | Flat and hierarchical retrieval, ranking, context compression. |
| `narrative/` | Characters, conflict, humor, surprise, the creative planner. |
| `generation/` | Generator abstraction, the Qwen adapter, prompt templates. |
| `verification/` | Claim extraction/verification and visual re-checking. |
| `evaluation/` | Grounding, continuity, narrative quality, the aggregate evaluator. |
| `pipeline/` | The orchestrator and the stage names it records. |
| `experiments/` | Manifests, artifact persistence, experiment runners, scale benchmarks. |
| `observability/` | Logging, stage timing, run traces. |

## Test

```bash
make test               # whole suite
make test-unit          # fast, no model inference
make test-challenge     # Challenge V1, its own deps
```

Tests are split by responsibility:

- `tests/unit/` - pure logic: schemas, ranking, memory, compression, config presets
- `tests/integration/` - the pipeline with stubbed models, plus runtime proofs
- `tests/regression/` - pinned prior behaviour, architecture invariants, challenge isolation
- `tests/fixtures/` - golden cases and shared fixtures

Two invariant suites are worth knowing about:

- `tests/regression/test_architecture_invariants.py` fails if a domain contract
  gains a second definition site, if the entry points multiply, or if `domain`
  starts importing pipeline stages.
- `tests/unit/test_config_defaults.py` pins every mode preset, so changes to
  `MODE_PRESETS` have to be deliberate.

## Challenge V1

```bash
cd challenges/image-story-v1
python -m pip install -r requirements.txt
python test_pipeline.py
python main.py ../../benchmarks/challenge-8-images --both
```

V1 has its own dependencies and its own entry point. It is a frozen record of
what was submitted and what it scored; its results in `challenges/image-story-v1/`
are the baseline the V2 work is measured against. It is not part of `make test`
and nothing in `src/` imports it.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The short version: one definition
per concept, tests by responsibility, no new top-level entry points, and V3
features land behind the interfaces in `docs/architecture/roadmap.md` rather than
as new parallel paths.

## License

MIT.