# Image → Story V2.3

**A scalable, hierarchical, evidence-grounded image-to-story system** with collection
processing, scene/entity memory, hierarchical retrieval, grounded creativity, claim
verification and evaluation.

## Documentation

| Document | Purpose |
|---|---|
| [docs/V2.3_ARCHITECTURE.md](docs/V2.3_ARCHITECTURE.md) | Full architecture: diagrams, runtime call graph, domain contracts, configuration, measured scale, limitations |
| [docs/DEVELOPER_HANDOFF.md](docs/DEVELOPER_HANDOFF.md) | New-developer orientation (read time: under 30 minutes) |
| [docs/COLLABORATION_CONTRACT.md](docs/COLLABORATION_CONTRACT.md) | Ownership boundaries, shared contracts, branch strategy, development rules |
| [docs/V3_ROADMAP.md](docs/V3_ROADMAP.md) | Proposed future roadmap — **not implemented** |
| [VALIDATION_REPORT_V22.md](VALIDATION_REPORT_V22.md) | V2.2 validation evidence |

## Architecture Overview

The current production path for image collections (V2.3):

```
IMAGE(S)
   ↓
INGESTION (validation, hashing, deduplication, ordering, resume/retry)
   ↓
VISUAL PERCEPTION (Florence-2 + GroundingDINO + OCR)
   ↓
GROUNDED EVIDENCE (structured EvidenceRecords)
   ↓
WORLD STATE (entities persistent across frames)
   ↓
COLLECTION MEMORY (scene grouping, entity memory, state transitions, narrative memory)
   ↓
HIERARCHICAL RETRIEVAL (scenes → evidence → entities → narrative elements)
   ↓
COLLECTION CONTEXT (compressed, budgeted: Hard Facts / Soft Inferences / Creative Space)
   ↓
CREATIVE PLANNER (Character, Conflict, Humor, Surprise, Open Loops, Callbacks)
   ↓
STORY BEAT PLANNER (7-beat structure)
   ↓
STORY GENERATION (Qwen local LLM)
   ↓
CLAIM EXTRACTION (OBSERVED / INFERRED / CREATIVE)
   ↓
CLAIM VERIFICATION (+ minimal repair of contradicted observed claims)
   ↓
CONTINUITY + CONTRADICTION + QUALITY EVALUATION
   ↓
FINAL GROUNDED CREATIVE STORY
```

### Execution paths

Two explicit, separately selectable paths exist:

| Path | Entry point | Retrieval | Context |
|------|-------------|-----------|---------|
| Legacy (V2.2, single/multi image) | `run_single_image()` / `run_multi_image()` | flat FAISS `EvidenceRetriever` | `ContextBuilder` / `SequenceContextBuilder` |
| Collection (V2.3) | `run_collection()` | `HierarchicalRetriever` over `CollectionMemory` | `CollectionContextBuilder` |

The collection path is opt-in (`--collection-memory`); the legacy path is never
silently upgraded, and `CollectionMemory` is only injected into the hierarchical
retriever on the collection path.

## Key Principles

> **The system is free to invent the story, but never free to invent the evidence.**

- **Hard Facts**: Directly supported by visual evidence (locked)
- **Soft Inferences**: Reasonable interpretations (qualified)
- **Creative Space**: Narrative invention allowed (personalities, motivations, dialogue, humor, twists)

### What each version added

| Version | Adds |
|---|---|
| **V2.2** | Grounded creativity: OBSERVED / INFERRED / CREATIVE claim classification, risk-aware creative budget (`8 / 3 / 0`), locked visual facts, grounded humor and surprise, targeted minimal repair, claim verification |
| **V2.3A** | Scalable collections: `StorySession`, `ImageCollection`, `ImageRecord`, ingestion (validation, hashing, deduplication, ordering), processing jobs, cache architecture, resume/retry |
| **V2.3B** | Hierarchical memory: `CollectionMemory`, `SceneSummary`, scene grouping, cross-scene `EntityMemory`, `StateTransition` detection, `NarrativeMemory`, `HierarchicalRetriever`, `CollectionContextBuilder`, FAISS provenance records |
| **V2.3C** | Production integration: `run_collection()` as the real collection path, `CollectionMemoryBuilder` / `HierarchicalRetriever` / `CollectionContextBuilder` wired in, `CreativePlanner` consuming collection memory, end-to-end runtime proof, legacy V2.2 path preserved |

> Future (V3) components — causal reasoning, narrative possibility space, narrative
> optimizer, candidate generation — are **not implemented**. See
> [docs/V3_ROADMAP.md](docs/V3_ROADMAP.md), which is a proposal only.

## Pipeline Modes

| Mode | Vision | Memory | Creative | Verification | Use Case |
|------|--------|--------|----------|--------------|----------|
| **Fast** | Florence-2 only | ❌ | ❌ | ❌ | CPU demos, quick tests |
| **Standard** | Florence-2 + GroundingDINO | FAISS | ✅ | ✅ | Primary development |
| **Full** | All + OCR | FAISS + narrative memory | ✅ | ✅ | Research, showcase |

## Installation

```bash
# Clone and install
pip install -e .

# Or with dependencies only
pip install -r requirements_v2.txt

# Download models (requires internet once)
python scripts/download_models.py

# Install spaCy model for claim extraction
python -m spacy download en_core_web_sm
```

## Usage

```bash
# Single image, standard mode
python main_v2.py images/thumb-chihiro001.png

# Multi-image sequence (legacy V2.2 path)
python main_v2.py images/ --multi

# V2.3 collection path (ingestion + hierarchical memory)
python main_v2.py images/ --collection-memory

# Fast mode (no GroundingDINO, no FAISS)
python main_v2.py images/ --mode fast

# Full mode (all features)
python main_v2.py images/ --mode full

# Custom config
python main_v2.py images/ --config configs/custom.yaml

# Skip evaluation
python main_v2.py images/ --no-eval

# Output directory
python main_v2.py images/ --output-dir my_results
```

## Project Structure

```
image_story_v2/
├── src/
│   └── image_story/
│       ├── domain/          # Schemas, enums, exceptions, collection models
│       ├── vision/          # Florence-2, GroundingDINO, OCR, Verifier
│       ├── ingestion/       # Validation, hashing, dedup, ordering, cache
│       ├── collections/     # CollectionPipeline, StorySession orchestration
│       ├── memory/          # Embeddings, FAISS, flat retrieval,
│       │                    #   + hierarchical: scene_grouper, entity_memory,
│       │                    #   state_transitions, narrative_memory,
│       │                    #   hierarchical_retriever, collection_builder
│       ├── context/         # World State, Ranker, Context + CollectionContext
│       ├── narrative/       # Character, Conflict, Humor, Surprise, Planner
│       ├── generation/      # Qwen adapter, Generation pipeline
│       ├── evaluation/      # Grounding, Continuity, Narrative, Claims
│       ├── pipeline/        # Orchestrator (legacy + run_collection)
│       └── config/          # Settings
├── benchmarks/              # Synthetic scale benchmark (100 / 1000 records)
├── configs/                 # YAML configs (fast, standard, full, baseline)
├── evals/
│   ├── golden/              # Regression test cases
│   └── fixtures/
├── tests/
│   ├── unit/                # Unit tests (no models)
│   ├── integration/         # V2.3C integration + end-to-end runtime proof
│   └── regression/          # Golden set regression
├── scripts/                 # Utility scripts
├── artifacts/               # Generated outputs (git-ignored)
├── notebooks/               # Analysis notebooks
└── main_v2.py               # CLI entry point
```

### Key V2.3 modules

| Module | Responsibility |
|--------|----------------|
| `memory/scene_grouper.py` | Groups observations into `SceneSummary` scenes |
| `memory/entity_memory.py` | Cross-scene `EntityMemory` with stable identities |
| `memory/state_transitions.py` | `StateTransition` detection (appear/disappear/move/state) |
| `memory/narrative_memory.py` | `NarrativeElement` memory for callbacks / open loops |
| `memory/collection_builder.py` | Orchestrates the four above into `CollectionMemory` |
| `memory/hierarchical_retriever.py` | Scene → evidence → entity → narrative retrieval |
| `context/collection_builder.py` | Compressed, budgeted `CollectionMemory` context |

## Running Tests

```bash
# Everything (233 tests)
pytest -q

# Unit tests only (no model inference)
pytest tests/unit -q

# V2.3C integration + end-to-end runtime proof
# (stubs only model boundaries: Florence2, embeddings, Qwen, CLIP/NLI)
pytest tests/integration -q

# Golden-set / grounded-creativity regression
pytest tests/regression -q
```

Current suite: **233 tests collected** — 53 V2.2, 100 V2.3B hierarchical-memory,
17 V2.3C integration, 63 V2.3C end-to-end runtime proof. Full breakdown in
[docs/V2.3_ARCHITECTURE.md §10](docs/V2.3_ARCHITECTURE.md#10-test-contract).

The runtime-proof suite exercises the real `run_collection()` orchestration and
asserts that each V2.3B component is actually invoked and that real objects flow
between stages. It stubs only the expensive ML boundaries (vision, embeddings,
Qwen, CLIP/NLI), never the components themselves.

```bash
# 202 model-independent tests, no weights or network required (~6 s)
HF_HUB_OFFLINE=1 pytest tests/unit tests/integration/test_v23c_runtime_proof.py -q
```

> Note: `tests/integration/test_v23c_integration.py` constructs real
> sentence-transformer embeddings, so it performs a Hugging Face model load on
> first use. It therefore needs network access (or a warm HF cache) the first
> time it runs. `test_v23c_runtime_proof.py` has no such dependency.

## Known Limitations

These are real, current limitations. They are documented rather than papered over.

1. **`StoryClaim.evidence_ids` is not populated.** `ClaimExtractor` creates claims
   with the field left empty; no code path writes to it. The claim → evidence link
   is carried by `VerificationResult.claim` / `.supporting_evidence` instead.
2. **`StoryDraft` has no direct scene/entity attribution.** Individual sentences
   cannot be mapped to the scene that motivated them without re-deriving that from
   claim evidence.
3. **`ExperimentManifest` is only partially populated.** `git_commit`,
   `vision_model`, `language_model`, `embedding_model`, `dataset` and `dataset_hash`
   are declared but left empty by the orchestrator, so manifests are not sufficient
   for full bit-level reproducibility.
4. **The vision result cache architecture exists but is not fully wired** into the
   production collection path.
5. **Scene grouping is O(N²)** (pairwise similarity over all observations), so very
   large single batches will degrade.
6. **FAISS insertion encodes per item.** `add_evidence_with_provenance` calls
   `encode_single` per record rather than batching, which costs throughput on large
   insertions.
7. **Scale validation is synthetic only.** The 100/1000-record checks in
   `benchmarks/scale_benchmark.py` and `tests/unit/test_faiss_collection.py` use
   synthetic observations. Large-scale empirical validation on real image
   collections has not been performed.
8. **No statistical claim is made** that V2.3 improves story quality. The reported
   metrics are grounding, contradiction, continuity and narrative-quality
   diagnostics; they are not evidence of statistically significant narrative
   improvement, and no generalisation to unseen images has been demonstrated.

## Evaluation Metrics

The system provides comprehensive evaluation:

- **Grounding**: CLIP image-story similarity, NLI contradiction, attribute conflicts
- **Claims**: Claim extraction, visual verification, claim grounding score
- **Continuity**: Entity consistency, transition markers, adjacent similarity
- **Narrative Quality**: Character depth, plot structure, emotional arc, humor, callbacks
- **Runtime**: Separate generation vs evaluation timing

## Golden Set Regression

Fixed test cases in `evals/golden/` ensure reproducibility:

```json
{
  "case_id": "golden_001",
  "image_path": "images/thumb-chihiro001.png",
  "expected_entities": ["girl", "car", "flowers"],
  "known_visual_facts": ["girl in car", "holding flowers"],
  "known_contradictions": ["boy", "dog"],
  "min_grounding_score": 0.6
}
```

## Ablation Framework

Track experiment manifests automatically:

```json
{
  "git_commit": "...",
  "vision_model": "florence2",
  "language_model": "qwen2.5-0.5b",
  "embedding_model": "minilm-l6",
  "dataset": "chihiro",
  "seed": 0,
  "config": {...}
}
```

Artifacts saved per run for full reproducibility.

## Citation

If you use this system, please cite the original Image → Story challenge and this V2 implementation.

## License

MIT License