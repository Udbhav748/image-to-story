# Architecture: what is actually implemented

This document describes the code that exists today. Planned work is in
[`roadmap.md`](roadmap.md) and is deliberately kept separate.

Version: V2.3. Source of truth: `src/image_story/`.

## Pipeline

`PipelineOrchestrator` (`pipeline/orchestrator.py`) has three entry points. They
share perception, planning, generation and evaluation; they differ in retrieval
and context building.

| Entry point | Retrieval | Context builder | Use for |
|---|---|---|---|
| `run_single_image(path)` | flat FAISS (`retrieval/flat.py`) | `ContextBuilder` (`retrieval/compression.py`) | one image |
| `run_multi_image(paths)` | flat FAISS | `SequenceContextBuilder` (`retrieval/sequence.py`) | an ordered sequence |
| `run_collection(paths)` | `HierarchicalRetriever` (`retrieval/hierarchical.py`) | `CollectionContextBuilder` (`retrieval/collection_context.py`) | collections with scene memory |

### Stage order and timings

Stage identifiers are declared in `pipeline/stages.py` and recorded in
`PipelineArtifacts.runtime` by `observability/timing.py:StageTimer`:

```
vision_s -> world_state_s -> [collection_memory_s] -> retrieval_s
         -> ranking_s -> planning_s -> context_s -> generation_s -> [evaluation_s]
```

`collection_memory_s` only appears on the collection path; `evaluation_s` only
when `evaluate=True`.

### Perception

`orchestrator._perceive()` is the single perception path for all three entry
points. It runs Florence-2, then GroundingDINO (using detected labels as
open-vocabulary prompts) and OCR when the config enables them, and returns one
`VisualObservations` per frame.

### Memory and retrieval

Two retrieval strategies share one FAISS store:

- **Flat** (`retrieval/flat.py:EvidenceRetriever`) indexes every evidence record
  and retrieves top-k for a query built from scenes, characters and OD labels.
- **Hierarchical** (`retrieval/hierarchical.py:HierarchicalRetriever`) requires a
  `CollectionMemory` built by `memory/collection.py:CollectionMemoryBuilder`,
  retrieves scenes first, then evidence within scenes, and also returns entity
  memories, state transitions and narrative elements.

`CollectionMemoryBuilder` populates its own internal retriever; the orchestrator
holds a separately configured retriever, so `run_collection` injects the built
memory explicitly via `set_collection_memory()`. Without that, the collection path
retrieves nothing and produces an ungrounded story.

### Context

Context builders separate three information classes, which is the mechanism that
lets a story be creative without contradicting the image:

- hard facts - must not be contradicted
- soft inferences - plausible, may be refined
- creative space - invention is encouraged

They extend `retrieval/compression.py:ContextBuilder`, which is also the class
`SequenceContextBuilder` and `CollectionContextBuilder` derive from.

### Planning and generation

`narrative/planner.py:CreativePlanner` produces a `CreativePlan` (characters,
conflict, open loops, callbacks, foreshadowing) and a 7-beat `StoryPlan`.
`generation/base.py:StoryGenerationPipeline` renders the plan into a prompt and
generates with Qwen2.5-0.5B-Instruct, optionally with three-attempt length
control.

### Verification

`verification/claims.py` extracts claims (spaCy, with a regex fallback) and
verifies them against evidence via `verification/visual.py:VisualVerifier`.
Results are `VerificationResult` records.

Known gaps, all documented in `docs/research/v3-architecture-notes.md`:
`StoryClaim.evidence_ids` and `StoryBeat.key_evidence_ids` are not populated,
and `ClaimVerifier.repair_story()` is unused.

### Evaluation

`evaluation/evaluator.py:ComprehensiveEvaluator` combines grounding (CLIP + NLI
+ attribute conflict + repetition), continuity and narrative quality into one
`EvaluationResult`.

### Artifacts and provenance

`experiments/artifacts.py:save_artifacts()` writes the artifacts JSON, the story,
the evaluation and the context for a run. `experiments/manifests.py:build_manifest()`
records the git commit, models, config, seed, device and dataset hash. The
orchestrator builds a manifest per run, so every artifact carries its own
provenance.

## Domain contracts

Each concept has exactly one definition. The invariant is enforced by
`tests/regression/test_architecture_invariants.py`.

| Contract | Definition |
|---|---|
| `BoundingBox`, `EvidenceRecord`, `VisualObservations`, `WorldEntity`, `WorldState`, `RetrievedEvidence`, `CreativePlan`, `StoryBeat`, `StoryPlan`, `StoryDraft`, `StoryClaim`, `VerificationResult`, `EvaluationResult`, `ImageRecord`, `ImageCollection`, `ProcessingJob`, `StorySession`, `CacheEntry`, `ProcessingConfig`, `CollectionPipelineConfig`, `PipelineConfig`, `ExperimentManifest`, `PipelineArtifacts` | `domain/schemas.py` |
| every vocabulary | `domain/enums.py` |
| errors | `domain/exceptions.py` |
| `SceneSummary`, `CollectionMemory`, `EntityMemory`, `StateTransition`, `NarrativeElement`, `RetrievalResult`, `RetrievalFilter` | `memory/hierarchical.py` |
| `IndexedEvidenceRecord`, `SceneEvidenceLink` | `evidence/provenance.py` |

`SceneSummary` and `IndexedEvidenceRecord` are re-exported from
`memory/hierarchical.py` because memory is their main consumer; there is still
only one definition.

### Configuration

`PipelineConfig` is the only configuration contract. Its mode presets live in
`config/defaults.py:MODE_PRESETS`, and `PipelineConfig.from_mode()` reads that
table. Adding a mode means adding one entry there. Defaults are deep-copied per
instance so runs cannot leak state into each other.

## What was removed during consolidation

Each of these existed twice before, with the copies diverging:

| Removed | Canonical location now |
|---|---|
| `domain/collections.py` (duplicate of five schemas) | `domain/schemas.py` |
| `SceneSummary` in `domain/schemas.py` | `memory/hierarchical.py` |
| collection enums, defined 5x each in one module | `domain/enums.py`, once each |
| `ImageHasher` + `Deduplicator` copies in `ingestion/validation.py` | `ingestion/hashing.py`, `ingestion/dedup.py` |
| `ingestion/collection.py` (stale fork of the collection pipeline, referencing an attribute that did not exist) | `ingestion/jobs.py` |
| `_evidence_to_text` in four modules, three formats | `evidence/builder.py:evidence_to_text()` |
| `Deduplicator.register_image` defined twice, first shadowed | one definition, returns `(duplicate_of, similar_to)` |
| inline `print()` in the ingestion pipeline, `save_artifacts` imported from `main_v2.py` | `observability/logging.py`, `experiments/artifacts.py` |

## Entry points

- Production: `python -m image_story` (`__main__.py` -> `cli.py`), or the
  `image-story` console script.
- Challenge V1: `challenges/image-story-v1/main.py`, with its own dependencies.

There is no `main.py`, `main_v2.py` or `challenge_main.py` at the repository root.

## Layering

```
domain  <-  everything
config  <-  domain
ingestion, vision, evidence, world, memory, retrieval, narrative,
generation, verification, evaluation  <-  domain, config, observability
pipeline  <-  all of the above
experiments, cli  <-  pipeline
```

`domain` imports nothing from the stage packages.
`memory/collection.py` imports `retrieval.hierarchical` lazily inside functions,
because hierarchical retrieval is built on the memory contracts; that is the one
place where the dependency is inverted on purpose, to keep `import image_story`
acyclic.