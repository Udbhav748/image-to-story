# Developer Handoff

**Target read time: under 30 minutes.** Read in this order and stop where the time
runs out; each section stands alone.

---

## 1. What this project does (30 seconds)

It turns one or more images into a short creative story, while keeping every visual
statement traceable to evidence. Images are perceived into structured evidence,
grouped into scenes, remembered across a collection, retrieved hierarchically,
compressed into a budgeted context, planned creatively (hard facts locked, creative
space free), written by a local LLM, then every claim in the output is extracted,
classified, and verified against the evidence.

The defining constraint: **the system may invent the story freely, but it may never
invent the evidence.**

---

## 2. Current architecture (3 minutes)

Full detail: `docs/V2.3_ARCHITECTURE.md`. Summary:

```
Images → Ingestion → Vision → Evidence → World State
       → Collection Memory → Hierarchical Retrieval → Collection Context
       → Creative Planning → Story Generation → Claim Extraction
       → Verification → Evaluation → Artifacts
```

Two real execution paths:

| | Legacy | Collection (V2.3) |
|---|---|---|
| Entry | `run_single_image()` / `run_multi_image()` | `run_collection()` |
| Retrieval | flat FAISS `EvidenceRetriever` | `HierarchicalRetriever` |
| Context | `ContextBuilder` / `SequenceContextBuilder` | `CollectionContextBuilder` |
| Memory | none | `CollectionMemory` (scenes, entities, transitions, narrative) |

Both are supported. The legacy path is never silently upgraded.

---

## 3. Where to start reading code (5 minutes)

Read in this order — each file is short and the next depends on it.

1. `src/image_story/domain/schemas.py` — all data contracts.
   Start with `EvidenceRecord`, `VisualObservations`, `PipelineConfig`,
   `PipelineArtifacts`. ~1000 lines; skim.
2. `src/image_story/pipeline/orchestrator.py` — the production orchestration.
   **Read `run_collection()` closely** (it is the V2.3 path end to end).
   `run_single_image()` and `run_multi_image()` are the legacy variants.
3. `src/image_story/memory/collection_builder.py` — the V2.3B orchestrator
   (`CollectionMemoryBuilder`). Shows how the four memory subsystems combine.
4. `src/image_story/memory/hierarchical.py` — `CollectionMemory`, `SceneSummary`,
   `EntityMemory`, `StateTransition`, `NarrativeElement`, `RetrievalResult`.
5. `src/image_story/memory/hierarchical_retriever.py` — `retrieve_full()`.
6. `src/image_story/context/collection_builder.py` — context assembly and budgeting.
7. `src/image_story/narrative/planner.py` — `CreativePlanner`. Note the optional
   `collection_memory` / `retrieval_result` parameters that switch it into V2.3 mode.
8. `src/image_story/evaluation/claims.py` — `ClaimExtractor` and `ClaimVerifier`,
   including the OBSERVED/INFERRED/CREATIVE rules and `repair_story`.

Do **not** start in `src/image_story/narrative/` sub-modules
(`character.py`, `conflict.py`, `humor.py`, `surprise.py`) — they are stable and
uninteresting until Phase 3+.

---

## 4. Runtime entry points (2 minutes)

**CLI**
```bash
python main_v2.py images/                                  # single image
python main_v2.py images/ --multi                          # legacy sequence
python main_v2.py images/ --collection-memory              # V2.3 collection path
python main_v2.py images/ --mode fast|standard|full|baseline
```

**Python**
```python
from image_story import PipelineOrchestrator, get_pipeline_config

config = get_pipeline_config("standard")
orch = PipelineOrchestrator(config)
artifacts = orch.run_collection(["a.jpg", "b.jpg"], evaluate=True)
print(artifacts.story_draft.text)
print(artifacts.collection_memory.scene_summaries)
```

**Collection with sessions**
```python
from image_story.collections import CollectionPipeline
p = CollectionPipeline(config=config)
collection = p.create_collection_from_paths(["a.jpg", "b.jpg"])
result = p.process_collection(collection, use_collection_memory=True)
```

Key artifacts on the collection path:
`artifacts.collection_memory`, `artifacts.retrieval_result`,
`artifacts.context`, `artifacts.story_draft`, `artifacts.claims`,
`artifacts.verification_results`, `artifacts.evaluation`.

---

## 5. Important classes (quick reference)

| Class | File | Role |
|---|---|---|
| `PipelineOrchestrator` | `pipeline/orchestrator.py` | Runs all three paths |
| `CollectionPipeline` | `collections/pipeline.py` | Ingestion + session + orchestration |
| `IngestionPipeline` | `ingestion/pipeline.py` | validate → hash → deduplicate |
| `CollectionMemoryBuilder` | `memory/collection_builder.py` | Builds `CollectionMemory` |
| `SceneGrouper` | `memory/scene_grouper.py` | Observations → `SceneSummary` |
| `EntityMemoryTracker` | `memory/entity_memory.py` | Cross-scene entity identity |
| `StateTransitionDetector` | `memory/state_transitions.py` | appear/disappear/move/state changes |
| `NarrativeMemory` | `memory/narrative_memory.py` | Callback + open-loop candidates |
| `FAISSVectorStore` | `memory/faiss_store.py` | Index + provenance records |
| `HierarchicalRetriever` | `memory/hierarchical_retriever.py` | `retrieve_full()` |
| `CollectionContextBuilder` | `context/collection_builder.py` | Budgeted collection context |
| `CreativePlanner` | `narrative/planner.py` | `CreativePlan` + 7-beat `StoryPlan` |
| `StoryGenerationPipeline` | `generation/base.py` | Prompt → `StoryDraft` |
| `ClaimExtractor` | `evaluation/claims.py` | Story → `StoryClaim[]` |
| `ClaimVerifier` | `evaluation/claims.py` | Claims → `VerificationResult[]` |
| `ComprehensiveEvaluator` | `evaluation/evaluator.py` | `EvaluationResult` |

---

## 6. Important tests (3 minutes)

**233 tests total.** Run them all before touching anything:

```bash
pytest -q
```

Fast subset (no model weights or network needed, ~6 s):
```bash
HF_HUB_OFFLINE=1 pytest tests/unit tests/integration/test_v23c_runtime_proof.py -q
```
That is 202 tests: all V2.2, all V2.3B, and the entire runtime-proof suite.

**The three tests that matter most:**

1. `tests/regression/test_grounded_creativity_v21.py` (14 tests) — the V2.2
   behavioural contract: claim classification, creative budget, locked facts,
   grounded humour/surprise. **These define what "grounded" means. Do not break them.**

2. `tests/integration/test_v23c_runtime_proof.py` (63 tests) — executes the **real**
   `run_collection()` with only model boundaries stubbed. It asserts each V2.3B
   component is genuinely invoked, that real objects flow between stages, and that
   flat FAISS retrieval is *not* used on the collection path.
   Treat a failure here as an architecture break, not a flaky test.

3. `tests/unit/test_faiss_collection.py::TestPerformanceScaling` (3 tests) —
   architectural scale guards using a deterministic mock embedding model.

Mutation-tested: disabling any single critical stage, or reintroducing flat
retrieval, makes the runtime-proof suite fail. It is a real guard, not decoration.

---

## 7. Current branch and version

| | |
|---|---|
| Branch | `release/v2.3-production-ready` |
| Commit | `f792965` — `feat: complete v2.3 production pipeline` |
| Base | `937e698` (V2.3A, `main`) |
| Version | `2.3.0` (`pyproject.toml`) |
| Tests | 233 passed, 0 failed, 0 skipped |
| `main` | Unmodified at `937e698` |

---

## 8. Known limitations (read before estimating work)

Full table with severity and ownership in `docs/V2.3_ARCHITECTURE.md` §12.

| Limitation | Why it matters to you |
|---|---|
| `StoryClaim.evidence_ids` never populated | The claim→evidence link runs through `VerificationResult` instead. If you build anything that reads `claim.evidence_ids`, it will silently get `[]`. |
| `StoryDraft` has no scene/entity attribution | You cannot map a sentence to the scene behind it without re-deriving it. Blocks fine-grained narrative debugging. |
| `ExperimentManifest` partially populated | Manifests are not sufficient for bit-level reproducibility. |
| Vision cache inactive | `VisionCache` is defined and exported but never instantiated. Re-running vision is always paid for. |
| Scene grouping is O(N²) | Visible at 1000 frames; will dominate at 10K+. |
| FAISS insertion encodes per item | **Largest measured cost**: 311.7 s for 5000 records. Batching this is the single biggest cheap win available. |
| Minimal repair not automated | `ClaimVerifier.repair_story` exists but no production path calls it. |
| Evaluation uses only `images[0]` | CLIP grounding ignores every frame except the first, even for collections. |
| Synthetic-only scale evidence | Largest verified run is 5000 records / 1000 synthetic frames. **No claim of 100K-scale readiness.** |

---

## 9. Ownership

Full contract: `docs/COLLABORATION_CONTRACT.md`.

- **Developer A (Udbhav)** — ingestion, collections, vision, evidence, world model,
  memory, FAISS, hierarchical retrieval, provenance infrastructure, pipeline and
  experiment infrastructure.
- **Developer B (Narrative/Reasoning)** — causal reasoning, narrative possibility
  space, narrative optimization, candidate generation and ranking, bidirectional
  verification extensions, multi-objective narrative evaluation.

**Shared, requires coordination before changing:** `domain/schemas.py`,
`PipelineConfig`, `PipelineOrchestrator` stage sequencing, evaluation interfaces and
`EvaluationResult`, artifact schemas and serialization, the evidence contract, and the
runtime-proof test suite.

---

## 10. Prohibited changes

Do not, without explicit agreement:

1. Rewrite, replace or "improve" a V2.3 memory, narrative or evaluation algorithm
   while adding a feature. Extend; do not substitute.
2. Add a second orchestrator, a second memory system, or a parallel V3 pipeline.
   V3 stages go **into** the existing `run_collection()` sequence.
3. Change a public signature that the runtime-proof suite or other modules depend on
   (`retrieve_full()`, `build_from_observations()`, `create_creative_plan()`,
   `evaluate()`, `to_dict()`).
4. Bypass `ClaimVerifier` or `ComprehensiveEvaluator`. Extend them.
5. Change the OBSERVED/INFERRED/CREATIVE semantics or the creative budget defaults
   (`8 / 3 / 0`) silently. They are a V2.2 behavioural contract with 14 regression tests.
6. Weaken, skip or delete a test to make a change pass. Fix the code or the
   expectation, visibly.
7. Commit model weights, HF caches, FAISS indexes, `__pycache__`, `.bak` files or
   scratch `temp_*.py` (all now gitignored — keep them that way).
8. Emit creative narrative content as an `EvidenceRecord`.
9. Merge to `main`. Feature work goes on a feature branch, then `integration`.
10. Claim quality improvements without unseen-image and human evaluation
    (Phases 9–10 of `docs/V3_ROADMAP.md`).

---

## 11. First recommended development task

**Close the `StoryClaim.evidence_ids` gap** (Phase 1 in `docs/V3_ROADMAP.md`).

Why this one:

- It is the highest-severity open gap and the cheapest to fix.
- It is small, well-scoped, and verifiable — a good first contribution.
- Phases 2 (causal reconstruction) and 6 (bidirectional verification) both consume
  claim-level provenance, so this unblocks the whole V3 chain.
- It is a shared-contract change, so it exercises the coordination workflow early
  rather than at integration time.

Concretely: decide and document what `evidence_ids` means on a claim, populate it in
`ClaimExtractor` (or formally deprecate it in favour of the `VerificationResult`
link), and update
`tests/integration/test_v23c_runtime_proof.py::test_claim_evidence_ids_field_is_unpopulated_known_gap`
— which currently asserts the gap still exists and will fail the moment you fix it.

Suggested branch:
```bash
git checkout -b feature/claim-evidence-links release/v2.3-production-ready
```

Then open a PR against `integration` once the 233-test suite is green with no
reductions.