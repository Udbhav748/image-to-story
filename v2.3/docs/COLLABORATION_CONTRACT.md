# Collaboration Contract

**Baseline:** V2.3 at `release/v2.3-production-ready` (commit `f792965`).
Read `docs/V2.3_ARCHITECTURE.md` first — this document assumes it.

This contract exists so two developers can extend the system without duplicating,
silently replacing, or breaking what already works.

---

## Developer A — Udbhav

**Owns the evidence and infrastructure spine.**

| Area | Modules |
|---|---|
| Ingestion | `src/image_story/ingestion/` — `pipeline.py`, `validation.py`, `hashing.py`, `collection.py`, `cache.py` |
| Collections | `src/image_story/collections/` — `CollectionPipeline`, session handling |
| Vision | `src/image_story/vision/` — `florence.py`, `grounding.py`, `ocr.py`, `verifier.py`, `base.py` |
| Evidence | `EvidenceRecord` production, `EvidenceType`, `InformationClass`, confidence classes |
| World model | `src/image_story/context/world_state.py` — `WorldStateBuilder`, `EntityTracker` |
| Memory | `src/image_story/memory/` — `faiss_store.py`, `embeddings.py`, `scene_grouper.py`, `entity_memory.py`, `state_transitions.py`, `narrative_memory.py`, `hierarchical_retriever.py`, `collection_builder.py`, `hierarchical.py` |
| FAISS | index types, provenance records, filtered search, persistence |
| Hierarchical retrieval | `HierarchicalRetriever`, `RetrievalResult` assembly, scene/entity selection |
| Provenance infrastructure | `IndexedEvidenceRecord`, `key_evidence_ids`, `RetrievalResult` ↔ `CollectionMemory` linkage |
| Pipeline infrastructure | `src/image_story/pipeline/orchestrator.py`, `main_v2.py`, `src/image_story/config/` |
| Experiment infrastructure | `ExperimentManifest`, `save_artifacts`, `benchmarks/`, `evals/`, configs |

## Developer B — Narrative/Reasoning Engineer

**Owns everything above the evidence line.**

| Area | Modules |
|---|---|
| Causal reasoning | *(to be created)* — event reconstruction from `EvidenceRecord` + `StateTransition` |
| Narrative possibility space | *(to be created)* — enumerating admissible narrative options |
| Narrative optimization | *(to be created)* — selecting among candidates against explicit objectives |
| Candidate generation | *(to be created)* — N story drafts per collection |
| Candidate ranking | *(to be created)* |
| Bidirectional verification extensions | `src/image_story/evaluation/claims.py`, `src/image_story/vision/verifier.py` (extensions only) |
| Multi-objective narrative evaluation | `src/image_story/evaluation/narrative.py`, `continuity.py` (extensions only) |

**These ownership areas are a starting point, not a settlement.** They should be
adjusted only after explicit agreement between both developers. If work in one
 developer's area requires changing files in the other's, that is a coordination
event — see §Shared Contracts.

---

## Shared Contracts

These files/components require coordination. Both developers may propose changes;
neither may land them unilaterally on a shared area without agreement.

### 1. `src/image_story/domain/schemas.py`

- **Who can modify:** either, **with coordination**.
- **Why shared:** every module imports from it. Both developers' work depends on it.
- **Expectation:** purely additive by default. New fields must have defaults so
  existing constructors keep working. Removing or renaming a field breaks the other
  developer immediately.
- **Specifically contested fields today:**
  - `StoryClaim.evidence_ids` — currently a known gap. Whichever developer closes it
    must not change the `VerificationResult`-based link that tests currently assert.
  - `StoryDraft` — adding scene/entity attribution is Developer B's gap to close and
    will require a new field here. Coordinate the shape with A so artifact
    serialization stays stable.
  - `ExperimentManifest` — Developer A's gap; B should not populate it independently.

### 2. `PipelineConfig` (`domain/schemas.py`)

- **Who can modify:** either, **with coordination**.
- **Why shared:** read by the orchestrator, memory builder, retriever and context
  builder. New V3 tunables (optimizer weights, candidate counts) will land here.
- **Expectation:** additive with defaults. `PipelineConfig.from_mode()` presets
  (`fast`/`standard`/`full`/`baseline`) must keep working; if B adds fields, decide
  explicitly whether each preset changes.

### 3. `PipelineOrchestrator` (`pipeline/orchestrator.py`)

- **Who can modify:** Developer A owns it.
- **Requires coordination when:** B needs a new stage in the collection chain. A
  integrates it; B does not edit the orchestrator directly.
- **Critical constraint:** `run_collection()` must remain the single collection entry
  point. Do not add a second orchestrator or a parallel V3 pipeline. V3 stages are
  inserted into the existing sequence.
- **Also note:** `run_single_image`, `run_multi_image` and `run_collection` currently
  duplicate stage sequencing. Any stage change must be mirrored across paths
  deliberately — or the paths will silently diverge.

### 4. Evaluation interfaces (`src/image_story/evaluation/`)

- **Who can modify:** Developer B owns extensions; Developer A owns
  `ComprehensiveEvaluator.evaluate()` orchestration and `claims.py` classification.
- **Requires coordination when:** changing `EvaluationResult` fields, changing claim
  classification semantics, or changing which artifact fields the evaluator reads.
- **Expectation:** the three-way `observed`/`inferred`/`creative` classification and
  the `compute_claim_grounding_score` weighting are a **V2.2 behavioural contract**.
  Changing them breaks the 14 regression tests and is not a silent change.

### 5. Artifact schemas (`PipelineArtifacts`, `save_artifacts`)

- **Who can modify:** Developer A owns the schema and the writer.
- **Requires coordination when:** B needs new artifacts (candidate drafts,
  optimizer traces, repair reports). A adds the fields and serialization; B consumes.
- **Expectation:** `to_dict()` must stay JSON-serializable. `collection_memory` and
  `retrieval_result` are already serialized via their own `to_dict()`.

### 6. Evidence contract (`EvidenceRecord`, `InformationClass`, `EvidenceType`)

- **Who can modify:** Developer A owns.
- **Requires coordination when:** B needs a new kind of evidence or a new
  information class. Adding is possible; reclassifying existing evidence is not a
  quiet change.
- **Hard rule:** creative narrative content must never be emitted as an
  `EvidenceRecord`. Evidence is what was seen; creativity lives in `CreativePlan`
  and `StoryDraft`.

### 7. Memory/retrieval interfaces

- **Who can modify:** Developer A owns.
- **Requires coordination when:** B needs different retrieval (e.g. retrieve by causal
  event rather than scene). Extend via new `HierarchicalRetriever` methods; do not
  change `retrieve_full()`'s existing signature, which
  `tests/integration/test_v23c_runtime_proof.py` asserts.

### 8. Tests as shared interface

- `tests/integration/test_v23c_runtime_proof.py` is a **contract test**, not just a
  test. It asserts real objects flow between stages and that flat FAISS is bypassed.
  Treat a failure there as an architecture break, not a flaky test.
- New shared behaviour needs a test in the appropriate layer before merge.

---

## Branch Strategy

```
main                                  # protected; V2.3A history, never feature work
  │
  └── v2.3.0 stable baseline           # release/v2.3-production-ready (f792965)
        │
        ├── feature/platform-memory         # Developer A
        ├── feature/narrative-reasoning     # Developer B
        ├── feature/verification-evaluation # Developer B
        │
        └── integration                     # both; resolves cross-module changes
              │
              └── main                      # only via reviewed PR
```

### Rules

1. **No direct feature work on `main`.** `main` receives only reviewed merges from
   `integration`.
2. **Both developers branch from the same V2.3 baseline.** Do not branch from each
   other's feature branches.
3. **No duplicate implementations.** Before creating a module, check
   `docs/V2.3_ARCHITECTURE.md` §3.3 and grep. If a capability exists, extend it. Two
   memories, two retrievers or two context builders will silently diverge.
4. **Merge only after tests.** `pytest -q` must be at 233 passed or more, with no
   reductions. Never fix a failing test by weakening or deleting it.
5. **Shared schema changes require coordination.** See §Shared Contracts. A schema
   change touching the other developer's modules needs explicit agreement first.
6. **`integration` resolves cross-module changes.** It is the only branch where both
   developers' work meets. Whoever integrates owns resolving conflicts.
7. **Feature branches stay scoped.** One coherent change per branch.
8. **Never weaken the runtime-proof suite.** It is the guard that the production path
   actually uses the architecture it claims to.

---

## V3 Development Rules

These constrain all V3 work.

1. **Evidence first.** Every narrative element must trace to evidence, a declared
   inference, or an explicit creative marker. Nothing else is admissible.
2. **Do not convert hypotheses into visual facts.** A reconstruction is an inference
   until it is evidenced; label it `inferred`, never `observed`.
3. **Preserve provenance.** Any new subsystem must be able to say which
   `EvidenceRecord`s, `SceneSummary`s and frames it used. If it cannot, it is not
   admissible.
4. **Preserve V2.2 creative freedom.** Adding memory or reasoning must not
   constrains writing. The budget stays `safe_creative=8`, `risky_inferred=3`,
   `forbidden_visual=0`.
5. **Do not optimize only one metric.** An optimizer that improves coherence while
   destroying grounding has failed, even if the objective improved.
6. **Candidate stories must be independently evaluated.** Every candidate gets its
   own full verification and evaluation. Do not evaluate one and assume the rest.
7. **Repair must be targeted.** Only contradicted *observed* claims may be repaired.
   Never regenerate wholesale to satisfy a metric.
8. **Repaired stories must be re-verified.** A repair that is not re-verified is an
   unverified claim about the story.
9. **Repair must have a bounded iteration budget.** Declare the bound up front and
   stop. Unbounded repair loops are a failure mode, not a feature.
10. **Never hide failed validation.** If a candidate fails, if a benchmark regresses,
    if a limitation is hit — record it. Do not quietly narrow the evaluation to make
    results look better.
11. **Do not add architecture simply for complexity.** Every new subsystem must
    resolve a measured problem. "It is more modular" is not a justification.
12. **Every major change requires a measurable hypothesis.** State what should
    improve, on which metric, by how much, and how it will be measured *before*
    implementing.

---

## Definition of Done (per change)

- [ ] Tests added at the correct layer; `pytest -q` shows no regressions
- [ ] Runtime-proof suite still passes unmodified
- [ ] No public signature changed without coordination
- [ ] `docs/V2.3_ARCHITECTURE.md` updated if behaviour or configuration changed
- [ ] New limitations documented, not hidden
- [ ] Commit message states what changed and why
- [ ] Branch pushed, PR opened against `integration`