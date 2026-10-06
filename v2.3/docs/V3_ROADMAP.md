# V3 Roadmap

> **⚠️ NOTHING IN THIS DOCUMENT IS IMPLEMENTED.**
> V3.0 has not been started. No causal reasoning, narrative possibility space,
> narrative optimizer, candidate generation, bidirectional verification or
> multi-objective evaluator exists in the codebase. Every phase below is a
> **proposal** for future work, not a description of current behaviour. Do not write
> code, tests or documentation against these subsystem names as though they were
> available.

**Read first:** `docs/V2.3_ARCHITECTURE.md` (what exists) and
`docs/COLLABORATION_CONTRACT.md` (who owns what, and the development rules).

**Scope:** Phases 2–8 are Developer B's remit; phases 1 and 9–12 are shared or
Developer A's, per the collaboration contract. Phase 0 is complete.

**Governing rules for all phases** (from `docs/COLLABORATION_CONTRACT.md` §V3
Development Rules): evidence first; hypotheses never become visual facts; provenance
preserved; V2.2 creative freedom preserved; never optimize a single metric; repair
must be targeted, bounded, and re-verified; never hide failed validation; no
architecture without a measurable hypothesis.

**Gate between every phase:** the V2.3 suite must be at ≥233 passed with no
reductions, and the runtime-proof suite must pass unmodified.

---

## PHASE 0 — Stable V2.3 baseline

**Status: COMPLETE** (branch `release/v2.3-production-ready`, commit `f792965`).

| | |
|---|---|
| Objective | Freeze a trustworthy, documented, tested collaboration baseline |
| Inputs | V2.2 grounded creativity, V2.3A ingestion, V2.3B hierarchical memory, V2.3C production integration |
| Outputs | 233 passing tests; architecture/collaboration/handoff docs; known-limitations table; measured synthetic scale data |
| Tests required | 233 (53 V2.2 + 100 V2.3B + 17 integration + 63 runtime proof) |
| Success criteria | ✅ Suite green; ✅ runtime proof executes real `run_collection()`; ✅ flat-FAISS bypass asserted; ✅ limitations documented honestly |

**Carry into V3:** the seven gaps in the limitations table remain open. Phases 1 and 2
depend on closing two of them.

---

## PHASE 1 — Shared domain / evidence contracts

**Owner: shared (A leads schemas, B signs off on consumption).**

| | |
|---|---|
| Objective | Make the evidence line a hard, testable interface that V3 cannot erode |
| Inputs | Existing `EvidenceRecord`, `InformationClass`, `StoryClaim`, `PipelineArtifacts` |
| Outputs | Populated `ExperimentManifest`; defined `StoryClaim.evidence_ids` semantics; a `StoryDraft` scene/entity attribution field; a versioned evidence-contract test |
| Work items | 1) Define what `evidence_ids` on a claim means and populate it (or formally deprecate it). 2) Add scene/entity attribution to `StoryDraft`. 3) Populate `git_commit`, model names, `dataset`/`dataset_hash` in `ExperimentManifest`. 4) Add a contract test asserting no creative text is ever emitted as `EvidenceRecord`. |
| Tests required | New contract tests; existing 233 unchanged; a test that fails if `evidence_ids` silently stays empty |
| Success criteria | Every artifact field is populated or explicitly documented as optional; both developers agree the evidence line is stable; no V3 subsystem can add a claim without provenance |

**Why first:** Phases 2 and 6 both consume provenance. Building causal or verification
work on an unstable provenance contract guarantees rework.

---

## PHASE 2 — Causal event reconstruction

**Owner: Developer B. Consumes Developer A's evidence contract (Phase 1).**

| | |
|---|---|
| Objective | Move from "entity X was in scene Y" to "event E happened, with causes and effects" — grounded, not speculative |
| Inputs | `EvidenceRecord`, `SceneSummary`, `EntityMemory`, `StateTransition`, temporal `WorldState` |
| Outputs | A `CausalEvent` model (proposed: `event_id`, `participants`, `scene_ids`, `evidence_ids`, `preconditions`, `postconditions`, `confidence`, `inference_class`) and a reconstruction component producing ranked candidate events |
| Constraints | Every event must reference the evidence it was reconstructed from. Events are `inferred` unless directly evidenced. No new visual facts. |
| Tests required | Unit tests for reconstruction rules; tests that no event references non-existent evidence/scene ids; tests that `inferred` events never surface as `observed` claims |
| Success criteria | A measurable improvement in event-reconstruction precision/recall against a hand-annotated fixture set, measured and reported — including cases where it declines to reconstruct |

**Risk:** this is the highest-risk phase. It can easily become unconstrained
story-inference that launders hypotheses as facts. Rule 2 applies hardest here.

---

## PHASE 3 — Narrative possibility space

**Owner: Developer B.**

| | |
|---|---|
| Objective | Replace single-shot planning with an explicit, inspectable space of admissible narrative options |
| Inputs | `CreativePlan` fields, `CausalEvent` set, narrative elements, open loops |
| Outputs | A `PossibilitySpace`: enumerated narrative options, each with its evidence basis, its risk class, and its expected payoff — bounded in size |
| Constraints | Options must be traceable to evidence or to a declared creative marker. Budget classes (`safe_creative`/`risky_inferred`/`forbidden_visual`) still apply. The space must be bounded — no unbounded enumeration. |
| Tests required | Tests that every option carries provenance; tests that `forbidden_visual` options are never generated; tests that space size respects the declared bound |
| Success criteria | Measurable increase in the fraction of options that survive verification, versus the current single-plan baseline |

---

## PHASE 4 — Narrative optimizer

**Owner: Developer B.**

| | |
|---|---|
| Objective | Select among options and shape the blueprint against an explicit multi-objective score |
| Inputs | `PossibilitySpace`, 7-beat structure, creative budget |
| Outputs | A scored, selected narrative direction plus a recorded rationale |
| Constraints | **Must not optimize a single metric** (rule 5). Grounding must be a hard constraint, not a weighted term that can be traded away. The optimizer may choose; it may not invent visual facts. |
| Tests required | Tests that grounding cannot be traded below threshold for other objectives; tests that the optimizer's chosen option is always drawn from the possibility space; determinism tests under fixed seed |
| Success criteria | Demonstrable Pareto improvement over the V2.3 baseline on at least two objectives with **no** regression on grounding; the score decomposition is logged and inspectable |

---

## PHASE 5 — Multiple candidate generation

**Owner: Developer B.**

| | |
|---|---|
| Objective | Generate N distinct story drafts so quality is selected rather than hoped for |
| Inputs | Optimized narrative direction, `StoryPlan`, collection context |
| Outputs | N `StoryDraft` candidates with a diversity guarantee (candidates must not be near-duplicates) |
| Constraints | Each candidate is generated through the existing `StoryGenerationPipeline`. Generation cost grows linearly in N — N must be a configuration value with a default, not a constant. |
| Tests required | Tests that all N candidates verify independently; diversity tests (candidates are not identical); cost tests at the configured N |
| Success criteria | Measurable gain in best-of-N quality over the single-draft baseline, reported together with the generation cost |

---

## PHASE 6 — Bidirectional verification

**Owner: Developer B. Requires Phase 1 provenance.**

| | |
|---|---|
| Objective | Verify in both directions: does evidence support the claim, **and** does the claim contradict evidence? Also detect evidence the story ignored. |
| Inputs | `StoryClaim`, candidates, `EvidenceRecord` set, `CausalEvent` set |
| Outputs | Forward support/contradiction verdicts plus an "unmentioned evidence" report |
| Constraints | Must preserve the `observed`/`inferred`/`creative` classification semantics (rule 4, and the V2.2 regression contract). Creative claims must never be penalized. Must not bypass `ClaimVerifier` — extend it. |
| Tests required | The 14 V2.2 regression tests unchanged and passing; new tests for contradiction detection; tests that creative claims remain unpenalized |
| Success criteria | Measurable reduction in undetected contradicted observed claims versus the V2.3 baseline, with no increase in false contradiction reports |

---

## PHASE 7 — Multi-objective evaluator

**Owner: Developer B, with Developer A for the shared `EvaluationResult` schema.**

| | |
|---|---|
| Objective | Replace the fixed metric bundle with an explicit multi-objective report, and make single-metric optimization visible and discouraged |
| Inputs | Candidates, verification results, continuity, narrative metrics |
| Outputs | An extended `EvaluationResult` with per-objective scores, explicit trade-offs, and a documented aggregation policy |
| Constraints | Extends `ComprehensiveEvaluator`; must not bypass it. Shared-schema change → coordination required. Existing metric semantics must not be silently redefined. |
| Tests required | Existing evaluation tests unchanged; new tests for each objective; tests that the aggregation policy is applied deterministically |
| Success criteria | Objectives are independently inspectable; no single objective can be silently maximized at the expense of grounding |

---

## PHASE 8 — Bounded targeted repair loop

**Owner: Developer B. Requires Phases 6 and 7.**

| | |
|---|---|
| Objective | Close the loop: verify → repair only what is contradicted → re-verify → stop |
| Inputs | Failing candidate, verification results, `EvidenceRecord` |
| Outputs | Repaired `StoryDraft` plus a repair report (what changed and why) |
| Constraints | **Targeted only** — repair contradicted *observed* claims, never regenerate wholesale (rule 7). **Bounded** — declared iteration budget, then stop (rule 9). **Re-verify after every repair** (rule 8). Creative content must not be edited to satisfy a metric. |
| Tests required | Tests that repair never touches creative sentences; tests that the iteration budget is enforced; tests that every repaired claim is re-verified; tests that repair cannot introduce new contradictions |
| Success criteria | Measurable reduction in contradicted observed claims; bounded, documented, reproducible repair traces; zero silent quality regressions |

---

## PHASE 9 — Unseen-image evaluation

**Owner: shared (A provides harness, B provides narrative metrics).**

| | |
|---|---|
| Objective | Replace synthetic-only evidence with evaluation on real images not used during development |
| Inputs | Held-out real image collections; full V3 pipeline |
| Outputs | A held-out evaluation set with published per-metric results and failure analysis |
| Constraints | No tuning on the held-out set. Failures must be reported, not filtered. |
| Tests required | The evaluation must be a reproducible script plus recorded artifacts, not a one-off run |
| Success criteria | Published results on unseen images with a written failure analysis — **including** cases where V3 does not beat V2.3 |

**Why it matters:** every current scale and quality number is synthetic or
self-selected. This phase is what makes any V3 quality claim meaningful.

---

## PHASE 10 — Human narrative-quality evaluation

**Owner: Developer B, with shared protocol design.**

| | |
|---|---|
| Objective | Measure what automated metrics cannot: whether humans actually prefer these stories |
| Inputs | Paired V2.3 and V3 stories; blinded human raters; a written rubric |
| Outputs | Preference study with protocol, sample size, rubric, and results including null/negative outcomes |
| Constraints | Blinded and randomized. Rubric fixed before data collection. Inter-rater agreement reported. |
| Tests required | Protocol review; agreement thresholds; full reproducibility of the analysis |
| Success criteria | A defensible preference result — or an honest null result. Either is a success; a quietly dropped negative result is not |

---

## PHASE 11 — Ablation / model comparison

**Owner: shared.**

| | |
|---|---|
| Objective | Establish which components actually contribute, rather than assuming all of them do |
| Inputs | Full V3 system; each major subsystem independently disabled |
| Outputs | Per-component ablation table; model comparison (e.g. embedding and LLM alternatives) |
| Constraints | Ablations must be run on the same held-out set. Removed components must be genuinely removed, not silently bypassed. |
| Tests required | Automated ablation harness with recorded configurations |
| Success criteria | Every major subsystem shows a measurable contribution, **or** is documented as not earning its complexity (rule 11) |

---

## PHASE 12 — Final research-quality benchmark

**Owner: shared.**

| | |
|---|---|
| Objective | Produce a defensible, reproducible final evaluation of V3 against V2.3 and, where available, published baselines |
| Inputs | Complete V3 system; Phases 9–11 results; frozen configuration |
| Outputs | Final benchmark report: metrics, human evaluation, ablations, cost, limitations, failure cases |
| Constraints | All limitations carried forward from earlier phases must appear. No metric may be dropped after seeing results. Negative results published. |
| Tests required | Full suite green; benchmark reproducible from a frozen commit with a single documented command |
| Success criteria | A result that can be independently reproduced and critically assessed — including an honest account of where V3 does not beat V2.3 |

---

## Dependency summary

```
PHASE 0  (done)
   ↓
PHASE 1  shared contracts          ← gates everything below
   ↓
PHASE 2  causal reconstruction     (B)
   ↓
PHASE 3  possibility space         (B)
   ↓
PHASE 4  optimizer                 (B)
   ↓
PHASE 5  candidate generation      (B)
   ↓
PHASE 6  bidirectional verification (B) ──┐
   ↓                                    ├── PHASE 8 bounded repair loop (B)
PHASE 7  multi-objective evaluator  (B) ──┘
   ↓
PHASE 9  unseen-image evaluation   (shared)
   ↓
PHASE 10 human evaluation          (shared)
   ↓
PHASE 11 ablations                 (shared)
   ↓
PHASE 12 final benchmark           (shared)
```

Phases 9–12 are deliberately last: a V3 quality claim is worthless without
unseen-image and human evaluation, and a component that does not survive ablation
should not exist.