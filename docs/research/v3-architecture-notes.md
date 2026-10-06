# V3 Architecture Notes

> ## DRAFT / DISCUSSION DOCUMENT
>
> This is a set of **notes and ideas**, not an approved design. Nothing described
> here has been built, and nothing here is a decision.
>
> **No V3 code exists in this repository.** V2.3 (`v2.3.0`) is the frozen, working
> system. Everything below is starting material for a discussion between the
> developers.
>
> Nothing in this file needs sign-off, approval, or ownership assignment before it
> can be explored. If an idea turns out to be wrong, change or drop it.
>
> **Planning document:** [`../architecture/roadmap.md`](../architecture/roadmap.md)
> **What exists today:** [`../architecture/current.md`](../architecture/current.md)
>
> **Note on paths.** The findings below were made against the pre-reorganisation
> tree. Where a path is quoted, translate it: `context/world_state.py` →
> `world/world_state.py`, `context/collection_builder.py` →
> `retrieval/collection_context.py`, `memory/collection_builder.py` →
> `memory/collection.py`, `memory/hierarchical_retriever.py` →
> `retrieval/hierarchical.py`, `evaluation/claims.py` →
> `verification/claims.py`, `vision/verifier.py` → `verification/visual.py`,
> `collections/pipeline.py` → `ingestion/jobs.py`. Findings still hold unless
> `current.md` says otherwise — the reorganisation consolidated duplicates, it did
> not change behaviour.

---

## 0. What this document is

Two developers are exploring how to extend a working image→story system. This file
collects:

1. **Findings about the current V2.3 code** that came out of actually reading it.
   These are verified against `main` @ `e2b94e6` and are the most valuable part —
   several are surprising and will shape any V3 work.
2. **Rough ideas** for the subsystems being considered (causal reasoning, multiple
   candidates, verification, repair).

The findings are solid. The ideas are sketches. Keep them separate.

---

## 1. Findings about the current code

These were found by reading `main` at `v2.3.0`. They are the practical starting
point, because several of them change what V3 is allowed to assume.

### 1.1 `EvidenceRecord` cannot be traced back to an image on its own

`EvidenceRecord` (`src/image_story/domain/schemas.py:37`) has `frame_id`, but **no
`image_id` and no `scene_id`**:

```python
@dataclass
class EvidenceRecord:
    id: str            # "obs_<8hex>"
    entity: str = ""
    frame_id: int = 0
    ...
```

The image/scene binding lives in a different object,
`IndexedEvidenceRecord` (`src/image_story/memory/hierarchical.py:290`), which wraps
an evidence record together with:

```python
@dataclass
class IndexedEvidenceRecord:
    evidence: EvidenceRecord
    image_id: str
    scene_id: str
    frame_id: int
    collection_id: str
    index_position: int = -1
```

**Why it matters:** any V3 object that wants to answer *"which image supports
this?"* cannot get there from an `EvidenceRecord` id alone. It needs a lookup built
from the indexed records. So something like a small `ProvenanceIndex`
(`evidence_id → image_id, scene_id, frame_id`) built once per run would be a
prerequisite for most V3 work. It is a small, self-contained piece of plumbing.

Note `SceneSummary` already carries both `image_ids` and `key_evidence_ids`, so the
scene→evidence→image chain is already available from the memory layer.

### 1.2 `StoryClaim.evidence_ids` is never populated

The field exists and is serialised:

```python
@dataclass
class StoryClaim:
    ...
    evidence_ids: list[str] = field(default_factory=list)
```

But **nothing in `src/image_story/` ever writes to it**. `ClaimExtractor` creates
claims and never touches it. Verified: the string `evidence_ids` does not appear in
`src/image_story/evaluation/`.

So today the claim→evidence link is carried indirectly, by `VerificationResult`
(which holds actual `supporting_evidence` objects), not by the claim itself.

There is a test that records this on purpose:
`tests/integration/test_v23c_runtime_proof.py::test_claim_evidence_ids_field_is_unpopulated_known_gap`
(lines 958–972). It asserts the list *stays* empty and says it "will need updating
when the field is wired up". **Whoever wires it up will have to change that test** —
worth knowing in advance.

### 1.3 `StoryBeat.key_evidence_ids` is also never populated

`StoryBeat` (`domain/schemas.py:311`) declares both `key_entities` and
`key_evidence_ids`. `CreativePlanner.create_story_plan`
(`narrative/planner.py:225-294`) fills in `key_entities` for all seven beats but
leaves `key_evidence_ids` empty every time.

`SceneSummary.key_evidence_ids` **is** populated (`memory/scene_grouper.py:86,313`),
so this is a planner gap, not a memory gap.

The field is already serialised by `StoryPlan.to_dict()`, so if it were populated
the plumbing would mostly work.

### 1.4 `StoryDraft` has no scene or entity attribution

```python
@dataclass
class StoryDraft:
    text: str
    story_plan: StoryPlan | None = None
    word_count: int = 0
    generation_time_s: float = 0.0
    model_used: str = "qwen2.5-0.5b-instruct"
    prompt_used: str = ""
```

Nothing links a sentence, or a beat, to a scene or an entity. `StoryBeat` has
`key_entities` (labels, not evidence ids), and claims carry `original_sentence`.

So if you want to ask *"which scenes did this story actually use?"*, it has to be
re-derived after the fact. Two ways this could go:

- **At claim level** — extend `StoryClaim` with a sentence index and beat numbers.
  Stable, because claims are the unit repair would operate on.
- **At sentence level** — precise, but fragile: any repair rewrites sentences and
  offsets stop matching.

Claim level seems more robust, and it reuses the `StoryBeat` structure that already
exists. Worth discussing rather than assuming.

### 1.5 `ClaimVerifier.repair_story()` is broken and unused

`ClaimVerifier.repair_story` (`evaluation/claims.py:353`) calls:

```python
sentence_claims = self._extract_claims_from_sentence(sentence, 0)
```

but `_extract_claims_from_sentence` is a method of **`ClaimExtractor`**, not
`ClaimVerifier`. So the call raises `AttributeError` on the very first sentence.
Confirmed by running it.

It also has **zero callers and zero tests**.

And even with that fixed, the function re-extracts claims from scratch — which
generates fresh random ids like `claim_a1b2c3d4` — and then compares
`vr.claim.id == claim.id`. Fresh ids can never equal the ids from the earlier
extraction, so it would find nothing to repair anyway.

**Conclusion:** there is no working repair loop in V2.3. If V3 wants one, it is
being written from scratch, not extended. The *intent* is good though — repair only
touching contradicted observed claims — so the docstring is worth reading as a
specification.

### 1.6 Textual verification cannot detect contradictions

`ClaimVerifier._textual_verification` (`evaluation/claims.py:266-311`) only ever
appends to `supporting`:

```python
for evidence in evidence_records:
    ...
    if matches >= 2:
        supporting.append(evidence)
    elif matches == 1 and evidence.confidence > 0.8:
        supporting.append(evidence)
```

There is no `contradicting.append(...)` anywhere in the function, yet it contains:

```python
if contradicting:
    status = ClaimStatus.CONTRADICTED
```

That branch is dead. A claim with no matching evidence comes back `unsupported`; it
can never come back `contradicted`. Verified with a deliberately mismatched claim.

Contradiction detection *does* exist on the image path —
`VisualVerifier.verify_claim` (`vision/verifier.py:53-96`) populates
`contradicting_evidence` properly. So the gap is specific to the text-only
fallback, which is what runs when no image is available.

### 1.7 The creative budget is declared but never enforced

`CreativePlan.creative_budget` and `PipelineConfig.creative_budget` both define:

```python
"safe_creative":    {"max": 8, "used": 0},
"risky_inferred":   {"max": 3, "used": 0},
"forbidden_visual": {"max": 0, "used": 0},
```

Across all five config presets. And `used` is **never incremented anywhere** in
`src/`. There is no code path that counts creative claims against these limits.

The `forbidden_visual: 0` limit — the one that stops the system inventing new
objects, colours, materials or people — is therefore not actually being enforced at
runtime. It's documentation right now.

This is probably the highest-value thing to fix early, because it makes an existing
safety property real rather than aspirational.

### 1.8 Two classification vocabularies coexist and disagree

`StoryClaim` has two classification fields:

```python
claim_type: InformationClass = InformationClass.HARD_FACT   # hard_fact / soft_inference / creative_space
claim_classification: str = "inferred"                      # "observed" / "inferred" / "creative"
```

They're produced by two separate classifiers in `ClaimExtractor`
(`_classify_claim_type` and `_classify_claim_classification`) with overlapping and
conflicting word lists. For example `"looked like"` and `"appeared to be"` appear in
**both** the visual-word list and the inference-marker list, so which one wins
depends on evaluation order.

Worth knowing: `claim_classification` is a plain `str`, not the `ClaimClassification`
enum that already exists in `domain/enums.py:154` (which is currently unused by
`StoryClaim`). The V2.2 regression tests assert on the returned *string*, so
switching to the enum would break them.

**Recommendation:** V3 should not add a third vocabulary on top. If a third axis is
needed for V3, it should be a separate, clearly-scoped enum.

### 1.9 The grading weights that must not drift

`ClaimVerifier.compute_claim_grounding_score` (`evaluation/claims.py:313`) implements
the V2.2 behaviour contract, and 14 regression tests depend on it:

- `creative` claims are **excluded entirely** from the denominator
- `observed` claims count fully when supported
- `inferred` claims count half
- contradicted claims subtract half
- a story with only creative claims scores `1.0`

The spirit of it: invention is not penalised. Anything that makes creative content
count against grounding would break both the tests and the design intent.

### 1.10 Duplicate definitions exist

Two classes with the same name:

```
src/image_story/domain/schemas.py:147      class SceneSummary:   # shadow, unused, no to_dict
src/image_story/memory/hierarchical.py:14  class SceneSummary:   # the one actually used
```

Everything in `memory/` and all tests import the `memory.hierarchical` one. The
`schemas.py` copy looks like dead weight.

And `domain/enums.py` (401 lines) redefines several enums repeatedly:

| Enum | Times defined |
|---|---|
| `CollectionStatus` | 6 |
| `ImageOrderingMode` | 4 |
| `ImageValidationStatus` | 4 |
| `ImageProcessingStatus` | 4 |
| `ProcessingJobStatus` | 4 |
| `SessionStatus` | 2 |

All copies are identical and the last one wins, so behaviour is currently correct —
but it's a trap for anyone adding a member, and it's why new V3 enums should go in a
new file rather than in `enums.py`.

### 1.11 Serialisation is field-by-field, so new fields are easy to drop

`StoryPlan.to_dict()` (`domain/schemas.py:329`) lists each beat field explicitly:

```python
"beats": [
    {
        "beat_number": b.beat_number,
        "beat_type": b.beat_type,
        ...
        "key_evidence_ids": b.key_evidence_ids,
        ...
    }
    for b in self.beats
],
```

`StoryDraft.to_dict()` and `EvaluationResult.to_dict()` have the same shape. Adding a
dataclass field without also adding it here means it silently vanishes from saved
artifacts. Not a bug today — just something to remember.

### 1.12 `ExperimentManifest` is never populated

`ExperimentManifest` (`domain/schemas.py:1016`) has `git_commit`, `vision_model`,
`language_model`, `embedding_model`, `dataset`, `dataset_hash` — all defaulting to
`""`. `run_collection()` only fills in `run_id`, `config`, `seed` and `device`.

So runs are not currently self-describing. Not V3-specific, but it would make
V3 experiments harder to reproduce, so it's convenient to fix whenever it's next
touched.

### 1.13 The pipeline produces exactly one story

`run_collection()` (`pipeline/orchestrator.py:416`) is the single collection entry
point and runs 8 stages, generating **one** `StoryDraft` and evaluating **that one**:

```python
story_draft = self._gen_pipeline.generate_story(prompt, story_plan)   # Stage 7
...
eval_result = self._evaluator.evaluate(artifacts, ...)                # Stage 8
```

`ComprehensiveEvaluator.evaluate()` reads `artifacts.story_draft` (singular) and
writes one `artifacts.claims` / `artifacts.verification_results` list. So "generate
N candidates and compare them" isn't a small change — it's a shape change to how
artifacts are organised.

Also worth noting: `run_single_image`, `run_multi_image` and `run_collection` each
duplicate the stage sequencing. Any new stage has to be added to all three or the
paths quietly diverge.

---

## 2. Information classes: the idea worth being strict about

This is the single most important thing to carry into V3, and it comes straight from
what already works in V2.2.

### 2.1 Three kinds of content, and they must stay separate

| Class | Meaning | Can it cite evidence? |
|---|---|---|
| **OBSERVED** | Directly visible in an image | Yes |
| **INFERRED** | A reasonable reading of what's visible | Yes, but must be labelled inferred |
| **CREATIVE** | Invented — motivations, humour, dialogue, metaphor | **No.** Never |

V2.2 already does this through the `safe_creative` / `risky_inferred` /
`forbidden_visual` budget (§1.7).

### 2.2 Why this matters more once V3 adds causal reasoning

Right now the risk is contained: the pipeline extracts claims, classifies them, and
stops. The moment V3 reconstructs events, something new becomes possible:

> *"The character dropped the balloon, which is why the dog barked."*

That sentence contains an **inference** (why the dog barked) presented as though it
were **observed** (the dropping). If a reasoning step produces a plausible causal
chain and the generation step then treats it as fact, the whole grounding story
quietly collapses. Nothing would fail. The story would just be wrong in a way the
evaluator can't see.

Call this **hypothesis laundering**: a guess becomes a fact by passing through the
system without ever being labelled.

### 2.3 Possible ways to make it structural rather than aspirational

Not a decision — just the options, roughly in order of how much they cost:

1. **Separate types.** A `CausalHypothesis` is a different class from a
   `CausalEvent`. They can't be passed interchangeably because they aren't the same
   type.
2. **Force the label.** A hypothesis object's constructor sets its class to
   `HYPOTHESIS` unconditionally, so there's no code path that yields a
   hypothesis-labelled-as-evidence.
3. **Keep them in separate lists.** Evidence lives in one collection, hypotheses in
   another. A hypothesis is never an `EvidenceRecord`.
4. **Namespaces that don't collide.** If evidence ids are `obs_*` and hypothesis ids
   are `hyp_*`, a hypothesis id simply won't resolve when someone tries to look it
   up as evidence.
5. **A test.** Assert that no creative or hypothetical text is ever emitted as an
   `EvidenceRecord`, and that no hypothesis id appears in an evidence id list.

Options 1–4 are cheap. The guarantee they give is that the *type system* prevents
laundering rather than a convention that everyone remembers.

### 2.4 Creative freedom has to survive

Whatever V3 adds must not make the writing worse or more timid. The V2.2 instinct —
creative content is excluded from grounding, not penalised by it — should carry
forward. A more sophisticated evaluator that makes safe invention look risky would
push generation toward dull, hedged prose.

---

## 3. Provenance: an idea for discussion

V3 objects that can influence the story should be able to say where they came from.

**For claims:**

```
StoryCandidate → StoryClaim.evidence_ids → EvidenceRecord → image_id / scene_id
```

This needs §1.1's `ProvenanceIndex` at the second step.

**For causal reasoning:**

```
NarrativeOpportunity → CausalEvent / CausalHypothesis → evidence_ids → image
```

Worth being careful about the direction: a hypothesis's evidence ids record *what it
was built from*, never *what proves it*. If those two are the same field, a claim
supported only by a hypothesis will start looking grounded.

**Two open questions:**

- **What should `evidence_ids` mean?** "Direct supporting evidence" is the strictest
  and most useful reading. "All evidence considered" is easier but nearly
  information-free — it would pass any non-empty check by default.
- **Should creative claims have evidence ids at all?** Arguably they must be empty.
  Putting evidence on invented content is another way to make invention look
  grounded.

---

## 4. Sketch ideas for the V3 subsystems

Loose shapes, not designs. Each is a question more than an answer.

### 4.1 Causal events

Instead of "entity X was in scene Y", something like "event E happened, involving
these participants, with these effects".

```python
@dataclass
class CausalEvent:
    event_id: str                    # "evt_<8hex>"
    description: str
    participants: list[str]          # entity ids
    scene_ids: list[str]
    evidence_ids: list[str]          # what it was reconstructed from
    information_class: str           # observed / inferred / hypothesis
    confidence: float
    causes: list[str]                # other event ids
    start_frame: int
    end_frame: int
```

Notes:
- Ordering is better as **links between events** than as a number.
- Image ids can be looked up from the scene, so storing them separately risks drift.
- `StateTransition` (`memory/hierarchical.py:168`) already captures single-entity
  state changes with evidence. It's a natural *input* to reconstruction — but an
  event with several participants is a different kind of assertion.

### 4.2 Causal hypotheses

A separate object from an event, so the two can't be confused:

```python
@dataclass
class CausalHypothesis:
    hypothesis_id: str               # "hyp_<8hex>"
    statement: str
    supports: list[str]              # event ids it would explain
    built_from_evidence_ids: list[str]
    falsified_by: list[str]          # what would show it's wrong
    plausibility: float
```

Recording what would *disprove* a hypothesis is the useful bit — it keeps the system
honest instead of just accumulating plausible-sounding guesses.

### 4.3 Narrative opportunities

An admissible opening for the story, rather than a fact:

```python
@dataclass
class NarrativeOpportunity:
    opportunity_id: str              # "opp_<8hex>"
    summary: str
    source_event_ids: list[str]
    source_entity_ids: list[str]
    evidence_ids: list[str]
    risk_class: str                  # safe_creative / risky_inferred / forbidden_visual
    creative_freedom: float
    expected_payoff: float
```

Using the **existing** `safe_creative` / `risky_inferred` / `forbidden_visual`
vocabulary here would be a good idea. A parallel risk scale would drift from the
budget in §1.7 and quietly weaken the only rule that forbids inventing new visual
facts.

### 4.4 Multiple candidates

Generate N drafts, evaluate each, keep the best.

The interesting questions are less about the idea and more about cost and shape:
generation cost scales linearly with N; candidates need to actually differ from each
other; and §1.13 means the artifact structure has to change to hold more than one.

An easy way to keep it compatible: keep `story_draft` pointing at the *selected*
candidate, and add a list alongside it. Downstream consumers keep working.

### 4.5 Verification in both directions

- **Forward** (story → world): does the evidence support the claim? Mostly exists.
- **Backward** (world → story): does the story contradict something visible, or
  quietly ignore something important?

The distinction that matters:

| Situation | Should it matter? |
|---|---|
| Claim contradicts evidence | Yes — blocking |
| Claim asserts a visible thing with no evidence | Yes — blocking |
| Plausible causal chain stated as fact | Yes — blocking (the laundering case) |
| Invented motivation, humour, metaphor | **No** — never a defect |
| Story ignores a background object | Advisory only — never a score |

**The "coverage" trap.** The tempting version of backward verification rewards the
story for mentioning everything, which produces keyword stuffing — all the content,
none of the meaning. Better to keep omission as a *reported diagnostic* than as
something to optimise, and to limit "important" evidence to something bounded like
`SceneSummary.key_evidence_ids`. Quality of support matters more than how much gets
mentioned.

### 4.6 Multi-objective evaluation

`EvaluationResult` (`domain/schemas.py:754`) is currently ~26 flat fields:
`grounding_score`, `grounding_pass`, `claim_grounding_score`, `continuity_score`,
`narrative_coherence`, `contradiction_count`, `repetition_rate`, timings.

Two ideas worth considering:

- **Keep the objectives separate.** V3 will add more dimensions, and collapsing them
  into one number early would hide the trade-offs.
- **Treat grounding as a gate, not a weight.** `grounding_pass` already exists as a
  boolean. A candidate that fails grounding being *ineligible* is easier to reason
  about than a low grounding score losing to a high creativity score. Otherwise
  "improve coherence by making the story vaguer" becomes a winning strategy.

If selection is ever needed: filter by gates, then compare on the Pareto front
(objectives that improve together without any worsening). That avoids inventing
weight numbers and makes the trade-offs visible.

Runtime cost should be recorded but not treated as a quality measure.

### 4.7 Targeted, bounded repair

```
verify → evaluate → pass → done
              ↓
             fail → repair only what failed → re-verify → re-evaluate
```

Reasonable bounds:

- A small iteration cap (2 is probably plenty; most issues are hedgeable in one pass).
- Only contradicted **observed** claims are repairable.
- Creative sentences are never edited to satisfy a metric.
- Re-verify after every change — an unverified "fix" is just a guess.
- If the budget runs out, **report the failure**. Don't quietly ship the last
  iteration, and don't lower a gate to make something pass.

The V2.2 intent in `repair_story`'s docstring is a good starting spec, even though
the implementation doesn't work (§1.5). Common repairs: soften a too-strong
assertion, point at the right evidence, or drop the assertion if nothing supports
it — cheapest and least destructive first.

---

## 5. Things worth deciding together

Not blockers, just the questions that came up:

1. **Claim attribution level** — claim-level or sentence-level (§1.4)? Claim level
   looks more robust under repair.
2. **What `evidence_ids` should mean** — direct support vs. everything considered
   (§3).
3. **Should creative claims have evidence ids?** Probably not (§3).
4. **Where the creative budget gets enforced** — this is the real safety property,
   and nothing enforces it yet (§1.7).
5. **How to hold N candidates** — new list field vs. one artifacts object each
   (§1.13, §4.4).
6. **Whether to fix `repair_story` or write a new repair loop** (§1.5).
7. **Whether hypothesis-derived content may appear in a story at all**, and if so,
   always labelled as inferred (§2.2).
8. **Cleanup of the duplicate `SceneSummary` and `enums.py`** (§1.10) — low
   priority, but annoying while it lasts.
9. **`ExperimentManifest` population** (§1.12), whenever it's next convenient.

---

## 6. Ground rules worth keeping

These come from what already works, and losing any of them would quietly degrade the
system:

- **Evidence first.** Every narrative element traces to evidence, a declared
  inference, or an explicit creative marker.
- **Hypotheses never become visual facts** (§2).
- **Provenance is preserved.** If a new component can't say what it used, it isn't
  finished.
- **Creative freedom is preserved** — invention is excluded from grounding, not
  penalised by it (§1.9, §2.4).
- **Don't optimise one metric at the expense of another.** Especially grounding.
- **Repair is targeted and bounded** (§4.7).
- **Failed validation gets recorded**, not quietly filtered out.
- **V2.3 stays working.** 233 tests passing is the baseline everything is measured
  against. Documentation and experiments should never move that number.

---

## 7. Status

**Nothing in sections 2–4 is implemented.** No causal reasoning, no candidate
generation, no verification changes, no repair loop. The dataclass sketches are
illustrative and will probably change.

Section 1 is verified against `main` @ `e2b94e6` (`v2.3.0`) and can be relied on.

This is a discussion document. Rewrite it freely.
