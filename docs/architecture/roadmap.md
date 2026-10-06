# Roadmap

Where V2 ends and what V3 adds. This is a plan; none of the V3 items exist in
code yet. When an item is implemented, move it into
[`current.md`](current.md) and update the table.

## Where we are

```
V2 complete foundation          DONE   perception -> evidence -> world -> memory
                                     -> retrieval -> planning -> generation
                                     -> verification -> evaluation -> artifacts
V2.3 collection / memory        DONE   scenes, entity memory, transitions,
                                     narrative memory, hierarchical retrieval
V3 causal reasoning             NEXT   the first V3 increment
```

## Phase 0 - V2 foundation (complete)

Perception, evidence with provenance, world state, FAISS, hierarchical memory,
retrieval, grounded context, 7-beat planning, claim extraction and verification,
evaluation, experiment manifests, reproducibility.

Consolidated into one canonical package under `src/image_story/` in the
`restructure/foundation-v2` branch. See [`current.md`](current.md).

## Phase 1 - V3 causal reasoning

Reconstruct what happened across a sequence, as hypotheses with confidence,
rather than leaving the ordering implicit in the frame list.

New package: `causal/`

```
causal/
  events.py          CausalEvent: what happened, between which entities
  hypotheses.py      CausalHypothesis: candidate cause, confidence, supporting evidence ids
  reconstruction.py  builds events + hypotheses from WorldState, StateTransition, observations
  confidence.py      scoring and calibration rules
```

Interface to land first: `CausalHypothesis.evidence_ids` must resolve to real
`EvidenceRecord.id` values. This is the same provenance requirement the V2 notes
flag as an open gap (sections 1.1-1.3), so fixing it here is a prerequisite, not
an optional extra.

Exit condition: given an ordered image set, the system produces hypotheses whose
evidence links resolve, and a hypothesis is falsified when its evidence is
contradicted.

## Phase 2 - Narrative possibility space

Enumerate what *could* be told, given the evidence and the causal hypotheses,
before committing to one story.

New modules under `narrative/`: `possibilities.py`

```
NarrativePossibility: a candidate direction, with the hypotheses it uses,
                      the evidence it rests on, and its risk profile
```

Output: a set of directions, not a plan. Direction selection is where the
optimizer in phase 3 gets its input.

Exit condition: at least three distinct, evidence-consistent directions per
collection, each traceable to the hypotheses and evidence it uses.

## Phase 3 - Narrative optimizer

Choose among the possibilities under explicit, inspectable objectives.

New module under `narrative/`: `optimizer.py`

The optimizer scores possibilities against the same information classes the
context builder already enforces, and its decision is recorded in the artifact
so a run can be explained after the fact.

Exit condition: the chosen direction and the scores of the rejected ones appear
in `PipelineArtifacts`.

## Phase 4 - Multiple candidates

Generate more than one story. `generation/candidates.py`.

```
CandidateStory: text, plan, possibility id, generation metadata
```

Exit condition: `run_candidates()` returns N candidates from one evidence set,
each with its own plan and manifest.

## Phase 5 - Bidirectional verification

Forward verification exists (does the evidence support this claim?). Backward
verification does not (did we use the evidence we had?).

New modules under `verification/`: `backward.py`, `bidirectional.py`

```
backward.py      evidence coverage: which retrieved evidence went unused, and why
bidirectional.py combines both directions into one verdict per candidate
```

Exit condition: every candidate carries both a forward and a backward verdict;
evidence coverage is reported, not just claim support.

## Phase 6 - Multi-objective evaluation

Score candidates across the existing metrics plus the new dimensions, and select
on a weighted objective rather than a single number.

New modules under `evaluation/`: `causality.py`, `character.py`, `creativity.py`,
`repetition.py`, `multi_objective.py`

Exit condition: selection is by declared weights, the weights appear in the
manifest, and no metric is silently dropped from the aggregate.

## Phase 7 - Targeted repair loop

Verify -> repair -> re-verify -> re-evaluate, bounded and auditable.

New module under `verification/`: `repair.py`, plus `refinement.py`

`ClaimVerifier.repair_story()` exists but is unused and, per the V2 notes
(section 1.5), broken. Repair must be targeted at specific failed claims with a
maximum number of attempts, and every attempt recorded.

Exit condition: a failed candidate can be repaired at most N times, each repair
names the claims it addressed, and the final verdict is recomputed, not reused.

## Sequencing

```
causal events
  -> narrative possibility space
    -> narrative optimizer
      -> candidate generation
        -> bidirectional verification
          -> multi-objective selection
            -> bounded repair loop
```

Each phase needs the previous one's output in the artifacts. Skipping a phase
breaks the provenance chain, which is the property that makes the results worth
anything.

## What must not regress

- Grounding must stay measurable on the same images and metrics, otherwise
  V2 and V3 numbers are not comparable. Use `benchmarks/challenge-8-images` and
  keep `challenges/image-story-v1/` as the frozen baseline.
- One story per run stays the default. Candidate generation is opt-in.
- The information-class separation (hard fact / soft inference / creative space)
  is the mechanism that keeps creativity from becoming hallucination. It must
  survive every phase above.