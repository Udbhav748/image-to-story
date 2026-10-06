# Benchmarks

Input image sets for experiments. These are data, not code.

- `challenge-8-images/` - the eight Challenge V1 evaluation frames. Used by
  `challenges/image-story-v1/README.md` and as the default input for
  `make run`.
- `unseen-images/` - held-out images for generalisation checks (not yet populated).
- `synthetic/` - generated inputs for scale benchmarks; the generator lives in
  `image_story.experiments.benchmarks` rather than here.

The pipeline treats any of these directories as an image set ordered by file
name. Multi-image runs use that order as the frame order.

## Running

```bash
make run IMAGES=benchmarks/challenge-8-images
make run-collection IMAGES=benchmarks/challenge-8-images
python scripts/run_experiment.py benchmarks/challenge-8-images --name v2-standard
```

## Reported results

Aggregate evaluation output lands in `artifacts/` (git-ignored). Summaries and
validation reports that are worth keeping are committed under `docs/experiments/`.