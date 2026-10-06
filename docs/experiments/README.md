# Experiments

Recorded experiment configurations and the reports they produced.

## Layout

- `../configs/` - pipeline presets (`fast`, `v2`, `full`, `baseline`).
- `experiments/` - one YAML per recorded experiment, checked in so a run can be
  reproduced from its config alone.
- `../artifacts/` - raw run output (git-ignored; regenerate with `make run-*`).
- Reports worth keeping are committed as markdown here.

## Reports

| Report | What it covers |
|---|---|
| `validation-report.md` | Initial V2 validation. |
| `validation-report-v21.md` | After the V2.1 grounded-creativity work. |
| `validation-report-v22.md` | After the V2.2 risk-aware creative budget. |
| `../architecture/v2.3-history.md` | The V2.3 architecture as it stood before the repository reorganisation. Useful for reading intent, not for reading current code. |

## Running one

```bash
python scripts/run_experiment.py ../benchmarks/challenge-8-images \
    --name v2-standard --output-dir ../artifacts \
    --summary ../artifacts/experiments/v2-standard.json

python ../scripts/build_report.py ../artifacts/experiments
```

Compare against the frozen Challenge V1 baseline in
`../../challenges/image-story-v1/` — its `results_final.csv` and
`results_final_fixlen.csv` are the numbers the V2 work is measured against.

## Comparability

Metrics are only comparable across runs when the model set, the images and the
seed are the same. Every run records a manifest (`src/image_story/experiments/manifests.py`)
with the git commit, models, config, seed, device and dataset hash, so a
mismatched comparison is visible rather than silent.

Do not overwrite a report. Add a new one, and say in the repository README or a
commit message what changed, otherwise the history of what was tried gets lost.