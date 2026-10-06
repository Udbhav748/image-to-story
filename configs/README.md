# Configs

`PipelineConfig` presets. Two ways to select one:

```bash
python -m image_story IMAGES --mode standard        # built-in preset
python -m image_story IMAGES --config configs/v2.yaml  # file, overrides --mode
```

| File | Mode | Notes |
|---|---|---|
| `fast.yaml` | `fast` | No GroundingDINO, no FAISS, no planner, no verification. The smoke test: it needs the fewest models, so it is what CI and `CONTRIBUTING.md` use to check the pipeline still runs. |
| `v2.yaml` | `standard` | The primary development mode. Named for the architecture version, not the mode. |
| `full.yaml` | `full` | Adds OCR. |
| `baseline.yaml` | `baseline` | No grounding, no retrieval, no planner, no verification. The comparison arm. |
| `experiments/` | — | One file per recorded experiment, so a run can be reproduced from its config. |

## Relationship to `MODE_PRESETS`

The built-in presets in `src/image_story/config/defaults.py` and the YAML files
here are two ways to express the same thing. YAML is for the values you want to
keep with an experiment; the Python table is for the defaults. A YAML file wins
because `--config` is loaded instead of a preset.

Every field a YAML file may set is a field of `PipelineConfig`
(`src/image_story/domain/schemas.py`). Unknown keys raise `TypeError` at load
time rather than being silently ignored.

The mode presets are pinned by `tests/unit/test_config_defaults.py`. Changing
one makes the numbers in `docs/experiments/` stale, which is the point of pinning
them.