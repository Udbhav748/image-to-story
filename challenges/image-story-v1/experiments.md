# Experiments: Image -> Story (Modules 7 and 10)

All numbers come from `results_final.csv` (normal run) and `results_final_fixlen.csv` (length-controlled run), produced by
`main.py` offline on a CPU-only laptop (Python 3.13, torch 2.13, transformers 5.15), 8 images from the *Spirited Away* frame set,
one run each, 16/16 evaluations successful. Historical result files are listed under Reproducibility.

## Problem

The baseline passes only one BLIP sentence to the story model. Most visual detail is lost before generation, and the story model invents
details to fill the gaps.

## Baseline

`Image -> BLIP (Salesforce/blip-image-captioning-base) -> one sentence -> Qwen2.5-0.5B-Instruct -> story` (greedy, seed 0, `repetition_penalty=1.05`,
prompt asks for 80-120 words). **The original instructor baseline code was unavailable, so this is a stand-in matching the documented architecture.**
It is not the exact official baseline.

## Observed failures

1. **Wrong length.** 0/8 baseline stories are 80-120 words (range 30-76). The baseline's only built-in check is word count.
2. **Details with no basis in the input.** `chihiro003.jpg`: the caption is "a painting of a street scene with a man walking down the street" and the story adds "jeans and a t-shirt that shows off his muscular build". `thumb-chihiro008.png`: the caption mentions "a man and a dog" and the story calls the dog "a golden retriever".
3. **Generic stories from a one-sentence caption.** `thumb-chihiro007.png`: the caption is "a restaurant with a table and chairs" and the story adds coffee, bread and a waiter that nothing in the input supports.

## Hypothesis

Giving the story model a structured description (detailed caption, detected objects, region captions) instead of one sentence will raise the CLIP
image-story similarity and lower the NLI contradiction, because the model has facts to use instead of gaps to fill.

## Module 7 change

**Replace BLIP's single-sentence image understanding with Florence-2 structured visual extraction.** The following are implementation components of this
richer seeing stage and were **not ablated separately**; the results cannot be attributed to any one of them:

- `<MORE_DETAILED_CAPTION>`, `<OD>` and `<DENSE_REGION_CAPTION>` from `florence-community/Florence-2-base` (greedy, CPU; `<OCR>` disabled because it returned junk)
- a structured JSON (`scene`, `characters`, `objects`, `od_labels`, `actions`, `spatial_relations`, `region_descriptions`, `style_or_mood`)
- a deterministic context builder that turns the JSON into the story prompt (no ML, no image-specific vocabulary)

The same Qwen model, seed, greedy decoding, repetition penalty, prompt template, threshold and metric formula are used for both pipelines, and `--fix-length`
is identical for both. The only intended difference is the seeing stage.

## Evaluation method (Module 10)

Every (image, context, story) triple is scored by `metric.py`:

- `grounding_score = 0.4*clip_n + 0.4*(1 - nli_contra_mean) + 0.2*(0 if attribute_conflict else 1)`, `clip_n = clip((cos - 0.15)/0.15, 0, 1)`.
- CLIP is the direct image-story similarity signal. NLI measures **textual consistency between the story and the supplied/generated visual context**; it is not
  independent image verification. The attribute check is a deterministic textual heuristic. The score is a composite automatic proxy, not ground truth.
- `length_valid = 80 <= words <= 120` is a benchmark requirement, not proof of narrative quality. `grounding_pass = score >= 0.60 AND no attribute conflict AND length_valid`.
- Runtime is measured with `time.perf_counter`, model loading excluded. Generation (`see_s + story_s`) and evaluation (`eval_clip_s`, `eval_nli_s`, `eval_rules_s`) are reported separately.
- Continuity (entity overlap and transition words) is computed only for the multi-image story, as a structural proxy. It is not reported for single-image runs.

## Normal evaluation (`results_final.csv`)

| Metric | Baseline | Improved | Delta |
|---|---|---|---|
| Mean grounding score | 0.783 | 0.856 | +0.072 |
| Mean CLIP similarity | 0.233 | 0.257 | +0.024 |
| Mean NLI contradiction | 0.098 | 0.075 | -0.023 |
| Mean repetition rate | 0.0034 | 0.0187 | +0.0154 |
| Length valid | 0/8 | 1/8 | |
| Grounding pass | 0/8 | 1/8 | |

## Length-controlled evaluation (`results_final_fixlen.csv`)

`--fix-length`: up to 3 greedy attempts with stricter length prompts, each cut to whole sentences and at most 120 words; the first attempt with 80-120 words is
returned, otherwise the attempt closest to 100 words. It does not produce equal word counts.

| Metric | Baseline | Improved | Delta |
|---|---|---|---|
| Mean grounding score | 0.756 | 0.827 | +0.070 |
| Mean CLIP similarity | 0.240 | 0.254 | +0.014 |
| Mean NLI contradiction | 0.207 | 0.061 | -0.145 |
| Mean repetition rate | 0.0068 | 0.0034 | -0.0034 |
| Length valid | 6/8 | 6/8 | |
| Grounding pass | 5/8 | 5/8 | |

## Before vs after

| Metric | Baseline (normal) | Improved (normal) | Delta | Baseline (length-controlled) | Improved (length-controlled) | Delta |
|---|---|---|---|---|---|---|
| Mean grounding score | 0.783 | 0.856 | +0.072 | 0.756 | 0.827 | +0.070 |
| Mean CLIP image-story similarity | 0.233 | 0.257 | +0.024 | 0.240 | 0.254 | +0.014 |
| Mean NLI contradiction (lower is better) | 0.098 | 0.075 | -0.023 | 0.207 | 0.061 | -0.145 |
| Mean repetition rate (lower is better) | 0.0034 | 0.0187 | +0.0154 | 0.0068 | 0.0034 | -0.0034 |
| Length valid (80-120 words) | 0/8 | 1/8 | | 6/8 | 6/8 | |
| Grounding pass | 0/8 | 1/8 | | 5/8 | 5/8 | |
| Mean words per story | 56.6 | 62.6 | | 92.8 | 93.1 | |
| Images improved / regressed (grounding score) | | 5 / 3 | | | 5 / 3 | |

Repetition is higher for the improved pipeline in the normal run (0.0034 -> 0.0187) and lower in the length-controlled run (0.0068 -> 0.0034); repeated trigrams are rare
in absolute terms, so this is not a finding. The normal-run pass rate (0/8 -> 1/8) mostly reflects length: only 1/8 improved stories are 80-120 words.

## Improvement cases

Grounding score improved on 5 of 8 images in the normal run (chihiro003, thumb-chihiro002, thumb-chihiro005, thumb-chihiro006, thumb-chihiro008) and on 5 of 8 in the length-controlled run (chihiro003, thumb-chihiro002, thumb-chihiro005, thumb-chihiro007, thumb-chihiro008).
Per-image values:

Normal:

| Image | Baseline grounding | Improved grounding | Delta | Delta CLIP | Delta NLI | Words (baseline / improved) |
|---|---|---|---|---|---|---|
| chihiro003.jpg | 0.740 | 0.856 | +0.116 | +0.032 | -0.075 | 71 / 99 |
| thumb-chihiro001.png | 0.856 | 0.817 | -0.039 | -0.010 | +0.033 | 45 / 66 |
| thumb-chihiro002.png | 0.644 | 0.911 | +0.267 | +0.068 | -0.218 | 47 / 65 |
| thumb-chihiro004.png | 0.895 | 0.823 | -0.073 | -0.020 | +0.050 | 30 / 59 |
| thumb-chihiro005.png | 0.722 | 0.841 | +0.119 | +0.051 | +0.042 | 50 / 41 |
| thumb-chihiro006.png | 0.878 | 0.975 | +0.097 | +0.029 | -0.047 | 58 / 66 |
| thumb-chihiro007.png | 0.787 | 0.750 | -0.037 | -0.011 | +0.019 | 76 / 52 |
| thumb-chihiro008.png | 0.746 | 0.873 | +0.127 | +0.050 | +0.013 | 76 / 53 |

Length-controlled:

| Image | Baseline grounding | Improved grounding | Delta | Delta CLIP | Delta NLI | Words (baseline / improved) |
|---|---|---|---|---|---|---|
| chihiro003.jpg | 0.800 | 0.887 | +0.087 | +0.033 | +0.001 | 107 / 110 |
| thumb-chihiro001.png | 0.811 | 0.783 | -0.027 | +0.009 | +0.129 | 66 / 114 |
| thumb-chihiro002.png | 0.691 | 0.712 | +0.021 | +0.049 | -0.224 | 86 / 116 |
| thumb-chihiro004.png | 0.902 | 0.827 | -0.075 | -0.020 | +0.054 | 52 / 59 |
| thumb-chihiro005.png | 0.677 | 0.821 | +0.145 | +0.019 | -0.236 | 112 / 61 |
| thumb-chihiro006.png | 0.905 | 0.851 | -0.054 | -0.033 | -0.086 | 82 / 106 |
| thumb-chihiro007.png | 0.777 | 0.911 | +0.133 | +0.031 | -0.127 | 117 / 92 |
| thumb-chihiro008.png | 0.488 | 0.820 | +0.333 | +0.024 | -0.675 | 120 / 87 |

## Regression cases

- Normal run: 3 of 8 images regressed (thumb-chihiro001, thumb-chihiro004, thumb-chihiro007). Length-controlled run: 3 of 8 regressed (thumb-chihiro001, thumb-chihiro004, thumb-chihiro006).
- `thumb-chihiro001.png` (both modes): the structured context contains both "girl" and "boy" for the same figure (`Characters: girl, boy, human`).
- `thumb-chihiro004.png` (both modes): the characters list contains "dog", extracted from the region caption "man eating hot dog" by the keyword matcher in `seeing.py`.
  This is an extraction artifact (the word "dog" inside "hot dog"), not a detected dog. A fix would change the contexts and require new runs, so it is left documented.
- Attribute check: it fired on 1 of 32 scored rows (`thumb-chihiro002.png`, improved, length-controlled: "red"), so it contributes almost nothing here.

## Runtime

| Stage (mean seconds per image, model loading excluded) | Baseline (normal) | Improved (normal) | Baseline (length-controlled) | Improved (length-controlled) |
|---|---|---|---|---|
| Seeing / vision | 1.4 | 15.5 | 1.5 | 16.5 |
| Story generation | 6.3 | 7.1 | 22.2 | 21.9 |
| **Generation total** | **7.7** | **22.6** | **23.7** | **38.5** |
| Evaluation (CLIP + NLI + rules; not part of generation) | 0.92 | 1.22 | 1.17 | 1.94 |

Model loading is excluded. The improved pipeline is slower because Florence-2 runs three tasks per image. Length control raises story time through retries.
Per-image evaluation time varies; the median `eval_total_s` is 0.93 s (normal) and 1.27 s (length-controlled).

## Limitations

- Stand-in baseline; 8 images, one run, seed 0; untuned threshold and CLIP range.
- CLIP is the only image-aware signal. NLI and the attribute check compare the story with generated text, so seeing-stage errors propagate into stories and are not penalised by them.
- Closed vocabularies (character/action keywords in `seeing.py`; colour, material and transition words in `metric.py`).
- The attribute check is a heuristic that can raise false alarms and misses most conflicts.
- Continuity is an entity-overlap and transition-word proxy; it does not evaluate narrative coherence. Generic recurring labels count like any other.
- The components of the seeing stage were not ablated, so no causal claim is made about which one helps.
- The size of the gain depends on how the context is formatted (see the historical rows below); with 8 images and one run this cannot be separated from variation.
- No fluency or narrative-quality metric. Length control does not guarantee a valid length (6/8 improved stories are in range even with it).

## Reproducibility

```powershell
$env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
python main.py images/ --both --output results_final.csv --vision-json vision_final.json
python main.py images/ --both --fix-length --output results_final_fixlen.csv --vision-json vision_final_fixlen.json
python main.py images/ --multi
python metric.py
python test_pipeline.py --models     # 24 unit + 2 model-dependent tests; `pytest -q` runs the 24 unit tests (2 skipped)
```

`HF_HOME` is respected if set; otherwise the default Hugging Face cache is used. Models: BLIP base, Qwen2.5-0.5B-Instruct, Florence-2-base (florence-community),
CLIP ViT-B/32, NLI MiniLM2. Greedy decoding with seed 0.

**Which file is which**

| Result file | Mean grounding (baseline -> improved) | Grounding pass | Length valid |
|---|---|---|---|
| `results_final.csv` (FINAL, normal) | 0.783 -> 0.856 (+0.072) | 0/8 -> 1/8 | 0/8 -> 1/8 |
| `results_final_fixlen.csv` (FINAL, length-controlled) | 0.756 -> 0.827 (+0.070) | 5/8 -> 5/8 | 6/8 -> 6/8 |
| `results_new.csv` (historical: earlier structured version, normal) | 0.783 -> 0.866 (+0.083) | 0/8 -> 2/8 | 0/8 -> 2/8 |
| `results_fixlen_new.csv` (historical: earlier structured version, length-controlled) | 0.756 -> 0.789 (+0.032) | 5/8 -> 4/8 | 6/8 -> 6/8 |
| `results.csv` (historical: earlier simple pipeline, normal) | 0.783 -> 0.845 (+0.062) | 0/8 -> 3/8 | 0/8 -> 3/8 |
| `results_fixlen.csv` (historical: earlier simple pipeline, length-controlled) | 0.756 -> 0.837 (+0.080) | 5/8 -> 5/8 | 6/8 -> 5/8 |

The historical rows were produced by earlier code (an earlier simple context; a context builder with an object whitelist) and an earlier metric version. They are kept as
evidence and are not comparable with the final rows. Earlier documents quoted `results_new.csv` and `results_fixlen_new.csv` as the headline results; those quotes are superseded by this document.

## Multi-image story

`python main.py images/ --multi` (filename order, 8 frames, 333 words, cut-off final sentence removed: True).
File names are not included in the prompt (an earlier run leaked a character name from a file name into the story). Continuity proxy: recurring entities
(11): boy, chair, child, footwear, girl, human, human face, man, person, window, woman; entity_consistency 0.4545; transition_markers 0;
adjacent_similarity 0.2285. These measure lexical overlap only and do not show that the story is coherent. The story text is in `combined_story.json`;
it still invents a named character ("Leo") and relatives that no frame shows.

## Final finding

In this 8-image, single-run comparison, replacing BLIP's one-sentence understanding with Florence-2 structured extraction is associated with a higher mean grounding score
(+0.072 normal, +0.070 length-controlled), higher CLIP similarity (+0.024 and +0.014) and lower NLI contradiction (-0.023 and -0.145).
The composite pass rate does not improve once length is controlled (5/8 vs 5/8), 3 of 8 images regress in that mode, and the improved pipeline costs about
2.9x the generation time (normal) and 1.6x (length-controlled). The comparison used a stand-in baseline, the components of the new seeing stage were
not ablated, and NLI and attribute checks compare against generated context, so these results show the direction on this sample, not a proven improvement.
