import json
import os

def parse_results(dir_path):
    results = []
    for f in os.listdir(dir_path):
        if f.endswith('_evaluation.json'):
            with open(os.path.join(dir_path, f)) as fp:
                data = json.load(fp)
                results.append(data)
    return results

baseline_results = parse_results('artifacts/benchmark/baseline')
v2_results = parse_results('artifacts/benchmark/v2_standard')
v2_1_results = parse_results('artifacts/benchmark/v2_1_standard')

print('=== BASELINE ===')
for r in baseline_results:
    print('  grounding={:.3f} clip={:.3f} nli={:.3f} claims_s={} claims_u={} claims_c={} narr={:.3f}'.format(
        r["grounding_score"], r["clip_image_story_mean"], r["nli_contra_mean"],
        r["supported_claims"], r["unsupported_claims"], r["contradicted_claims"],
        r["narrative_coherence"]))

print()
print('=== V2 STANDARD ===')
for r in v2_results:
    print('  grounding={:.3f} clip={:.3f} nli={:.3f} claims_s={} claims_u={} claims_c={} narr={:.3f}'.format(
        r["grounding_score"], r["clip_image_story_mean"], r["nli_contra_mean"],
        r["supported_claims"], r["unsupported_claims"], r["contradicted_claims"],
        r["narrative_coherence"]))

print()
print('=== V2.1 STANDARD ===')
for r in v2_1_results:
    print('  grounding={:.3f} clip={:.3f} nli={:.3f} claims_s={} claims_u={} claims_c={} narr={:.3f}'.format(
        r["grounding_score"], r["clip_image_story_mean"], r["nli_contra_mean"],
        r["supported_claims"], r["unsupported_claims"], r["contradicted_claims"],
        r["narrative_coherence"]))

def mean(data, key):
    return sum(d[key] for d in data) / len(data) if data else 0

print()
print('=== MEANS ===')
print('Baseline:        grounding={:.3f} clip={:.3f} nli={:.3f} claims_s={:.1f} claims_u={:.1f} narr={:.3f}'.format(
    mean(baseline_results, "grounding_score"), mean(baseline_results, "clip_image_story_mean"),
    mean(baseline_results, "nli_contra_mean"), mean(baseline_results, "supported_claims"),
    mean(baseline_results, "unsupported_claims"), mean(baseline_results, "narrative_coherence")))
print('V2 Standard:     grounding={:.3f} clip={:.3f} nli={:.3f} claims_s={:.1f} claims_u={:.1f} narr={:.3f}'.format(
    mean(v2_results, "grounding_score"), mean(v2_results, "clip_image_story_mean"),
    mean(v2_results, "nli_contra_mean"), mean(v2_results, "supported_claims"),
    mean(v2_results, "unsupported_claims"), mean(v2_results, "narrative_coherence")))
print('V2.1 Standard:   grounding={:.3f} clip={:.3f} nli={:.3f} claims_s={:.1f} claims_u={:.1f} narr={:.3f}'.format(
    mean(v2_1_results, "grounding_score"), mean(v2_1_results, "clip_image_story_mean"),
    mean(v2_1_results, "nli_contra_mean"), mean(v2_1_results, "supported_claims"),
    mean(v2_1_results, "unsupported_claims"), mean(v2_1_results, "narrative_coherence")))

# Runtime
def mean_runtime(data, key):
    return sum(d.get("runtime", {}).get(key, 0) for d in data) / len(data) if data else 0

print()
print('=== RUNTIME (seconds) ===')
print('Baseline total:   {:.1f}s'.format(mean_runtime(baseline_results, "generation_s") + mean_runtime(baseline_results, "evaluation_s")))
print('V2 Standard:     {:.1f}s'.format(mean_runtime(v2_results, "generation_s") + mean_runtime(v2_results, "evaluation_s")))
print('V2.1 Standard:   {:.1f}s'.format(mean_runtime(v2_1_results, "generation_s") + mean_runtime(v2_1_results, "evaluation_s")))