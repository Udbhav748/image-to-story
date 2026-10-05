#!/usr/bin/env python
"""Synthetic scale benchmarks for V2.3C hierarchical memory pipeline."""
import sys
import os
import time
import gc
import tracemalloc
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from image_story.domain.schemas import (
    PipelineConfig,
    VisualObservations,
    EvidenceRecord,
    EvidenceType,
    SourceModel,
    InformationClass,
    CreativePlan,
)
from image_story.memory.collection_builder import CollectionMemoryBuilder, create_collection_memory_builder
from image_story.memory.hierarchical_retriever import create_hierarchical_retriever
from image_story.memory.faiss_store import FAISSVectorStore
from image_story.memory.embeddings import create_embedding_model
from image_story.context.collection_builder import create_collection_context_builder
from image_story.narrative.planner import CreativePlanner
from image_story.memory.hierarchical import CollectionMemory, RetrievalResult
from image_story.context.world_state import WorldStateBuilder
from image_story.context.ranker import EvidenceRanker


# Global model instances to avoid reloading
_embedding_model = None
_vector_store = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = create_embedding_model(device="cpu")
        _embedding_model.load()
    return _embedding_model


def get_vector_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore(get_embedding_model(), device="cpu")
        _vector_store.initialize()
    return _vector_store


def create_synthetic_observations(num_frames: int, evidence_per_frame: int = 5) -> List[VisualObservations]:
    """Create synthetic visual observations for benchmarking."""
    observations = []
    entity_pool = [f"entity_{i}" for i in range(min(50, num_frames * 2))]
    scene_pool = ["indoor", "outdoor", "kitchen", "garden", "street", "room", "park", "office"]
    
    for frame_id in range(num_frames):
        scene = scene_pool[frame_id % len(scene_pool)]
        evidence_records = []
        
        for i in range(evidence_per_frame):
            entity = entity_pool[(frame_id * evidence_per_frame + i) % len(entity_pool)]
            ev = EvidenceRecord(
                entity=entity,
                type=EvidenceType.OBJECT if i % 2 == 0 else EvidenceType.PERSON,
                frame_id=frame_id,
                confidence=0.8 + (i % 3) * 0.05,
                source=SourceModel.FLORENCE2,
                evidence_text=f"Detected {entity} in frame {frame_id}",
                information_class=InformationClass.HARD_FACT,
            )
            evidence_records.append(ev)
        
        obs = VisualObservations(
            image_id=f"img_{frame_id}",
            frame_id=frame_id,
            scene=scene,
            detailed_caption=f"Frame {frame_id} showing {scene} with various objects",
            objects=[f"obj_{frame_id}_{j}" for j in range(3)],
            od_labels=[f"label_{frame_id}_{j}" for j in range(3)],
            characters=[f"char_{frame_id}"],
            actions=[f"action_{frame_id}"],
            spatial_relations=[f"spatial_{frame_id}"],
            region_descriptions=[f"region_{frame_id}"],
            style_or_mood="neutral",
            evidence_records=evidence_records,
        )
        observations.append(obs)
    
    return observations


def benchmark_collection_memory_build(observations: List[VisualObservations], config: PipelineConfig) -> Dict[str, Any]:
    """Benchmark CollectionMemoryBuilder.build_from_observations."""
    gc.collect()
    tracemalloc.start()
    
    builder = create_collection_memory_builder(
        device="cpu",
        similarity_threshold=config.scene_similarity_threshold,
        top_k_scenes=config.top_k_scenes,
        top_k_evidence_per_scene=config.top_k_evidence_per_scene,
    )
    
    start = time.perf_counter()
    collection = builder.build_from_observations(observations, config)
    build_time = time.perf_counter() - start
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "time_seconds": build_time,
        "peak_memory_mb": peak / (1024 * 1024),
        "num_scenes": len(collection.scene_summaries),
        "num_entities": len(collection.global_entities),
        "num_transitions": len(collection.state_transitions),
        "num_narrative": len(collection.narrative_elements),
    }


def benchmark_faiss_batch_insertion(observations: List[VisualObservations], config: PipelineConfig) -> Dict[str, Any]:
    """Benchmark FAISS vector store insertion."""
    gc.collect()
    tracemalloc.start()
    
    vector_store = get_vector_store()
    embedding_model = get_embedding_model()
    
    # Collect all evidence
    all_evidence = []
    for obs in observations:
        all_evidence.extend(obs.evidence_records)
    
    start = time.perf_counter()
    for evidence in all_evidence:
        text = f"Entity: {evidence.entity}. Frame: {evidence.frame_id}. Type: {evidence.type.value}."
        vector_store.add_evidence_with_provenance(
            evidence, text, f"img_{evidence.frame_id}", f"scene_{evidence.frame_id // 10}", evidence.frame_id, "bench_collection"
        )
    insert_time = time.perf_counter() - start
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "insert_time_seconds": insert_time,
        "peak_memory_mb": peak / (1024 * 1024),
        "num_vectors": len(all_evidence),
    }


def benchmark_hierarchical_retrieval(collection_memory: CollectionMemory, vector_store: FAISSVectorStore, 
                                      embedding_model, config: PipelineConfig) -> Dict[str, Any]:
    """Benchmark hierarchical retrieval latency."""
    gc.collect()
    
    retriever = create_hierarchical_retriever(
        embedding_model,
        vector_store,
        top_k_scenes=config.top_k_scenes,
        top_k_evidence_per_scene=config.top_k_evidence_per_scene,
        max_total_evidence=config.faiss_top_k * 5,
    )
    retriever.set_collection_memory(collection_memory)
    
    # Warmup
    _ = retriever.retrieve_full("test query")
    
    # Benchmark
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        result = retriever.retrieve_full("scene with objects and characters")
        latencies.append(time.perf_counter() - start)
    
    return {
        "mean_latency_ms": np.mean(latencies) * 1000,
        "p50_latency_ms": np.percentile(latencies, 50) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "p99_latency_ms": np.percentile(latencies, 99) * 1000,
        "num_scenes_retrieved": len(result.scenes),
        "num_evidence_retrieved": len(result.evidence),
    }


def benchmark_context_building(collection_memory: CollectionMemory, retrieval_result: RetrievalResult,
                                config: PipelineConfig) -> Dict[str, Any]:
    """Benchmark CollectionContextBuilder.build_collection_context."""
    gc.collect()
    tracemalloc.start()
    
    context_builder = create_collection_context_builder(
        max_words=config.context_max_words,
        max_scenes=config.max_scenes_in_context,
    )
    
    creative_plan = CreativePlan(
        genre=config.genre,
        tone=config.tone,
        locked_facts=["test fact"],
    )
    
    start = time.perf_counter()
    context = context_builder.build_collection_context(
        collection_memory,
        retrieval_result,
        creative_plan,
        current_scene_id=None,
    )
    build_time = time.perf_counter() - start
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "time_seconds": build_time,
        "peak_memory_mb": peak / (1024 * 1024),
        "context_length_chars": len(context),
        "context_words": len(context.split()),
    }


def benchmark_creative_planner(observations: List[VisualObservations], config: PipelineConfig) -> Dict[str, Any]:
    """Benchmark CreativePlanner.create_creative_plan."""
    gc.collect()
    
    world_builder = WorldStateBuilder()
    world_state = world_builder.add_observations_batch(observations)
    
    ranker = EvidenceRanker()
    
    # Mock ranked evidence
    all_evidence = []
    for obs in observations:
        for ev in obs.evidence_records:
            all_evidence.append(type('obj', (object,), {'record': ev, 'semantic_similarity': 0.9})())
    
    planner = CreativePlanner.from_config(config)
    
    start = time.perf_counter()
    creative_plan = planner.create_creative_plan(world_state, observations, all_evidence)
    plan_time = time.perf_counter() - start
    
    # CreativePlan has characters, open_loops, foreshadowing_elements, etc.
    return {
        "time_seconds": plan_time,
        "num_characters": len(creative_plan.characters),
        "num_open_loops": len(creative_plan.open_loops),
        "has_central_conflict": creative_plan.central_conflict is not None,
        "num_foreshadowing": len(creative_plan.foreshadowing_elements),
    }


def run_scale_benchmark(scale: int, evidence_per_frame: int = 5) -> Dict[str, Any]:
    """Run full benchmark at a given scale."""
    print(f"\n{'='*60}")
    print(f"SCALE BENCHMARK: {scale} frames ({scale * evidence_per_frame} evidence records)")
    print(f"{'='*60}")
    
    config = PipelineConfig.from_mode("standard")
    
    # Create synthetic data
    print("Creating synthetic observations...")
    observations = create_synthetic_observations(scale, evidence_per_frame)
    total_evidence = sum(len(o.evidence_records) for o in observations)
    print(f"Created {len(observations)} observations with {total_evidence} evidence records")
    
    results = {"scale": scale, "evidence_per_frame": evidence_per_frame, "total_evidence": total_evidence}
    
    # 1. Collection Memory Build
    print("Benchmarking CollectionMemoryBuilder...")
    mem_results = benchmark_collection_memory_build(observations, config)
    results["collection_memory_build"] = mem_results
    print(f"  Time: {mem_results['time_seconds']:.3f}s, Peak RAM: {mem_results['peak_memory_mb']:.1f}MB")
    print(f"  Scenes: {mem_results['num_scenes']}, Entities: {mem_results['num_entities']}")
    
    # 2. FAISS Batch Insertion
    print("Benchmarking FAISS batch insertion...")
    faiss_results = benchmark_faiss_batch_insertion(observations, config)
    results["faiss_batch_insertion"] = faiss_results
    print(f"  Insert: {faiss_results['insert_time_seconds']:.3f}s, Peak RAM: {faiss_results['peak_memory_mb']:.1f}MB")
    
    # 3. Hierarchical Retrieval
    print("Benchmarking hierarchical retrieval...")
    embedding_model = get_embedding_model()
    vector_store = get_vector_store()
    
    # Build collection memory first
    builder = create_collection_memory_builder(device="cpu")
    collection_memory = builder.build_from_observations(observations, config)
    
    # Populate vector store (reuse from FAISS benchmark)
    for obs in observations:
        for evidence in obs.evidence_records:
            text = f"Entity: {evidence.entity}. Frame: {evidence.frame_id}. Type: {evidence.type.value}."
            vector_store.add_evidence_with_provenance(
                evidence, text, f"img_{evidence.frame_id}", f"scene_{evidence.frame_id // 10}", evidence.frame_id, "bench"
            )
    
    retrieval_results = benchmark_hierarchical_retrieval(collection_memory, vector_store, embedding_model, config)
    results["hierarchical_retrieval"] = retrieval_results
    print(f"  Mean: {retrieval_results['mean_latency_ms']:.1f}ms, P50: {retrieval_results['p50_latency_ms']:.1f}ms, P95: {retrieval_results['p95_latency_ms']:.1f}ms")
    
    # 4. Context Building
    print("Benchmarking context building...")
    # Create a retriever for context building
    retriever = create_hierarchical_retriever(
        embedding_model,
        vector_store,
        top_k_scenes=config.top_k_scenes,
        top_k_evidence_per_scene=config.top_k_evidence_per_scene,
        max_total_evidence=config.faiss_top_k * 5,
    )
    retriever.set_collection_memory(collection_memory)
    retrieval_result = retriever.retrieve_full("benchmark query for context")
    context_results = benchmark_context_building(collection_memory, retrieval_result, config)
    results["context_building"] = context_results
    print(f"  Time: {context_results['time_seconds']:.3f}s, Peak RAM: {context_results['peak_memory_mb']:.1f}MB")
    print(f"  Context: {context_results['context_words']} words ({context_results['context_length_chars']} chars)")
    
    # 5. Creative Planner
    print("Benchmarking creative planner...")
    planner_results = benchmark_creative_planner(observations, config)
    results["creative_planner"] = planner_results
    print(f"  Time: {planner_results['time_seconds']:.3f}s")
    
    return results


def main():
    """Run benchmarks at multiple scales."""
    scales = [100, 1000]
    evidence_per_frame = 5
    
    all_results = []
    
    for scale in scales:
        try:
            results = run_scale_benchmark(scale, evidence_per_frame)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR at scale {scale}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"scale": scale, "error": str(e)})
    
    # Print summary table
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY TABLE")
    print("="*100)
    print(f"{'Records':>10} | {'Mem Build(s)':>12} | {'FAISS Insert(s)':>15} | {'Retrieval(ms)':>14} | {'Context(s)':>11} | {'Planner(s)':>11} | {'Context Words':>13}")
    print("-"*100)
    
    for r in all_results:
        if "error" in r:
            print(f"{r['scale']:>10} | ERROR: {r['error']}")
            continue
        mem = r["collection_memory_build"]["time_seconds"]
        faiss = r["faiss_batch_insertion"]["insert_time_seconds"]
        retr = r["hierarchical_retrieval"]["mean_latency_ms"]
        ctx = r["context_building"]["time_seconds"]
        plan = r["creative_planner"]["time_seconds"]
        ctx_words = r["context_building"]["context_words"]
        print(f"{r['total_evidence']:>10} | {mem:>12.3f} | {faiss:>15.3f} | {retr:>14.1f} | {ctx:>11.3f} | {plan:>11.3f} | {ctx_words:>13}")
    
    return all_results


if __name__ == "__main__":
    main()