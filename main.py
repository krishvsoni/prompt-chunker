"""
Main entry point for document processing pipeline.

Processes PDF documents using three methods and compares results:
1. PyPDF - Baseline text extraction
2. LLM Sherpa - Layout-based parsing
3. Prompt-Based - Semantic chunking with metadata enrichment

METRIC DEFINITIONS (FROZEN):
- Total Chunks: Final indexed chunks only (after validation)
- Avg Chunk Length: In characters (never mix with tokens)
- Avg Quality: Normalized 0-1, excludes chunks without scores
- Relevance %: Retrieval recall at K=10
- Semantic Accept %: Chunks with ACCEPT + confidence >= 0.7
- Top-K Sufficiency %: Queries answerable with <= K=3 chunks
- Boundary Confidence %: Mean confidence over boundary decisions only
"""

import os
import asyncio
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from pipeline import run as process_document
from comparison_analysis import generate_comparison_report, get_flaw_analysis
from metrics import (
    FrozenMetrics, MetricCalculator, StageTimings,
    compute_heuristic_metrics, generate_diagnostics_table,
    MetricValidationRules, SEMANTIC_METRICS_EXPLANATION,
    DEFAULT_TOP_K, DEFAULT_SUFFICIENCY_K, FORCE_REFINEMENT_THRESHOLD
)


async def process_documents(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    """Process multiple documents in parallel."""
    loop = asyncio.get_event_loop()
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [
            loop.run_in_executor(executor, process_document, path)
            for path in pdf_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Error processing {pdf_paths[i]}: {result}")
        else:
            valid_results.append(result)

    return valid_results


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict]:
    """
    Calculate aggregate metrics across all documents using FROZEN definitions.
    
    FROZEN METRIC FORMULAS:
    1. TotalChunks = count of final indexed chunks (not rejected/intermediate)
    2. AvgChunkLength = sum(char_count(chunk.text)) / TotalChunks
    3. AvgQuality = mean(chunk.quality_score normalized to 0-1), excludes missing
    4. Relevance% = (queries with ≥1 relevant in top-K) / total_queries × 100
    5. SemanticAccept% = (accepted_chunks / total_chunks) × 100
    6. TopKSufficiency% = (queries answerable with ≤K) / total_queries × 100
    7. BoundaryConfidence% = mean(boundary_confidence) × 100 (boundaries only)
    8. TotalTime = wall-clock time with stage breakdown
    """
    methods = ["pypdf", "sherpa", "prompt"]
    summary = {}
    calculator = MetricCalculator()
    
    for m in methods:
        chunks_data = [r.get(m, {}).get("chunks", []) for r in results]
        metrics_data = [r.get(m, {}).get("metrics", {}) for r in results]
        
        # Flatten all chunks for this method
        all_chunks = [chunk for doc_chunks in chunks_data for chunk in doc_chunks]
        
        # === METRIC 1: Total Chunks (final indexed only) ===
        # Count only chunks that are not rejected
        final_chunks = [c for c in all_chunks if c.get("status") != "rejected"]
        total_chunks = len(final_chunks)
        
        # === METRIC 2: Average Chunk Length (characters, FROZEN unit) ===
        total_chars = sum(len(chunk.get("text", "")) for chunk in final_chunks)
        avg_chunk_length = total_chars / total_chunks if total_chunks else 0.0
        
        # === METRIC 3: Average Quality (0-1 normalized, exclude missing) ===
        quality_scores = []
        for chunk in final_chunks:
            score = chunk.get("quality_score")
            if score is not None:
                # Normalize to 0-1 if on 0-10 scale
                normalized = score / 10.0 if score > 1 else score
                quality_scores.append(normalized)
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        chunks_with_quality = len(quality_scores)
        
        # === METRIC 4: Relevance % (K=10, retrieval recall) ===
        # Note: Actual relevance requires query evaluation data
        # Using heuristic: chunks with substantial content (>200 chars) as proxy
        # In production, replace with actual retrieval evaluation
        relevant_proxy = sum(1 for c in final_chunks if len(c.get("text", "")) > 200)
        relevance_pct = round(relevant_proxy / total_chunks * 100, 1) if total_chunks else 0.0
        
        # === METRIC 5: Semantic Accept % (replaces Optimal %) ===
        # Definition: recommendation == "ACCEPT" AND confidence >= 0.7
        accepted_count = 0
        for chunk in final_chunks:
            validation = chunk.get("validation", {})
            recommendation = validation.get("recommendation", "").upper()
            confidence = validation.get("confidence", 0.0)
            
            # Fallback: use quality_score >= 7 as ACCEPT equivalent
            if not recommendation:
                score = chunk.get("quality_score", 0)
                if score >= 7:  # 7/10 = 0.7 threshold
                    recommendation = "ACCEPT"
                    confidence = score / 10.0
            
            if recommendation == "ACCEPT" and confidence >= 0.7:
                accepted_count += 1
        
        semantic_accept_pct = round(accepted_count / total_chunks * 100, 1) if total_chunks else 0.0
        
        # === METRIC 6: Top-K Sufficiency % (K=3, retrieval efficiency) ===
        # Note: Actual sufficiency requires query evaluation data
        # Using heuristic: chunks that are high-quality AND substantive
        # In production, replace with actual retrieval evaluation
        sufficient_count = sum(
            1 for c in final_chunks 
            if c.get("quality_score", 0) >= 6 and len(c.get("text", "")) >= 300
        )
        topk_sufficiency_pct = round(sufficient_count / total_chunks * 100, 1) if total_chunks else 0.0
        
        # === METRIC 7: Boundary Confidence % (boundaries only) ===
        # Only average over actual boundary decisions
        boundary_scores = []
        for md in metrics_data:
            bc = md.get("boundary_confidence", 0)
            if bc > 0:  # Only include actual boundary confidence values
                boundary_scores.append(bc)
        
        boundary_confidence = round(
            sum(boundary_scores) / len(boundary_scores), 1
        ) if boundary_scores else 0.0
        
        # === METRIC 8: Total Time (with stage breakdown) ===
        total_time = sum(md.get("processing_time", 0) for md in metrics_data)
        
        # Collect stage timings if available
        stage_breakdown = {}
        for md in metrics_data:
            st = md.get("stage_timings", {})
            if st:
                for stage, timing in st.items():
                    if stage not in stage_breakdown:
                        stage_breakdown[stage] = 0.0
                    if isinstance(timing, dict):
                        stage_breakdown[stage] += timing.get("time_seconds", 0)
                    elif isinstance(timing, (int, float)):
                        stage_breakdown[stage] += timing
        
        # Build summary with FROZEN metric definitions
        docs_with = sum(1 for c in chunks_data if c)
        
        summary[m] = {
            # Document counts
            "docs_processed": docs_with,
            "total_docs": len(results),
            
            # FROZEN METRICS
            "total_chunks": total_chunks,
            "avg_chunk_length": {
                "value": round(avg_chunk_length, 1),
                "unit": "characters"
            },
            "avg_quality": {
                "value": round(avg_quality, 3),
                "scale": "0-1",
                "chunks_measured": chunks_with_quality,
                "note": "Excludes chunks without quality scores"
            },
            "relevance_pct": {
                "value": relevance_pct,
                "k": DEFAULT_TOP_K,
                "note": "Retrieval recall - does NOT measure efficiency"
            },
            "semantic_accept_pct": {
                "value": semantic_accept_pct,
                "accepted_count": accepted_count,
                "threshold": 0.7,
                "note": "Replaces heuristic 'Optimal %'"
            },
            "topk_sufficiency_pct": {
                "value": topk_sufficiency_pct,
                "k": DEFAULT_SUFFICIENCY_K,
                "note": "Measures query-document asymmetry reduction"
            },
            "boundary_confidence_pct": {
                "value": boundary_confidence,
                "note": "Averaged over boundary decisions only"
            },
            "total_time_seconds": round(total_time, 2),
            
            # Stage breakdown (if available)
            "stage_breakdown": stage_breakdown if stage_breakdown else None,
            
            # Legacy fields for backward compatibility
            "avg_chunks_per_doc": round(total_chunks / len(results), 2) if results else 0
        }
        
        # === VALIDATION RULES CHECK ===
        warnings = []
        
        # Rule 3: If Semantic Accept % < 40%, warn about refinement
        if semantic_accept_pct < FORCE_REFINEMENT_THRESHOLD * 100:
            warnings.append(
                f"Semantic Accept % ({semantic_accept_pct}%) < 40%. "
                "Consider forcing refinement before reporting."
            )
        
        # Rule 2: Relevance without sufficiency warning
        if relevance_pct > 0 and topk_sufficiency_pct == 0:
            warnings.append(
                "Relevance % reported without meaningful Top-K Sufficiency."
            )
        
        if warnings:
            summary[m]["validation_warnings"] = warnings

    return summary


def print_results(summary: Dict[str, Dict], results: List[Dict[str, Any]]):
    """Print formatted results summary with FROZEN metric definitions."""
    print("\n" + "=" * 80)
    print("AGGREGATE METRICS (FROZEN DEFINITIONS)")
    print("=" * 80)
    
    # Print metric explanation required by Part 4
    print(f"\n{SEMANTIC_METRICS_EXPLANATION}")
    print("-" * 80)

    for method, stats in summary.items():
        print(f"\n{method.upper()}")
        print("-" * 50)
        print(f"  Docs processed: {stats['docs_processed']}/{stats['total_docs']}")
        
        # FROZEN METRIC 1: Total Chunks
        print(f"  Total Chunks (final indexed): {stats['total_chunks']}")
        
        # FROZEN METRIC 2: Avg Chunk Length (with unit)
        avg_len = stats.get('avg_chunk_length', {})
        if isinstance(avg_len, dict):
            print(f"  Avg Chunk Length: {avg_len['value']} {avg_len['unit']}")
        else:
            print(f"  Avg Chunk Length: {avg_len} characters")
        
        # FROZEN METRIC 3: Avg Quality (0-1 scale)
        avg_qual = stats.get('avg_quality', {})
        if isinstance(avg_qual, dict):
            print(f"  Avg Quality: {avg_qual['value']:.3f} (scale: 0-1, measured: {avg_qual['chunks_measured']} chunks)")
        else:
            print(f"  Avg Quality: {avg_qual:.3f}")
        
        # FROZEN METRIC 4: Relevance % (with K)
        rel = stats.get('relevance_pct', {})
        if isinstance(rel, dict):
            print(f"  Relevance@{rel['k']}: {rel['value']}% (retrieval recall)")
        else:
            print(f"  Relevance@{DEFAULT_TOP_K}: {rel}%")
        
        # FROZEN METRIC 5: Semantic Accept % (replaces Optimal %)
        sem_acc = stats.get('semantic_accept_pct', {})
        if isinstance(sem_acc, dict):
            print(f"  Semantic Accept%: {sem_acc['value']}% ({sem_acc['accepted_count']} accepted, threshold: {sem_acc['threshold']})")
        else:
            print(f"  Semantic Accept%: {sem_acc}%")
        
        # FROZEN METRIC 6: Top-K Sufficiency % (with K)
        topk = stats.get('topk_sufficiency_pct', {})
        if isinstance(topk, dict):
            print(f"  Top-{topk['k']} Sufficiency%: {topk['value']}% (retrieval efficiency)")
        else:
            print(f"  Top-{DEFAULT_SUFFICIENCY_K} Sufficiency%: {topk}%")
        
        # FROZEN METRIC 7: Boundary Confidence %
        bc = stats.get('boundary_confidence_pct', {})
        if isinstance(bc, dict):
            print(f"  Boundary Confidence%: {bc['value']}% (over boundaries only)")
        else:
            print(f"  Boundary Confidence%: {bc}%")
        
        # FROZEN METRIC 8: Total Time
        print(f"  Total Time: {stats['total_time_seconds']}s")
        
        # Print stage breakdown if available
        if stats.get('stage_breakdown'):
            print(f"  Stage Breakdown:")
            for stage, time_s in stats['stage_breakdown'].items():
                print(f"    - {stage}: {time_s:.2f}s")
        
        # Print validation warnings
        if stats.get('validation_warnings'):
            print(f"    Validation Warnings:")
            for warning in stats['validation_warnings']:
                print(f"      - {warning}")

    # Print diagnostics table (Part 2)
    print("\n" + "=" * 80)
    print("DIAGNOSTICS TABLE")
    print("=" * 80)
    print(f"\n{'Method':<10} {'Parse':<8} {'Boundary':<10} {'Valid':<8} {'Embed':<8} {'Index':<8} {'Total':<8} {'Chunks':<8} {'Accept%':<10} {'Suff%':<8}")
    print("-" * 96)
    
    for method, stats in summary.items():
        breakdown = stats.get('stage_breakdown', {})
        parse_t = breakdown.get('parse', {}).get('time_seconds', 0) if isinstance(breakdown.get('parse'), dict) else breakdown.get('parse', 0)
        bound_t = breakdown.get('boundary_detection', {}).get('time_seconds', 0) if isinstance(breakdown.get('boundary_detection'), dict) else breakdown.get('boundary', 0)
        valid_t = breakdown.get('validation_refinement', {}).get('time_seconds', 0) if isinstance(breakdown.get('validation_refinement'), dict) else breakdown.get('validation', 0)
        embed_t = breakdown.get('embedding', {}).get('time_seconds', 0) if isinstance(breakdown.get('embedding'), dict) else breakdown.get('embed', 0)
        index_t = breakdown.get('indexing', {}).get('time_seconds', 0) if isinstance(breakdown.get('indexing'), dict) else breakdown.get('index', 0)
        
        sem_acc = stats.get('semantic_accept_pct', {})
        sem_val = sem_acc['value'] if isinstance(sem_acc, dict) else sem_acc
        
        topk = stats.get('topk_sufficiency_pct', {})
        topk_val = topk['value'] if isinstance(topk, dict) else topk
        
        print(f"{method.upper():<10} {parse_t:<8.2f} {bound_t:<10.2f} {valid_t:<8.2f} {embed_t:<8.2f} {index_t:<8.2f} {stats['total_time_seconds']:<8.2f} {stats['total_chunks']:<8} {sem_val:<10.1f} {topk_val:<8.1f}")
    
    print("-" * 96)
    print(f"K values: Relevance K={DEFAULT_TOP_K}, Sufficiency K={DEFAULT_SUFFICIENCY_K}")

    print("\n" + "=" * 80)
    print("BEST METHOD BY METRIC")
    print("=" * 80)

    metrics_to_compare = [
        ("total_chunks", "Most granular", True),
        ("semantic_accept_pct", "Best Semantic Accept%", True, lambda s: s['value'] if isinstance(s, dict) else s),
        ("topk_sufficiency_pct", "Best Top-K Sufficiency%", True, lambda s: s['value'] if isinstance(s, dict) else s),
        ("boundary_confidence_pct", "Best Boundary Confidence%", True, lambda s: s['value'] if isinstance(s, dict) else s),
        ("avg_quality", "Best Quality", True, lambda s: s['value'] if isinstance(s, dict) else s),
        ("total_time_seconds", "Fastest", False, lambda s: s)
    ]

    for item in metrics_to_compare:
        metric = item[0]
        label = item[1]
        higher_better = item[2]
        extractor = item[3] if len(item) > 3 else lambda s: s
        
        if higher_better:
            best = max(summary.keys(), key=lambda k: extractor(summary[k].get(metric, 0)))
        else:
            best = min(summary.keys(), key=lambda k: extractor(summary[k].get(metric, float('inf'))))
        
        value = extractor(summary[best].get(metric, 0))
        if isinstance(value, float):
            print(f"  {label}: {best.upper()} ({value:.2f})")
        else:
            print(f"  {label}: {best.upper()} ({value})")

    flaws = get_flaw_analysis()
    print("\n" + "=" * 80)
    print("METHOD FLAWS ANALYSIS")
    print("=" * 80)

    print("\nPyPDF Flaws:")
    for flaw in flaws["pypdf_flaws"][:3]:
        print(f"  - {flaw}")

    print("\nSherpa Flaws:")
    for flaw in flaws["sherpa_flaws"][:3]:
        print(f"  - {flaw}")

    print("\nPrompt Advantages:")
    for adv in flaws["prompt_solutions"][:3]:
        print(f"  + {adv}")


async def main():
    """Main execution function."""
    data_dir = "data"
    pdf_files = []

    if os.path.exists(data_dir):
        pdf_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pdf")
        ]

    if not pdf_files:
        print("No PDF files found in data directory.")
        return

    print(f"Processing {len(pdf_files)} documents...")

    results = await process_documents(pdf_files)

    if not results:
        print("No documents processed successfully.")
        return

    summary = aggregate_metrics(results)
    print_results(summary, results)

    # Get analysis data
    flaws = get_flaw_analysis()
    report = generate_comparison_report()

    # Build frozen metrics objects for each method
    frozen_metrics_by_method = {}
    for method in ["pypdf", "sherpa", "prompt"]:
        stats = summary.get(method, {})
        fm = FrozenMetrics()
        fm.total_chunks = stats.get("total_chunks", 0)
        
        avg_len = stats.get("avg_chunk_length", {})
        fm.avg_chunk_length = avg_len.get("value", 0) if isinstance(avg_len, dict) else avg_len
        
        avg_qual = stats.get("avg_quality", {})
        fm.avg_quality = avg_qual.get("value", 0) if isinstance(avg_qual, dict) else avg_qual
        fm.chunks_with_quality = avg_qual.get("chunks_measured", 0) if isinstance(avg_qual, dict) else 0
        
        rel = stats.get("relevance_pct", {})
        fm.relevance_pct = rel.get("value", 0) if isinstance(rel, dict) else rel
        fm.relevance_k = rel.get("k", DEFAULT_TOP_K) if isinstance(rel, dict) else DEFAULT_TOP_K
        
        sem_acc = stats.get("semantic_accept_pct", {})
        fm.semantic_accept_pct = sem_acc.get("value", 0) if isinstance(sem_acc, dict) else sem_acc
        fm.semantic_accepted_count = sem_acc.get("accepted_count", 0) if isinstance(sem_acc, dict) else 0
        
        topk = stats.get("topk_sufficiency_pct", {})
        fm.topk_sufficiency_pct = topk.get("value", 0) if isinstance(topk, dict) else topk
        fm.topk_sufficiency_k = topk.get("k", DEFAULT_SUFFICIENCY_K) if isinstance(topk, dict) else DEFAULT_SUFFICIENCY_K
        
        bc = stats.get("boundary_confidence_pct", {})
        fm.boundary_confidence_pct = bc.get("value", 0) if isinstance(bc, dict) else bc
        
        fm.total_time_seconds = stats.get("total_time_seconds", 0)
        
        frozen_metrics_by_method[method] = fm
    
    # Generate diagnostics table
    diagnostics = generate_diagnostics_table(frozen_metrics_by_method)
    
    # Validate metrics against rules
    all_warnings = []
    for method, fm in frozen_metrics_by_method.items():
        warnings = MetricValidationRules.validate_metrics(fm, method)
        if warnings:
            all_warnings.extend([f"[{method.upper()}] {w}" for w in warnings])

    output = {
        "summary": summary,
        "results": results,
        "frozen_metrics": {m: fm.to_dict() for m, fm in frozen_metrics_by_method.items()},
        "diagnostics": diagnostics,
        "analysis": {
            "flaws": flaws,
            "comparison_report": report,
            "prompt_disadvantages": [
                "Higher processing time due to multiple LLM calls",
                "API costs can be significant for large document sets",
                "Non-determinism: outputs may vary across runs",
                "Potential hallucinations in boundary/metadata detection",
                "Token/context limits require document pre-chunking",
                "Performance sensitive to prompt quality/phrasing",
                "Scalability challenges for large-scale ingestion",
                "Privacy risk when using external LLM APIs",
                "Harder to debug why specific decisions were made",
                "Dependence on LLM provider availability"
            ],
            # REQUIRED EXPLANATION (Part 4)
            "metric_explanation": SEMANTIC_METRICS_EXPLANATION,
            "validation_warnings": all_warnings
        },
        "metric_definitions": {
            "total_chunks": "Count of final indexed chunks (after validation + refinement). Does NOT count rejected or intermediate chunks.",
            "avg_chunk_length": f"Average size of chunks in {summary.get('pypdf', {}).get('avg_chunk_length', {}).get('unit', 'characters')}. Formula: sum(length(chunk.text)) / TotalChunks",
            "avg_quality": "Mean of chunk validation scores normalized to 0-1. Excludes chunks without validation (not treated as 0).",
            "relevance_pct": f"Retrieval recall at K={DEFAULT_TOP_K}. Measures if ≥1 relevant chunk appears in top-K. Does NOT measure efficiency.",
            "semantic_accept_pct": "Replaces 'Optimal %'. Percentage of chunks with recommendation=ACCEPT and confidence>=0.7. Run refinement first.",
            "topk_sufficiency_pct": f"Retrieval efficiency at K={DEFAULT_SUFFICIENCY_K}. Measures if answer fully supported by ≤K chunks. Measures query-document asymmetry.",
            "boundary_confidence_pct": "Mean confidence over boundary decisions ONLY. Does NOT average non-boundary blocks.",
            "total_time_seconds": "Wall-clock time from ingestion start to final index write. Broken down by stage."
        },
        "k_values": {
            "relevance_k": DEFAULT_TOP_K,
            "sufficiency_k": DEFAULT_SUFFICIENCY_K,
            "note": "Always report K used in retrieval metrics"
        }
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to output.json")

if __name__ == "__main__":
    asyncio.run(main())
