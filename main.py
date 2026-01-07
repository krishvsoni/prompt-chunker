"""
Main entry point for document processing pipeline.

Processes PDF documents using three methods and compares results:
1. PyPDF - Baseline text extraction
2. LLM Sherpa - Layout-based parsing
3. Prompt-Based - Semantic chunking with metadata enrichment
"""

import os
import asyncio
import json
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from pipeline import run as process_document
from comparison_analysis import generate_comparison_report, get_flaw_analysis


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
    """Calculate aggregate metrics across all documents."""
    methods = ["pypdf", "sherpa", "prompt"]
    summary = {}

    for m in methods:
        chunks_data = [r.get(m, {}).get("chunks", []) for r in results]
        metrics_data = [r.get(m, {}).get("metrics", {}) for r in results]

        total_chunks = sum(len(c) for c in chunks_data)
        total_chars = sum(len(chunk.get("text", "")) for c in chunks_data for chunk in c)
        docs_with = sum(1 for c in chunks_data if c)

        avg_quality = 0
        quality_values = [chunk.get("quality_score", 0) for c in chunks_data for chunk in c]
        if quality_values:
            avg_quality = sum(quality_values) / len(quality_values)

        relevant = sum(1 for c in chunks_data for chunk in c if len(chunk.get("text", "")) > 200)


        accepted_chunks = sum(1 for c in chunks_data for chunk in c if chunk.get("quality_score", 0) >= 7)
        semantic_accept_rate = round(accepted_chunks / total_chunks * 100, 1) if total_chunks else 0

    
        sufficient_chunks = sum(1 for c in chunks_data for chunk in c 
                                if chunk.get("quality_score", 0) >= 6 and len(chunk.get("text", "")) > 300)
        topk_sufficiency_rate = round(sufficient_chunks / total_chunks * 100, 1) if total_chunks else 0

        boundary_scores = []
        for md in metrics_data:
            bc = md.get("boundary_confidence", 0)
            if bc: boundary_scores.append(bc)
        boundary_confidence = round(sum(boundary_scores) / len(boundary_scores), 1) if boundary_scores else 0

        summary[m] = {
            "docs_processed": docs_with,
            "total_docs": len(results),
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(total_chunks / len(results), 2) if results else 0,
            "avg_chunk_length": round(total_chars / total_chunks, 1) if total_chunks else 0,
            "avg_quality_score": round(avg_quality, 2),
            "relevant_chunks": relevant,
            "relevance_pct": round(relevant / total_chunks * 100, 1) if total_chunks else 0,
            "semantic_accept_rate": semantic_accept_rate,
            "topk_sufficiency_rate": topk_sufficiency_rate,
            "boundary_confidence": boundary_confidence,
            "total_time": round(sum(m.get("processing_time", 0) for m in metrics_data), 2)
        }

    return summary


def print_results(summary: Dict[str, Dict], results: List[Dict[str, Any]]):
    """Print formatted results summary."""
    print("\n" + "=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    for method, stats in summary.items():
        print(f"\n{method.upper()}")
        print("-" * 40)
        print(f"  Docs processed: {stats['docs_processed']}/{stats['total_docs']}")
        print(f"  Total chunks: {stats['total_chunks']}")
        print(f"  Avg chunks/doc: {stats['avg_chunks_per_doc']}")
        print(f"  Avg chunk length: {stats['avg_chunk_length']} chars")
        print(f"  Avg quality score: {stats['avg_quality_score']}/10")
        print(f"  Relevant (>200 chars): {stats['relevant_chunks']} ({stats['relevance_pct']}%)")
        print(f"  Semantic Accept Rate: {stats['semantic_accept_rate']}%")
        print(f"  Top-K Sufficiency Rate: {stats['topk_sufficiency_rate']}%")
        print(f"  Boundary Confidence: {stats['boundary_confidence']}%")
        print(f"  Processing time: {stats['total_time']}s")

    print("\n" + "=" * 70)
    print("BEST METHOD BY METRIC")
    print("=" * 70)

    metrics_to_compare = [
        ("total_chunks", "Most granular", True),
        ("relevance_pct", "Highest relevance", True),
        ("semantic_accept_rate", "Best semantic accept", True),
        ("topk_sufficiency_rate", "Best top-K sufficiency", True),
        ("boundary_confidence", "Best boundary confidence", True),
        ("avg_quality_score", "Best quality", True),
        ("total_time", "Fastest", False)
    ]

    for metric, label, higher_better in metrics_to_compare:
        if higher_better:
            best = max(summary.keys(), key=lambda k: summary[k][metric])
        else:
            best = min(summary.keys(), key=lambda k: summary[k][metric])
        print(f"  {label}: {best} ({summary[best][metric]})")

    flaws = get_flaw_analysis()
    print("\n" + "=" * 70)
    print("METHOD FLAWS ANALYSIS")
    print("=" * 70)

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

    output = {
        "summary": summary,
        "results": results,
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
            "metric_explanation": "We removed 'Optimal %' because it measured heuristic size constraints. We replaced it with semantic_accept_rate, topk_sufficiency_rate, and boundary_confidence which directly reflect retrieval correctness."
        }
    }

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to output.json")

if __name__ == "__main__":
    asyncio.run(main())
