import asyncio
from compare_pipeline import run

async def main():
    results = await asyncio.gather(
        asyncio.to_thread(run, "data/doc001.pdf"),
        asyncio.to_thread(run, "data/doc002.pdf"),
        asyncio.to_thread(run, "data/doc003.pdf"),
        asyncio.to_thread(run, "data/ai_queue_design.pdf")
    )

    # Aggregate final metrics across all documents and methods
    methods = ["pypdf", "sherpa", "prompt"]
    summary = {}
    for m in methods:
        docs_with = sum(1 for r in results if r.get(m) and len(r.get(m)) > 0)
        total_chunks = sum(len(r.get(m) or []) for r in results)
        total_chars = sum(len(c.get("text", "")) for r in results for c in (r.get(m) or []))
        avg_chunks_per_doc = total_chunks / len(results) if results else 0
        avg_chunk_length = (total_chars / total_chunks) if total_chunks else 0
        relevant_chunks = sum(1 for r in results for c in (r.get(m) or []) if len(c.get("text", "")) > 200)
        relevance_pct = (relevant_chunks / total_chunks * 100) if total_chunks else 0

        summary[m] = {
            "docs_with_chunks": docs_with,
            "total_chunks": total_chunks,
            "avg_chunks_per_doc": round(avg_chunks_per_doc, 2),
            "avg_chunk_length": round(avg_chunk_length, 1),
            "relevant_chunks": relevant_chunks,
            "relevance_pct": round(relevance_pct, 1)
        }

    print("\nFinal Summary Across Documents:")
    for m, stats in summary.items():
        print(f"\nMethod: {m}")
        print(f" - Docs w/ chunks: {stats['docs_with_chunks']}/{len(results)}")
        print(f" - Total chunks: {stats['total_chunks']}")
        print(f" - Avg chunks/doc: {stats['avg_chunks_per_doc']}")
        print(f" - Avg chunk length (chars): {stats['avg_chunk_length']}")
        print(f" - Relevant chunks (>200 chars): {stats['relevant_chunks']} ({stats['relevance_pct']}%)")

    # Best per metric
    def _best(metric):
        best_m = max(summary.keys(), key=lambda k: summary[k][metric])
        return best_m, summary[best_m][metric]

    best_docs_method, best_docs_val = _best("docs_with_chunks")
    best_chunks_method, best_chunks_val = _best("total_chunks")
    best_relevance_method, best_relevance_val = _best("relevance_pct")

    print("\nBest Methods by Metric:")
    print(f" - Most documents covered: {best_docs_method} ({best_docs_val}/{len(results)})")
    print(f" - Most chunks produced: {best_chunks_method} ({best_chunks_val})")
    print(f" - Highest relevance %: {best_relevance_method} ({best_relevance_val}%)")

asyncio.run(main())
