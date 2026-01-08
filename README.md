
# Semantic Chunking & RAG Ingestion

Overview
--------

We replaced chunking authority (not parsing) with a reasoning-first, prompt-driven pipeline that produces semantically complete, retrieval-ready chunks. Parsers (PyPDF / Sherpa / LlamaParser) remain upstream extractors — necessary but not sufficient. The goal: improve RAG precision, reduce hallucinations, and align document chunks with user queries.

What we implemented (summary)
--------------------------------

- Minimal parsing: PyPDF (and optionally LlamaParser/Sherpa) for text and layout extraction.
- Prompt-driven semantic chunker that:
	- Produces paragraph-level micro-blocks with provenance.
	- Runs batched boundary-detection prompts to propose chunk boundaries.
	- Runs structure-analysis prompts to infer hierarchy (section/subsection).
	- Assembles chunks with `hierarchy_path` and evidence spans.
	- Runs metadata-enrichment prompts per chunk (summary, tags, entities).
	- Validates chunks with a quality/pass-coherence prompt that returns `ACCEPT|REFINE|REJECT`, scores, and refinement actions.
	- Automatic refinement loop: apply suggested split/merge once, then revalidate.
	- Routing logic: use Sherpa when layout is strong; otherwise route to the semantic chunker.
- Metrics & instrumentation: stage timing (parse, boundary, structure, validate, embed, index) and semantic metrics (see below).

Why this was needed (brief)
---------------------------

Parsers often produce either giant blobs (low precision) or brittle fragments (over-fragmentation) depending on layout quality. Retrieval quality depends on semantic completeness of chunks, not page boundaries or fixed token windows. Prompt-driven chunking reasons about meaning and therefore:

- Keeps definitions with explanations.
- Binds tables to their narrative.
- Groups continuations (appendix, quoted text).
- Produces smaller, answer-sized pieces for precise retrieval.

Prompt patterns we designed (and why)
-----------------------------------

We implemented a small set of specialized, schema-first prompts. Run each with `temperature=0` (classification/structure) and strict JSON-schema output. Batch inputs (20–30 blocks) to avoid truncation.

### `boundary-detection`

- Task: For each micro-block, output `{block_id, new_chunk: bool, role, confidence, reason, evidence_span}`.
- Purpose: Propose semantic chunk boundaries; drives chunk assembly.

### `structure-analysis`

- Task: Build a hierarchical TOC-like tree for windows of blocks, provide `title`, `level`, `ambiguity_flag`, `line_offsets`.
- Purpose: Preserve hierarchy in chunk metadata (`hierarchy_path`, `full_path`).

### `metadata-enrichment`

- Task: For each chunk, extract `summary`, `semantic_tags`, `entities` with `source_spans` and `confidence`.
- Purpose: Provide retrieval filters and reranking signals.

### `quality-validation` / `pass-coherence`

- Task: For each chunk return `{chunk_id, recommendation: ACCEPT|REFINE|REJECT, scores: {self_containment, coherence, boundary_quality}, confidence, refinements[]}`.
- Purpose: Gate chunks; trigger auto-refinement and human review for Tier-1 docs.

### `refinement` (auto-split/merge)

- Task: Given a chunk and refinement hint, return split/merge points (block ids) and new chunk proposals.
- Purpose: Automatic repair pass before indexing.

Rules enforced across prompts
----------------------------

- JSON-only output; reject non-JSON and retry with smaller batch.
- Include `evidence_span` for every extraction (format: `chunk_id:char_start-char_end`).
- Return per-decision confidence (0.0–1.0).
- Short deterministic `reason` fields (≤30 chars); no chain-of-thought.
- Batch, retry on invalid output, cap tokens, log raw responses.

Query–document asymmetry (short)
--------------------------------

Users ask short questions; documents are long and multi-topic. If chunks are too big, answers are noisy; if chunks are too small or layout-driven, answers require stitching multiple fragments. Semantic chunking aligns chunk size/meaning to query intent: one query → one chunk in most cases. This reduces the top-K needed, lowers hallucination risk, and improves answer precision.

Metrics (what to measure and why)
--------------------------------

Replace the old parser-centric “Optimal %” with semantics-first metrics:

- **Total Chunks** — final indexed chunks (after validation/refinement).
- **Avg Chunk Length** — tokens per chunk (use tokens consistently).
- **Avg Quality** — mean validation score (normalized 0–1).
- **Relevance %** — percent of queries with ≥1 relevant chunk in top-K (specify K).
- **Semantic Accept %** — percent of chunks with `recommendation == ACCEPT` and `confidence >= threshold` (run one auto-refinement pass first).
- **Top-K Sufficiency %** — percent of queries answerable with ≤K chunks (K typically 3).
- **Boundary Confidence %** — mean boundary decision confidence across chunks (0–100).
- **Total Time (s)** — wall-clock ingestion time, broken down by stage.

Instrumentation: log per-stage timing for Parse/Extract, Boundary Detection, Structure Analysis, Chunk Assembly, Validation/Refinement, Embedding, Indexing.

Diagnostics rules / reporting notes
---------------------------------

- Do not report Relevance% without Top-K Sufficiency. Relevance alone is misleading.
- If Semantic Accept % < 40%, force auto-refinement before reporting.
- Use `K` in all reports and display it prominently.
- Track sample failed chunks and their refinements for quick fixes.

Targets (Tier-1)
----------------

- Semantic Accept ≥ 60% (after auto-refinement).
- Top-K Sufficiency (K=3) ≥ 80%.

Practical run & operational suggestions
--------------------------------------

1. Always run micro-block creation first (paragraphs with provenance).
2. Use parsers for extraction and layout hints. If Sherpa returns ≤1 chunk, route to the prompt chunker.
3. Batch prompts (20–30 blocks) and validate JSON strictly.
4. Apply one automated refinement cycle before acceptance: validate → refine → revalidate.
5. Index only `ACCEPT`ed chunks; queue others for refinement/human review depending on tier.

What we did not try to solve
---------------------------

- Low-level binary extraction, OCR, table detection from pixels — these remain parser/OCR responsibilities.
- Replacing parsers with prompts. Prompts require text input; extraction is out-of-band.

Justification (short)
---------------------

Parsers extract text and visuals; they cannot reason about meaning. Retrieval quality depends on semantic completeness, not layout or token windows. Our prompt-driven chunker reasons about discourse, groups idea-sized units, and validates them. That directly reduces query–document asymmetry and improves RAG precision at the cost of compute (reasoning time). Tiered routing keeps cost under control.

TODO (prioritized)
-----------------------

1. Add strict JSON schemas and few-shot examples to each prompt file.
2. Implement automatic refinement pass and human-in-the-loop for Tier-1 failures.
3. Instrument embedding + indexing times; compute and publish full stage breakdowns.
4. Build small labeled test set for metric calibration (semantic accept thresholds, Top-K targets).
5. Optimize prompt batching and cache repeated LLM calls where possible.

Contact / ownership
-------------------

- Implementation: Krish ( engineer)
- Design review: include @Utsab and @Shorya