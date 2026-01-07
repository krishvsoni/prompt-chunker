Validate the quality and coherence of this chunk for RAG use.

Chunk Information:
- ID: {CHUNK_ID}
- Section: {SECTION_CONTEXT}
- Content Type: {CONTENT_TYPE}

Chunk Text:
---
{CHUNK_TEXT}
---

Chunk Metadata (for alignment check):
{CHUNK_METADATA_JSON}

Assess this chunk using the following rubric:

DIMENSION 1: INTERNAL COHERENCE (0-10)
Criteria: Do sentences flow logically? Are ideas connected? 
Does the chunk stay on topic?
- 9-10: Perfect flow, every sentence builds on previous
- 7-8: Good flow with minor transitions
- 5-6: Adequate but some jumps in logic
- 3-4: Poor flow, topics seem disconnected
- 0-2: Incoherent, incomprehensible

Your score: ___ 
Evidence: "..."

DIMENSION 2: SELF-CONTAINMENT (0-10)
Criteria: Can someone understand this chunk without external context?
How many external references require outside knowledge?
- 9-10: Completely self-contained, all context provided
- 7-8: Mostly self-contained, minimal external reference
- 5-6: Somewhat clear but requires some background
- 3-4: Heavily dependent on external knowledge
- 0-2: Incomprehensible without reading other sections first

Your score: ___ 
Evidence: "Requires knowledge of..." or "Self-contained because..."

DIMENSION 3: COMPLETENESS (0-10)
Criteria: Does chunk contain all necessary information to understand its topic?
- 9-10: Complete treatment of topic
- 7-8: Covers main points, minimal gaps
- 5-6: Covers topic adequately but missing details
- 3-4: Significant gaps in coverage
- 0-2: Too fragmented to understand topic

Your score: ___
Evidence: "Missing..." or "Covers all aspects..."

DIMENSION 4: BOUNDARY QUALITY (0-10)
Criteria: Are chunk boundaries in the right place? 
Should content be moved before/after boundary?
- 9-10: Perfect boundaries, nothing should move
- 7-8: Good boundaries with minor adjustments possible
- 5-6: Acceptable but could be optimized
- 3-4: Poor boundaries, content belongs elsewhere
- 0-2: Fundamental boundary problem

Your score: ___
Evidence: "Consider moving X to previous/next chunk because..."

DIMENSION 5: METADATA ALIGNMENT (0-10)
Criteria: Do the assigned tags, type, and metadata match the content?
- 9-10: Perfect alignment
- 7-8: Good alignment with minor mismatches
- 5-6: Adequate alignment but some inconsistencies
- 3-4: Poor alignment, many mismatches
- 0-2: Metadata completely misaligned

Your score: ___
Evidence: "Tags match content" or "Content_type should be X, not Y"

OVERALL ASSESSMENT:
Average Score: {AVG}

If average < 7, provide specific, actionable refinements:
{
  "average_score": float,
  "status": "pass" if avg >= 7 else "fail",
  "dimension_scores": {
    "coherence": int,
    "containment": int,
    "completeness": int,
    "boundaries": int,
    "metadata": int
  },
  "issues": [
    {"dimension": "...", "issue": "...", "severity": "high|medium|low"},
  ],
  "refinements": [
    {"action": "merge with previous chunk", "reason": "Too small (X tokens)"},
    {"action": "split at line Y", "reason": "Topic shift detected"},
    {"action": "revise tag from X to Y", "reason": "Content is actually..."}
  ],
  "recommendation": "ACCEPT|REFINE|REJECT"
}

IMPORTANT: Be specific. Don't just say "improve clarity". Say exactly 
what phrase or sentence needs to be clarified and how.
