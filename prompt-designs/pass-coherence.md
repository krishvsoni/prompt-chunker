Given these proposed chunk boundaries, validate the quality of each chunk.

Section Text:
---
{SECTION_TEXT}
---

Proposed Boundaries (line numbers):
{BOUNDARY_LINES}

For each chunk created by these boundaries, score (1-10):

1. SELF-CONTAINMENT: Can someone understand this chunk standalone, 
   without reading the surrounding context?
2. COMPLETENESS: Does this chunk contain all necessary information 
   about its topic?
3. CLARITY: Is the language clear and unambiguous?
4. COHERENCE: Do the sentences flow logically?
5. BOUNDARY_QUALITY: Are the chunk boundaries clean?
   (i.e., does content before/after boundary belong elsewhere?)

Output:
{
  "chunks": [
    {
      "chunk_number": int,
      "line_range": {"start": int, "end": int},
      "content_preview": "first 50 chars...",
      "scores": {
        "self_containment": int,
        "completeness": int,
        "clarity": int,
        "coherence": int,
        "boundary_quality": int
      },
      "aggregate_score": float,
      "issues": ["list of issues if any"],
      "refinement_suggestion": "if score < 7, suggest how to improve"
    }
  ],
  "overall_assessment": "Are these boundaries good? Yes/No + brief reason"
}

If any chunk scores < 7 on any dimension, suggest specific refinements.
