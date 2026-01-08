# Rules to Avoid Misleading Metrics

These rules MUST be enforced before generating any metrics report.

---

## Rule 1: Never Compare Parsers and Semantic Chunkers Using Size-Based Optimality

**Why:** Size-based metrics (e.g., "chunk is between 200-800 chars") do not reflect semantic quality.
A 500-character chunk that splits a sentence mid-thought is worse than a 1200-character chunk
that contains a complete concept.

**Enforcement:** Use Semantic Accept % instead of size-based "Optimal %" metrics.

---

## Rule 2: Relevance % Alone is Meaningless Without Top-K Sufficiency

**Why:** Relevance % only measures recall ("did we retrieve *something* relevant?").
It does not measure efficiency ("how many chunks were needed?").

**Example of misleading report:**
- Method A: 95% Relevance (but needs 50 chunks to answer)
- Method B: 90% Relevance (but needs only 3 chunks to answer)

Method B is actually better for RAG despite lower Relevance %.

**Enforcement:** Always report both Relevance % AND Top-K Sufficiency % together.

---

## Rule 3: If Semantic Accept % < 40%, Force Refinement Before Reporting

**Why:** Low acceptance rates indicate the chunking quality is poor.
Reporting these metrics without refinement misrepresents production quality.

**Enforcement:**
1. Check if Semantic Accept % < 40%
2. If true, run auto-refinement pass (split/merge operations)
3. Re-compute metrics after refinement
4. Report both "before refinement" and "after refinement" rates

---

## Rule 4: Always Report K Used in Retrieval Metrics

**Why:** Relevance@K=3 vs Relevance@K=100 are completely different metrics.
Without knowing K, the metric is uninterpretable.

**Enforcement:**
- Relevance % must include K value: "Relevance@10: 85%"
- Top-K Sufficiency % must include K value: "Top-3 Sufficiency: 72%"
- Use consistent K values across all methods in a comparison

**Default Values:**
- Relevance K: 10
- Sufficiency K: 3

---

## Rule 5: Never Average Metrics Across Documents of Different Tiers Without Labeling

**Why:** Tier-1 (complex, high-value) documents have different quality requirements
than Tier-3 (simple, bulk) documents. Averaging them together hides important information.

**Example of misleading report:**
- "Average Semantic Accept: 75%"
- Reality: Tier-1 at 50%, Tier-3 at 90%

The Tier-1 documents (which matter most) have poor quality hidden by Tier-3 numbers.

**Enforcement:**
1. Label document tiers in input
2. Report metrics separately per tier
3. If averaging, clearly label: "Weighted Average (Tier-1: 30%, Tier-2: 30%, Tier-3: 40%)"

---

## Rule 6: Never Mix Length Units in the Same Report

**Why:** "Average chunk length: 500" is meaningless without unit.
Characters vs tokens have ~4x difference.

**Enforcement:**
- Pick ONE unit (characters OR tokens) at the start
- Use consistently throughout all metrics
- Label explicitly: "Avg Length: 500 characters" or "Avg Length: 125 tokens"

**Recommendation:** Use characters for simplicity, tokens for embedding-aware analysis.

---

## Rule 7: Exclude Rejected/Intermediate Chunks from Final Metrics

**Why:** Counting rejected chunks inflates totals and skews averages.
Counting intermediate chunks (before validation) misrepresents final quality.

**Enforcement:**
- Total Chunks = final indexed chunks only
- Average Quality = mean over validated chunks only (exclude missing scores, don't treat as 0)
- Semantic Accept % = accepted / validated total (not raw total)

---

## Rule 8: Boundary Confidence Only Over Boundary Decisions

**Why:** Averaging confidence over ALL blocks (including non-boundaries) dilutes the signal.
A block that's clearly middle-of-paragraph shouldn't affect boundary confidence.

**Enforcement:**
- Filter to only decisions where `new_chunk=True`
- Calculate: mean(boundary.confidence) for boundary decisions only

---

## Required Statement in All Reports

Include this exact statement in every metrics report:

> "Semantic metrics (Semantic Accept Rate and Top-K Sufficiency) replace heuristic 'optimality'
> because they directly measure query–document alignment and retrieval efficiency."

---

## Validation Checklist

Before publishing any report, verify:

- [ ] Semantic Accept % is used instead of size-based Optimal %
- [ ] Relevance % is paired with Top-K Sufficiency %
- [ ] K values are explicitly stated
- [ ] Semantic Accept % >= 40% (or refinement was forced)
- [ ] Document tiers are labeled if mixed
- [ ] Length units are consistent and labeled
- [ ] Only final indexed chunks are counted
- [ ] Boundary confidence is over boundaries only
- [ ] Explanation statement is included