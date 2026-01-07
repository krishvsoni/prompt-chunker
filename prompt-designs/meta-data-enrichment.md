Extract and structure metadata from this chunk for use in RAG systems.

Document Context: {DOCUMENT_TITLE} / Section: {SECTION_NAME}

Chunk Text:
---
{CHUNK_TEXT}
---

Extract the following metadata:

1. CONTENT TYPE (choose one or multiple):
   - Definition: Explains a concept or term
   - Procedure: Steps or process description
   - Example: Concrete example or case study
   - Result/Finding: Outcome or conclusion
   - Warning/Caution: Important note or limitation
   - Context: Background or prerequisite information
   - Question: Addresses a specific question
   - Other: Describe

2. KEY ENTITIES:
   - People: [names if any]
   - Organizations: [names if any]
   - Products/Tools: [names if any]
   - Concepts: [technical or business concepts]
   - Metrics: [numbers or measurements mentioned]

3. CONCEPTS & DEFINITIONS:
   - What concepts are defined in this chunk? 
     Format: {"term": "X", "definition": "...", "context": "..."}
   - What concepts are assumed known (not defined)?
     Format: {"term": "Y", "assumed_defined_in": "section X"}

4. DIFFICULTY LEVEL (1-5):
   - 1: Entry-level, no prior knowledge needed
   - 3: Intermediate, assumes some background
   - 5: Advanced, requires significant domain expertise

5. RETRIEVAL HINTS:
   - What questions would retrieve this chunk?
   - Format: ["What is X?", "How to do Y?", "Why is Z important?"]

6. PREREQUISITES:
   - What should a reader know before this chunk?
   - Format: ["Understand concept A", "Know about process B"]

7. RELATED CHUNKS:
   - What other sections of the document relate to this?
   - Format: {"section": "Methodology", "relationship": "provides foundation for"}

8. SEMANTIC TAGS (select relevant):
   - Domain: [technical, business, legal, medical, etc.]
   - Function: [explanatory, procedural, analytical, prescriptive]
   - Importance: [critical, important, supplementary]
   - Temporal: [current, historical, future, timeless]

9. SUMMARY (1-2 sentences):
   - What is the key point of this chunk?

Output JSON:
{
  "content_type": ["type1", "type2"],
  "entities": {
    "people": [],
    "organizations": [],
    "products": [],
    "concepts": [],
    "metrics": []
  },
  "concepts": {
    "defined": [
      {"term": "...", "definition": "...", "context": "..."}
    ],
    "assumed": [
      {"term": "...", "assumed_defined_in": "..."}
    ]
  },
  "difficulty_level": int,
  "retrieval_hints": ["hint1", "hint2", "hint3"],
  "prerequisites": ["prereq1", "prereq2"],
  "related_sections": [
    {"section": "...", "relationship": "..."}
  ],
  "semantic_tags": {
    "domain": "...",
    "function": ["func1", "func2"],
    "importance": "...",
    "temporal": "..."
  },
  "summary": "..."
}
