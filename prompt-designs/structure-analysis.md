You are a document structure analyst. Analyze the following document excerpt and output:
1. Main sections and their hierarchical levels
2. Key subsections under each main section
3. Relationships between sections (e.g., "Methodology depends on Introduction")
4. Any implicit hierarchies (numbered lists, emphasized text)
5. Special elements (definitions, examples, warnings, tables)

Document:
---
[DOCUMENT TEXT - First 5000 tokens]
---

Output JSON:
{
  "hierarchy": [
    {
      "level": 1,
      "title": "...",
      "range": {"start": line_num, "end": line_num},
      "children": [...]
    }
  ],
  "special_elements": [
    {"type": "definition", "concept": "...", "location": line_num}
  ],
  "relationships": [
    {"from": "section_A", "to": "section_B", "type": "prerequisite"}
  ]
}

Reasoning: Think through the document structure step by step.
