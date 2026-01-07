import os
import time
import json
import difflib
from typing import List, Dict


def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def fuzzy_match(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def assert_chunks(chunks, name):
    assert len(chunks) > 1, f"{name} failed: produced only 1 chunk"

# =========================
# PyPDF (INTENTIONALLY MINIMAL)
# =========================

def extract_with_pypdf(pdf_path: str) -> List[Dict]:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n\n".join(p.extract_text() or "" for p in reader.pages)

    # naive: whole doc = one chunk
    return [{
        "chunk_id": "pypdf_0",
        "title": "__entire_document__",
        "text": text
    }]

# =========================
# Sherpa (STRUCTURE ONLY)
# =========================

def extract_with_sherpa(pdf_path: str) -> List[Dict]:
    """
    Simulated Sherpa behavior:
    - layout-based
    - section aware
    - NO semantic reasoning
    """
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n\n".join(p.extract_text() or "" for p in reader.pages)

    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]

    chunks = []
    current = []
    title = "__root__"

    for p in paragraphs:
        if p[:4].split()[0].rstrip(".").isdigit():
            if current:
                chunks.append({
                    "chunk_id": f"sherpa_{len(chunks)}",
                    "title": title,
                    "text": "\n\n".join(current)
                })
            title = p.split("\n")[0][:80]
            current = []
        else:
            current.append(p)

    if current:
        chunks.append({
            "chunk_id": f"sherpa_{len(chunks)}",
            "title": title,
            "text": "\n\n".join(current)
        })

    assert_chunks(chunks, "Sherpa")
    return chunks

# =========================
# PROMPT-DRIVEN SEMANTIC PIPELINE
# =========================

STRUCTURE_ANALYST_PROMPT = """
You are a document structure analyst.

You receive ordered text blocks (paragraph-level).
Your task is to decide SEMANTIC CHUNK BOUNDARIES.

For EACH block decide:
- Does this START a new semantic chunk?
- What is the role of this block?

Rules:
- Remove headers/footers.
- Do NOT merge unrelated topics.
- Prefer semantic completeness over size.
- Tables and referenced paragraphs stay together.
- Output STRICT JSON ONLY.

Output format:
[
  {
    "block_id": "b0",
    "new_chunk": true,
    "role": "section_header|subsection_header|body|list|table|noise",
    "reason": "short explanation",
    "confidence": 0.0-1.0
  }
]
"""

def call_llm(messages, model="gpt-4o-mini"):
    from openai import OpenAI
    client = OpenAI()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=2000
    )

def extract_with_prompt(pdf_path: str) -> List[Dict]:
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    raw_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)

    # TRUE MICRO-BLOCKS (paragraphs)
    blocks = [
        {"id": f"b{i}", "text": p.strip()}
        for i, p in enumerate(raw_text.split("\n\n"))
        if len(p.strip()) > 50
    ]

    assert len(blocks) > 1, "Not enough blocks for semantic chunking"

    messages = [
        {"role": "system", "content": "You output STRICT JSON only."},
        {"role": "user", "content": STRUCTURE_ANALYST_PROMPT + "\n\nBlocks:\n" + json.dumps(blocks)}
    ]

    resp = call_llm(messages)
    decisions = json.loads(resp.choices[0].message.content)


    chunks = []
    current_chunk = []
    title = "__auto__"

    for dec, block in zip(decisions, blocks):
        if dec["role"] == "noise":
            continue

        if dec["new_chunk"] and current_chunk:
            chunks.append({
                "chunk_id": f"prompt_{len(chunks)}",
                "title": title,
                "text": "\n\n".join(current_chunk)
            })
            current_chunk = []

        if dec["role"] in ["section_header", "subsection_header"]:
            title = block["text"][:80]

        current_chunk.append(block["text"])

    if current_chunk:
        chunks.append({
            "chunk_id": f"prompt_{len(chunks)}",
            "title": title,
            "text": "\n\n".join(current_chunk)
        })

    assert_chunks(chunks, "Prompt-based")
    return chunks


def run(pdf_path: str):
    print("\n--- Running Comparison ---\n")

    t0 = time.time()
    pypdf_chunks = extract_with_pypdf(pdf_path)
    print(f"PyPDF    → {len(pypdf_chunks)} chunks | {time.time()-t0:.2f}s")

    t0 = time.time()
    sherpa_chunks = extract_with_sherpa(pdf_path)
    print(f"Sherpa   → {len(sherpa_chunks)} chunks | {time.time()-t0:.2f}s")

    t0 = time.time()
    prompt_chunks = extract_with_prompt(pdf_path)
    print(f"Prompt   → {len(prompt_chunks)} chunks | {time.time()-t0:.2f}s")

    return {
        "pypdf": pypdf_chunks,
        "sherpa": sherpa_chunks,
        "prompt": prompt_chunks
    }


