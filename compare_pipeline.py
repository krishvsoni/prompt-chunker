import os
import time
import json
import difflib
from typing import List, Dict



def normalize(s: str) -> str:
    return " ".join(s.lower().strip().split())

def fuzzy_match(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()




def extract_with_pypdf(pdf_path: str) -> List[Dict]:
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
    return [{
        "chunk_id": "pypdf_0",
        "title": "__entire_document__",
        "text": text
    }]



def extract_with_sherpa(pdf_path: str) -> List[Dict]:
    """
    Proper LLM Sherpa usage via HTTP API.
    Layout-based parsing only.
    """
    import requests

    SHERPA_URL = "http://localhost:5010/api/parseDocument"

    with open(pdf_path, "rb") as f:
        files = {"file": f}
        params = {
            "renderFormat": "all"
        }
        resp = requests.post(SHERPA_URL, files=files, params=params)

    if resp.status_code != 200:
        print("Sherpa API failed, status:", resp.status_code)
        return []

    data = resp.json()

    chunks = []

    for i, block in enumerate(data.get("sections", [])):
        text = block.get("text", "").strip()
        if not text:
            continue

        title = block.get("title", "__layout__")

        chunks.append({
            "chunk_id": f"sherpa_{i}",
            "title": title,
            "text": text
        })

    if len(chunks) <= 1:
        print("Sherpa: layout weak or flattened, produced <=1 chunk")

    return chunks
    """
    Proper LLM Sherpa usage via LangChain.
    Layout-based only. No semantic reasoning.
    """
    from langchain_community.document_loaders import LLMSherpaPDFLoader

    sherpa_url = "http://localhost:5010/api/parseDocument?renderFormat=all"

    loader = LLMSherpaPDFLoader(
        file_path=pdf_path,
        llmsherpa_api_url=sherpa_url
    )

    docs = loader.load()

    chunks = []
    for i, d in enumerate(docs):
        text = d.page_content.strip()
        if not text:
            continue
        title = d.metadata.get("section_title", "__layout__")
        chunks.append({
            "chunk_id": f"sherpa_{i}",
            "title": title,
            "text": text
        })

    # no assert here — 1 chunk is a valid layout outcome
    if len(chunks) <= 1:
        print("Sherpa: layout weak or flattened, produced <=1 chunk")

    return chunks



STRUCTURE_ANALYST_PROMPT = """
You are a document structure analyst.

You receive ordered paragraph blocks.
Decide semantic chunk boundaries.

For each block:
- Does it start a NEW semantic chunk?
- What is its role?

Rules:
- Ignore headers, footers, signatures, boilerplate.
- Do not merge unrelated topics.
- Prefer semantic completeness over size.
- Output STRICT JSON only.

Output format:
[
  {
    "block_id": "b0",
    "new_chunk": true,
    "role": "section_header|subsection_header|body|list|table|noise",
    "reason": "short",
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

    blocks = [
        {"id": f"b{i}", "text": p.strip()}
        for i, p in enumerate(raw_text.split("\n\n"))
        if len(p.strip()) > 40
    ]

    if len(blocks) <= 1:
        raise RuntimeError("Not enough content for semantic chunking")

    messages = [
        {"role": "system", "content": "Output strict JSON only."},
        {"role": "user", "content": STRUCTURE_ANALYST_PROMPT + "\n\nBlocks:\n" + json.dumps(blocks)}
    ]

    resp = call_llm(messages)

    print("LLM Response:", resp)

    try:
        if not resp or not hasattr(resp, "choices") or not resp.choices:
            raise ValueError("Invalid response from LLM: Missing 'choices'")
        decisions = json.loads(resp.choices[0].message.content)
    except (AttributeError, JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Failed to parse LLM response: {e}")

    chunks = []
    current = []
    title = "__auto__"

    for dec, block in zip(decisions, blocks):
        if dec["role"] == "noise":
            continue

        if dec["new_chunk"] and current:
            chunks.append({
                "chunk_id": f"prompt_{len(chunks)}",
                "title": title,
                "text": "\n\n".join(current)
            })
            current = []

        if dec["role"] in ["section_header", "subsection_header"]:
            title = block["text"][:80]

        current.append(block["text"])

    if current:
        chunks.append({
            "chunk_id": f"prompt_{len(chunks)}",
            "title": title,
            "text": "\n\n".join(current)
        })

    if len(chunks) <= 1:
        raise RuntimeError("Prompt chunking failed — investigate prompt or document")

    return chunks




def run(pdf_path: str):
    print("\n--- Running Comparison ---\n")

    t0 = time.time()
    pypdf_chunks = extract_with_pypdf(pdf_path)
    print(f"PyPDF    -> {len(pypdf_chunks)} chunks | {time.time()-t0:.2f}s")

    t0 = time.time()
    sherpa_chunks = extract_with_sherpa(pdf_path)
    print(f"Sherpa   -> {len(sherpa_chunks)} chunks | {time.time()-t0:.2f}s")

    # Route to prompt chunker if Sherpa is weak
    t0 = time.time()
    if len(sherpa_chunks) <= 1:
        print("Routing to semantic prompt chunker")
        prompt_chunks = extract_with_prompt(pdf_path)
    else:
        prompt_chunks = extract_with_prompt(pdf_path)
    print(f"Prompt   -> {len(prompt_chunks)} chunks | {time.time()-t0:.2f}s")

    return {
        "pypdf": pypdf_chunks,
        "sherpa": sherpa_chunks,
        "prompt": prompt_chunks
    }

