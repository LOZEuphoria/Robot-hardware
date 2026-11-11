#!/usr/bin/env python3
"""Add the Paper 315 supplementary PDF to the production RAG corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from crawl_huflit_rag_v15_integrated import build_rag_chunks, extract_pdf_text


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = BASE_DIR / "documents" / "HIRA_Paper315_Supplementary_ResultsOnly.pdf"
DATABASE_PATH = BASE_DIR / "huflit_rag_db_production.json"
DOC_ID = "hira_paper315_supplementary_results"
TITLE = "Supplementary Results for HIRA Paper 315"
GITHUB_BLOB_URL = (
    "https://github.com/LOZEuphoria/Robot-hardware/blob/main/"
    "RAG/documents/HIRA_Paper315_Supplementary_ResultsOnly.pdf"
)
GITHUB_RAW_URL = (
    "https://raw.githubusercontent.com/LOZEuphoria/Robot-hardware/main/"
    "RAG/documents/HIRA_Paper315_Supplementary_ResultsOnly.pdf"
)


def make_document(pdf_path: Path) -> dict[str, Any]:
    extracted = extract_pdf_text(pdf_path)
    full_text = extracted["full_text"].strip()
    if not full_text:
        raise RuntimeError(f"No selectable text found in PDF: {pdf_path}")

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    source_url = (
        GITHUB_BLOB_URL
        if pdf_path.resolve() == DEFAULT_PDF.resolve()
        else pdf_path.resolve().as_uri()
    )
    file_url = GITHUB_RAW_URL if pdf_path.resolve() == DEFAULT_PDF.resolve() else source_url

    return {
        "doc_id": DOC_ID,
        "doc_type": "pdf",
        "category": "research_supplement",
        "title": TITLE,
        "source_url": source_url,
        "file_url": file_url,
        "published_at": None,
        "scraped_at": now,
        "first_seen_at": now,
        "last_seen_at": now,
        "source_order": {"detail_id": None, "discovered_rank": None},
        "time_confidence": "unknown",
        "full_text": full_text,
        "paragraphs": extracted["paragraphs"],
        "page_texts": extracted["page_texts"],
        "extraction": extracted["extraction"],
        "links": [],
        "attachments": [],
        "status": {
            "is_empty": False,
            "is_low_quality": False,
            "is_link_only": False,
            "is_duplicate": False,
            "has_missing_links": False,
            "suspected_date_error": False,
            "needs_table_parse": True,
        },
        "content_hash": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
    }


def update_corpus(document: dict[str, Any]) -> int:
    data = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))
    source_documents = data.setdefault("source_documents", [])
    data["source_documents"] = [
        item
        for item in source_documents
        if item.get("doc_id") != DOC_ID
        and item.get("source_url") != document["source_url"]
    ]
    data["source_documents"].append(document)

    data["rag_chunks"] = [
        item for item in data.get("rag_chunks", []) if item.get("doc_id") != DOC_ID
    ]
    chunks = build_rag_chunks([document])
    data["rag_chunks"].extend(chunks)

    metadata = data.setdefault("metadata", {})
    metadata["version"] = datetime.now().astimezone().strftime("%Y-%m-%d")
    metadata["total_documents"] = len(data["source_documents"])
    metadata["total_articles"] = sum(
        item.get("doc_type") == "article" for item in data["source_documents"]
    )
    metadata["total_pdfs"] = sum(
        item.get("doc_type") == "pdf" for item in data["source_documents"]
    )
    metadata["total_linked_resources"] = sum(
        item.get("doc_type") == "linked_resource"
        for item in data["source_documents"]
    )
    metadata["total_chunks"] = len(data["rag_chunks"])

    temporary_path = DATABASE_PATH.with_suffix(DATABASE_PATH.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary_path, DATABASE_PATH)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="?", type=Path, default=DEFAULT_PDF)
    args = parser.parse_args()

    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    document = make_document(pdf_path)
    chunk_count = update_corpus(document)
    print(
        f"Indexed {document['title']}: {len(document['page_texts'])} pages, "
        f"{chunk_count} chunks."
    )


if __name__ == "__main__":
    main()
