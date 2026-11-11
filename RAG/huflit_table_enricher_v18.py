#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HUFLIT Table Enricher v18

Mục tiêu:
- Đọc file HUFLIT RAG JSON hiện có.
- Tìm các document có file đã tải về: PDF/DOCX/XLSX.
- Extract bảng thành structured_records.tables.
- Tạo thêm table chunks để chatbot hỏi chính xác các câu dạng bảng:
  "lớp nào bị hủy?", "mã học phần nào?", "danh sách ngành nào?", ...
- Rebuild rag_chunks theo page cho document dài.

Cài đặt:
    pip install -r requirements_huflit_table_v18.txt

Chạy:
    python huflit_table_enricher_v18.py ^
      --input huflit_rag_kb_full.json ^
      --output huflit_rag_kb_full_tables.json ^
      --download-dir downloads ^
      --add-table-chunks ^
      --rebuild-chunks

Ghi chú:
- PDF table tốt nhất khi file PDF thật đã download được.
- Nếu PDF là scan ảnh, script này không OCR; dùng viewer OCR riêng.
- Với bảng OCR, nên xử lý bằng script OCR rồi chạy script này lại nếu OCR có sinh structured table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Optional libs
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import openpyxl
except Exception:
    openpyxl = None


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def short_hash(text: str, n: int = 10) -> str:
    return sha256_text(text)[:n]


def clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\b(20\d)\s+(\d)\b", r"\1\2", text)
    text = re.sub(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", r"\1/\2/\3", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_cell(value: Any) -> str:
    return clean_text(value).replace("\n", " ").strip()


def is_empty_row(row: List[str]) -> bool:
    return all(not normalize_cell(x) for x in row)


def dedupe_columns(cols: List[str]) -> List[str]:
    result = []
    seen = {}
    for i, c in enumerate(cols):
        c = normalize_cell(c) or f"col_{i+1}"
        base = c
        count = seen.get(base, 0)
        if count:
            c = f"{base}_{count+1}"
        seen[base] = count + 1
        result.append(c)
    return result


def rows_to_markdown(columns: List[str], rows: List[Dict[str, str]], max_rows: Optional[int] = None) -> str:
    if not columns:
        return ""
    rows = rows if max_rows is None else rows[:max_rows]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for r in rows:
        cells = []
        for c in columns:
            val = normalize_cell(r.get(c, ""))
            val = val.replace("|", "\\|")
            cells.append(val)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def split_text_to_chunks(text: str, max_chars: int = 2800, overlap_chars: int = 250) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) + 1 <= max_chars:
            cur = f"{cur}\n{p}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    if overlap_chars and len(chunks) > 1:
        out, tail = [], ""
        for ch in chunks:
            out.append((tail + "\n" + ch).strip() if tail else ch)
            tail = ch[-overlap_chars:]
        chunks = out
    return [c for c in chunks if len(c) >= 40]


def chunk_by_pages(doc: Dict[str, Any], pages_per_chunk: int = 2) -> List[Dict[str, Any]]:
    page_texts = doc.get("page_texts") or []
    page_texts = [p for p in page_texts if clean_text(p.get("text", ""))]
    if not page_texts:
        return []

    chunks = []
    group = []

    def flush(g):
        if not g:
            return
        text = clean_text("\n\n".join(p.get("text", "") for p in g))
        if len(text) < 40:
            return
        chunks.append({
            "page_start": g[0].get("page"),
            "page_end": g[-1].get("page"),
            "chunk_text": text,
        })

    for p in page_texts:
        group.append(p)
        if len(group) >= pages_per_chunk:
            flush(group)
            group = []
    flush(group)
    return chunks


# ---------------------------------------------------------------------
# Locate files
# ---------------------------------------------------------------------

def resolve_local_file(doc: Dict[str, Any], download_dirs: List[Path]) -> Optional[Path]:
    candidates = []

    for key in ["local_path", "file_path"]:
        if doc.get(key):
            candidates.append(Path(doc[key]))

    if doc.get("file_name"):
        for d in download_dirs:
            candidates.append(d / doc["file_name"])
            candidates.append(d / "profile_check" / doc["file_name"])

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            pass

    return None


def file_ext(path: Path) -> str:
    return path.suffix.lower()


# ---------------------------------------------------------------------
# Table extraction: PDF
# ---------------------------------------------------------------------

def table_matrix_to_record(
    matrix: List[List[Any]],
    doc: Dict[str, Any],
    source_file: Path,
    page: Optional[int],
    table_index: int,
    method: str,
    confidence: str,
) -> Optional[Dict[str, Any]]:
    # Normalize + remove empty rows
    cleaned = []
    for row in matrix or []:
        normalized = [normalize_cell(x) for x in row]
        if not is_empty_row(normalized):
            cleaned.append(normalized)

    if len(cleaned) < 2:
        return None

    # Use first row as header if reasonable; otherwise create generic columns.
    header = cleaned[0]
    data_rows = cleaned[1:]

    non_empty_header = sum(1 for h in header if h)
    if non_empty_header < max(1, len(header) // 2):
        max_cols = max(len(r) for r in cleaned)
        header = [f"col_{i+1}" for i in range(max_cols)]
        data_rows = cleaned
    else:
        header = dedupe_columns(header)

    # Normalize row length
    width = len(header)
    rows = []
    for row in data_rows:
        row = row + [""] * (width - len(row))
        row = row[:width]
        if is_empty_row(row):
            continue
        rows.append({header[i]: normalize_cell(row[i]) for i in range(width)})

    if not rows:
        return None

    table_text = rows_to_markdown(header, rows)
    doc_id = doc.get("doc_id", "unknown")
    table_id = f"table_{doc_id}_p{page or 0}_{table_index:03d}_{short_hash(table_text, 6)}"

    return {
        "table_id": table_id,
        "doc_id": doc_id,
        "parent_doc_id": doc.get("parent_doc_id"),
        "doc_type": doc.get("doc_type"),
        "category": doc.get("category"),
        "title": doc.get("title"),
        "source_url": doc.get("source_url"),
        "file_url": doc.get("file_url"),
        "file_name": doc.get("file_name") or source_file.name,
        "local_path": str(source_file),
        "page": page,
        "table_index": table_index,
        "columns": header,
        "rows": rows,
        "row_count": len(rows),
        "column_count": len(header),
        "table_markdown": table_text,
        "extraction_method": method,
        "confidence": confidence,
        "created_at": now_str(),
    }


def extract_pdf_tables_pdfplumber(path: Path, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    if pdfplumber is None:
        return []

    tables = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                # Try line-based extraction first, then text strategy.
                raw_tables = []
                settings_list = [
                    {
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    },
                    {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    },
                ]

                for settings in settings_list:
                    try:
                        extracted = page.extract_tables(table_settings=settings) or []
                        if extracted:
                            raw_tables = extracted
                            break
                    except Exception:
                        continue

                for ti, matrix in enumerate(raw_tables, start=1):
                    rec = table_matrix_to_record(
                        matrix=matrix,
                        doc=doc,
                        source_file=path,
                        page=page_idx,
                        table_index=ti,
                        method="pdfplumber",
                        confidence="medium",
                    )
                    if rec:
                        tables.append(rec)
    except Exception:
        return tables

    return tables


def extract_pdf_tables_heuristic(path: Path, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fallback for PDFs where table text is extracted line-by-line.
    This is best-effort and lower confidence.
    """
    if fitz is None:
        return []

    tables = []
    try:
        pdf = fitz.open(str(path))
        for page_idx in range(len(pdf)):
            text = clean_text(pdf[page_idx].get_text("text") or "")
            lines = [clean_text(x) for x in text.split("\n") if clean_text(x)]

            # Candidate rows: lines with repeated spacing, pipes, tabs, or multiple code-like fields
            candidate_rows = []
            for line in lines:
                if (
                    "|" in line or "\t" in line or
                    re.search(r"\s{2,}", line) or
                    len(re.findall(r"\b[A-ZĐ0-9][A-ZĐ0-9._/-]{2,}\b", line)) >= 2
                ):
                    candidate_rows.append(line)

            # Group contiguous rows.
            groups = []
            group = []
            prev_idx = None
            for i, line in enumerate(lines):
                if line in candidate_rows:
                    if prev_idx is None or i <= prev_idx + 2:
                        group.append(line)
                    else:
                        if len(group) >= 3:
                            groups.append(group)
                        group = [line]
                    prev_idx = i
            if len(group) >= 3:
                groups.append(group)

            for gi, g in enumerate(groups, start=1):
                matrix = []
                for line in g:
                    if "|" in line:
                        row = [x.strip() for x in line.split("|")]
                    elif "\t" in line:
                        row = [x.strip() for x in line.split("\t")]
                    else:
                        row = [x.strip() for x in re.split(r"\s{2,}", line)]
                    matrix.append(row)

                rec = table_matrix_to_record(
                    matrix=matrix,
                    doc=doc,
                    source_file=path,
                    page=page_idx + 1,
                    table_index=gi,
                    method="pymupdf_text_heuristic",
                    confidence="low",
                )
                if rec:
                    tables.append(rec)
    except Exception:
        return tables

    return tables


# ---------------------------------------------------------------------
# Table extraction: XLSX / DOCX
# ---------------------------------------------------------------------

def extract_xlsx_tables(path: Path, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    if openpyxl is None:
        return []

    tables = []
    try:
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        for sheet_idx, ws in enumerate(wb.worksheets, start=1):
            rows = []
            for row in ws.iter_rows(values_only=True):
                vals = [normalize_cell(v) for v in row]
                # Trim trailing empty cells
                while vals and not vals[-1]:
                    vals.pop()
                if vals and not is_empty_row(vals):
                    rows.append(vals)

            if len(rows) < 2:
                continue

            rec = table_matrix_to_record(
                matrix=rows,
                doc=doc,
                source_file=path,
                page=sheet_idx,
                table_index=1,
                method="openpyxl",
                confidence="high",
            )
            if rec:
                rec["sheet_name"] = ws.title
                tables.append(rec)
    except Exception:
        return tables

    return tables


def extract_docx_tables(path: Path, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    tables = []
    try:
        with zipfile.ZipFile(path) as z:
            if "word/document.xml" not in z.namelist():
                return []
            root = ET.fromstring(z.read("word/document.xml"))

            tbl_count = 0
            for tbl in root.iter():
                if not tbl.tag.endswith("}tbl"):
                    continue
                tbl_count += 1
                matrix = []
                for tr in tbl:
                    if not tr.tag.endswith("}tr"):
                        continue
                    row = []
                    for tc in tr:
                        if not tc.tag.endswith("}tc"):
                            continue
                        texts = []
                        for node in tc.iter():
                            if node.tag.endswith("}t") and node.text:
                                texts.append(node.text)
                        row.append(normalize_cell(" ".join(texts)))
                    if row and not is_empty_row(row):
                        matrix.append(row)

                rec = table_matrix_to_record(
                    matrix=matrix,
                    doc=doc,
                    source_file=path,
                    page=None,
                    table_index=tbl_count,
                    method="docx_xml",
                    confidence="high",
                )
                if rec:
                    tables.append(rec)
    except Exception:
        return tables

    return tables


def extract_tables_from_file(path: Path, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    ext = file_ext(path)
    if ext == ".pdf":
        tables = extract_pdf_tables_pdfplumber(path, doc)
        if not tables:
            tables = extract_pdf_tables_heuristic(path, doc)
        return tables
    if ext == ".xlsx":
        return extract_xlsx_tables(path, doc)
    if ext == ".docx":
        return extract_docx_tables(path, doc)
    return []


# ---------------------------------------------------------------------
# Main enrichment
# ---------------------------------------------------------------------

def remove_existing_tables_for_docs(data: Dict[str, Any], doc_ids: set) -> None:
    sr = data.setdefault("structured_records", {})
    tables = sr.setdefault("tables", [])
    sr["tables"] = [t for t in tables if t.get("doc_id") not in doc_ids]


def enrich_tables(data: Dict[str, Any], download_dirs: List[Path], only_doc_ids: Optional[set] = None) -> Dict[str, Any]:
    sr = data.setdefault("structured_records", {})
    sr.setdefault("tables", [])

    docs = data.get("source_documents", [])
    extracted_tables = []
    processed_doc_ids = set()
    missing_files = []
    table_docs = 0

    for doc in docs:
        doc_id = doc.get("doc_id")
        if only_doc_ids and doc_id not in only_doc_ids:
            continue

        if doc.get("doc_type") not in {"pdf", "linked_resource"}:
            continue

        path = resolve_local_file(doc, download_dirs)
        if not path:
            # If local file is missing but full_text exists, no table extraction from file.
            if doc.get("file_name"):
                missing_files.append({"doc_id": doc_id, "file_name": doc.get("file_name")})
            continue

        tables = extract_tables_from_file(path, doc)
        processed_doc_ids.add(doc_id)

        if tables:
            table_docs += 1
            extracted_tables.extend(tables)
            doc.setdefault("status", {})
            doc["status"]["has_tables"] = True
            doc["status"]["table_count"] = len(tables)
            doc["status"]["needs_table_parse"] = False
        else:
            doc.setdefault("status", {})
            doc["status"]["has_tables"] = False
            doc["status"]["table_count"] = 0

    # Replace tables for processed docs.
    remove_existing_tables_for_docs(data, processed_doc_ids)
    data["structured_records"]["tables"].extend(extracted_tables)

    summary = {
        "processed_docs": len(processed_doc_ids),
        "docs_with_tables": table_docs,
        "tables_added": len(extracted_tables),
        "missing_files": len(missing_files),
        "missing_file_examples": missing_files[:20],
        "updated_at": now_str(),
    }
    data.setdefault("postprocess", {})
    data["postprocess"]["table_enrichment"] = summary

    return summary


def table_to_chunk(table: Dict[str, Any]) -> Dict[str, Any]:
    text = table.get("table_markdown", "")
    heading = table.get("title") or table.get("file_name") or table.get("doc_id")
    page = table.get("page")
    prefix = f"Bảng từ tài liệu: {heading}"
    if page:
        prefix += f" | Trang: {page}"
    chunk_text = clean_text(prefix + "\n" + text)

    return {
        "chunk_id": f"{table['table_id']}_chunk_001",
        "doc_id": table.get("doc_id"),
        "doc_type": "table",
        "category": table.get("category"),
        "title": table.get("title"),
        "source_url": table.get("source_url"),
        "file_url": table.get("file_url"),
        "parent_doc_id": table.get("parent_doc_id"),
        "file_name": table.get("file_name"),
        "table_id": table.get("table_id"),
        "page": page,
        "chunk_index": 1,
        "chunk_text": chunk_text,
        "quality": {
            "has_source": bool(table.get("source_url") or table.get("file_url")),
            "is_table": True,
            "table_confidence": table.get("confidence"),
            "confidence": table.get("confidence") or "medium",
        },
    }


def rebuild_chunks(data: Dict[str, Any], add_table_chunks: bool, pages_per_chunk: int) -> None:
    chunks = []

    for doc in data.get("source_documents", []):
        status = doc.get("status", {}) or {}
        if status.get("is_empty") or status.get("is_low_quality"):
            continue
        text = doc.get("full_text", "")
        if not text:
            continue

        page_chunks = chunk_by_pages(doc, pages_per_chunk=pages_per_chunk) if doc.get("page_texts") else []
        if page_chunks:
            for idx, pc in enumerate(page_chunks, start=1):
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_chunk_{idx:03d}",
                    "doc_id": doc.get("doc_id"),
                    "doc_type": doc.get("doc_type"),
                    "category": doc.get("category"),
                    "title": doc.get("title"),
                    "source_url": doc.get("source_url"),
                    "file_url": doc.get("file_url"),
                    "parent_doc_id": doc.get("parent_doc_id"),
                    "file_name": doc.get("file_name"),
                    "published_at": doc.get("published_at"),
                    "scraped_at": doc.get("scraped_at"),
                    "first_seen_at": doc.get("first_seen_at"),
                    "last_seen_at": doc.get("last_seen_at"),
                    "source_order": doc.get("source_order"),
                    "detail_id": (doc.get("source_order") or {}).get("detail_id"),
                    "time_confidence": doc.get("time_confidence"),
                    "page_start": pc.get("page_start"),
                    "page_end": pc.get("page_end"),
                    "chunk_index": idx,
                    "chunk_text": pc.get("chunk_text"),
                    "quality": {
                        "has_source": bool(doc.get("source_url") or doc.get("file_url")),
                        "has_link": bool(doc.get("links")),
                        "needs_table_parse": bool(status.get("needs_table_parse")),
                        "has_tables": bool(status.get("has_tables")),
                        "is_long_document": bool(status.get("is_long_document")),
                        "confidence": "medium",
                    },
                })
        else:
            for idx, chunk_text in enumerate(split_text_to_chunks(text), start=1):
                chunks.append({
                    "chunk_id": f"{doc['doc_id']}_chunk_{idx:03d}",
                    "doc_id": doc.get("doc_id"),
                    "doc_type": doc.get("doc_type"),
                    "category": doc.get("category"),
                    "title": doc.get("title"),
                    "source_url": doc.get("source_url"),
                    "file_url": doc.get("file_url"),
                    "parent_doc_id": doc.get("parent_doc_id"),
                    "file_name": doc.get("file_name"),
                    "published_at": doc.get("published_at"),
                    "scraped_at": doc.get("scraped_at"),
                    "first_seen_at": doc.get("first_seen_at"),
                    "last_seen_at": doc.get("last_seen_at"),
                    "source_order": doc.get("source_order"),
                    "detail_id": (doc.get("source_order") or {}).get("detail_id"),
                    "time_confidence": doc.get("time_confidence"),
                    "page": None,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "quality": {
                        "has_source": bool(doc.get("source_url") or doc.get("file_url")),
                        "has_link": bool(doc.get("links")),
                        "needs_table_parse": bool(status.get("needs_table_parse")),
                        "has_tables": bool(status.get("has_tables")),
                        "is_long_document": bool(status.get("is_long_document")),
                        "confidence": "medium",
                    },
                })

    if add_table_chunks:
        for table in data.get("structured_records", {}).get("tables", []):
            chunks.append(table_to_chunk(table))

    data["rag_chunks"] = chunks
    data.setdefault("metadata", {})["total_chunks"] = len(chunks)


def export_tables_csv(data: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = data.get("structured_records", {}).get("tables", [])

    for t in tables:
        table_id = safe_filename(t.get("table_id", "table"))
        columns = t.get("columns", [])
        rows = t.get("rows", [])
        path = out_dir / f"{table_id}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for r in rows:
                writer.writerow({c: r.get(c, "") for c in columns})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--download-dir", action="append", default=[], help="Folder containing downloaded files. Can repeat.")
    ap.add_argument("--doc-id", action="append", default=[], help="Only process specific doc_id. Can repeat.")
    ap.add_argument("--add-table-chunks", action="store_true", help="Append table markdown chunks to rag_chunks.")
    ap.add_argument("--rebuild-chunks", action="store_true", help="Rebuild all rag_chunks after table extraction.")
    ap.add_argument("--pages-per-chunk", type=int, default=2)
    ap.add_argument("--export-csv-dir", default=None, help="Optional folder to export each extracted table as CSV.")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    download_dirs = [Path(d) for d in args.download_dir] if args.download_dir else [
        Path("downloads"),
        Path("downloads/profile_check"),
    ]

    only = set(args.doc_id) if args.doc_id else None
    summary = enrich_tables(data, download_dirs, only_doc_ids=only)

    if args.rebuild_chunks:
        rebuild_chunks(data, add_table_chunks=args.add_table_chunks, pages_per_chunk=args.pages_per_chunk)
    elif args.add_table_chunks:
        # Keep existing chunks and add/replace table chunks.
        existing = [c for c in data.get("rag_chunks", []) if c.get("doc_type") != "table"]
        table_chunks = [table_to_chunk(t) for t in data.get("structured_records", {}).get("tables", [])]
        data["rag_chunks"] = existing + table_chunks
        data.setdefault("metadata", {})["total_chunks"] = len(data["rag_chunks"])

    if args.export_csv_dir:
        export_tables_csv(data, Path(args.export_csv_dir))

    data.setdefault("metadata", {})["total_tables"] = len(data.get("structured_records", {}).get("tables", []))
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Output:", args.output)
    if args.export_csv_dir:
        print("CSV dir:", args.export_csv_dir)


if __name__ == "__main__":
    main()
