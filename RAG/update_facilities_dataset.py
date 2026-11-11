import hashlib
import json
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from crawl_huflit_rag_v15_integrated import build_rag_chunks, clean_text


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "huflit_rag_db_production.json"
SOURCE_URL = (
    "https://huflit.edu.vn/vi/tin-tuc/"
    "hoc-phi-lo-trinh-hoc-tap-va-co-so-dao-tao-tai-huflit-2026-"
    "thong-tin-tan-sinh-vien-can-biet-truoc-khi-nhap-hoc/"
)
DOC_ID = "huflit_facilities_2026"


def main() -> None:
    response = requests.get(SOURCE_URL, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    containers = soup.select(".container-inner")
    if not containers:
        raise RuntimeError("Không tìm thấy vùng nội dung chính trên trang HUFLIT")
    container = max(
        containers,
        key=lambda item: len(item.get_text(" ", strip=True)),
    )

    full_text = clean_text(container.get_text("\n", strip=True))
    required_addresses = (
        "828 Sư Vạn Hạnh",
        "32 Trường Sơn",
        "52-70 Ba Gia",
        "806 Lê Quang Đạo",
    )
    if not all(address in full_text for address in required_addresses):
        raise RuntimeError("Trang HUFLIT chưa trích xuất đủ bốn địa chỉ cơ sở")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    document = {
        "doc_id": DOC_ID,
        "doc_type": "article",
        "category": "co_so_hoc_tap",
        "title": "Các cơ sở đào tạo của HUFLIT năm 2026",
        "source_url": SOURCE_URL,
        "published_at": "2026-07-13",
        "scraped_at": now,
        "first_seen_at": now,
        "last_seen_at": now,
        "source_order": {"detail_id": None, "discovered_rank": 1},
        "time_confidence": "exact_published_at",
        "full_text": full_text,
        "paragraphs": [line for line in full_text.splitlines() if line.strip()],
        "links": [],
        "attachments": [],
        "status": {
            "is_empty": False,
            "is_low_quality": False,
            "is_link_only": False,
            "is_duplicate": False,
            "has_missing_links": False,
            "suspected_date_error": False,
            "needs_table_parse": False,
        },
        "content_hash": hashlib.sha256(full_text.encode("utf-8")).hexdigest(),
    }

    data = json.loads(DATABASE_PATH.read_text(encoding="utf-8"))
    data["source_documents"] = [
        item
        for item in data.get("source_documents", [])
        if item.get("doc_id") != DOC_ID and item.get("source_url") != SOURCE_URL
    ]
    data["source_documents"].append(document)

    chunks = build_rag_chunks([document])
    data["rag_chunks"] = [
        item
        for item in data.get("rag_chunks", [])
        if item.get("doc_id") != DOC_ID
    ]
    data["rag_chunks"].extend(chunks)

    metadata = data.setdefault("metadata", {})
    metadata["version"] = datetime.now().strftime("%Y-%m-%d")
    metadata["total_documents"] = len(data["source_documents"])
    metadata["total_articles"] = sum(
        item.get("doc_type") == "article" for item in data["source_documents"]
    )
    metadata["total_chunks"] = len(data["rag_chunks"])
    DATABASE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Đã cập nhật {document['title']}: "
        f"{len(chunks)} chunk, {metadata['total_documents']} tài liệu."
    )


if __name__ == "__main__":
    main()
