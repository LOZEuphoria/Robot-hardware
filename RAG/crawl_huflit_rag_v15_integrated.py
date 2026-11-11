#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HUFLIT Portal crawler for RAG-ready JSON.

Output schema:
{
  "metadata": {...},
  "source_documents": [...],
  "rag_chunks": [...],
  "structured_records": {
    "forms": [...],
    "hospital_kcb": [...],
    "links": [...]
  },
  "errors": [...]
}

Install:
    pip install requests beautifulsoup4 lxml pymupdf tqdm python-dateutil

Run:
    python crawl_huflit_rag.py --max-pages 250 --out huflit_rag_kb.json

Notes:
- This script crawls HUFLIT portal article URLs containing /News/Detail/.
- It preserves anchor hrefs, so "tại đây" links are not lost.
- It extracts direct PDF links when possible.
- It logs PDF/text/link errors instead of silently setting errors=0.
- It stores published_at when available, plus first_seen_at/last_seen_at/detail_id fallback for recency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import zipfile
import xml.etree.ElementTree as ET
import os
import re
import sys
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag, quote, unquote, parse_qs

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

try:
    import pymupdf as fitz  # PyMuPDF
except Exception:
    try:
        import fitz  # Older PyMuPDF alias
    except Exception:
        fitz = None


# Portal sinh viên cũ + website chính mới của HUFLIT.
# BASE_URL giữ portal để không phá các hàm cũ đang dùng default base.
PORTAL_BASE_URL = "https://portal.huflit.edu.vn"
MAIN_SITE_URL = "https://huflit.edu.vn/vi/"
BASE_URL = PORTAL_BASE_URL

DEFAULT_START_URLS = [
    "https://portal.huflit.edu.vn/",
    "https://portal.huflit.edu.vn/News",
    "https://huflit.edu.vn/vi/",
    "https://huflit.edu.vn/vi/tin-tuc-va-su-kien/",
]

# Khi muốn lấy đầy đủ cả 2 nguồn trong 1 file, crawler sẽ chạy từng nhóm nguồn
# với quota riêng rồi mới gộp source_documents/rag_chunks lại.
# Như vậy portal không chiếm hết max_articles của website chính và ngược lại.
DEFAULT_SOURCE_GROUPS = [
    {
        "name": "portal",
        "label": "HUFLIT Portal",
        "start_urls": [
            "https://portal.huflit.edu.vn/",
            "https://portal.huflit.edu.vn/News",
        ],
    },
    {
        "name": "main_site",
        "label": "HUFLIT Main Website",
        "start_urls": [
            "https://huflit.edu.vn/vi/",
            "https://huflit.edu.vn/vi/tin-tuc-va-su-kien/",
        ],
    },
]

# Chỉ crawl 2 domain chính này để tránh crawler chạy lan sang các subdomain lớn
# như tuyển sinh, thư viện, tạp chí, career hub... ngoài phạm vi KB học vụ.
ALLOWED_CRAWL_DOMAINS = {
    "portal.huflit.edu.vn",
    "huflit.edu.vn",
    "www.huflit.edu.vn",
}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

FILE_EXTS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".rar", ".7z"
}


# ----------------------------
# Utility
# ----------------------------

def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

def normalize_text_for_dedupe(text: str) -> str:
    """
    Normalize text for duplicate detection.
    This is stricter than clean_text: lowercase, remove repeated spaces,
    and remove most punctuation/layout characters.
    """
    text = clean_text(text).lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\sÀ-ỹ]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_fingerprint(text: str) -> str:
    """
    Fingerprint for exact/near-exact duplicate content after normalization.
    """
    return sha256_text(normalize_text_for_dedupe(text))


def document_quality_score(doc: Dict[str, Any]) -> int:
    """
    Score used when deciding which duplicate to keep.
    Higher is better.
    """
    text = doc.get("full_text", "") or ""
    title = doc.get("title", "") or ""
    links = doc.get("links", []) or []
    attachments = doc.get("attachments", []) or []
    status = doc.get("status", {}) or {}

    score = 0
    score += min(len(text), 20000)
    score += 2000 if title else 0
    score += 1000 if doc.get("published_at") else 0
    score += 500 if doc.get("source_order", {}).get("detail_id") is not None else 0
    score += min(len(links), 20) * 50
    score += len(attachments) * 300

    if status.get("is_low_quality"):
        score -= 5000
    if status.get("is_empty"):
        score -= 10000

    return score



def short_hash(text: str, n: int = 12) -> str:
    return sha256_text(text)[:n]


def normalize_url(url: str, base: str = BASE_URL) -> str:
    if not url:
        return ""
    url = url.strip()
    url = urljoin(base, url)
    url, _frag = urldefrag(url)

    # Normalize accidental spaces and unicode safely.
    parsed = urlparse(url)
    path = quote(unquote(parsed.path), safe="/:%[]@!$&'()*+,;=-")
    query = quote(unquote(parsed.query), safe="=&:%[]@!$'()*+,;/?-")
    normalized = parsed._replace(path=path, query=query).geturl()
    return normalized


def get_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def is_same_domain(url: str, allowed_domain: str = "") -> bool:
    """
    Backward-compatible domain checker.
    - Nếu truyền allowed_domain: kiểm tra đúng domain đó.
    - Nếu không truyền: cho phép các domain trong ALLOWED_CRAWL_DOMAINS.
    """
    try:
        host = urlparse(url).netloc.lower()
        if not host:
            return False
        if allowed_domain:
            allowed = allowed_domain.lower()
            return host == allowed or host.endswith("." + allowed)
        return host in ALLOWED_CRAWL_DOMAINS
    except Exception:
        return False


def is_portal_article_url(url: str) -> bool:
    """Portal cũ: /News/Detail/<id>/<slug>."""
    path = urlparse(url).path
    return bool(re.match(r"^/News/Detail/\d+(?:/[^/]+)?/?$", path, re.IGNORECASE))


def is_main_site_content_url(url: str) -> bool:
    """
    Website chính HUFLIT dùng WordPress-like slug, ví dụ:
    /vi/su-kien/huflit-bo-nhiem-hieu-truong.../
    /vi/tin-tuc/giai-ma-vu-khi.../

    Hàm này nhận diện các trang nội dung tiếng Việt trên huflit.edu.vn,
    đồng thời loại bớt trang kỹ thuật/search/feed để tránh nhiễu KB.
    """
    if "?" in url:
        return False

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.rstrip("/") + "/"

    if host not in {"huflit.edu.vn", "www.huflit.edu.vn"}:
        return False
    if not path.startswith("/vi/"):
        return False

    # Bỏ trang chủ và các URL kỹ thuật.
    if path in {"/vi/"}:
        return False
    if any(x in path.lower() for x in [
        "/wp-admin", "/wp-login", "/feed", "/tag/", "/author/",
        "/search", "/page/", "/?", "/en/",
    ]):
        return False
    if "sharepoint.com" in host:
        return False

    # Các trang danh sách/category vẫn cần follow để tìm bài, nhưng không nên parse như article.
    listing_paths = {
        "/vi/tin-tuc-va-su-kien/",
        "/vi/khoa/",
        "/vi/dao-tao/",
        "/vi/tuyen-sinh/",
        "/vi/ve-huflit/",
    }
    if path in listing_paths:
        return False

    # Chỉ nhận các path có slug thật, tránh link trống/menu ngắn.
    parts = [p for p in path.split("/") if p]
    return len(parts) >= 2


def is_article_url(url: str) -> bool:
    """Accept both portal /News/Detail URLs and main-site /vi/... content URLs."""
    return is_portal_article_url(url) or is_main_site_content_url(url)


def should_follow_crawl_url(url: str) -> bool:
    """
    Quyết định URL HTML nào được đưa vào queue crawl.
    Cho portal cũ: chỉ đi quanh /News.
    Cho site chính: đi trong /vi để lấy menu/listing/article.
    """
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path

    if get_file_ext_from_url(url) in FILE_EXTS:
        return False

    lower_path = path.lower()
    if any(x in lower_path for x in [
        "/public/tracuuthoikhoabieu", "/login", "/home",
        "/wp-admin", "/wp-login", "/feed", "/tag/", "/author/", "/search",
    ]):
        return False

    if host == "portal.huflit.edu.vn":
        return "/News" in path or url.rstrip("/") == PORTAL_BASE_URL.rstrip("/")

    if host in {"huflit.edu.vn", "www.huflit.edu.vn"}:
        return path.startswith("/vi")

    return False


def get_file_ext_from_url(url: str) -> str:
    path = urlparse(url).path
    ext = os.path.splitext(path)[1].lower()
    return ext

def extract_detail_id(url: str) -> Optional[int]:
    """
    Extract HUFLIT News detail id from URL.
    Example:
      /News/Detail/3860/... -> 3860

    If published_at is missing, detail_id is useful as a relative recency signal:
    larger detail_id usually means newer portal article.
    """
    if not url:
        return None
    m = re.search(r"/News/Detail/(\d+)", urlparse(url).path)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None



def slug_to_title(url: str) -> str:
    """
    Fallback title from /News/Detail/<id>/<slug>.
    This is only used when the page HTML does not expose a title cleanly.
    """
    try:
        path = unquote(urlparse(url).path)
        m = re.search(r"/News/Detail/\d+/(.+)$", path)
        if not m:
            return ""
        slug = m.group(1)
        slug = slug.replace("-", " ")
        slug = re.sub(r"\s+", " ", slug).strip()
        return slug
    except Exception:
        return ""


def is_nav_or_category_link(url: str, anchor: str = "") -> bool:
    """
    Skip sidebar/category/breadcrumb links so article links do not get polluted.
    """
    a = clean_text(anchor).lower()
    parsed = urlparse(url)
    path = parsed.path

    if "/News/Type/" in path:
        return True

    if any(x in path.lower() for x in ["/public/tracuuthoikhoabieu", "/login", "/home"]):
        return True

    nav_anchors = {
        "tin tức",
        "thông báo chung",
        "kế hoạch năm học",
        "quy định - quy chế",
        "thông báo phòng đào tạo đại học",
        "thông báo phòng chính trị - công tác sinh viên",
        "đánh giá kết quả rèn luyện sinh viên",
        "thông báo tuần sinh hoạt công dân - sinh viên",
        "thông báo bộ phận y tế",
        "học bổng - khen thưởng",
        "thông báo phòng khảo thí - bảo đảm chất lượng giáo dục",
        "thông báo phòng kế hoạch - tài chính",
        "hướng dẫn sử dụng",
        "các câu hỏi thường gặp",
        "kí hiệu phòng học và cơ sở",
        "mẫu đơn",
        "curriculums",
        "chương trình đào tạo",
        "đề án mở ngành",
        "course outlines",
        "đề cương chi tiết học phần",
        "cẩm nang - sổ tay sinh viên",
    }
    if a in nav_anchors:
        return True

    return False


def guess_link_type(url: str, anchor: str = "") -> str:
    ext = get_file_ext_from_url(url)
    if ext == ".pdf":
        return "pdf"
    if ext in {".doc", ".docx"}:
        return "word_document"
    if ext in {".xls", ".xlsx"}:
        return "spreadsheet"
    if ext in {".ppt", ".pptx"}:
        return "presentation"
    if "forms.office.com" in url.lower():
        return "form"
    if any(k in anchor.lower() for k in ["tải", "download", "file", "biểu mẫu", "mẫu đơn", "tại đây", "hướng dẫn"]):
        return "possible_attachment"
    return "external_link"


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")

    # Normalize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix common extraction spacing issues seen in HUFLIT pages/PDFs.
    replacements = {
        r"\bE\s+mail\b": "Email",
        r"\be\s+mail\b": "email",
        r"Đ\s+ối": "Đối",
        r"đ\s+ối": "đối",
        r"Phòng\s+Đ\s+ối": "Phòng Đối",
        r"\bT\s+ừ\b": "Từ",
        r"\bthe\s+o\b": "theo",
        r"\bqu\s+y\b": "quy",
        r"Ch\s+ươ\s+ng\s+tr\s+ì\s+nh": "Chương trình",
        r"chuy\s+ể\s+n\s+ti\s+ế\s+p": "chuyển tiếp",
        r"Sư\s+V\s+ạ\s+n\s+H\s+ạ\s+nh": "Sư Vạn Hạnh",
    }
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Fix broken years: 202 6 -> 2026
    text = re.sub(r"\b(20\d)\s+(\d)\b", r"\1\2", text)

    # Fix dates with broken spaces: 23 /0 6 /2025 -> 23/06/2025
    text = re.sub(r"(\d{1,2})\s*/\s*0\s*(\d)\s*/\s*(\d{4})", r"\1/0\2/\3", text)
    text = re.sub(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", r"\1/\2/\3", text)
    text = re.sub(r"(\d{1,2})/(\d{1,2})\s*/\s*(\d{4})", r"\1/\2/\3", text)
    text = re.sub(r"(\d{1,2})/(\d)\s*/\s*(\d{4})", r"\1/0\2/\3", text)
    text = re.sub(r"(\d{1,2})/(\d{1,2})\s*/\s*202\s+(\d)", r"\1/\2/202\3", text)

    # Spaces before punctuation.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)

    # Collapse excessive spaces but preserve paragraphs.
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def text_len(text: str) -> int:
    return len(clean_text(text))

def parse_vietnamese_date(text: str) -> Optional[str]:
    """
    Parse common Vietnamese date formats into YYYY-MM-DD.
    Supported examples:
      - 20/05/2025
      - 20-05-2025
      - ngày 20 tháng 05 năm 2025
      - 2025-05-20
    """
    if not text:
        return None

    text = clean_text(text)

    # ISO-like date.
    m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", text)
    if m:
        year, month, day = map(int, m.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    patterns = [
        r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b",
        r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue

        day, month, year = map(int, m.groups())
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return None

    return None


def extract_published_at(soup: BeautifulSoup, content_container: Optional[Tag] = None) -> Optional[str]:
    """
    Extract the article published date from HTML.

    Important:
    - published_at = date the portal article was posted.
    - scraped_at = date our crawler fetched the article.
    Do not use scraped_at to decide whether a notice is old/new.
    """

    # 1) Meta tags are the cleanest if available.
    meta_selectors = [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="article:published_time"]', "content"),
        ('meta[property="og:published_time"]', "content"),
        ('meta[name="pubdate"]', "content"),
        ('meta[name="date"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
    ]

    for selector, attr in meta_selectors:
        node = soup.select_one(selector)
        if node and node.get(attr):
            parsed = parse_vietnamese_date(node.get(attr, ""))
            if parsed:
                return parsed

    # 2) Common visible date elements.
    selectors = [
        "time[datetime]",
        "time",
        ".date",
        ".news-date",
        ".post-date",
        ".created-date",
        ".publish-date",
        ".published-date",
        ".datetime",
        ".article-date",
        ".time",
        ".date-post",
        ".post-time",
    ]

    for selector in selectors:
        for node in soup.select(selector):
            candidates = []
            if node.has_attr("datetime"):
                candidates.append(node.get("datetime", ""))
            candidates.append(node.get_text(" ", strip=True))

            for c in candidates:
                parsed = parse_vietnamese_date(c)
                if parsed:
                    return parsed

    # 3) Search only for date labels, not every date in article body.
    # This avoids accidentally using deadlines/event dates as published date.
    search_areas: List[Tag] = []
    if content_container is not None:
        # Look at parent area around the article title/content.
        if content_container.parent and isinstance(content_container.parent, Tag):
            search_areas.append(content_container.parent)
        search_areas.append(content_container)
    if soup.body:
        search_areas.append(soup.body)

    label_patterns = [
        r"(?:ngày\s*đăng|đăng\s*ngày|ngày\s*cập\s*nhật|cập\s*nhật|thời\s*gian\s*đăng|ngày|date)\s*:?\s*.{0,60}?\d{1,2}[/-]\d{1,2}[/-]\d{4}",
        r"(?:ngày\s*đăng|đăng\s*ngày|ngày\s*cập\s*nhật|cập\s*nhật|thời\s*gian\s*đăng|ngày|date)\s*:?\s*.{0,80}?ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}",
    ]

    for area in search_areas:
        raw = clean_text(area.get_text(" ", strip=True))
        for pattern in label_patterns:
            m = re.search(pattern, raw, flags=re.IGNORECASE)
            if m:
                parsed = parse_vietnamese_date(m.group(0))
                if parsed:
                    return parsed

    return None



def make_session(timeout: int = 30) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    })
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch(session: requests.Session, url: str, timeout: int = 30, max_size: int = 50 * 1024 * 1024) -> requests.Response:
    resp = session.get(url, timeout=timeout, verify=False, stream=True)
    resp.raise_for_status()
    
    content = b""
    for chunk in resp.iter_content(chunk_size=8192):
        if chunk:
            content += chunk
        if len(content) > max_size:
            resp.close()
            raise ValueError(f"File too large, exceeding {max_size // (1024*1024)}MB limit: {url}")
            
    resp._content = content
    resp._content_consumed = True
    return resp


def safe_filename(name: str, max_len: int = 160) -> str:
    name = clean_text(name)
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > max_len:
        root, ext = os.path.splitext(name)
        name = root[: max_len - len(ext) - 1] + ext
    return name or "file"


# ----------------------------
# HTML parsing
# ----------------------------

def remove_noise(soup: BeautifulSoup) -> None:
    for selector in [
        "script", "style", "noscript", "iframe",
        "header", "footer", "nav",
        ".navbar", ".menu", ".sidebar", ".breadcrumb",
        ".footer", ".header", ".social", ".share",
        ".pagination", ".ads", ".advertisement",
    ]:
        for node in soup.select(selector):
            node.decompose()



def get_visible_text_lines(root: Tag) -> List[str]:
    """
    Convert visible text into clean non-empty lines, preserving inline tags
    and formatting table rows as pipe-separated columns.
    """
    block_tags = {
        "p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
        "table", "pre", "blockquote", "section", "article", "br",
        "header", "footer", "nav", "aside"
    }
    
    parts = []
    
    def traverse(node):
        if isinstance(node, str):
            clean_node = node.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
            parts.append(clean_node)
        elif isinstance(node, Tag):
            name = node.name.lower()
            if name == "tr":
                cells = [clean_text(c.get_text(" ", strip=True)) for c in node.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    txt = " | ".join(cells)
                    parts.append("\n" + txt + "\n")
                return
            
            is_block = name in block_tags
            if is_block:
                parts.append("\n")
            for child in node.children:
                traverse(child)
            if is_block:
                parts.append("\n")

    traverse(root)
    raw = "".join(parts)
    
    lines: List[str] = []
    prev = None

    for line in raw.split("\n"):
        line = clean_text(line)
        if not line:
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line

    return lines


def is_probably_menu_line(line: str) -> bool:
    """
    Filter common portal menu/category lines.
    """
    l = clean_text(line).lower()
    menu_lines = {
        "tin tức",
        "thông báo chung",
        "kế hoạch năm học",
        "quy định - quy chế",
        "thông báo phòng đào tạo đại học",
        "thông báo phòng chính trị - công tác sinh viên",
        "đánh giá kết quả rèn luyện sinh viên",
        "thông báo tuần sinh hoạt công dân - sinh viên",
        "thông báo bộ phận y tế",
        "học bổng - khen thưởng",
        "thông báo phòng khảo thí - bảo đảm chất lượng giáo dục",
        "thông báo phòng kế hoạch - tài chính",
        "hướng dẫn sử dụng",
        "các câu hỏi thường gặp",
        "kí hiệu phòng học và cơ sở",
        "mẫu đơn",
        "curriculums",
        "chương trình đào tạo",
        "đề án mở ngành",
        "course outlines",
        "đề cương chi tiết học phần",
        "cẩm nang - sổ tay sinh viên",
    }
    if l in menu_lines:
        return True
    if l.startswith("https://portal.huflit.edu.vn/news/type/"):
        return True
    return False


def is_footer_or_after_article_line(line: str) -> bool:
    """
    Stop markers after the article body.
    These are intentionally conservative; they only catch obvious footer/layout text.
    """
    l = clean_text(line).lower()
    stop_prefixes = [
        "trụ sở",
        "cơ sở",
        "điện thoại:",
        "email:",
        "copyright",
        "©",
        "bản quyền",
        "huflit portal",
        "trường đại học ngoại ngữ - tin học thành phố hồ chí minh",
    ]
    stop_exact = {
        "thông báo",
        "×",
        "x",
        "đóng",
        "ok",
        "xem thêm",
        "tin liên quan",
        "bài viết liên quan",
    }
    if l in stop_exact:
        return True
    if any(l.startswith(p) for p in stop_prefixes):
        return True
    return False


def extract_article_by_news_marker(soup: BeautifulSoup) -> Tuple[str, List[str]]:
    """
    HUFLIT detail pages often render the article as plain text:

      Tin tức - thông báo
      <ARTICLE TITLE>
      <ARTICLE BODY...>

    This function extracts:
    - title = first meaningful line right after "Tin tức - thông báo"
    - paragraphs = lines after title until footer/layout markers

    It is used as a high-priority parser because generic content-container scoring
    can accidentally include sidebar/category links.
    """
    body = soup.body or soup
    lines = get_visible_text_lines(body)

    marker_idx = None
    for i, line in enumerate(lines):
        if clean_text(line).lower() == "tin tức - thông báo":
            marker_idx = i
            break

    if marker_idx is None:
        return "", []

    # Title = first meaningful line after marker.
    title = ""
    title_idx = None
    for j in range(marker_idx + 1, len(lines)):
        candidate = clean_text(lines[j])
        if not candidate:
            continue
        if is_probably_menu_line(candidate):
            continue
        if is_footer_or_after_article_line(candidate):
            continue
        # Avoid a raw URL becoming title.
        if candidate.startswith("http"):
            continue
        title = candidate
        title_idx = j
        break

    if not title or title_idx is None:
        return "", []

    paragraphs: List[str] = []
    for line in lines[title_idx + 1:]:
        line = clean_text(line)
        if not line:
            continue
        if is_footer_or_after_article_line(line):
            break
        if is_probably_menu_line(line):
            continue
        # Avoid accidentally re-adding breadcrumb title.
        if line == title:
            continue
        paragraphs.append(line)

    # Remove tiny junk-only results.
    if len("\n".join(paragraphs)) < 30:
        return title, []

    return title, paragraphs

def get_title(soup: BeautifulSoup, page_url: str = "") -> str:
    candidates = []

    # Meta tags often contain the real page title when visible title is hard to select.
    for selector, attr in [
        ('meta[property="og:title"]', "content"),
        ('meta[name="twitter:title"]', "content"),
        ('meta[name="title"]', "content"),
    ]:
        node = soup.select_one(selector)
        if node and node.get(attr):
            txt = clean_text(node.get(attr, ""))
            if txt:
                candidates.append(txt)

    for sel in [
        "h1", ".news-title", ".title", ".post-title", ".article-title",
        ".detail-title", ".page-title", "h2", "title"
    ]:
        for node in soup.select(sel):
            txt = clean_text(node.get_text(" ", strip=True))
            if txt:
                candidates.append(txt)

    bad = {"huflit", "portal", "tin tức", "thông báo", "news"}
    clean_candidates = []
    for t in candidates:
        tl = t.strip().lower()
        if not t or tl in bad:
            continue
        t = re.sub(r"\s*[-|]\s*HUFLIT\s*$", "", t, flags=re.IGNORECASE).strip()
        clean_candidates.append(t)

    for t in clean_candidates:
        if len(t) > 10:
            return t

    marker_title, _marker_paragraphs = extract_article_by_news_marker(soup)
    if marker_title:
        return marker_title

    fallback = slug_to_title(page_url)
    if fallback:
        return fallback

    return clean_candidates[0] if clean_candidates else ""


def score_content_node(node: Tag) -> int:
    text = clean_text(node.get_text("\n", strip=True))
    if not text:
        return 0

    # Prefer nodes with many article-like blocks and moderate links.
    p_count = len(node.find_all(["p", "li"]))
    text_score = min(len(text), 20000)
    link_count = len(node.find_all("a"))
    table_count = len(node.find_all("table"))

    score = text_score + p_count * 150 + table_count * 500

    # Penalize navigation-like nodes.
    if link_count > 80 and p_count < 20:
        score -= 5000

    cls = " ".join(node.get("class", [])) if node.has_attr("class") else ""
    node_id = node.get("id", "")
    marker = f"{cls} {node_id}".lower()

    if any(k in marker for k in ["detail", "content", "article", "news", "post"]):
        score += 3000
    if any(k in marker for k in ["menu", "nav", "sidebar", "footer", "header"]):
        score -= 8000

    return score


def find_content_container(soup: BeautifulSoup) -> Tag:
    # Strong selectors first.
    selectors = [
        ".elementor-widget-theme-post-content",
        ".elementor-text-editor",
        ".post_wrapper",
        ".single-content",
        ".page-content",
        "article",
        ".news-detail",
        ".news_detail",
        ".detail-news",
        ".news-content",
        ".content-news",
        ".article-content",
        ".post-content",
        ".entry-content",
        ".detail-content",
        ".content-detail",
        "main",
        "#content",
        ".content",
    ]
    for sel in selectors:
        nodes = soup.select(sel)
        if nodes:
            nodes = sorted(nodes, key=score_content_node, reverse=True)
            if score_content_node(nodes[0]) > 500:
                return nodes[0]

    # Fallback: best div in body.
    body = soup.body or soup
    divs = body.find_all(["div", "section"], recursive=True)
    if divs:
        best = max(divs, key=score_content_node)
        if score_content_node(best) > 500:
            return best

    return body


def extract_blocks(container: Tag) -> List[str]:
    lines = get_visible_text_lines(container)
    blocks = []
    for line in lines:
        if len(line) < 2:
            continue
        if line not in blocks:
            blocks.append(line)
    return blocks



PRIORITY_LINK_PHRASES = [
    "tại đây",
    "xem tại đây",
    "vui lòng xem",
    "xem hướng dẫn",
    "hướng dẫn tại đây",
    "tải file",
    "tải về",
    "download",
    "biểu mẫu",
    "mẫu đơn",
    "link đăng ký",
    "link khảo sát",
]


def has_priority_link_phrase(text: str) -> bool:
    text = clean_text(text).lower()
    return any(p in text for p in PRIORITY_LINK_PHRASES)


def attr_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value)
    return str(value)


def extract_url_candidates_from_value(value: str, page_url: str) -> List[str]:
    """
    Extract URLs from href/data attributes/onclick JavaScript.

    Handles:
      - https://...
      - /Upload/File.pdf
      - ../Files/abc.docx
      - href values directly
    """
    value = attr_to_text(value).strip()
    if not value:
        return []

    candidates: List[str] = []

    # Direct href-like value.
    if not value.lower().startswith(("javascript:", "void(", "#")):
        if value.startswith(("http://", "https://", "mailto:", "/")) or re.search(r"\.(pdf|docx?|xlsx?|pptx?|zip|rar)(?:$|[?#])", value, flags=re.I):
            candidates.append(value)

    # URLs inside JavaScript or attributes.
    for m in re.finditer(r"""https?://[^\s'"<>),]+""", value, flags=re.I):
        candidates.append(m.group(0))

    # Relative file URLs inside JavaScript.
    for m in re.finditer(r"""(?:\.\./|/)?[A-Za-z0-9_\-/%().\[\]À-ỹ ]+\.(?:pdf|docx?|xlsx?|pptx?|zip|rar)(?:\?[^\s'"<>),]+)?""", value, flags=re.I):
        candidates.append(m.group(0).strip())

    normalized: List[str] = []
    seen: Set[str] = set()

    for c in candidates:
        c = c.strip().strip("'\"")
        if not c:
            continue
        u = normalize_url(c, page_url)
        if u not in seen:
            seen.add(u)
            normalized.append(u)

    return normalized


def extract_url_candidates_from_tag(tag: Tag, page_url: str) -> List[str]:
    attrs_to_check = [
        "href", "src",
        "data-href", "data-url", "data-link", "data-file", "data-src",
        "data-original", "data-original-url", "data-download-url",
        "onclick", "ng-click",
    ]

    candidates: List[str] = []

    for attr in attrs_to_check:
        if tag.has_attr(attr):
            candidates.extend(extract_url_candidates_from_value(tag.get(attr), page_url))

    # Sometimes links are placed in arbitrary data-* attributes.
    for attr, value in tag.attrs.items():
        if attr.startswith("data-") and attr not in attrs_to_check:
            candidates.extend(extract_url_candidates_from_value(value, page_url))

    # Deduplicate.
    result: List[str] = []
    seen: Set[str] = set()
    for u in candidates:
        if u not in seen:
            seen.add(u)
            result.append(u)

    return result


def is_valid_article_link(url: str, anchor: str, page_url: str) -> bool:
    if not url:
        return False

    parsed = urlparse(url)
    if parsed.scheme and parsed.scheme not in {"http", "https", "mailto"}:
        return False

    if is_nav_or_category_link(url, anchor):
        return False

    # Skip self/breadcrumb links.
    if normalize_url(url) == normalize_url(page_url):
        return False

    return True


def merge_links(links: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    for link in links:
        anchor = clean_text(link.get("anchor_text", ""))
        url = normalize_url(link.get("url", ""))
        key = (anchor.lower(), url)
        if key in seen:
            continue
        seen.add(key)

        link["url"] = url
        link["anchor_text"] = anchor or url
        link["link_id"] = f"link_{link.get('doc_id')}_{len(merged)+1:03d}"
        merged.append(link)

    return merged


def extract_priority_links_from_soup(
    soup: BeautifulSoup,
    page_url: str,
    doc_id: str,
    article_text: str = "",
) -> List[Dict[str, Any]]:
    """
    Fallback extractor for cases like:
      <<vui lòng xem hướng dẫn tại đây>>

    Some HUFLIT pages put the actual URL in a global <a>, data-* attribute,
    or onclick handler outside the selected content container. This scans the
    whole page, but keeps only links whose anchor/context looks article-related.
    """
    article_norm = clean_text(article_text).lower()
    links: List[Dict[str, Any]] = []

    # Scan tags that are likely to carry links.
    for tag in soup.find_all(["a", "button", "span", "div"]):
        anchor = clean_text(tag.get_text(" ", strip=True))
        parent = tag.find_parent(["p", "li", "tr", "div"])
        context = clean_text(parent.get_text(" ", strip=True)) if parent else anchor

        urls = extract_url_candidates_from_tag(tag, page_url)
        if not urls:
            continue

        phrase_hit = has_priority_link_phrase(anchor) or has_priority_link_phrase(context)
        context_in_article = bool(article_norm and clean_text(context).lower() in article_norm)
        anchor_in_article = bool(article_norm and anchor and anchor.lower() in article_norm)

        for u in urls:
            if not is_valid_article_link(u, anchor, page_url):
                continue

            link_type = guess_link_type(u, anchor)
            useful_by_type = link_type in {"pdf", "word_document", "spreadsheet", "presentation", "form", "possible_attachment"}

            # Keep if it clearly belongs to article text, or it is a useful file/form/link phrase.
            if not (phrase_hit or context_in_article or anchor_in_article or useful_by_type):
                continue

            links.append({
                "link_id": f"link_{doc_id}_{len(links)+1:03d}",
                "doc_id": doc_id,
                "anchor_text": anchor or u,
                "url": u,
                "context": context or anchor or u,
                "link_type": link_type,
            })

    return merge_links(links)



def extract_links(container: Tag, page_url: str, doc_id: str, article_text: str = "") -> List[Dict[str, Any]]:
    links: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()

    article_text_norm = clean_text(article_text).lower()

    for i, a in enumerate(container.select("a[href]"), start=1):
        href = a.get("href", "").strip()
        if not href:
            continue

        abs_url = normalize_url(href, page_url)
        anchor = clean_text(a.get_text(" ", strip=True)) or abs_url

        if not is_valid_article_link(abs_url, anchor, page_url):
            continue

        parent = a.find_parent(["p", "li", "tr", "div"])
        context = clean_text(parent.get_text(" ", strip=True)) if parent else anchor

        # If a link lives outside the real article body, remove it.
        if article_text_norm:
            c_norm = clean_text(context).lower()
            a_norm = anchor.lower()
            u_norm = abs_url.lower()
            if c_norm and c_norm not in article_text_norm and a_norm not in article_text_norm and u_norm not in article_text_norm:
                if guess_link_type(abs_url, anchor) not in {"pdf", "word_document", "spreadsheet", "presentation", "form"}:
                    continue

        key = (anchor.lower(), abs_url)
        if key in seen:
            continue
        seen.add(key)

        links.append({
            "link_id": f"link_{doc_id}_{len(links)+1:03d}",
            "doc_id": doc_id,
            "anchor_text": anchor,
            "url": abs_url,
            "context": context,
            "link_type": guess_link_type(abs_url, anchor),
        })

    return merge_links(links)


def is_possible_pdf_url(url: str, link_type: str = "") -> bool:
    if get_file_ext_from_url(url) == ".pdf":
        return True
    if link_type == "pdf":
        return True
    return False


def categorize(title: str, text: str, url: str = "") -> str:
    s = f"{title}\n{text}\n{url}".lower()

    rules = [
        ("bhyt_kcb", ["bhyt", "bảo hiểm y tế", "khám chữa bệnh", "kcb"]),
        ("hoc_phi", ["học phí", "gia hạn đóng học phí", "miễn giảm học phí"]),
        ("diem_ren_luyen", ["điểm rèn luyện", "đrl", "rèn luyện"]),
        ("trao_doi_sinh_vien", ["trao đổi sinh viên", "chuyển tiếp", "học kỳ thu", "đại học youngsan", "kobe", "wonkwang"]),
        ("quy_che_quy_dinh", ["quy chế", "quy định", "quyết định", "công tác sinh viên"]),
        ("mau_don_bieu_mau", ["mẫu đơn", "biểu mẫu", "tải file", "xin hoãn thi"]),
        ("co_so_hoc_tap", ["hóc môn", "sư vạn hạnh", "cơ sở"]),
        ("hoc_bong", ["học bổng"]),
    ]

    for category, keys in rules:
        if any(k in s for k in keys):
            return category

    return "general"


def extract_deadlines(text: str) -> List[str]:
    # Return raw date/time mentions for later normalization.
    patterns = [
        r"(?:trước|đến|hết|từ ngày ra thông báo đến)\s+[^.\n]{0,80}?\d{1,2}[/-]\d{1,2}[/-]\d{4}",
        r"\d{1,2}g\d{0,2}\s+ngày\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}",
        r"\d{1,2}:\d{2},?\s+ngày\s+\d{1,2}[/-]\d{1,2}[/-]\d{4}",
        r"\d{1,2}[/-]\d{1,2}[/-]\d{4}",
    ]
    found: List[str] = []
    for p in patterns:
        for m in re.finditer(p, text, flags=re.IGNORECASE):
            raw = clean_text(m.group(0))
            if raw not in found:
                found.append(raw)
    return found[:10]


def suspected_date_error(title: str, text: str) -> bool:
    # Simple heuristic: title/text mentions 2025 but important deadline in 2024.
    s = f"{title}\n{text}"
    if "2025" in s and re.search(r"(hạn|trước|nộp hồ sơ|đăng ký)[^.\n]{0,100}2024", s, flags=re.IGNORECASE):
        return True
    return False


# ----------------------------
# PDF extraction
# ----------------------------

def download_file(session: requests.Session, url: str, out_dir: Path, timeout: int = 60) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    resp = session.get(url, timeout=timeout, stream=True, verify=False)
    resp.raise_for_status()

    # Filename from content disposition or URL.
    cd = resp.headers.get("content-disposition", "")
    filename = ""
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', cd, flags=re.IGNORECASE)
    if m:
        filename = unquote(m.group(1))
    if not filename:
        filename = os.path.basename(urlparse(url).path) or f"download_{short_hash(url)}"

    filename = safe_filename(filename)
    path = out_dir / filename

    # Avoid overwrite collision.
    if path.exists():
        root, ext = os.path.splitext(filename)
        path = out_dir / f"{root}_{short_hash(url, 6)}{ext}"

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            if chunk:
                f.write(chunk)

    return path


def extract_pdf_text(pdf_path: Path) -> Dict[str, Any]:
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed. Run: pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    pages: List[Dict[str, Any]] = []
    all_text_parts: List[str] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        text = clean_text(page.get_text("text") or "")
        pages.append({
            "page": page_index + 1,
            "text": text,
            "char_count": len(text),
        })
        if text:
            all_text_parts.append(text)

    full_text = clean_text("\n\n".join(all_text_parts))

    return {
        "pages": len(doc),
        "full_text": full_text,
        "paragraphs": [p for p in full_text.split("\n") if p.strip()],
        "page_texts": pages,
        "extraction": {
            "method": "pymupdf",
            "success": bool(full_text.strip()),
            "is_empty_text": not bool(full_text.strip()),
            "table_preserved": False,
        }
    }


def extract_docx_text(file_path: Path) -> Dict[str, Any]:
    """
    Extract text from .docx using only stdlib zip/xml.
    """
    texts: List[str] = []
    xml_files = [
        "word/document.xml",
        "word/footnotes.xml",
        "word/endnotes.xml",
    ]

    with zipfile.ZipFile(file_path) as z:
        for xml_name in xml_files:
            if xml_name not in z.namelist():
                continue
            root = ET.fromstring(z.read(xml_name))
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    texts.append(node.text)

    full_text = clean_text("\n".join(texts))
    return {
        "pages": None,
        "full_text": full_text,
        "paragraphs": [p for p in full_text.split("\n") if p.strip()],
        "page_texts": [],
        "extraction": {
            "method": "stdlib_docx_zip_xml",
            "success": bool(full_text.strip()),
            "is_empty_text": not bool(full_text.strip()),
            "table_preserved": False,
        }
    }


def extract_pptx_text(file_path: Path) -> Dict[str, Any]:
    """
    Extract text from .pptx using only stdlib zip/xml.
    """
    slides: List[Dict[str, Any]] = []
    all_text: List[str] = []

    with zipfile.ZipFile(file_path) as z:
        slide_files = sorted(
            [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)]
        )

        for idx, slide_name in enumerate(slide_files, start=1):
            root = ET.fromstring(z.read(slide_name))
            texts = []
            for node in root.iter():
                if node.tag.endswith("}t") and node.text:
                    texts.append(node.text)
            slide_text = clean_text("\n".join(texts))
            slides.append({
                "page": idx,
                "text": slide_text,
                "char_count": len(slide_text),
            })
            if slide_text:
                all_text.append(slide_text)

    full_text = clean_text("\n\n".join(all_text))
    return {
        "pages": len(slides),
        "full_text": full_text,
        "paragraphs": [p for p in full_text.split("\n") if p.strip()],
        "page_texts": slides,
        "extraction": {
            "method": "stdlib_pptx_zip_xml",
            "success": bool(full_text.strip()),
            "is_empty_text": not bool(full_text.strip()),
            "table_preserved": False,
        }
    }


def extract_xlsx_text(file_path: Path, max_rows_per_sheet: int = 5000) -> Dict[str, Any]:
    """
    Extract text from .xlsx using stdlib zip/xml.
    This does not preserve formulas/styles, but is enough for RAG text extraction.
    """
    def cell_ref_to_col(cell_ref: str) -> int:
        m = re.match(r"([A-Z]+)", cell_ref or "")
        if not m:
            return 0
        col = 0
        for ch in m.group(1):
            col = col * 26 + (ord(ch) - ord("A") + 1)
        return col

    with zipfile.ZipFile(file_path) as z:
        names = z.namelist()

        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter():
                if si.tag.endswith("}si"):
                    parts = []
                    for node in si.iter():
                        if node.tag.endswith("}t") and node.text:
                            parts.append(node.text)
                    shared_strings.append("".join(parts))

        sheet_files = sorted([n for n in names if re.match(r"xl/worksheets/sheet\d+\.xml$", n)])
        sheets: List[Dict[str, Any]] = []
        all_text: List[str] = []

        for sheet_index, sheet_name in enumerate(sheet_files, start=1):
            root = ET.fromstring(z.read(sheet_name))
            rows_text: List[str] = []
            row_count = 0

            for row in root.iter():
                if not row.tag.endswith("}row"):
                    continue
                row_count += 1
                if row_count > max_rows_per_sheet:
                    break

                cells = []
                for c in row:
                    if not c.tag.endswith("}c"):
                        continue

                    cell_type = c.attrib.get("t")
                    value = ""
                    v = None
                    inline_texts = []

                    for child in c.iter():
                        if child.tag.endswith("}v") and child.text is not None:
                            v = child.text
                        elif child.tag.endswith("}t") and child.text:
                            inline_texts.append(child.text)

                    if cell_type == "s" and v is not None:
                        try:
                            value = shared_strings[int(v)]
                        except Exception:
                            value = v
                    elif inline_texts:
                        value = "".join(inline_texts)
                    elif v is not None:
                        value = v

                    value = clean_text(value)
                    if value:
                        cells.append(value)

                if cells:
                    rows_text.append(" | ".join(cells))

            sheet_text = clean_text("\n".join(rows_text))
            sheets.append({
                "page": sheet_index,
                "text": sheet_text,
                "char_count": len(sheet_text),
            })
            if sheet_text:
                all_text.append(sheet_text)

    full_text = clean_text("\n\n".join(all_text))
    return {
        "pages": len(sheets),
        "full_text": full_text,
        "paragraphs": [p for p in full_text.split("\n") if p.strip()],
        "page_texts": sheets,
        "extraction": {
            "method": "stdlib_xlsx_zip_xml",
            "success": bool(full_text.strip()),
            "is_empty_text": not bool(full_text.strip()),
            "table_preserved": False,
        }
    }


def extract_html_text(html: str, page_url: str = "") -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    remove_noise(soup)

    marker_title, marker_paragraphs = extract_article_by_news_marker(soup)
    if marker_paragraphs:
        paragraphs = marker_paragraphs
    else:
        container = find_content_container(soup)
        paragraphs = extract_blocks(container)

    full_text = clean_text("\n".join(paragraphs))
    return {
        "pages": None,
        "title": marker_title or get_title(soup, page_url),
        "full_text": full_text,
        "paragraphs": paragraphs,
        "page_texts": [],
        "extraction": {
            "method": "beautifulsoup_html",
            "success": bool(full_text.strip()),
            "is_empty_text": not bool(full_text.strip()),
            "table_preserved": False,
        }
    }


def extract_download_filename(resp: requests.Response, url: str, fallback: str = "download") -> str:
    cd = resp.headers.get("content-disposition", "")
    filename = ""

    m = re.search(r"filename\*=(?:UTF-8'')?([^;]+)", cd, flags=re.IGNORECASE)
    if m:
        filename = unquote(m.group(1).strip().strip('"'))

    if not filename:
        m = re.search(r'filename="?([^";]+)"?', cd, flags=re.IGNORECASE)
        if m:
            filename = unquote(m.group(1).strip().strip('"'))

    if not filename:
        filename = os.path.basename(urlparse(resp.url or url).path)

    if not filename:
        filename = fallback

    ctype = resp.headers.get("content-type", "").lower().split(";")[0].strip()
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        guessed = mimetypes.guess_extension(ctype) or ""
        if ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            guessed = ".docx"
        elif ctype == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            guessed = ".xlsx"
        elif ctype == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            guessed = ".pptx"
        if guessed:
            filename += guessed

    return safe_filename(filename)


def extract_file_by_extension(file_path: Path) -> Dict[str, Any]:
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        return extract_pdf_text(file_path)
    if ext == ".docx":
        return extract_docx_text(file_path)
    if ext == ".xlsx":
        return extract_xlsx_text(file_path)
    if ext == ".pptx":
        return extract_pptx_text(file_path)

    raise RuntimeError(f"Unsupported file type for text extraction: {ext}")


def is_processable_resource_link(link: Dict[str, Any]) -> bool:
    """
    Return True if crawler should open the link and attempt to extract data.
    We intentionally skip forms because extracting submitted form content is not possible.
    """
    url = link.get("url", "")
    link_type = link.get("link_type", "")
    ext = get_file_ext_from_url(url)

    if link_type == "form":
        return False

    if ext in {".pdf", ".docx", ".xlsx", ".pptx"}:
        return True

    if link_type in {"pdf", "word_document", "spreadsheet", "presentation", "possible_attachment"}:
        return True

    return False


def build_resource_candidate_urls(url: str) -> List[str]:
    """
    Return ordered URLs to try when opening a linked resource.
    """
    return [url]


def cookies_to_requests_session(cookies: List[Dict[str, Any]], base_session: Optional[requests.Session] = None) -> requests.Session:
    """
    Convert Playwright cookies to a requests.Session.
    """
    session = base_session or make_session()
    for c in cookies:
        try:
            session.cookies.set(
                name=c.get("name"),
                value=c.get("value"),
                domain=c.get("domain"),
                path=c.get("path", "/"),
            )
        except Exception:
            continue
    return session


def response_looks_like_file(resp: requests.Response) -> bool:
    """
    Detect whether a response is an actual downloadable file, not an Office viewer page.
    """
    ctype = resp.headers.get("content-type", "").lower()
    cd = resp.headers.get("content-disposition", "").lower()
    url_ext = get_file_ext_from_url(resp.url)

    if resp.content[:4] == b"%PDF":
        return True
    if "attachment" in cd or "filename" in cd:
        return True
    if url_ext in {".pdf", ".docx", ".xlsx", ".pptx"}:
        return True
    if any(x in ctype for x in [
        "application/pdf",
        "wordprocessingml",
        "spreadsheetml",
        "presentationml",
        "application/vnd.ms-excel",
        "application/msword",
    ]):
        return True

    return False


def html_looks_like_viewer_or_login(resp: requests.Response) -> bool:
    """
    Detect Office login/viewer HTML.
    """
    ctype = resp.headers.get("content-type", "").lower()
    if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
        return False

    text = (resp.text or "").lower()
    markers = [
        "sign in",
        "đăng nhập",
        "login",
        "microsoftonline",
        "access denied",
        "requires javascript",
        "onedrive.aspx",
        "_layouts/15",
       
    ]
    return any(m in text for m in markers)






# ----------------------------
# Structured records
# ----------------------------

def build_form_records(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    title = doc.get("title", "")
    text = doc.get("full_text", "")
    links = doc.get("links", [])
    if "mẫu đơn" not in f"{title} {text}".lower() and "tải file" not in text.lower():
        return []

    records: List[Dict[str, Any]] = []
    lines = [clean_text(x) for x in re.split(r"[\n;]+", text) if clean_text(x)]

    form_lines = [
        line for line in lines
        if "mẫu đơn" in line.lower() or "tải file" in line.lower() or "đơn " in line.lower()
    ]
    if not form_lines:
        form_lines = [title]

    # Candidate download links.
    download_links = [
        l for l in links
        if l.get("link_type") in {"pdf", "word_document", "possible_attachment"}
        or any(k in l.get("anchor_text", "").lower() for k in ["tải", "file", "tại đây"])
    ]

    for i, line in enumerate(form_lines, start=1):
        # Clean "[1]. Mẫu đơn xin hoãn thi << tải file tại đây >>"
        form_name = re.sub(r"^\s*\[\d+\]\.?\s*", "", line).strip()
        form_name = re.sub(r"<<.*?>>", "", form_name).strip()
        if not form_name:
            form_name = title

        link = download_links[i - 1] if i - 1 < len(download_links) else (download_links[0] if len(download_links) == 1 else None)

        records.append({
            "form_id": f"form_{doc['doc_id']}_{i:03d}",
            "doc_id": doc["doc_id"],
            "form_name": form_name,
            "department": infer_department(title, text),
            "source_article_url": doc.get("source_url"),
            "download_url": link.get("url") if link else None,
            "link_anchor_text": link.get("anchor_text") if link else None,
            "file_type": get_file_ext_from_url(link.get("url", "")) if link else None,
            "status": {
                "has_download_url": bool(link),
                "error": None if link else "Missing href/download URL for form anchor",
            }
        })

    return records


def infer_department(title: str, text: str) -> Optional[str]:
    s = f"{title}\n{text}"
    departments = [
        "Phòng Khảo thí - Bảo đảm chất lượng giáo dục",
        "Phòng Chính trị - Công tác sinh viên",
        "Phòng Đối ngoại",
        "Phòng Đào tạo",
        "Phòng Tài chính",
    ]
    for d in departments:
        if d.lower() in s.lower():
            return d
    return None


def parse_hospital_kcb_records(pdf_doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Best-effort parse for BHYT KCB tables.
    Because PDF table extraction often loses columns, this parser stores each detected row
    as record_text plus best-effort fields: stt, ten_benh_vien, ma_benh_vien, tuyen.
    For production, use pdfplumber/Camelot manual parsing per PDF layout.
    """
    title = pdf_doc.get("title", "")
    text = pdf_doc.get("full_text", "")
    if not re.search(r"\bKCB\b|khám chữa bệnh|BHYT|Mã Bệnh", f"{title}\n{text}", flags=re.IGNORECASE):
        return []

    lines = [clean_text(x) for x in text.split("\n") if clean_text(x)]
    rows: List[str] = []
    current: List[str] = []

    row_start = re.compile(r"^\s*(\d{1,3})\s+(.+?)\s+(79-\d{3})\s+", flags=re.IGNORECASE)

    for line in lines:
        if row_start.search(line):
            if current:
                rows.append(clean_text(" ".join(current)))
            current = [line]
        else:
            if current:
                current.append(line)

    if current:
        rows.append(clean_text(" ".join(current)))

    records: List[Dict[str, Any]] = []
    for row in rows:
        m = re.search(r"^\s*(\d{1,3})\s+(.+?)\s+(79-\d{3})\s+(Trung ương|Tỉnh|Quận,\s*Huyện)", row, flags=re.IGNORECASE)
        if not m:
            continue

        stt = int(m.group(1))
        ten = clean_text(m.group(2))
        ma = m.group(3)
        tuyen = clean_text(m.group(4))

        records.append({
            "record_id": f"kcb_{pdf_doc['doc_id']}_{ma.replace('-', '_')}",
            "doc_id": pdf_doc["doc_id"],
            "source_file": pdf_doc.get("file_name"),
            "source_url": pdf_doc.get("source_url"),
            "page": None,
            "stt": stt,
            "ten_benh_vien": ten,
            "ma_benh_vien": ma,
            "tuyen": tuyen,
            "dang_ky_kcb_ban_dau": None,
            "dia_chi": None,
            "ghi_chu": None,
            "record_text": row,
            "parse_quality": "partial_from_flat_text"
        })

    return records


# ----------------------------
# Chunking
# ----------------------------

def split_text_to_chunks(
    text: str,
    max_chars: int = 2800,
    overlap_chars: int = 250,
) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for p in paragraphs:
        if len(current) + len(p) + 1 <= max_chars:
            current = f"{current}\n{p}".strip()
        else:
            if current:
                chunks.append(current)
            if len(p) > max_chars:
                # Hard split long paragraph.
                start = 0
                while start < len(p):
                    end = start + max_chars
                    chunks.append(p[start:end].strip())
                    start = max(0, end - overlap_chars)
                    if start >= len(p):
                        break
                current = ""
            else:
                current = p

    if current:
        chunks.append(current)

    # Add overlap between adjacent chunks if useful.
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = []
        prev_tail = ""
        for ch in chunks:
            if prev_tail:
                merged = f"{prev_tail}\n{ch}"
            else:
                merged = ch
            overlapped.append(merged.strip())
            prev_tail = ch[-overlap_chars:]
        chunks = overlapped

    return [c for c in chunks if len(c.strip()) >= 40]


def build_rag_chunks(source_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    for doc in source_documents:
        status = doc.get("status", {})
        if status.get("is_empty") or status.get("is_low_quality"):
            continue

        text = doc.get("full_text", "")
        if not text:
            continue

        # Do not embed raw table PDF as one huge table only; still create chunks but flag them.
        raw_chunks = split_text_to_chunks(text)

        for idx, chunk_text in enumerate(raw_chunks, start=1):
            chunk_id = f"{doc['doc_id']}_chunk_{idx:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc["doc_id"],
                "doc_type": doc.get("doc_type"),
                "category": doc.get("category"),
                "title": doc.get("title"),
                "source_url": doc.get("source_url"),
                "file_url": doc.get("file_url"),
                "published_at": doc.get("published_at"),
                "scraped_at": doc.get("scraped_at"),
                "first_seen_at": doc.get("first_seen_at"),
                "last_seen_at": doc.get("last_seen_at"),
                "source_order": doc.get("source_order"),
                "detail_id": (doc.get("source_order") or {}).get("detail_id"),
                "time_confidence": doc.get("time_confidence"),
                "page": None,
                "section": None,
                "chunk_index": idx,
                "chunk_text": chunk_text,
                "keywords": extract_keywords(doc.get("title", ""), chunk_text),
                "dates": {
                    "raw_deadlines": extract_deadlines(chunk_text),
                },
                "quality": {
                    "has_source": bool(doc.get("source_url") or doc.get("file_url")),
                    "has_link": bool(doc.get("links")),
                    "is_expired": None,
                    "needs_table_parse": bool(doc.get("status", {}).get("needs_table_parse")),
                    "confidence": "medium",
                }
            })

    return chunks


def extract_keywords(title: str, text: str, limit: int = 12) -> List[str]:
    # Lightweight keyword extraction for metadata only.
    s = f"{title}\n{text}"
    candidates = []
    patterns = [
        r"HUFLIT",
        r"ĐRL|Điểm rèn luyện",
        r"BHYT|KCB",
        r"Phòng [A-ZĐÂÊÔƠƯa-zà-ỹ\s\-]+",
        r"Quận \d+|Hóc Môn|Sư Vạn Hạnh|Kobe|Youngsan|Wonkwang",
        r"học kỳ [IVX0-9]+|năm học \d{4}\s*-\s*\d{4}",
        r"\d{1,2}/\d{1,2}/\d{4}",
    ]
    for pat in patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            kw = clean_text(m.group(0))
            if kw and kw not in candidates:
                candidates.append(kw)
    return candidates[:limit]


# ----------------------------
# Crawler
# ----------------------------

@dataclass
class CrawlConfig:
    start_urls: List[str]
    max_pages: int
    max_articles: int
    delay: float
    download_pdfs: bool
    download_dir: Path
    out_path: Path
    previous_json: Optional[Path] = None

    same_domain_only: bool = True


class HuflitCrawler:
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = make_session()
        self.visited_pages: Set[str] = set()
        self.article_urls: List[str] = []
        self.errors: List[Dict[str, Any]] = []
        self.duplicates: List[Dict[str, Any]] = []
        self.previous_times: Dict[str, Dict[str, Any]] = self.load_previous_times(config.previous_json)
        self.discovered_dates: Dict[str, str] = {}

    def log_error(self, stage: str, url: str, error_type: str, message: str, doc_id: Optional[str] = None) -> None:
        self.errors.append({
            "error_id": f"err_{len(self.errors)+1:05d}",
            "doc_id": doc_id,
            "url": url,
            "stage": stage,
            "error_type": error_type,
            "message": str(message),
            "created_at": now_str(),
        })

    def record_duplicate(self, dropped_doc: Dict[str, Any], kept_doc: Dict[str, Any], reason: str) -> None:
        """
        Duplicate records are data-quality information, not crawler errors.
        Store them separately so total_errors stays meaningful.
        """
        self.duplicates.append({
            "duplicate_id": f"dup_{len(self.duplicates)+1:05d}",
            "reason": reason,
            "dropped_doc_id": dropped_doc.get("doc_id"),
            "dropped_title": dropped_doc.get("title"),
            "dropped_url": dropped_doc.get("file_url") or dropped_doc.get("source_url"),
            "kept_doc_id": kept_doc.get("doc_id"),
            "kept_title": kept_doc.get("title"),
            "kept_url": kept_doc.get("file_url") or kept_doc.get("source_url"),
            "created_at": now_str(),
        })


    def load_previous_times(self, previous_json: Optional[Path]) -> Dict[str, Dict[str, Any]]:
        """
        Load previous crawl result to preserve first_seen_at.
        This is useful when the portal does not expose published_at.
        """
        if not previous_json:
            return {}

        try:
            previous_json = Path(previous_json)
            if not previous_json.exists():
                return {}

            data = json.loads(previous_json.read_text(encoding="utf-8"))
            docs = data.get("source_documents", [])
            result: Dict[str, Dict[str, Any]] = {}

            for d in docs:
                doc_id = d.get("doc_id")
                if not doc_id:
                    continue
                result[doc_id] = {
                    "published_at": d.get("published_at"),
                    "first_seen_at": d.get("first_seen_at") or d.get("scraped_at"),
                    "last_seen_at": d.get("last_seen_at") or d.get("scraped_at"),
                    "source_order": d.get("source_order"),
                    "time_confidence": d.get("time_confidence"),
                }

            return result
        except Exception as e:
            self.log_error("load_previous_json", str(previous_json), type(e).__name__, str(e))
            return {}

    def build_time_fields(
        self,
        doc_id: str,
        url: str,
        scraped_at: str,
        published_at: Optional[str],
        discovered_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build time metadata for documents.

        published_at:
            True article post date, only if visible in HTML/meta.
        scraped_at:
            Current crawler fetch time.
        first_seen_at:
            First time our system saw this doc. Preserved from previous_json if available.
            If no previous_json, it equals scraped_at.
        last_seen_at:
            Last crawl time this doc was still visible.
        detail_id:
            Portal News/Detail id, useful for relative recency if published_at is missing.
        """
        previous = self.previous_times.get(doc_id, {})
        detail_id = extract_detail_id(url)

        first_seen_at = previous.get("first_seen_at") or scraped_at
        last_seen_at = scraped_at

        if published_at:
            confidence = "exact_published_at"
        elif detail_id is not None:
            confidence = "inferred_from_detail_id"
        else:
            confidence = "first_seen_fallback"

        return {
            "published_at": published_at,
            "scraped_at": scraped_at,
            "first_seen_at": first_seen_at,
            "last_seen_at": last_seen_at,
            "source_order": {
                "detail_id": detail_id,
                "discovered_rank": discovered_rank,
            },
            "time_confidence": confidence,
        }


    def discover_article_urls(self) -> List[str]:
        q = deque(normalize_url(u) for u in self.config.start_urls)
        seen_urls: Set[str] = set(q)
        article_seen: Set[str] = set()

        pbar = tqdm(total=self.config.max_pages, desc="Discover pages", unit="page")

        while q and len(self.visited_pages) < self.config.max_pages:
            url = q.popleft()
            if url in self.visited_pages:
                continue

            if self.config.same_domain_only and not is_same_domain(url):
                continue

            try:
                resp = fetch(self.session, url)
                ctype = resp.headers.get("content-type", "").lower()
                if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
                    continue

                self.visited_pages.add(url)
                pbar.update(1)

                soup = BeautifulSoup(resp.text, "lxml")
                for a in soup.select("a[href]"):
                    href = normalize_url(a.get("href", ""), url)
                    if not href:
                        continue

                    if is_article_url(href):
                        # Extract published_at from list page structure (usually in <i> element nested in <a>)
                        published_date = None
                        for i_tag in a.find_all("i"):
                            i_text = i_tag.get_text()
                            parsed_d = parse_vietnamese_date(i_text)
                            if parsed_d:
                                if "đăng" in i_text.lower():
                                    published_date = parsed_d
                                    break
                                elif not published_date:
                                    published_date = parsed_d
                        
                        if published_date:
                            self.discovered_dates[normalize_url(href)] = published_date

                        if href not in article_seen:
                            article_seen.add(href)
                            self.article_urls.append(href)
                            if len(self.article_urls) >= self.config.max_articles:
                                pbar.close()
                                return self.article_urls

                    # Continue crawling allowed HUFLIT HTML pages, but avoid file links.
                    if href not in seen_urls and is_same_domain(href) and should_follow_crawl_url(href):
                        seen_urls.add(href)
                        q.append(href)

                time.sleep(self.config.delay)

            except Exception as e:
                self.log_error("discover", url, type(e).__name__, str(e))

        pbar.close()
        return self.article_urls

    def parse_article(self, url: str, discovered_rank: Optional[int] = None) -> Optional[Dict[str, Any]]:
        try:
            resp = fetch(self.session, url)
            soup = BeautifulSoup(resp.text, "lxml")
            remove_noise(soup)

            marker_title, marker_paragraphs = extract_article_by_news_marker(soup)

            title = marker_title or get_title(soup, url)
            container = find_content_container(soup)

            if marker_paragraphs:
                paragraphs = marker_paragraphs
            else:
                paragraphs = extract_blocks(container)

            full_text = clean_text("\n".join(paragraphs))
            doc_id = self.make_doc_id(url, "article")

            links = extract_links(container, url, doc_id, full_text)
            fallback_links = extract_priority_links_from_soup(soup, url, doc_id, full_text)
            links = merge_links(links + fallback_links)

            category = categorize(title, full_text, url)
            scraped_at = now_str()
            published_at = extract_published_at(soup, container)
            if not published_at:
                published_at = self.discovered_dates.get(normalize_url(url))
            time_fields = self.build_time_fields(
                doc_id=doc_id,
                url=url,
                scraped_at=scraped_at,
                published_at=published_at,
                discovered_rank=discovered_rank,
            )

            status = {
                "is_empty": not bool(full_text.strip()),
                "is_low_quality": len(full_text.strip()) < 80 or full_text.strip().lower() in {"thông báo", "tin tức"},
                "is_link_only": bool(links) and len(full_text.strip()) < 120,
                "is_duplicate": False,
                "has_missing_links": self.has_missing_link_text(full_text, links),
                "suspected_date_error": suspected_date_error(title, full_text),
                "needs_table_parse": False,
            }

            doc = {
                "doc_id": doc_id,
                "doc_type": "article",
                "category": category,
                "title": title,
                "source_url": url,
                **time_fields,
                "full_text": full_text,
                "paragraphs": paragraphs,
                "links": links,
                "attachments": [],
                "status": status,
                "content_hash": sha256_text(full_text),
            }

            # Open resource links such as "tại đây" and extract their data when possible.
            # Direct PDFs/DOCX/XLSX/PPTX/SharePoint-like file links become source_documents too.
            for link in links:
                if is_processable_resource_link(link):
                    attachment = self.process_resource_link(
                        link,
                        parent_doc_id=doc_id,
                        source_article_url=url,
                        title_hint=link.get("anchor_text") or title,
                    )
                    if attachment:
                        doc["attachments"].append(attachment)

            if not title and not full_text:
                self.log_error("parse_article", url, "empty_article", "No title and no full_text", doc_id)

            return doc

        except Exception as e:
            self.log_error("parse_article", url, type(e).__name__, str(e))
            return None

    @staticmethod
    def has_missing_link_text(full_text: str, links: List[Dict[str, Any]]) -> bool:
        s = full_text.lower()
        has_anchor_phrase = any(k in s for k in ["tại đây", "tải file", "xem tại đây", "download"])
        return has_anchor_phrase and len(links) == 0

    @staticmethod
    def make_doc_id(url_or_name: str, prefix: str) -> str:
        # Use News/Detail ID if available.
        m = re.search(r"/News/Detail/(\d+)", url_or_name)
        if m:
            return f"{prefix}_{m.group(1)}"
        return f"{prefix}_{short_hash(url_or_name)}"




    def process_resource_link(
        self,
        link: Dict[str, Any],
        parent_doc_id: Optional[str] = None,
        source_article_url: Optional[str] = None,
        title_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Open a link such as "tại đây" and extract data if it points to a resource.

        Handles:
        - Direct PDF/DOCX/XLSX/PPTX files
        - Same-domain / external HTML pages
        - SharePoint/OneDrive viewer links when they can be converted to direct download URLs

        Does not extract Microsoft Forms responses; form links are kept as links only.
        """
        url = link.get("url", "")
        link_type = link.get("link_type", "")
        if not url:
            return None

        if link_type == "form":
            return None


        # Direct PDFs that already end with .pdf still use the stable PDF path.
        if is_possible_pdf_url(url, link_type):
            return self.process_pdf_link(
                url,
                parent_doc_id=parent_doc_id,
                source_article_url=source_article_url,
                title_hint=title_hint or link.get("anchor_text"),
            )

        if "sharepoint.com" in url.lower():
            return None

        doc_id = self.make_doc_id(url, "resource")
        scraped_at = now_str()
        time_fields = self.build_time_fields(
            doc_id=doc_id,
            url=source_article_url or url,
            scraped_at=scraped_at,
            published_at=None,
            discovered_rank=None,
        )

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "doc_type": "linked_resource",
            "category": categorize(title_hint or link.get("anchor_text", ""), "", url),
            "title": title_hint or link.get("anchor_text") or os.path.basename(urlparse(url).path) or url,
            "file_name": None,
            "source_url": source_article_url or url,
            "file_url": url,
            "resolved_url": None,
            "attempted_urls": build_resource_candidate_urls(url),
            "parent_doc_id": parent_doc_id,
            "link_anchor_text": link.get("anchor_text"),
            "link_context": link.get("context"),
            **time_fields,
            "full_text": "",
            "pages": None,
            "links": [],
            "attachments": [],
            "status": {
                "is_empty": True,
                "needs_table_parse": False,
                "resource_data_null": False,
                "unsupported_resource": False,
                "auth_or_js_required": False,
            },
            "extraction": {
                "method": None,
                "success": False,
                "is_empty_text": True,
                "table_preserved": False,
            }
        }


        last_error: Optional[Exception] = None
        candidate_urls = result["attempted_urls"]

        for attempt_index, attempt_url in enumerate(candidate_urls, start=1):
            is_last_attempt = attempt_index == len(candidate_urls)

            try:
                resp = self.session.get(attempt_url, timeout=60, allow_redirects=True, verify=False)
                resp.raise_for_status()

                ctype = resp.headers.get("content-type", "").lower()
                final_url = resp.url or attempt_url
                result["resolved_url"] = final_url

                ext = get_file_ext_from_url(final_url) or get_file_ext_from_url(attempt_url) or get_file_ext_from_url(url)

                # HTML page.
                if "text/html" in ctype or "application/xhtml+xml" in ctype:
                    html_text = resp.text
                    lower_html = html_text.lower()

                    login_markers = ["sign in", "đăng nhập", "login", "microsoftonline", "access denied", "requires javascript"]
                    is_auth_like = any(m in lower_html for m in login_markers)

                    if is_auth_like:
                        result["status"]["auth_or_js_required"] = True
                        self.log_error(
                            "linked_resource_extraction",
                            url,
                            "auth_or_js_required",
                            "HTML link appears to require authentication or JavaScript",
                            doc_id,
                        )
                        return result

                    html_data = extract_html_text(html_text, final_url)
                    extracted_text = html_data["full_text"]


                    result["title"] = html_data.get("title") or result["title"]
                    result["full_text"] = extracted_text
                    result["paragraphs"] = html_data["paragraphs"]
                    result["pages"] = html_data.get("pages")
                    result["page_texts"] = html_data.get("page_texts", [])
                    result["extraction"] = html_data["extraction"]
                    result["status"]["is_empty"] = not bool(result["full_text"].strip())
                    result["content_hash"] = sha256_text(result["full_text"])

                    if result["status"]["is_empty"]:
                        self.log_error("linked_resource_extraction", url, "empty_html_text", "Linked HTML page extracted empty text", doc_id)

                    return result

                # File download.
                filename = extract_download_filename(resp, attempt_url, fallback=f"{doc_id}")
                if not os.path.splitext(filename)[1] and ext:
                    filename += ext

                path = self.config.download_dir / safe_filename(filename)
                self.config.download_dir.mkdir(parents=True, exist_ok=True)

                if path.exists():
                    root, suffix = path.stem, path.suffix
                    path = self.config.download_dir / f"{root}_{short_hash(attempt_url, 6)}{suffix}"

                path.write_bytes(resp.content)

                result["local_path"] = str(path)
                result["file_name"] = path.name
                result["file_url"] = attempt_url

                try:
                    extracted = extract_file_by_extension(path)
                except Exception as e:
                    last_error = e
                    # Try next candidate if current downloaded HTML/unknown file is not extractable.
                    if attempt_index < len(candidate_urls):
                        continue
                    result["status"]["unsupported_resource"] = True
                    self.log_error("linked_resource_extraction", url, type(e).__name__, str(e), doc_id)
                    return result

                result["full_text"] = extracted["full_text"]
                result["paragraphs"] = extracted["paragraphs"]
                result["pages"] = extracted.get("pages")
                result["page_texts"] = extracted.get("page_texts", [])
                result["extraction"] = extracted["extraction"]
                result["status"]["is_empty"] = not bool(result["full_text"].strip())
                result["status"]["needs_table_parse"] = bool(re.search(
                    r"\bKCB\b|BHYT|Mã Bệnh|Tên Bệnh viện",
                    result["full_text"],
                    flags=re.IGNORECASE
                ))
                result["content_hash"] = sha256_text(result["full_text"])

                if result["status"]["is_empty"]:
                    if attempt_index < len(candidate_urls):
                        continue
                    self.log_error("linked_resource_extraction", url, "empty_resource_text", "Linked resource extracted empty text", doc_id)

                return result

            except Exception as e:
                last_error = e
                if not is_last_attempt:
                    continue

        result["status"]["resource_data_null"] = True
        self.log_error(
            "linked_resource_extraction",
            url,
            type(last_error).__name__ if last_error else "UnknownError",
            str(last_error) if last_error else "Unable to extract linked resource",
            doc_id,
        )
        return result


    def process_pdf_link(
        self,
        pdf_url: str,
        parent_doc_id: Optional[str] = None,
        source_article_url: Optional[str] = None,
        title_hint: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        doc_id = self.make_doc_id(pdf_url, "pdf")
        file_name = safe_filename(os.path.basename(urlparse(pdf_url).path) or f"{doc_id}.pdf")

        scraped_at = now_str()
        time_fields = self.build_time_fields(
            doc_id=doc_id,
            url=source_article_url or pdf_url,
            scraped_at=scraped_at,
            published_at=None,
            discovered_rank=None,
        )

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "doc_type": "pdf",
            "category": categorize(title_hint or file_name, "", pdf_url),
            "title": title_hint or file_name,
            "file_name": file_name,
            "source_url": source_article_url or pdf_url,
            "file_url": pdf_url,
            "parent_doc_id": parent_doc_id,
            **time_fields,
            "full_text": "",
            "pages": None,
            "links": [],
            "attachments": [],
            "status": {
                "is_empty": True,
                "needs_table_parse": False,
                "pdf_data_null": False,
            },
            "extraction": {
                "method": None,
                "success": False,
                "is_empty_text": True,
                "table_preserved": False,
            }
        }

        if not self.config.download_pdfs:
            result["status"]["pdf_data_null"] = True
            return result

        try:
            path = download_file(self.session, pdf_url, self.config.download_dir)
            result["local_path"] = str(path)

            pdf_data = extract_pdf_text(path)
            result["full_text"] = pdf_data["full_text"]
            result["pages"] = pdf_data["pages"]
            result["paragraphs"] = pdf_data["paragraphs"]
            result["page_texts"] = pdf_data["page_texts"]
            result["extraction"] = pdf_data["extraction"]

            is_empty = not bool(result["full_text"].strip())
            result["status"]["is_empty"] = is_empty
            result["status"]["needs_table_parse"] = bool(re.search(
                r"\bKCB\b|BHYT|Mã Bệnh|Tên Bệnh viện",
                result["full_text"],
                flags=re.IGNORECASE
            ))

            result["content_hash"] = sha256_text(result["full_text"])

            if is_empty:
                self.log_error(
                    "pdf_extraction",
                    pdf_url,
                    "empty_text",
                    f"PDF has {result.get('pages')} pages but extracted full_text is empty",
                    doc_id,
                )

            return result

        except Exception as e:
            result["status"]["pdf_data_null"] = True
            self.log_error("pdf_extraction", pdf_url, type(e).__name__, str(e), doc_id)
            return result

    def crawl(self) -> Dict[str, Any]:
        article_urls = self.discover_article_urls()
        print(f"Discovered article URLs: {len(article_urls)}")

        source_documents: List[Dict[str, Any]] = []
        pdf_docs_by_url: Dict[str, Dict[str, Any]] = {}

        for rank, url in enumerate(tqdm(article_urls, desc="Parse articles", unit="article"), start=1):
            doc = self.parse_article(url, discovered_rank=rank)
            if not doc:
                continue

            # Attachments are also source_documents.
            for att in doc.get("attachments", []):
                if att.get("file_url") not in pdf_docs_by_url:
                    pdf_docs_by_url[att.get("file_url")] = att

            source_documents.append(doc)
            time.sleep(self.config.delay)

        # Add top-level PDF source docs.
        source_documents.extend(pdf_docs_by_url.values())

        # Deduplicate documents by normalized source/file URL and content hash.
        source_documents = self.deduplicate_documents(source_documents)


        # Structured records.
        all_links: List[Dict[str, Any]] = []
        form_records: List[Dict[str, Any]] = []
        hospital_records: List[Dict[str, Any]] = []

        for doc in source_documents:
            all_links.extend(doc.get("links", []))
            form_records.extend(build_form_records(doc))
            if doc.get("doc_type") in {"pdf", "linked_resource"}:
                hospital_records.extend(parse_hospital_kcb_records(doc))

        rag_chunks = build_rag_chunks(source_documents)

        output = {
            "metadata": {
                "source": list(dict.fromkeys(self.config.start_urls)),
                "version": datetime.now().strftime("%Y-%m-%d"),
                "created_at": now_str(),
                "total_input": len(article_urls),
                "total_documents": len(source_documents),
                "total_articles": sum(1 for d in source_documents if d.get("doc_type") == "article"),
                "total_pdfs": sum(1 for d in source_documents if d.get("doc_type") == "pdf"),
                "total_linked_resources": sum(1 for d in source_documents if d.get("doc_type") == "linked_resource"),
                "total_chunks": len(rag_chunks),
                "total_errors": len(self.errors),
                "total_duplicates_removed": len(self.duplicates),
                "language": "vi",
            },
            "source_documents": source_documents,
            "rag_chunks": rag_chunks,
            "structured_records": {
                "forms": form_records,
                "hospital_kcb": hospital_records,
                "links": all_links,
            },
            "duplicates": self.duplicates,
            "postprocess": {},
            "errors": self.errors,
        }

        return output

    def deduplicate_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents from source_documents before chunking.

        Duplicate rules:
        1. Same canonical source/file URL -> keep the richer document.
        2. Same normalized content fingerprint -> keep the richer document.
        3. Duplicates are reported in self.duplicates, not self.errors.

        This keeps the final KB cleaner and prevents vector DB from being polluted
        by repeated chunks.
        """

        def canonical_url_key(doc: Dict[str, Any]) -> str:
            url = doc.get("file_url") or doc.get("source_url") or doc.get("doc_id") or ""
            return f"{doc.get('doc_type')}::{normalize_url(url)}"

        def choose_better(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            """
            Return (kept, dropped).
            """
            if document_quality_score(b) > document_quality_score(a):
                return b, a
            return a, b

        # Pass 1: dedupe by URL.
        by_url: Dict[str, Dict[str, Any]] = {}

        for doc in docs:
            key = canonical_url_key(doc)

            if key not in by_url:
                by_url[key] = doc
                continue

            kept, dropped = choose_better(by_url[key], doc)
            dropped.setdefault("status", {})["is_duplicate"] = True
            dropped["duplicate_of"] = kept.get("doc_id")
            self.record_duplicate(dropped, kept, "same_url")

            by_url[key] = kept

        url_deduped = list(by_url.values())

        # Pass 2: dedupe by normalized content fingerprint.
        by_content: Dict[str, Dict[str, Any]] = {}

        for doc in url_deduped:
            text = doc.get("full_text", "") or ""
            norm = normalize_text_for_dedupe(text)

            # Empty docs are handled by quality flags; do not group all empty docs together here.
            if not norm:
                continue

            fp = sha256_text(norm)
            if fp not in by_content:
                by_content[fp] = doc
                continue

            kept, dropped = choose_better(by_content[fp], doc)
            dropped.setdefault("status", {})["is_duplicate"] = True
            dropped["duplicate_of"] = kept.get("doc_id")
            self.record_duplicate(dropped, kept, "same_normalized_content")

            by_content[fp] = kept

        # Important: by_content contains only docs with non-empty norm.
        # Add docs with empty normalized text once, if any.
        content_ids = {id(d) for d in by_content.values()}
        final_docs: List[Dict[str, Any]] = list(by_content.values())

        for doc in url_deduped:
            if not normalize_text_for_dedupe(doc.get("full_text", "") or ""):
                if id(doc) not in content_ids:
                    final_docs.append(doc)

        # Keep deterministic order:
        # - article: newest first by detail_id/discovered rank if possible
        # - pdf/others: after related articles if no detail_id.
        def sort_key(doc: Dict[str, Any]):
            order = doc.get("source_order") or {}
            detail_id = order.get("detail_id")
            rank = order.get("discovered_rank")
            # Negative for descending detail_id. None goes last.
            detail_sort = -int(detail_id) if isinstance(detail_id, int) else 10**12
            rank_sort = int(rank) if isinstance(rank, int) else 10**9
            return (doc.get("doc_type") != "article", detail_sort, rank_sort, doc.get("doc_id", ""))

        final_docs.sort(key=sort_key)

        # Ensure final kept docs are not marked duplicate.
        for doc in final_docs:
            doc.setdefault("status", {})["is_duplicate"] = False
            doc.pop("duplicate_of", None)

        return final_docs


# ----------------------------
# Optional: load old huflit_data.json and convert to new schema
# ----------------------------

def convert_old_json_to_new_schema(input_path: Path, out_path: Path) -> None:
    """
    Converts previous schema:
    {
      "metadata": {},
      "text_articles": [],
      "pdf_documents": []
    }
    or list[article/pdf]
    into the new RAG schema.
    """
    data = json.loads(input_path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        text_articles = data.get("text_articles", [])
        pdf_documents = data.get("pdf_documents", [])
        old_meta = data.get("metadata", {})
    elif isinstance(data, list):
        text_articles = [x for x in data if x.get("type") == "text_article"]
        pdf_documents = [x for x in data if x.get("type") == "pdf_document"]
        old_meta = {}
    else:
        raise ValueError("Unsupported JSON root type")

    errors: List[Dict[str, Any]] = []
    source_documents: List[Dict[str, Any]] = []

    for a in text_articles:
        url = normalize_url(a.get("url", ""))
        title = clean_text(a.get("title", ""))
        full_text = clean_text(a.get("full_text", ""))
        doc_id = HuflitCrawler.make_doc_id(url or title, "article")

        links = []
        # If old data has no links, detect missing link phrases.
        status = {
            "is_empty": not bool(full_text),
            "is_low_quality": len(full_text.strip()) < 80 or full_text.strip().lower() in {"thông báo", "tin tức"},
            "is_duplicate": False,
            "has_missing_links": bool(re.search(r"tại đây|tải file|xem tại đây", full_text, flags=re.IGNORECASE)),
            "suspected_date_error": suspected_date_error(title, full_text),
            "needs_table_parse": False,
        }

        if status["is_empty"]:
            errors.append({
                "error_id": f"err_{len(errors)+1:05d}",
                "doc_id": doc_id,
                "url": url,
                "stage": "convert",
                "error_type": "empty_article",
                "message": "Article full_text is empty",
                "created_at": now_str(),
            })

        scraped_at = a.get("scraped_at") or now_str()
        published_at = a.get("published_at")
        detail_id = extract_detail_id(url)
        source_order = a.get("source_order") or {
            "detail_id": detail_id,
            "discovered_rank": None,
        }

        source_documents.append({
            "doc_id": doc_id,
            "doc_type": "article",
            "category": categorize(title, full_text, url),
            "title": title,
            "source_url": url,
            "published_at": published_at,
            "scraped_at": scraped_at,
            "first_seen_at": a.get("first_seen_at") or scraped_at,
            "last_seen_at": a.get("last_seen_at") or scraped_at,
            "source_order": source_order,
            "time_confidence": a.get("time_confidence") or ("exact_published_at" if published_at else ("inferred_from_detail_id" if detail_id is not None else "first_seen_fallback")),
            "full_text": full_text,
            "paragraphs": a.get("paragraphs", []),
            "links": links,
            "attachments": a.get("pdfs", []),
            "status": status,
            "content_hash": sha256_text(full_text),
        })

    for p in pdf_documents:
        title = clean_text(p.get("title", "") or p.get("file_name", ""))
        file_url = normalize_url(p.get("url", "") or p.get("file_url", ""))
        doc_id = HuflitCrawler.make_doc_id(file_url or title, "pdf")

        pdf_data = p.get("pdf_data") or {}
        full_text = clean_text(pdf_data.get("full_text", "") or p.get("full_text", ""))
        pages = pdf_data.get("pages") or p.get("pages")

        is_empty = not bool(full_text)
        status = {
            "is_empty": is_empty,
            "needs_table_parse": bool(re.search(r"\bKCB\b|BHYT|Mã Bệnh|Tên Bệnh viện", full_text, flags=re.IGNORECASE)),
            "pdf_data_null": p.get("pdf_data") is None,
        }

        if is_empty:
            errors.append({
                "error_id": f"err_{len(errors)+1:05d}",
                "doc_id": doc_id,
                "url": file_url,
                "stage": "convert",
                "error_type": "empty_pdf_text",
                "message": f"PDF pages={pages}, but full_text is empty",
                "created_at": now_str(),
            })

        scraped_at = p.get("scraped_at") or now_str()
        published_at = p.get("published_at")
        source_url = p.get("source_url") or file_url
        detail_id = extract_detail_id(source_url)

        source_documents.append({
            "doc_id": doc_id,
            "doc_type": "pdf",
            "category": categorize(title, full_text, file_url),
            "title": title,
            "file_name": p.get("file_name") or title,
            "source_url": source_url,
            "file_url": file_url,
            "published_at": published_at,
            "scraped_at": scraped_at,
            "first_seen_at": p.get("first_seen_at") or scraped_at,
            "last_seen_at": p.get("last_seen_at") or scraped_at,
            "source_order": p.get("source_order") or {
                "detail_id": detail_id,
                "discovered_rank": None,
            },
            "time_confidence": p.get("time_confidence") or ("exact_published_at" if published_at else ("inferred_from_detail_id" if detail_id is not None else "first_seen_fallback")),
            "pages": pages,
            "full_text": full_text,
            "paragraphs": pdf_data.get("paragraphs", []),
            "links": [],
            "attachments": [],
            "status": status,
            "extraction": {
                "method": "unknown_old_json",
                "success": bool(full_text),
                "is_empty_text": is_empty,
                "table_preserved": False,
            },
            "content_hash": sha256_text(full_text),
        })

    # Deduplicate lightly.
    seen = set()
    deduped = []
    for d in source_documents:
        key = (d.get("doc_type"), normalize_url(d.get("file_url") or d.get("source_url") or ""), d.get("content_hash"))
        if key in seen:
            d["status"]["is_duplicate"] = True
            continue
        seen.add(key)
        deduped.append(d)

    links = []
    forms = []
    hospitals = []
    for d in deduped:
        links.extend(d.get("links", []))
        forms.extend(build_form_records(d))
        if d.get("doc_type") == "pdf":
            hospitals.extend(parse_hospital_kcb_records(d))

    rag_chunks = build_rag_chunks(deduped)

    output = {
        "metadata": {
            "source": old_meta.get("source", BASE_URL),
            "version": datetime.now().strftime("%Y-%m-%d"),
            "created_at": now_str(),
            "total_documents": len(deduped),
            "total_articles": sum(1 for d in deduped if d.get("doc_type") == "article"),
            "total_pdfs": sum(1 for d in deduped if d.get("doc_type") == "pdf"),
            "total_chunks": len(rag_chunks),
            "total_errors": len(errors),
            "language": "vi",
        },
        "source_documents": deduped,
        "rag_chunks": rag_chunks,
        "structured_records": {
            "forms": forms,
            "hospital_kcb": hospitals,
            "links": links,
        },
        "errors": errors,
    }

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Converted JSON saved to: {out_path}")



# ----------------------------
# Multi-source merge
# ----------------------------

def rebuild_output_from_documents(
    source_documents: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    duplicates: List[Dict[str, Any]],
    source_meta: List[Dict[str, Any]],
    source_urls: List[str],
    merge_crawler: HuflitCrawler,
) -> Dict[str, Any]:
    """
    Rebuild one final RAG JSON after crawling multiple source groups.
    This is used by crawl-mode separate so both portal and huflit.edu.vn/vi get
    their own crawl quota, but final output is still one JSON file.
    """
    source_documents = merge_crawler.deduplicate_documents(source_documents)

    all_links: List[Dict[str, Any]] = []
    form_records: List[Dict[str, Any]] = []
    hospital_records: List[Dict[str, Any]] = []

    for doc in source_documents:
        all_links.extend(doc.get("links", []))
        form_records.extend(build_form_records(doc))
        if doc.get("doc_type") in {"pdf", "linked_resource"}:
            hospital_records.extend(parse_hospital_kcb_records(doc))

    rag_chunks = build_rag_chunks(source_documents)

    # Re-number errors after combining multiple crawlers.
    renumbered_errors: List[Dict[str, Any]] = []
    for err in errors:
        e = dict(err)
        e["error_id"] = f"err_{len(renumbered_errors)+1:05d}"
        renumbered_errors.append(e)

    combined_duplicates = list(duplicates) + list(merge_crawler.duplicates)
    renumbered_duplicates: List[Dict[str, Any]] = []
    for dup in combined_duplicates:
        d = dict(dup)
        d["duplicate_id"] = f"dup_{len(renumbered_duplicates)+1:05d}"
        renumbered_duplicates.append(d)

    return {
        "metadata": {
            "source": list(dict.fromkeys(source_urls)),
            "source_mode": "separate_then_merge",
            "source_groups": source_meta,
            "version": datetime.now().strftime("%Y-%m-%d"),
            "created_at": now_str(),
            "total_input": sum(x.get("total_input", 0) for x in source_meta),
            "total_documents": len(source_documents),
            "total_articles": sum(1 for d in source_documents if d.get("doc_type") == "article"),
            "total_pdfs": sum(1 for d in source_documents if d.get("doc_type") == "pdf"),
            "total_linked_resources": sum(1 for d in source_documents if d.get("doc_type") == "linked_resource"),
            "total_chunks": len(rag_chunks),
            "total_errors": len(renumbered_errors),
            "total_duplicates_removed": len(renumbered_duplicates),
            "language": "vi",
        },
        "source_documents": source_documents,
        "rag_chunks": rag_chunks,
        "structured_records": {
            "forms": form_records,
            "hospital_kcb": hospital_records,
            "links": all_links,
        },
        "duplicates": renumbered_duplicates,
        "postprocess": {},
        "errors": renumbered_errors,
    }


def crawl_source_groups_to_one_file(args: argparse.Namespace, out_path: Path) -> Dict[str, Any]:
    """
    Crawl portal and main site separately, each with its own max_pages/max_articles,
    then merge into one JSON output.
    """
    all_docs: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, Any]] = []
    all_duplicates: List[Dict[str, Any]] = []
    source_meta: List[Dict[str, Any]] = []
    source_urls: List[str] = []

    for group in DEFAULT_SOURCE_GROUPS:
        name = group["name"]
        start_urls = group["start_urls"]
        source_urls.extend(start_urls)

        print(f"\n=== Crawling source group: {group['label']} ===")
        print("Start URLs:")
        for u in start_urls:
            print(f"- {u}")

        group_download_dir = Path(args.download_dir) / name
        config = CrawlConfig(
            start_urls=start_urls,
            max_pages=args.max_pages,
            max_articles=args.max_articles,
            delay=args.delay,
            download_pdfs=not args.no_download_pdfs,
            download_dir=group_download_dir,
            out_path=out_path,
            previous_json=Path(args.previous_json) if args.previous_json else None,
        )

        crawler = HuflitCrawler(config)
        output = crawler.crawl()
        meta = output.get("metadata", {})

        source_meta.append({
            "name": name,
            "label": group["label"],
            "start_urls": start_urls,
            "max_pages": args.max_pages,
            "max_articles": args.max_articles,
            "total_input": meta.get("total_input", 0),
            "total_documents_before_final_merge": meta.get("total_documents", 0),
            "total_articles_before_final_merge": meta.get("total_articles", 0),
            "total_chunks_before_final_merge": meta.get("total_chunks", 0),
            "total_errors": meta.get("total_errors", 0),
        })

        all_docs.extend(output.get("source_documents", []))
        all_errors.extend(output.get("errors", []))
        all_duplicates.extend(output.get("duplicates", []))

    merge_config = CrawlConfig(
        start_urls=source_urls,
        max_pages=args.max_pages,
        max_articles=args.max_articles,
        delay=args.delay,
        download_pdfs=not args.no_download_pdfs,
        download_dir=Path(args.download_dir),
        out_path=out_path,
        previous_json=Path(args.previous_json) if args.previous_json else None,
    )
    merge_crawler = HuflitCrawler(merge_config)

    return rebuild_output_from_documents(
        source_documents=all_docs,
        errors=all_errors,
        duplicates=all_duplicates,
        source_meta=source_meta,
        source_urls=source_urls,
        merge_crawler=merge_crawler,
    )

# ----------------------------
# Main
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl HUFLIT portal + main HUFLIT website into one RAG-ready JSON")
    parser.add_argument("--start-url", action="append", default=None, help="Start URL. Can be repeated. If supplied, crawler uses mixed mode for these URLs.")
    parser.add_argument("--crawl-mode", choices=["separate", "mixed"], default="separate", help="separate = crawl portal and main site with separate quotas then merge into one JSON; mixed = old shared queue behavior.")
    parser.add_argument("--max-pages", type=int, default=250, help="Max HTML pages to crawl while discovering articles. In separate mode, this applies to each source group.")
    parser.add_argument("--max-articles", type=int, default=250, help="Max article pages to parse. In separate mode, this applies to each source group.")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds.")
    parser.add_argument("--out", type=str, default="huflit_rag_kb.json", help="Output JSON path.")
    parser.add_argument("--download-dir", type=str, default="downloads", help="Folder to store downloaded files.")
    parser.add_argument("--no-download-pdfs", action="store_true", help="Do not download/extract PDFs and Office files.")
    parser.add_argument("--previous-json", type=str, default=None, help="Previous crawl JSON to preserve first_seen_at/last_seen_at.")

    parser.add_argument("--convert-old-json", type=str, default=None, help="Convert old huflit_data.json to new schema instead of crawling.")
    args = parser.parse_args()

    out_path = Path(args.out)

    if args.convert_old_json:
        convert_old_json_to_new_schema(Path(args.convert_old_json), out_path)
        return

    # Default behavior: crawl both HUFLIT Portal and huflit.edu.vn/vi fully,
    # with separate quotas, then merge into one file.
    if args.crawl_mode == "separate" and not args.start_url:
        output = crawl_source_groups_to_one_file(args, out_path)
    else:
        # Mixed mode: keep old shared queue behavior. Useful for custom --start-url runs.
        config = CrawlConfig(
            start_urls=args.start_url or DEFAULT_START_URLS,
            max_pages=args.max_pages,
            max_articles=args.max_articles,
            delay=args.delay,
            download_pdfs=not args.no_download_pdfs,
            download_dir=Path(args.download_dir),
            out_path=out_path,
            previous_json=Path(args.previous_json) if args.previous_json else None,
        )

        crawler = HuflitCrawler(config)
        output = crawler.crawl()
        output.setdefault("metadata", {})["source_mode"] = "mixed_shared_queue"

    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Output: {out_path.resolve()}")
    print(f"Source mode: {output['metadata'].get('source_mode')}")
    print(f"Documents: {output['metadata']['total_documents']}")
    print(f"Chunks: {output['metadata']['total_chunks']}")
    print(f"Errors: {output['metadata']['total_errors']}")


if __name__ == "__main__":
    main()
