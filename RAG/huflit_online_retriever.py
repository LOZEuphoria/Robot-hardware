import logging
import re
import unicodedata
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from crawl_huflit_rag_v15_integrated import extract_html_text


ALLOWED_HOSTS = {"huflit.edu.vn", "www.huflit.edu.vn", "portal.huflit.edu.vn"}
PORTAL_NEWS_URL = "https://portal.huflit.edu.vn/News"
MAIN_SEARCH_URL = "https://huflit.edu.vn/wp-json/wp/v2/search"
GENERIC_WORDS = {
    "thong", "tin", "cho", "toi", "hoi", "ve", "cua", "la", "gi",
    "nhu", "the", "nao", "huong", "dan", "xin", "duoc",
}


def _normalize(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    value = unicodedata.normalize("NFD", value)
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _keywords(query: str) -> set[str]:
    return {
        word for word in _normalize(query).split()
        if len(word) > 1 and word not in GENERIC_WORDS
    }


def _is_allowed(url: str) -> bool:
    return urlparse(url).hostname in ALLOWED_HOSTS


class HUFLITOnlineRetriever:
    """Find and read only pages hosted on HUFLIT's official domains."""

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        )

    def _portal_candidates(self, query: str) -> List[Dict[str, Any]]:
        response = self.session.get(PORTAL_NEWS_URL, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
        query_words = _keywords(query)
        candidates = []
        seen = set()
        for anchor in soup.select('a[href*="/News/Detail/"]'):
            url = urljoin(PORTAL_NEWS_URL, anchor.get("href", ""))
            title = anchor.get_text(" ", strip=True)
            if not title or url in seen or not _is_allowed(url):
                continue
            seen.add(url)
            title_words = set(_normalize(title).split())
            overlap = len(query_words & title_words)
            if overlap:
                candidates.append(
                    {"title": title, "url": url, "score": overlap * 3}
                )
        return candidates

    def _main_site_candidates(self, query: str) -> List[Dict[str, Any]]:
        response = self.session.get(
            MAIN_SEARCH_URL,
            params={"search": query, "per_page": 7},
            timeout=self.timeout,
        )
        response.raise_for_status()
        candidates = []
        for item in response.json():
            url = item.get("url", "")
            if url and _is_allowed(url):
                candidates.append(
                    {
                        "title": item.get("title", "Tài liệu HUFLIT"),
                        "url": url,
                        # Results are already ranked by HUFLIT's own search API.
                        "score": 1,
                    }
                )
        return candidates

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        if not query.strip():
            return []

        candidates: List[Dict[str, Any]] = []
        for discover in (self._portal_candidates, self._main_site_candidates):
            try:
                candidates.extend(discover(query))
            except (requests.RequestException, ValueError, TypeError) as exc:
                logging.warning("Tìm kiếm HUFLIT trực tuyến thất bại: %s", exc)

        candidates.sort(key=lambda item: item["score"], reverse=True)
        documents = []
        seen = set()
        for candidate in candidates:
            if len(documents) >= top_k:
                break
            url = candidate["url"]
            if url in seen:
                continue
            seen.add(url)
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                extracted = extract_html_text(response.text, url)
                text = (extracted.get("full_text") or "").strip()
                if urlparse(url).hostname in {"huflit.edu.vn", "www.huflit.edu.vn"}:
                    soup = BeautifulSoup(response.text, "lxml")
                    content_candidates = soup.select(".container-inner")
                    main_content = max(
                        content_candidates,
                        key=lambda item: len(item.get_text(" ", strip=True)),
                        default=None,
                    )
                    if main_content is not None:
                        text = main_content.get_text("\n", strip=True)
                if len(text) < 80:
                    continue
                documents.append(
                    {
                        "doc_type": "online",
                        "title": extracted.get("title") or candidate["title"],
                        "source_url": url,
                        "url": url,
                        "chunk_text": text[:6000],
                        "retrieval_score": candidate["score"],
                        "online": True,
                    }
                )
            except (requests.RequestException, ValueError, TypeError) as exc:
                logging.warning("Không đọc được trang HUFLIT %s: %s", url, exc)

        logging.info("Tìm thấy %d tài liệu HUFLIT trực tuyến.", len(documents))
        return documents
