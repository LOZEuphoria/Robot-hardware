import os
import re
from typing import Annotated, Any, Dict, List, Optional

import requests
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from huflit_rag_retriever import HUFLITRetriever
from huflit_online_retriever import HUFLITOnlineRetriever


class LMStudioLLM:
    _instance = None

    def __init__(self):
        self.base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        self.model = os.getenv("LM_STUDIO_MODEL", "").strip()
        self.api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        self.timeout = float(os.getenv("LM_STUDIO_TIMEOUT", "120"))
        self.reasoning_effort = os.getenv("LM_STUDIO_REASONING_EFFORT", "none").strip()
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
        )

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _resolve_model(self) -> str:
        if self.model:
            return self.model

        try:
            response = self.session.get(f"{self.base_url}/models", timeout=self.timeout)
            response.raise_for_status()
            models = response.json().get("data", [])
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"Khong ket noi duoc LM Studio tai {self.base_url}. "
                "Hay mo LM Studio, nap model va bat Local Server."
            ) from exc

        if not models:
            raise RuntimeError(
                "LM Studio chua co model kha dung. Hay nap mot model trong LM Studio "
                "hoac dat bien moi truong LM_STUDIO_MODEL."
            )

        self.model = models[0]["id"]
        print(f"[LM Studio] Using model: {self.model}")
        return self.model

    def chat(self, system: str, user: str, max_tokens: int) -> str:
        payload = {
            "model": self._resolve_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort

        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content: Optional[str] = response.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            detail = ""
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                detail = f" Phan hoi LM Studio: {exc.response.text[:500]}"
            raise RuntimeError(f"Goi LM Studio API that bai.{detail}") from exc

        return (content or "").strip()


def call_local_llm(llm: LMStudioLLM, system: str, user: str, max_tokens: int = 150) -> str:
    return llm.chat(system=system, user=user, max_tokens=max_tokens)


def merge_messages(left: List[BaseMessage], right: List[BaseMessage]) -> List[BaseMessage]:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], merge_messages]
    rewritten_query: str
    retrieved_documents: List[Dict[str, Any]]
    next_step: str
    generation: str


def _is_social_query(text: str) -> bool:
    query = re.sub(r"\s+", " ", str(text or "").lower()).strip(" .,!?")
    social_prefixes = (
        "xin chào",
        "xin chao",
        "xin kìa",
        "xin kia",
        "chào bạn",
        "chao ban",
        "hello",
        "hi",
        "cảm ơn",
        "cam on",
        "thank",
        "bạn là ai",
    )
    return any(query.startswith(prefix) for prefix in social_prefixes)


def query_rewriter_node(state: AgentState) -> Dict[str, Any]:
    messages = state["messages"]
    latest_query = messages[-1].content
    if _is_social_query(str(latest_query)):
        print(f"[Agent] Social query detected: '{latest_query}'")
        return {"rewritten_query": str(latest_query)}

    history_lines = [
        f"Câu hỏi cũ: {str(message.content)[:300]}"
        for message in messages[-4:-1]
        if isinstance(message, HumanMessage)
    ]
    history = "\n".join(history_lines)

    system = (
        "Bạn là trợ lý học vụ của HUFLIT. Hãy viết lại câu hỏi của sinh viên thành một câu hỏi "
        "tra cứu ngắn gọn bằng tiếng Việt. Giữ nguyên các từ khóa quan trọng về học phí, tuyển sinh, "
        "học bổng, quy chế, biểu mẫu, BHYT, KCB và tên cơ sở. Chỉ trả về câu hỏi đã viết lại."
    )
    if history:
        user = f"Lịch sử chat:\n{history}\n\nCâu hỏi mới: {latest_query}\n\nCâu hỏi viết lại:"
    else:
        user = f"Câu hỏi mới: {latest_query}\n\nCâu hỏi viết lại:"

    rewritten = call_local_llm(LMStudioLLM.get_instance(), system, user, max_tokens=80)
    print(f"[Agent] Rewrote query: '{latest_query}' -> '{rewritten}'")
    return {"rewritten_query": rewritten or str(latest_query)}


def router_node(state: AgentState) -> Dict[str, Any]:
    next_step = "respond_directly" if _is_social_query(state.get("rewritten_query", "")) else "retrieve"
    print(f"[Router] Decided next step: {next_step}")
    return {"next_step": next_step}


_retriever_instance = None
_online_retriever_instance = None


def retriever_node(state: AgentState) -> Dict[str, Any]:
    global _retriever_instance, _online_retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HUFLITRetriever()

    result = _retriever_instance.retrieve(state.get("rewritten_query", ""), top_k=3)
    documents: List[Dict[str, Any]] = []
    for form in result["forms"]:
        download_url = form.get("download_url")
        source_url = form.get("source_article_url")
        documents.append(
            {
                "doc_type": "form",
                "title": form.get("form_name"),
                "file_name": form.get("form_name"),
                "file_url": download_url,
                "source_url": source_url,
                "chunk_text": (
                    f"Biểu mẫu: {form.get('form_name')}\n"
                    f"Mô tả: {form.get('description')}\n"
                    f"Đường dẫn tải: {download_url}\n"
                    f"Bài viết: {source_url}"
                ),
            }
        )
    documents.extend(result["chunks"])

    online_enabled = os.getenv("RAG_ONLINE_FALLBACK", "true").strip().lower()
    if online_enabled in {"1", "true", "yes", "on"}:
        if _online_retriever_instance is None:
            _online_retriever_instance = HUFLITOnlineRetriever()
        online_documents = _online_retriever_instance.search(
            state.get("rewritten_query", ""),
            top_k=2,
        )
        existing_urls = {
            document.get("source_url") or document.get("url")
            for document in documents
        }
        documents.extend(
            document for document in online_documents
            if (document.get("source_url") or document.get("url")) not in existing_urls
        )

    print(f"[Retriever] Retrieved {len(documents)} documents/forms.")
    return {"retrieved_documents": documents}


def grader_node(state: AgentState) -> Dict[str, Any]:
    documents = state.get("retrieved_documents", [])
    print(f"[Grader] Kept all {len(documents)} retrieved documents/forms.")
    return {"retrieved_documents": documents}


def _append_links(answer: str, documents: List[Dict[str, Any]]) -> str:
    if os.getenv("RAG_APPEND_LINKS", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        return answer.strip()

    links = []
    seen = set()
    for document in documents:
        title = document.get("title") or document.get("file_name") or "Tài liệu HUFLIT"
        candidates = [document.get("file_url"), document.get("source_url"), document.get("url")]
        candidates.extend(re.findall(r"https?://[^\s\])}]+", document.get("chunk_text") or ""))
        for url in candidates:
            if not url or str(url).lower() == "none" or url in seen:
                continue
            seen.add(url)
            links.append(f"- [{title}]({str(url).rstrip('.,;')})")
    if links:
        answer = answer.strip() + "\n\n**Tài liệu tham khảo chính thức:**\n" + "\n".join(links[:5])
    return answer


def generator_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("rewritten_query", "")
    documents = state.get("retrieved_documents", [])
    if not documents:
        return {
            "generation": (
                "Tôi không tìm thấy thông tin này trong tài liệu HUFLIT hiện có. "
                "Bạn vui lòng liên hệ Phòng Đào tạo hoặc Phòng Chính trị - Công tác Sinh viên để được hỗ trợ."
            )
        }

    context_parts = []
    for document in documents:
        title = document.get("title") or document.get("file_name") or "Tài liệu HUFLIT"
        excerpt_limit = 1800 if document.get("online") else 700
        context_parts.append(
            f"--- {title} ---\n"
            f"{(document.get('chunk_text') or '')[:excerpt_limit]}"
        )

    system = (
        "Bạn là trợ lý học vụ HUFLIT. Trả lời bằng tiếng Việt, rõ ràng và chỉ dựa trên tài liệu được cung cấp. "
        "Không tự suy đoán. Nếu tài liệu không đủ thông tin, hãy nói rằng chưa tìm thấy thông tin chính xác."
    )
    user = (
        "Tài liệu HUFLIT:\n"
        + "\n\n".join(context_parts)
        + f"\n\nCâu hỏi: {query}\n\nCâu trả lời:"
    )
    # Use clean UTF-8 prompts here; some legacy prompt literals in old datasets
    # may have passed through a non-UTF-8 console and become mojibake.
    system = (
        "Bạn là trợ lý học vụ HUFLIT. Hãy trả lời bằng tiếng Việt, rõ ràng, "
        "hoàn chỉnh và chỉ dựa trên tài liệu được cung cấp. Ưu tiên trả lời "
        "trực tiếp các mức tiền, thời hạn và quy trình nếu tài liệu có nêu. "
        "Không tự suy đoán. Dùng văn bản thuần phù hợp để đọc thành giọng nói; "
        "không dùng Markdown, bảng, dấu đầu dòng hoặc ký hiệu trang trí."
    )
    user = (
        "Tài liệu HUFLIT:\n"
        + "\n\n".join(context_parts)
        + f"\n\nCâu hỏi: {query}\n\nHãy trả lời trực tiếp và đầy đủ:"
    )
    print("[Generator] Qwen is generating the answer...")
    answer_max_tokens = int(os.getenv("RAG_ANSWER_MAX_TOKENS", "500"))
    answer = call_local_llm(
        LMStudioLLM.get_instance(),
        system,
        user,
        max_tokens=answer_max_tokens,
    )
    print("[Generator] Answer generated.")
    if not answer:
        answer = (
            "Tôi chưa thể tạo câu trả lời vào lúc này. "
            "Bạn vui lòng hỏi lại bằng một câu đầy đủ hơn."
        )
    return {"generation": _append_links(answer, documents)}


def direct_respond_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("rewritten_query", "")
    if "cảm ơn" in query.lower() or "cam on" in query.lower() or "thank" in query.lower():
        answer = "Không có gì. Tôi luôn sẵn sàng hỗ trợ bạn tra cứu thông tin HUFLIT."
    else:
        answer = (
            "Xin chào. Tôi có thể hỗ trợ bạn tra cứu thông tin HUFLIT về học vụ, học phí, "
            "tuyển sinh, biểu mẫu và bảo hiểm y tế."
        )
    return {"generation": answer}


def create_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("direct_respond", direct_respond_node)

    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "router")
    workflow.add_conditional_edges(
        "router",
        lambda state: state["next_step"],
        {"retrieve": "retriever", "respond_directly": "direct_respond"},
    )
    workflow.add_edge("retriever", "grader")
    workflow.add_edge("grader", "generator")
    workflow.add_edge("generator", END)
    workflow.add_edge("direct_respond", END)
    return workflow.compile(checkpointer=MemorySaver())
