import json
import logging
from typing import List, Dict, Any
from elasticsearch import Elasticsearch

class HUFLITRetriever:
    def __init__(self, es_url: str = "http://127.0.0.1:9200"):
        self.es = Elasticsearch(es_url)

    def search_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not query.strip():
            return []
            
        intents = ["học phí", "điểm chuẩn", "tuyển sinh", "học bổng", "bảo hiểm", "lịch thi", "quy chế", "đăng ký", "thời khóa biểu", "tốt nghiệp"]
        query_lower = query.lower()
        
        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "chunk_text"],
                                "type": "best_fields"
                            }
                        }
                    ],
                    "should": []
                }
            },
            "size": top_k
        }
        
        for intent in intents:
            if intent in query_lower:
                body["query"]["bool"]["should"].append({
                    "multi_match": {
                        "query": intent,
                        "fields": ["title^5", "chunk_text^2"],
                        "type": "phrase",
                        "boost": 10.0
                    }
                })

        try:
            res = self.es.search(index="huflit_chunks", body=body)
            results = []
            for hit in res["hits"]["hits"]:
                source = hit["_source"]
                source["retrieval_score"] = round(hit["_score"], 3)
                results.append(source)
            return results
        except Exception as e:
            logging.error(f"Search chunks failed: {e}")
            return []

    def search_forms(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        form_keywords = {"đơn", "biểu mẫu", "mẫu", "form", "giấy"}
        if not any(k in query_lower for k in form_keywords):
            return []
            
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["form_name^3", "description"],
                    "type": "best_fields"
                }
            },
            "size": 3
        }
        
        try:
            res = self.es.search(index="huflit_forms", body=body)
            results = []
            for hit in res["hits"]["hits"]:
                source = hit["_source"]
                source["match_score"] = round(hit["_score"], 3)
                results.append(source)
            return results
        except Exception as e:
            logging.error(f"Form search failed: {e}")
            return []

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        forms = self.search_forms(query)
        chunks = self.search_chunks(query, top_k=top_k)
        
        return {
            "forms": forms[:3],
            "chunks": chunks
        }


# Quick test run
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    retriever = HUFLITRetriever()
    print("Testing retriever...")
    res = retriever.retrieve("học phí ngành ngôn ngữ Anh")
    print(f"Chunks retrieved: {len(res['chunks'])}")
    for c in res['chunks'][:1]:
        print(f"  - [{c['chunk_id']}] score={c['retrieval_score']} | Title: {c['title']}")
        print(f"    Text: {c['chunk_text'][:200]}...")
        
    res_kcb = retriever.retrieve("79-001")
    print(f"KCB matches: {len(res_kcb['chunks'])}")
    for c in res_kcb['chunks'][:1]:
        print(f"  - [{c['chunk_id']}] score={c['retrieval_score']}")
        print(f"    Text: {c['chunk_text'][:200]}...")
