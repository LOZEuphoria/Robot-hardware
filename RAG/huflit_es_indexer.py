import json
import time
from elasticsearch import Elasticsearch, helpers
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Connect to ES
es = Elasticsearch("http://localhost:9200")

def wait_for_es():
    print("Waiting for Elasticsearch to start...")
    for _ in range(60):
        try:
            if es.ping():
                print("Elasticsearch is UP!")
                return True
        except Exception:
            pass
        time.sleep(2)
    print("Failed to connect to Elasticsearch.")
    sys.exit(1)

def setup_indices():
    # Define mapping with standard analyzer
    index_settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "title": {"type": "text"},
                "chunk_text": {"type": "text"},
                "source_url": {"type": "keyword", "index": False},
                "file_url": {"type": "keyword", "index": False},
                "published_at": {"type": "date", "format": "yyyy-MM-dd||strict_date_optional_time"},
                "category": {"type": "keyword"}
            }
        }
    }
    
    form_settings = {
        "mappings": {
            "properties": {
                "form_name": {"type": "text"},
                "description": {"type": "text"},
                "download_url": {"type": "keyword"}
            }
        }
    }

    if es.indices.exists(index="huflit_chunks"):
        es.indices.delete(index="huflit_chunks")
    es.indices.create(index="huflit_chunks", body=index_settings)

    if es.indices.exists(index="huflit_forms"):
        es.indices.delete(index="huflit_forms")
    es.indices.create(index="huflit_forms", body=form_settings)

def index_data(file_path):
    print(f"Loading data from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = data.get("rag_chunks", [])
    forms = data.get("structured_records", {}).get("forms", [])
    
    print(f"Found {len(chunks)} chunks and {len(forms)} forms.")
    
    # Prepare bulk actions for chunks
    chunk_actions = []
    for i, c in enumerate(chunks):
        action = {
            "_index": "huflit_chunks",
            "_id": c.get("chunk_id") or f"chunk_{i}",
            "_source": {
                "chunk_id": c.get("chunk_id", ""),
                "doc_id": c.get("doc_id", ""),
                "doc_type": c.get("doc_type", ""),
                "title": c.get("title", ""),
                "chunk_text": c.get("chunk_text", ""),
                "source_url": c.get("source_url", ""),
                "file_url": c.get("file_url", ""),
                "published_at": c.get("published_at") or None,
                "category": c.get("category", "")
            }
        }
        chunk_actions.append(action)
        
    # Prepare bulk actions for forms
    form_actions = []
    for i, f in enumerate(forms):
        action = {
            "_index": "huflit_forms",
            "_id": f"form_{i}",
            "_source": {
                "form_name": f.get("form_name", ""),
                "description": f.get("description", ""),
                "download_url": f.get("download_url", ""),
                "source_article_url": f.get("source_article_url", "")
            }
        }
        form_actions.append(action)
        
    print("Indexing chunks to ES...")
    helpers.bulk(es, chunk_actions)
    print("Indexing forms to ES...")
    helpers.bulk(es, form_actions)
    print("Done indexing!")

if __name__ == "__main__":
    wait_for_es()
    setup_indices()
    index_data("huflit_rag_db_production.json")
