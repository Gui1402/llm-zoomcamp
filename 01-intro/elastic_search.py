from elasticsearch import Elasticsearch
import json
from tqdm import tqdm


es = Elasticsearch("http://localhost:9200")

## Index in elastic search is like a table in a relational database
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
index_name = "course-questions"
es.indices.create(index=index_name, body=index_settings)

with open("01-intro/documents.json", "rt") as f:
    docs_raw = json.load(f)

documents = []

for course_dict in docs_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course']
        documents.append(doc)

for doc in tqdm(documents):
    es.index(index=index_name, document=doc)



## Querying

query = "When course starts?"
search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}

response = es.search(index=index_name, body=search_query)
result_docs = []
for hit in response["hits"]["hits"]:
    result_docs.append(hit["_source"])