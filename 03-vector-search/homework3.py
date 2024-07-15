from sentence_transformers import SentenceTransformer
import requests 
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch



class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]


model = SentenceTransformer("multi-qa-distilbert-cos-v1")
user_question = "I just discovered the course. Can I still join it?"
v = model.encode(user_question)


print(f"######### Question 1: {v[0]}")


base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()


# Filtering
documents = [doc for doc in documents if doc["course"] == "machine-learning-zoomcamp"]
print(f"Lenght of documents is {len(documents)}")

embeddings = []
# Creating embeddings
for doc in tqdm(documents):
    qa_text = f'{doc["question"]} {doc["text"]}'
    embedding = model.encode(qa_text)
    embeddings.append(embedding)
    doc["emb"] = embedding

X = np.array(embeddings)
print(f"######### Question 2: {X.shape}")


print(f"######### Question 3: {X.dot(v).max()}")


search_engine = VectorSearchEngine(documents=documents, embeddings=X)
search_engine.search(v, num_results=5)


base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

hits = []
hr = []
for g in tqdm(ground_truth):
    v = model.encode(g["question"])
    rank = search_engine.search(v, num_results=5)
    rate = [True if g["document"] == r["id"] else False for r in rank]
    hits.append(rate)
    hr.append(any(rate))
print(f"######### Question 4: {sum(hr)/len(hr)}")



es = Elasticsearch("http://localhost:9200")


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
            "course": {"type": "keyword"} ,
            "id": {"type": "text"},
            "emb": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
        }
    }
}

index_name = "course-questions"

es.indices.delete(index=index_name, ignore_unavailable=True)
es.indices.create(index=index_name, body=index_settings)

for doc in tqdm(documents):
    try:
        es.index(index=index_name, document=doc)
    except Exception as e:
        print(e)

# Query
user_question = "I just discovered the course. Can I still join it?"
v = model.encode(user_question)
query = {
    "field": "emb",
    "query_vector": v,
    "k": 5,
    "num_candidates": 10000, 
}
res = es.search(index="course-questions", knn=query, source=["text", "section", "question", "course", "id"])
print(res["hits"]["hits"])

print(f"######### Question 5: {res["hits"]["hits"][0]["_source"]["id"]}")


hits = []
hr = []
for g in tqdm(ground_truth):
    v = model.encode(g["question"])
    query = {
    "field": "emb",
    "query_vector": v,
    "k": 5,
    "num_candidates": 10000, 
    }
    rank = es.search(index="course-questions", knn=query, source=["text", "section", "question", "course", "id"])
    rank = rank["hits"]["hits"]
    rate = [True if g["document"] == r["_source"]["id"] else False for r in rank]
    hits.append(rate)
    hr.append(any(rate))

print(f"######### Question 6: {sum(hr)/len(hr)}")