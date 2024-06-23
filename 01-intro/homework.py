
import requests 
from elasticsearch import Elasticsearch, exceptions
from tqdm import tqdm
import textwrap
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()



def push_docs(client, index):
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    for course in tqdm(documents_raw):
        course_name = course['course']

        for doc in tqdm(course['documents']):
            doc['course'] = course_name
            insert_to_index(doc, client, index)


def create_index(client, index_name, index_config):
    client.indices.create(index=index_name, body=index_config)

def insert_to_index(doc, client, index_name):
    client.index(index=index_name, document=doc)

def search(index, query, client):
    response = client.search(index=index, body=query)
    return response


def build_prompt(question, hits):
    context_template = textwrap.dedent("""
    Q: {question}
    A: {text}
    """).strip()

    context_list = []
    for h in hits:
        context_list.append(context_template.format(
            question=h["_source"]["question"],
            text=h["_source"]["text"]
        ))

    context = "\n".join(context_list)

    prompt_template = textwrap.dedent("""
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """).strip()

    return prompt_template.format(
        question=question,
        context=context
    )

def generate_response(prompt):
    client = OpenAI()
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    response = []
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response.append(chunk.choices[0].delta.content)
    return "".join(response)

def main():
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
                "course": {"type": "keyword"} 
            }
        }
    }
    ## Creating index in elastic search
    index_name = "homework"

    if es.indices.exists(index=index_name):
        try:
            es.indices.delete(index=index_name)
            print(f"Index '{index_name}' deleted successfully.")
        except exceptions.RequestError as e:
            print(f"Failed to delete index '{index_name}': {e.info}")
            raise Exception(e)

    create_index(es, index_name, index_settings)
    push_docs(es, index_name)

    ## Search elements
    question = "How do I execute a command in a running docker container?"
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": question,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                }
                }
            }
        }
    ## Result of Question 3
    response = search(index_name,search_query, es)
    print(f"Question 3: {response["hits"]["max_score"]}")

    ## Result of Question 4
    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": question,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
                }
                }
            }
    }
    response = search(index_name,search_query, es)
    print(f"Question 4: {response["hits"]["hits"][-1]["_source"]["question"]}")

    ## Result of Question 5
    llm_prompt = build_prompt(question, response["hits"]["hits"])

    print(f"Question 5: {len(llm_prompt)}")

    ## Result of Question 6
    encoding = tiktoken.encoding_for_model("gpt-4o")

    print(f"Question 6: {len(encoding.encode(llm_prompt))}")

    ## Result of Question 7
    output_prompt = generate_response(llm_prompt)
    print(f"Question 7\n:{output_prompt}")

    ## Result of Question 8 
    out_tokens = len(encoding.encode(output_prompt))
    in_tokens = len(encoding.encode(llm_prompt))
    price = (0.005*in_tokens/1000) + (0.015*out_tokens/1000)
    print(f"Question 8: ${1000*price}")
    
    

if __name__ == "__main__":
    main()