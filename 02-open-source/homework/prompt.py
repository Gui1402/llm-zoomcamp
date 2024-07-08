from openai import OpenAI
import tiktoken


client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)


def llm(prompt):
    response = client.chat.completions.create(
        model='gemma:2b',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.
    )
    
    return response.choices[0].message.content

response = llm("What's the formula for energy?")

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(response)
len(tokens)