"""
//bash
  pip install openai langchain chromadb
//python 
  documents = ["React is a frontend library", "Python is used for AI"] 

  from langchain.vectorstores import Chroma 
  from langchain.embeddings import OpenAIEmbeddings

  db = Chroma.from_texts(documents, OpenAIEmbeddings()) 

  query = "What is React?" 
  docs = db.similarity_search(query) 
  
  from langchain.chat_models import ChatOpenAI 
  llm = ChatOpenAI() 
  
  response = llm.predict(f"Answer based on: {docs}") 
  print(response) 



bhanusaitejamarisa@Bhanus-MacBook-Air simple-rag % python3 -m venv venv
bhanusaitejamarisa@Bhanus-MacBook-Air simple-rag % source venv/bin/activate
(venv) bhanusaitejamarisa@Bhanus-MacBook-Air simple-rag % pip install openai langchain chromadb tiktoken
"""


# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_openai import ChatOpenAI

# # Step 1: Data
# documents = [
#     "React is a frontend library",
#     "Python is used for AI"
# ]

# # Step 2: Embeddings + Vector DB
# embeddings = HuggingFaceEmbeddings()

# db = Chroma.from_texts(documents, embeddings)

# # Step 3: Query
# query = input("Ask something: ")

# docs = db.similarity_search(query)

# # Step 4: LLM
# llm = ChatOpenAI()

# context = " ".join([doc.page_content for doc in docs])

# response = llm.invoke(f"Answer based on: {context}")

# print("\nAnswer:\n", response.content)

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

documents = [
    "React is a frontend library",
    "Python is used for AI"
]

embeddings = HuggingFaceEmbeddings()

db = Chroma.from_texts(documents, embeddings)
llm = ChatOpenAI()

while True:
    query = input("\nAsk: ")
    if query == "exit":
        break

    docs = db.similarity_search(query)
    context = " ".join([doc.page_content for doc in docs])
    response = llm.invoke(f"Answer based on: {context}")
    print("\nAnswer:", response.content)