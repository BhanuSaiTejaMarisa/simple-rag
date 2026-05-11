from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --- 1. DOCUMENTS (your "knowledge base") ---
# In a real app these come from PDFs, websites, databases etc.
documents = [
    "React is a JavaScript frontend library for building user interfaces.",
    "Python is a programming language widely used for AI and data science.",
    "LangChain is a framework for building LLM-powered applications.",
    "ChromaDB is a vector database used to store and search embeddings.",
]

# --- 2. EMBEDDINGS ---
# Converts text into a vector of 768 numbers representing semantic meaning.
# "React library" and "frontend framework" will have similar vectors.
embeddings = HuggingFaceEmbeddings()  # runs locally, no API key needed

# Uncomment to inspect raw embeddings:
vec = embeddings.embed_query("React")
print(f"Dims: {len(vec)}, Sample: {vec[:5]}")

# --- 3. VECTOR STORE ---
# Stores embedded documents. similarity_search uses cosine similarity.
db = Chroma.from_texts(documents, embeddings)

# See what's stored in ChromaDB
collection = db._collection.get(include=["embeddings", "documents"])
print(f"\nStored docs: {collection['documents']}")
print(f"Embedding shape: {len(collection['embeddings'][0])} dims")

# --- 4. LLM ---
# gpt-4o-mini is cheap and capable. gpt-3.5-turbo also works here.
llm = ChatOpenAI(model="gpt-4o-mini")

# --- 5. PROMPT TEMPLATE ---
# Separating prompt from code is best practice.
# The LLM is instructed to only use the provided context.
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided context."

Context: {context}

Question: {question}
""")

# --- 6. CHAIN (LCEL) ---
# prompt | llm pipes the formatted prompt directly into the LLM.
chain = prompt | llm

while True:
    query = input("\nAsk (or 'exit'): ").strip()
    if query.lower() == "exit":
        break

    # similarity_search_with_score returns (doc, score) — lower = more similar
    results = db.similarity_search_with_score(query, k=2)

    print("\n[Retrieved chunks]")
    for doc, score in results:
        print(f"  score={score:.4f} | {doc.page_content}")

    context = "\n".join([doc.page_content for doc, _ in results])
    response = chain.invoke({"context": context, "question": query})
    print("\nAnswer:", response.content)
