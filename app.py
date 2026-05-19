from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings()

# --- PERSISTENT CHROMADB ---
# If chroma_db/ exists, load it. Otherwise, build it from docs and save.
if os.path.exists(CHROMA_DIR):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    print("Building vector store from docs...")

    # Load all .txt files from docs/
    loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    # Chunk each doc into pieces of 500 chars, with 100 char overlap between chunks
    # Overlap ensures a sentence split across two chunks isn't lost
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} docs → split into {len(chunks)} chunks")

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("Vector store saved to chroma_db/")

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided context."

Context: {context}

Question: {question}
""")

chain = prompt | llm

while True:
    query = input("\nAsk (or 'exit'): ").strip()
    if query.lower() == "exit":
        break

    # MMR: fetch 10 candidates, return 3 most relevant AND diverse
    relevant = db.max_marginal_relevance_search(query, k=3, fetch_k=10)

    if not relevant:
        print("\nAnswer: I don't know based on the provided context.")
        continue

    print("\n[Retrieved chunks]")
    for doc in relevant:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  {source} | {doc.page_content[:80]}...")

    context = "\n".join([doc.page_content for doc in relevant])
    response = chain.invoke({"context": context, "question": query})
    print("\nAnswer:", response.content)
