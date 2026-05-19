from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings()

# chunks need to be in memory for BM25 — load them regardless
loader = DirectoryLoader(DOCS_DIR, glob="**/*.txt", loader_cls=TextLoader)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# --- PERSISTENT CHROMADB ---
if os.path.exists(CHROMA_DIR):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
else:
    print(f"Building vector store from docs... ({len(chunks)} chunks)")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("Vector store saved to chroma_db/")

# --- HYBRID RETRIEVER ---
# BM25: keyword search over raw chunks
# ChromaDB: semantic/vector search
# EnsembleRetriever merges both, weights control how much each contributes
bm25_retriever = BM25Retriever.from_documents(chunks, k=3)
vector_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided context."

Context: {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | llm

# conversation history for this session
history = []

while True:
    query = input("\nAsk (or 'exit'): ").strip()
    if query.lower() == "exit":
        break

    # optional metadata filter: prefix query with [docname] e.g. [transformer] what is attention
    # searches all docs if no prefix given
    filter_source = None
    if query.startswith("["):
        end = query.find("]")
        if end != -1:
            filter_source = query[1:end].strip() + ".txt"
            query = query[end+1:].strip()

    if filter_source:
        # scoped search: only vector search supports metadata filtering
        scoped_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "filter": {"source": f"docs/{filter_source}"}}
        )
        relevant = scoped_retriever.invoke(query)
        print(f"\n[Searching only: {filter_source}]")
    else:
        # full hybrid search across all docs
        relevant = retriever.invoke(query)

    if not relevant:
        print("\nAnswer: I don't know based on the provided context.")
        continue

    print("\n[Retrieved chunks]")
    for doc in relevant:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  {source} | {doc.page_content[:80]}...")

    context = "\n".join([doc.page_content for doc in relevant])
    response = chain.invoke({"context": context, "question": query, "history": history})
    print("\nAnswer:", response.content)

    # only store turns where the LLM actually had a useful answer
    if "don't know" not in response.content.lower():
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=response.content))
