from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import time
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

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
bm25_retriever = BM25Retriever.from_documents(chunks, k=6)
vector_retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 12})
base_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# --- RE-RANKER ---
# retrieves more candidates (k=6) then re-ranks and keeps top 3
# FlashrankRerank runs locally, no API key needed
reranker = FlashrankRerank(top_n=5)
retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)

llm = ChatOpenAI(model="gpt-4o-mini")

# --- QUERY REWRITER ---
# rewrites vague follow-up queries into standalone questions using conversation history
# only rewrites if there is history, otherwise uses the query as-is
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Your only job is to rewrite the follow-up question as a standalone question using the conversation history.
Do NOT answer the question.
Do NOT add any information.
Return ONLY the rewritten question as a single sentence ending with a question mark."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Rewrite this as a standalone question: {question}"),
])
rewrite_chain = rewrite_prompt | llm

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided context."
If you use information from the context, end your answer with 'Source: <filename>' citing only the file you actually used.

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

    # rewrite vague queries using history, skip if no history yet
    search_query = query
    if history:
        rewritten = rewrite_chain.invoke({"question": query, "history": history})
        search_query = rewritten.content.strip()
        if search_query != query:
            print(f"\n[Rewritten query]: {search_query}")

    if filter_source:
        scoped_retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "filter": {"source": f"docs/{filter_source}"}}
        )
        relevant = scoped_retriever.invoke(search_query)
        print(f"[Searching only: {filter_source}]")
    else:
        relevant = retriever.invoke(search_query)

    if not relevant:
        print("\nAnswer: I don't know based on the provided context.")
        continue

    print("\n[Retrieved chunks]")
    for doc in relevant:
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        print(f"  {source} | {doc.page_content[:80]}...")

    # build context with source labels so LLM can cite them
    context = "\n\n".join([
        f"[{os.path.basename(doc.metadata.get('source', 'unknown'))}]\n{doc.page_content}"
        for doc in relevant
    ])
    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    for chunk in chain.stream({"context": context, "question": query, "history": history}):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content
    print()

    if "don't know" not in full_response.lower():
        history.append(HumanMessage(content=query))
        history.append(AIMessage(content=full_response))
