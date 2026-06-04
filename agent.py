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
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import os
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()

DOCS_DIR = "docs"
CHROMA_DIR = "chroma_db"

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

llm = ChatOpenAI(model="gpt-4o-mini")

# --- AGENT STATE ---
class AgentState(TypedDict):
    query: str
    search_query: str
    filter_source: str | None
    chunks: List
    answer: str
    history: List
    retries: int

# --- NODE 1: RETRIEVE ---
def retrieve(state: AgentState) -> AgentState:
    print(f"\n[Retrieving]: {state['search_query']}")
    if state.get("filter_source"):
        scoped = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 12, "filter": {"source": f"docs/{state['filter_source']}"}}
        )
        results = scoped.invoke(state["search_query"])
        print(f"[Searching only: {state['filter_source']}]")
    else:
        results = retriever.invoke(state["search_query"])
    return {**state, "chunks": results}

# --- NODE 2: GRADE CHUNKS ---
# LLM decides if retrieved chunks are relevant enough to answer the query
grade_prompt = ChatPromptTemplate.from_template("""Are these chunks relevant to answer the question?
Question: {query}
Chunks: {chunks}
Reply with only 'yes' or 'no'.""")

def grade_chunks(state: AgentState) -> str:
    if not state["chunks"] or state["retries"] >= 2:
        return "generate"  # give up retrying, attempt answer or say I don't know

    chunk_text = "\n".join([c.page_content[:200] for c in state["chunks"]])
    result = llm.invoke(grade_prompt.format(query=state["query"], chunks=chunk_text))

    if "yes" in result.content.lower():
        print("[Chunks graded]: relevant ✓")
        return "generate"
    else:
        print("[Chunks graded]: not relevant, rewriting query...")
        return "rewrite"

# --- NODE 3: REWRITE QUERY ---
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """Your only job is to rewrite the follow-up question as a standalone question using the conversation history to improve document retrieval.
Do NOT answer the question.
Do NOT add any information.
Make it more specific and keyword-rich.
Return ONLY the rewritten question, nothing else."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "Rewrite for better retrieval: {question}"),
])

def rewrite_query(state: AgentState) -> AgentState:
    result = llm.invoke(rewrite_prompt.format_messages(
        question=state["search_query"],
        history=state["history"]
    ))
    new_query = result.content.strip()
    print(f"[Rewritten]: {new_query}")
    return {**state, "search_query": new_query, "retries": state["retries"] + 1}

# --- NODE 4: GENERATE ---
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided context."
If you use information from the context, end your answer with 'Source: <filename>' citing only the file you actually used.

Context: {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
answer_chain = answer_prompt | llm

def generate(state: AgentState) -> AgentState:
    if not state["chunks"]:
        return {**state, "answer": "I don't know based on the provided context."}

    context = "\n\n".join([
        f"[{os.path.basename(c.metadata.get('source', 'unknown'))}]\n{c.page_content}"
        for c in state["chunks"]
    ])

    full_response = ""
    for chunk in answer_chain.stream({
        "context": context,
        "question": state["query"],
        "history": state["history"]
    }):
        full_response += chunk.content

    return {**state, "answer": full_response}

# --- BUILD GRAPH ---
graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve)
graph.add_node("rewrite", rewrite_query)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")

# after retrieve → grade chunks → decide: generate or rewrite
graph.add_conditional_edges("retrieve", grade_chunks, {
    "generate": "generate",
    "rewrite": "rewrite"
})

# after rewrite → retrieve again
graph.add_edge("rewrite", "retrieve")

# after generate → done
graph.add_edge("generate", END)

app = graph.compile()

def ask(query: str, history: list, filter_source: str | None = None) -> dict:
    """Main entry point. Returns answer and updated history."""

    # optional metadata filter: [transformer] what is attention
    if query.startswith("["):
        end = query.find("]")
        if end != -1:
            filter_source = query[1:end].strip() + ".txt"
            query = query[end+1:].strip()

    # rewrite query upfront if history exists
    search_query = query
    if history:
        rewritten = llm.invoke(rewrite_prompt.format_messages(
            question=query,
            history=history
        ))
        search_query = rewritten.content.strip()

    result = app.invoke({
        "query": query,
        "search_query": search_query,
        "filter_source": filter_source,
        "chunks": [],
        "answer": "",
        "history": history,
        "retries": 0
    })

    answer = result["answer"]
    sources = list(dict.fromkeys([
        os.path.basename(c.metadata.get("source", "unknown"))
        for c in result["chunks"]
    ]))

    updated_history = history.copy()
    if "don't know" not in answer.lower():
        updated_history.append(HumanMessage(content=query))
        updated_history.append(AIMessage(content=answer))

    return {"answer": answer, "sources": sources, "history": updated_history}
