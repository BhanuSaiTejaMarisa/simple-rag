from fastapi import FastAPI
from pydantic import BaseModel
from agent import ask
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI()

# --- REQUEST / RESPONSE MODELS ---
class Message(BaseModel):
    role: str    # "human" or "ai"
    content: str

class AskRequest(BaseModel):
    query: str
    history: list[Message] = []

class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    history: list[Message]

# --- ENDPOINT ---
@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    # convert Message objects back to LangChain message types
    history = [
        HumanMessage(content=m.content) if m.role == "human" else AIMessage(content=m.content)
        for m in request.history
    ]

    result = ask(query=request.query, history=history)

    # convert LangChain messages back to serializable Message objects
    updated_history = [
        Message(role="human" if isinstance(m, HumanMessage) else "ai", content=m.content)
        for m in result["history"]
    ]

    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        history=updated_history
    )

@app.get("/health")
def health():
    return {"status": "ok"}
