from fastapi import APIRouter
from pydantic import BaseModel
from app.rag import retrieve, collection_count
from app.memory import ShortTermMemory, LongTermMemory
from app.persona import build_prompt
from app.gemini import generate

router = APIRouter()

_short_term: dict[str, ShortTermMemory] = {}
_long_term = LongTermMemory()


def _get_short_term(session_id: str) -> ShortTermMemory:
    if session_id not in _short_term:
        _short_term[session_id] = ShortTermMemory()
    return _short_term[session_id]


class ChatRequest(BaseModel):
    message: str
    session_id: str


class SessionEndRequest(BaseModel):
    session_id: str


@router.post("/chat")
def chat(req: ChatRequest):
    stm = _get_short_term(req.session_id)
    chunks = retrieve(req.message)
    long_term = _long_term.load(req.session_id)
    history = stm.get_history()

    prompt = build_prompt(req.message, chunks, history, long_term)
    response = generate(prompt)

    stm.add_turn("user", req.message)
    stm.add_turn("assistant", response)

    sources = list({c["source"] for c in chunks})
    return {"response": response, "sources": sources}


@router.post("/session/end")
def end_session(req: SessionEndRequest):
    stm = _get_short_term(req.session_id)
    history = stm.get_history()
    _long_term.summarize_and_save(req.session_id, history, generate)
    stm.clear()
    if req.session_id in _short_term:
        del _short_term[req.session_id]
    return {"status": "ok"}


@router.get("/memory/{session_id}")
def get_memory(session_id: str):
    entries = _long_term.load(session_id)
    return {"session_id": session_id, "memories": entries}


@router.get("/health")
def health():
    return {"status": "ok", "docs_loaded": collection_count()}
