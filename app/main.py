import json

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.chain import build_chain_and_streaming
from app.config import API_KEY, RAGEVAL_API_KEY

from rag_eval import track

app = FastAPI(title="RAG Portfolio API")

_chain = None
_stream_fn = None

api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


def _ensure_initialized():
    global _chain, _stream_fn
    if _chain is None:
        _chain, _stream_fn = build_chain_and_streaming()


def get_chain():
    _ensure_initialized()
    return _chain


def get_stream_fn():
    _ensure_initialized()
    return _stream_fn


class QueryRequest(BaseModel):
    question: str


class Chunk(BaseModel):
    content: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    chunks: list[Chunk]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_api_key)])
def query(request: QueryRequest):
    try:
        result = get_chain().invoke(request.question)
        track(question=request.question, answer=result["answer"], chunks=result["chunks"], api_key=RAGEVAL_API_KEY)
        return QueryResponse(answer=result["answer"], chunks=result["chunks"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream", dependencies=[Depends(verify_api_key)])
async def query_stream(request: QueryRequest):
    async def generate():
        answer = ""
        chunks: list[dict] = []
        try:
            async for event in get_stream_fn()(request.question):
                if event["type"] == "token":
                    answer += event["content"]
                elif event["type"] == "sources":
                    chunks = event["content"]
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            return
        if answer:
            track(question=request.question, answer=answer, chunks=chunks, api_key=RAGEVAL_API_KEY)

    return StreamingResponse(generate(), media_type="text/event-stream")
