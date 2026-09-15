import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from generation_engine_complete import ALM_RAG_System
from persistence.database import init_db, get_db_session
import persistence.repository as repo


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)

# Initialize your master RAG system pointing to your PDF documents folder
rag = ALM_RAG_System("./alm_docs")


class ChatRequest(BaseModel):
    prompt: str
    username: str
    session_id: uuid.UUID | None = None  # omit to start a new conversation


@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db_session)):
    user = await repo.get_or_create_user(db, request.username)

    if request.session_id:
        session = await repo.get_session(db, request.session_id)
        if not session or session.user_id != user.id:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = await repo.create_session(db, user.id)

    await repo.add_message(db, session.id, "user", request.prompt)
    await repo.maybe_set_session_title(db, session, request.prompt)

    # NOTE: guardrail was previously bypassed here -- this endpoint used to
    # reimplement rewrite/retrieve/generate manually instead of calling
    # rag.ask(), so PromptGuardrail.validate() never ran. Fixed below.
    if not rag.guard.validate(request.prompt):
        rejection = "\u274c Error: Query outside ALM scope."
        await repo.add_message(db, session.id, "assistant", rejection, extra={"guardrail_blocked": True})

        def rejection_stream():
            yield rejection

        return StreamingResponse(
            rejection_stream(), media_type="text/plain", headers={"X-Session-Id": str(session.id)}
        )

    optimized_query = rag.rewriter.rewrite(request.prompt)
    context = rag.retriever.retrieve(optimized_query)
    history = ""  # populated by the context-management step (next up)

    prompt = rag.generator.prompt_template.format(
        context="\n\n".join(context),
        history=history,
        query=optimized_query,
    )

    full_response = {"text": ""}

    def stream_generator():
        for chunk in rag.generator.llm.stream(prompt):
            full_response["text"] += chunk
            yield chunk

    async def persist_assistant_reply():
        async with AsyncSession(bind=db.bind) as bg_db:
            await repo.add_message(
                bg_db,
                session.id,
                "assistant",
                full_response["text"],
                extra={"sources_used": len(context)},
            )

    background_tasks.add_task(persist_assistant_reply)

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
        headers={"X-Session-Id": str(session.id)},
        background=background_tasks,
    )


@app.get("/sessions")
async def get_sessions(username: str, db: AsyncSession = Depends(get_db_session)):
    user = await repo.get_or_create_user(db, username)
    sessions = await repo.list_sessions(db, user.id)
    return [
        {"id": str(s.id), "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at}
        for s in sessions
    ]


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: uuid.UUID, db: AsyncSession = Depends(get_db_session)):
    session = await repo.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await repo.get_messages(db, session_id)
    return [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)