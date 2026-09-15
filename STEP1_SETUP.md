# Step 1 — Persistence layer: setup

## 1. New dependencies
Add to your requirements:
```
sqlalchemy[asyncio]>=2.0
asyncpg
```

## 2. Start Postgres
Place `docker-compose.yml` at your project root, then:
```
docker compose up -d
```
This starts Postgres on `localhost:5432` (db `alm_rag`, user/pass `alm_user`/`alm_password` — change these for anything beyond local dev).

## 3. Files added/changed
- `docker-compose.yml` — Postgres service
- `db/database.py` — async engine, session factory, `init_db()`
- `db/models.py` — `User`, `ChatSession`, `Message` tables
- `db/repository.py` — CRUD helpers used by the backend
- `backend.py` — replaces your current one:
  - creates tables on startup (`init_db()` via lifespan)
  - `/chat` now requires `username` in the request body, accepts optional `session_id`
  - fixes a bug where the input guardrail was never actually being called
  - persists every user + assistant message; returns `X-Session-Id` header
  - new `GET /sessions?username=` and `GET /sessions/{id}/messages`
- `streamlit_app.py` — replaces your current one: asks for a name once, sidebar lists/switches past sessions, loads history from the backend instead of only `st.session_state`

## 4. Not changed yet (later steps)
- `generation_engine_complete.py` — untouched
- `history` is still passed as `""` to the generator — step 2 wires real conversation history / summarization in here
- No per-folder document filtering yet — step 3
- Guardrail is still just the keyword allow-list — step 4/5 will replace it

## 5. Quick test
```
docker compose up -d
pip install -r requirements.txt
python backend.py        # http://localhost:8000
streamlit run streamlit_app.py
```
Type a name, ask a question, refresh the page, log in with the same name — your session should reappear in the sidebar with history intact.
