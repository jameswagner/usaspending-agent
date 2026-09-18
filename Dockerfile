FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.9.6 /uv /uvx /bin/

WORKDIR /app

# Dependency layer cached separately from app code so code-only changes
# don't reinstall everything.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY backend/ backend/
COPY data/raw/ data/raw/
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}"
ENV HF_HOME=/app/.cache/huggingface
ENV CONVERSATIONS_DB_PATH=/data/conversations.db

# Build the retrieval indices once, from the static source files in
# data/raw/, and bake them into the image - they're never written to again
# at runtime (see #71). Only the conversation-history DB, written at
# runtime, needs a mounted Volume. This also downloads and caches the
# sentence-transformers/cross-encoder model weights (via HF_HOME above),
# avoiding the ~15-20s cold-start download on every boot.
RUN python -m backend.app.retrieval.pipeline.ingest \
      --pdf data/raw/analyst-guide.pdf --out data/chunks/analysts_guide_chunks.jsonl \
 && python -m backend.app.retrieval.pipeline.ingest_glossary \
 && python -m backend.app.retrieval.pipeline.vector_index \
      --chunks data/chunks/analysts_guide_chunks.jsonl data/chunks/glossary_chunks.jsonl \
 && python -m backend.app.retrieval.pipeline.bm25_index \
      --chunks data/chunks/analysts_guide_chunks.jsonl data/chunks/glossary_chunks.jsonl \
 && python -m backend.app.retrieval.pipeline.ingest_naics \
 && python -m backend.app.retrieval.pipeline.ingest_psc \
 && python -m backend.app.retrieval.pipeline.ingest_cfda \
 && CHROMA_DB_DIR=./data/chroma_naics WHOOSH_INDEX_DIR=./data/whoosh_naics \
      python -m backend.app.retrieval.pipeline.vector_index --chunks data/chunks/naics_chunks.jsonl \
 && CHROMA_DB_DIR=./data/chroma_naics WHOOSH_INDEX_DIR=./data/whoosh_naics \
      python -m backend.app.retrieval.pipeline.bm25_index --chunks data/chunks/naics_chunks.jsonl \
 && CHROMA_DB_DIR=./data/chroma_psc WHOOSH_INDEX_DIR=./data/whoosh_psc \
      python -m backend.app.retrieval.pipeline.vector_index --chunks data/chunks/psc_chunks.jsonl \
 && CHROMA_DB_DIR=./data/chroma_psc WHOOSH_INDEX_DIR=./data/whoosh_psc \
      python -m backend.app.retrieval.pipeline.bm25_index --chunks data/chunks/psc_chunks.jsonl \
 && CHROMA_DB_DIR=./data/chroma_cfda WHOOSH_INDEX_DIR=./data/whoosh_cfda \
      python -m backend.app.retrieval.pipeline.vector_index --chunks data/chunks/cfda_chunks.jsonl \
 && CHROMA_DB_DIR=./data/chroma_cfda WHOOSH_INDEX_DIR=./data/whoosh_cfda \
      python -m backend.app.retrieval.pipeline.bm25_index --chunks data/chunks/cfda_chunks.jsonl

# Railway injects PORT at runtime; single-instance/single-worker only (see
# #71) - the SqliteSaver conversation-history checkpointer is one sqlite3
# connection held by one process, not safe for --workers > 1 or multiple
# replicas (see #72 for the Postgres swap that would allow that).
#
# --forwarded-allow-ips='*': the container is only reachable through
# Railway's edge proxy, never directly from the internet, so that proxy is
# the sole trusted hop - this makes uvicorn's ProxyHeadersMiddleware trust
# its X-Forwarded-For and rewrite request.client.host accordingly (see #3).
CMD ["sh", "-c", "mkdir -p $(dirname \"$CONVERSATIONS_DB_PATH\") && uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
