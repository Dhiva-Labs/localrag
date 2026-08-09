FROM python:3.11-slim

WORKDIR /app

# copy metadata first so dependency layers cache across code-only changes;
# hatchling reads README.md and LICENSE while building package metadata,
# so both must be present before the install step runs
COPY pyproject.toml README.md LICENSE ./
COPY localrag/ localrag/

RUN pip install --no-cache-dir .

# HF_HOME lives under /data so the embedding model download lands on the
# persistent volume instead of this ephemeral image layer
ENV LOCALRAG_HOST=0.0.0.0 \
    LOCALRAG_DATA_DIR=/data \
    HF_HOME=/data/hf

EXPOSE 8090

CMD ["localrag", "serve"]
