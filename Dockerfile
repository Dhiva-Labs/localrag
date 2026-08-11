FROM python:3.11-slim

WORKDIR /app

# CPU-only torch, installed first: the default PyPI build drags in ~10 GB of
# CUDA libraries this CPU-only container can never use, and satisfying the
# dependency up front stops the main install from pulling that build in
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

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
