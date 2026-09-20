# Hugging Face Spaces (Docker SDK) image for the Streamlit app.
# HF runs the container as uid 1000, so everything the app writes at runtime
# (SQLite store, Chroma dir, sentence-transformers model cache) lives under a
# HOME that user owns.

FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /home/user/app

# CPU-only torch first, so sentence-transformers doesn't pull the multi-GB
# CUDA build on a CPU Space.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user . .

USER user

# Bake the embedding model into the image so the first search isn't a cold
# download at request time.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860
CMD ["streamlit", "run", "app/main.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
