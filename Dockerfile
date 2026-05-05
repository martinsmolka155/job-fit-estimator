FROM python:3.11-slim

# poppler-utils required for pymupdf4llm PDF processing
# NOTE: tesseract intentionally omitted — OCR is not supported in MVP
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
