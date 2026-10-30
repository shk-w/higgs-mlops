# Base image: Python on a minimal Linux. Match the version you use locally (python --version)
FROM python:3.11.5

# PYTHONPATH lets "import higgs" work; the other two make logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install dependencies first (see the note on caching below)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Then copy your code and the trained artifacts
COPY src/ src/
COPY artifacts/ artifacts/

EXPOSE 8000

# Docker calls /health periodically to know whether the container is working
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# --host 0.0.0.0 is required inside a container, otherwise the API is unreachable from outside
CMD ["python", "-m", "uvicorn", "higgs.serve:app", "--host", "0.0.0.0", "--port", "8000"]