# Build from the repository root:  docker build -f backend/Dockerfile .

# --- Stage 1: build the React app ---------------------------------------------------------
FROM node:24-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Stage 2: the FastAPI app, which also serves the built frontend -----------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    FRONTEND_BUILD_DIR=/app/frontend/build

WORKDIR /app/backend

# Dependencies first, so editing application code does not reinstall them
COPY backend/pyproject.toml ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

COPY backend/ ./
COPY --from=frontend /frontend/build /app/frontend/build

RUN useradd --system --no-create-home careflow
USER careflow

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=2)"

# One worker on purpose: Socket.IO rooms live in this process (see Future Work in the PRD)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
