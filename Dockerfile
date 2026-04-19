# ── Stage 1: build Next.js static site ───────────────────────────────────────
FROM node:20-slim AS frontend

WORKDIR /app/web
COPY apps/web/package*.json ./
RUN npm install

COPY apps/web/ ./

# NEXT_PUBLIC_API_URL="" means same-origin calls — works because
# FastAPI serves both the frontend and the /api routes on one domain.
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build
# Output is in /app/web/out


# ── Stage 2: Python API + static files ───────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# System deps (needed by lightgbm / xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY apps/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# API source code
COPY apps/api/ .

# Copy built frontend from stage 1
COPY --from=frontend /app/web/out ./web/out

EXPOSE 8000

ENV STATIC_DIR=/app/web/out

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
