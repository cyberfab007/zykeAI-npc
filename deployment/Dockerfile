FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000

WORKDIR /app

# Copy root requirements to reuse full dependency set
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy deployment app
COPY deployment /app/deployment

EXPOSE 5000

# Healthcheck hits /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsSL http://127.0.0.1:${PORT}/health || exit 1

# Use gunicorn with threaded workers
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-k", "gthread", "--threads", "4", "--timeout", "60", "deployment.app:app"]
