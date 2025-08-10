# Dockerfile — Data Analyst Agent
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application
COPY . /app

# Optional non-root user
RUN useradd -m agentuser
USER agentuser

ENV PORT=8000
EXPOSE 8000

# Run with gunicorn + uvicorn worker, using a config file
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.main:app"]
