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
    libfontconfig1 \
    libfreetype6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy application
COPY . /app

# Create non-root user (but don't switch yet - we need permissions)
RUN useradd -m agentuser && chown -R agentuser:agentuser /app

# Switch to non-root user
USER agentuser

# Set environment variables
ENV PORT=8000
ENV PYTHONPATH=/app
ENV MATPLOTLIB_CACHE_DIR=/tmp/matplotlib

EXPOSE 8000

# Use the correct module path for your structure
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app.main:app"]

