# ============================================
# Product Review Intelligence System
# Docker image for FastAPI app + frontend
# ============================================
FROM python:3.9-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time
RUN python -c "import nltk; nltk.download('stopwords', download_dir='/usr/local/nltk_data'); nltk.download('wordnet', download_dir='/usr/local/nltk_data'); nltk.download('omw-1.4', download_dir='/usr/local/nltk_data')"

# Copy application code
COPY app/ /app/

# Copy local model files as fallback (if they exist during build)
COPY models/ /app/models/

# Expose the app port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
