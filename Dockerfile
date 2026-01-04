# AI Training Service Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install numpy and Cython first to avoid scikit-surprise build issues
RUN pip install --no-cache-dir "numpy==1.24.3" "Cython<3.0" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create results directory for model storage
RUN mkdir -p /app/results

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=training_service.py
ENV FLASK_ENV=production

# Expose port (can be overridden by SERVICE_PORT env var)
ARG SERVICE_PORT=5001
EXPOSE ${SERVICE_PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

# Change to api directory for running
WORKDIR /app/api

# Run the application
CMD ["python", "training_service.py"]

