# Railway Optimized Dockerfile - Minimal and Fast
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary files (not entire directory)
COPY gradio_isl_demo.py common.py ./
COPY checkpoints/best_gru.pt ./checkpoints/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "gradio_isl_demo.py"]
