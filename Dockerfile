# Multi-stage build for minimal Railway deployment
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_THREADPOOL_SIZE=1

# Copy requirements and install dependencies
COPY requirements.txt .
# Install CPU-only PyTorch first to avoid CUDA wheels
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.0.1 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir mediapipe==0.10.7 --no-deps \
    && find /opt/venv -type d -name "__pycache__" -prune -exec rm -rf {} + \
    && find /opt/venv -type d \( -name "tests" -o -name "test" \) -prune -exec rm -rf {} +

# Final stage - minimal runtime
FROM python:3.10-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONOPTIMIZE=1

# Set working directory
WORKDIR /app

# Copy only essential files
COPY gradio_isl_demo.py common.py ./
COPY checkpoints/best_gru.pt ./checkpoints/

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod +x gradio_isl_demo.py

USER appuser

## Environment variables set earlier

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "gradio_isl_demo.py"]
