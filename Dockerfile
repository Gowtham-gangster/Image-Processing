FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV and other ML libraries
# libgomp1 is required to prevent OpenMP segfaults when TensorFlow and PyTorch
# are loaded together in the same process (MTCNN uses TF, YOLO uses PyTorch)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Environment variables to prevent OpenMP segfaults (TF + PyTorch conflict)
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1
# Ensure Python output is flushed immediately to Railway logs
ENV PYTHONUNBUFFERED=1

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose a default port (Railway injects $PORT at runtime)
EXPOSE 8000

# Start server using shell form to allow environment variable interpolation
CMD sh -c "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}"
