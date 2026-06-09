FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required by OpenCV and other ML libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose a default port (Railway injects $PORT at runtime)
EXPOSE 8000

# Start server using shell form to allow environment variable interpolation
CMD sh -c "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}"
