# Build and run the Lung Disease Management API server.
# Usage:
#   docker build -t lung-disease-api .
#   docker run -p 8000:8000 lung-disease-api

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Python dependencies first (layer-cache friendly)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/

# Expose API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
