FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl libpq-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directories inside container (volume mount will overlay them)
RUN mkdir -p /app/data/uploads /app/data/runs /app/data/chroma /app/models

EXPOSE 8000

# No --reload: hot-reload defeats docker cp patching
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
