FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps required for some packages (kept minimal)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY api/requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files into the image
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run the FastAPI app. `api.main:app` refers to `api/main.py` in the project structure.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
