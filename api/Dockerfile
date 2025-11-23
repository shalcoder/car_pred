FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements from api folder
COPY api/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Copy src folder for imports
COPY src /app/src

# Copy api folder
COPY api /app/api

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
