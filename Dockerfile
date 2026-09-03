FROM python:3.13.15-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY dashboard ./dashboard
COPY scripts ./scripts
COPY data ./data
COPY results ./results
COPY models ./models

EXPOSE 8000 8501

CMD ["hypercorn", "api.main:app", "--bind", "0.0.0.0:8000"]