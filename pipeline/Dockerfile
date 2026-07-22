FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GCP_PROJECT=buyorwait-2026
ENV TOP_N=2000

CMD ["python3", "fetch_recent.py"]
