FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    unixodbc-dev \
    unixodbc \
    && rm -rf /var/lib/apt/lists
RUN pip install -r requirements.txt --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --timeout 120 --retries 5
COPY main.py .
COPY rag.py .
COPY semantic_search.py .

EXPOSE 8000

CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000"]
