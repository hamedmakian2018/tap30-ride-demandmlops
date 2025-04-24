FROM python:3.13.3-slim-bookworm


ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app


WORKDIR /app

COPY . .

RUN pip install --no-cache-dir --upgrade pip --timeout=300 -i https://mirror-pypi.runflare.com/simple
RUN pip install --no-cache-dir -e . --timeout=300 -i https://mirror-pypi.runflare.com/simple

RUN python pipeline/run.py

EXPOSE 8080





CMD ["uvicorn", "web.application:app", "--host", "0.0.0.0", "--port", "8080"]





