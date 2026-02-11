FROM python:3.12-slim

WORKDIR /app

# Системные зависимости для Pillow и psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY prismalab/ prismalab/
COPY create_admin.py .
COPY setup_env.py .
# Альбомы примеров (создаются через /getfileid; создай пустой [] если файла нет)
COPY examples_albums.json ./

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["python", "-m", "prismalab.bot"]
