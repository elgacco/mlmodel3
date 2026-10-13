FROM python:3.11-slim

WORKDIR /app

# сначала зависимости: этот слой кэшируется и не пересобирается при правках кода
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# затем код и артефакты модели
COPY src/ src/
COPY model/ model/

EXPOSE 8000

# порт берём из переменной PORT (её задают хостинги), по умолчанию 8000
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
