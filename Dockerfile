FROM python:3.11-slim

WORKDIR /app


RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Установка системных зависимостей для unstructured и git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Копирование файлов проекта
COPY requirements.txt .
COPY main.py .
COPY .env .
COPY static /app/static
COPY start.sh .

# Создание директории для документов
RUN mkdir -p docs faiss_index
RUN touch last_updated.txt rebuild_log.txt

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Сделать скрипт запуска исполняемым
RUN chmod +x start.sh

# Запуск FastAPI приложения
CMD ["./start.sh"]

# Порт для FastAPI
EXPOSE 8000