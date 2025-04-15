FROM python:3.11-slim

WORKDIR /app

# Копирование файла зависимостей
COPY requirements.txt .

# Обновление pip и установка зависимостей
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Установка системных зависимостей для unstructured и git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    git \
    libmagic1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Копирование остальных файлов проекта
COPY main.py .
COPY .env .
COPY static /app/static
COPY start.sh .

# Создание директории для документов
RUN mkdir -p docs faiss_index
RUN touch last_updated.txt rebuild_log.txt

# Сделать скрипт запуска исполняемым
RUN chmod +x start.sh

# Порт для FastAPI
EXPOSE 8000

# Запуск FastAPI приложения
CMD ["./start.sh"]