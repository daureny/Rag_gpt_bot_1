#!/bin/bash
# Скрипт для копирования индекса с Google Drive на persistent storage Render

echo "Начинаем копирование индекса..."
cd /tmp

echo "Скачиваем файл с Google Drive..."
curl -L "https://drive.google.com/uc?export=download&id=13w_RsfyJ6m1sbPEZle_tM8EFPxZv3vwT" -o index.zip

echo "Проверяем, что файл скачался..."
ls -la index.zip

echo "Распаковываем архив..."
unzip -o index.zip

echo "Создаем директорию для индекса, если её еще нет..."
mkdir -p /data/faiss_index

echo "Копируем все файлы в директорию persistent storage..."
cp -r index/* /data/faiss_index/

echo "Проверяем, что файлы скопированы..."
ls -la /data/faiss_index/

echo "Индекс успешно скопирован!"