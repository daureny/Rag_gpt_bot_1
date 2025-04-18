#!/usr/bin/env python3
"""
Скрипт для локальной индексации документов и сохранения индекса внутри проекта.

Использование:
    python build_index_local.py [--docs-dir DIR] [--openai-api-key KEY] [--max-docs NUM] [--direct-copy]

По умолчанию скрипт:
1. Использует документы из директории ./docs внутри проекта
2. Обрабатывает все документы и создает FAISS индекс
3. Сохраняет готовый индекс в локальную директорию ./index внутри проекта
"""

import os
import sys
import time
import json
import shutil
import tempfile
import argparse
from datetime import datetime
from pathlib import Path

# Проверка наличия необходимых библиотек
try:
    from dotenv import load_dotenv
    from pypdf import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import (
        TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
    )
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Для работы скрипта необходимо установить библиотеки. Запустите:")
    print("pip install langchain langchain_community langchain_openai pypdf python-dotenv")
    sys.exit(1)

# Загрузка переменных окружения из .env файла, если он существует
load_dotenv()

# Константы
DEFAULT_DOCS_DIR = "./docs"  # Директория с документами внутри проекта
INDEX_DIR = "./index"  # Путь для сохранения индекса внутри проекта
RENDER_INDEX_DIR = "/data"  # Путь к директории Render для возможности прямого копирования


def parse_arguments():
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Локальная индексация документов и сохранение внутри проекта.')
    parser.add_argument('--docs-dir', default=DEFAULT_DOCS_DIR,
                        help=f'Директория с документами (по умолчанию: {DEFAULT_DOCS_DIR})')
    parser.add_argument('--openai-api-key',
                        help='API ключ OpenAI (по умолчанию берется из переменной OPENAI_API_KEY)')
    parser.add_argument('--max-docs', type=int, default=0,
                        help='Максимальное количество документов для обработки (0 = все документы)')
    parser.add_argument('--direct-copy', action='store_true',
                        help='Копировать индекс напрямую в директорию Render (для запуска на Render)')

    return parser.parse_args()


def extract_title(text, filename):
    """Извлекает заголовок из документа для лучшего представления в поиске"""
    try:
        # Сначала пытаемся получить осмысленный заголовок из первых строк
        lines = text.splitlines()[:10]  # Проверяем первые 10 строк

        # Ищем типичные паттерны заголовков документов
        for line in lines:
            line = line.strip()
            if len(line.strip()) > 10 and any(
                    kw in line.upper() for kw in ["ЗАКОН", "ПРАВИЛ", "ПОСТАНОВЛ", "МСФО", "КОДЕКС", "РЕГУЛИРОВАНИЕ",
                                                  "ИНСТРУКЦ", "ПОЛОЖЕНИ", "ТРЕБОВАНИ"]):
                return f"{line} ({filename})"

        # Если паттерн не найден, ищем первую непустую, достаточно длинную строку
        for line in lines:
            line = line.strip()
            if len(line) > 20:  # Убедимся, что это существенная строка
                return f"{line[:100]}... ({filename})"

        # Запасной вариант - просто имя файла с пометкой, что это действительный документ
        return f"Документ: {filename}"
    except Exception as e:
        print(f"Ошибка при извлечении заголовка из {filename}: {e}")
        return f"Документ: {filename}"


def build_index(docs_dir, max_docs=0):
    """Строит FAISS индекс из всех документов в указанной директории"""
    print(f"Начинаем индексацию документов из {docs_dir}...")

    # Проверяем наличие директории с документами
    if not os.path.exists(docs_dir):
        print(f"Ошибка: директория {docs_dir} не существует")
        print(f"Создаю пустую директорию {docs_dir} для размещения документов...")
        try:
            os.makedirs(docs_dir, exist_ok=True)
            print(f"Директория {docs_dir} создана. Пожалуйста, разместите в ней документы и запустите скрипт снова.")
            return None
        except Exception as e:
            print(f"Ошибка при создании директории {docs_dir}: {e}")
            return None

    # Определяем путь к документам
    docs_path = Path(docs_dir)

    # Находим все файлы в директории
    all_files = list(docs_path.glob("**/*.*"))  # Ищем файлы и в поддиректориях
    print(f"Найдено всего файлов: {len(all_files)}")

    # Фильтруем только поддерживаемые форматы
    supported_extensions = [".pdf", ".docx", ".txt", ".html"]
    files_to_process = [f for f in all_files if f.suffix.lower() in supported_extensions]
    print(f"Файлы для обработки: {len(files_to_process)}")

    # Если указано ограничение, берем только указанное количество файлов
    if max_docs > 0 and max_docs < len(files_to_process):
        print(f"Ограничиваем количество документов до {max_docs}")
        files_to_process = files_to_process[:max_docs]

    # Выводим список файлов для обработки
    print("\nСписок файлов для индексации:")
    for i, file in enumerate(files_to_process, 1):
        print(f"{i}. {file.name}")
    print()

    # Если нет файлов для обработки, выходим
    if not files_to_process:
        print(f"Нет файлов для индексации в директории {docs_dir}")
        print("Пожалуйста, добавьте документы в форматах PDF, DOCX, TXT или HTML и запустите скрипт снова.")
        return None

    # Создаем векторайзер для эмбеддингов
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Инициализируем словарь для хранения чанков
    chunk_store = {}

    # Обрабатываем все файлы
    all_docs = []
    error_files = []

    for i, file in enumerate(files_to_process, 1):
        try:
            print(f"[{i}/{len(files_to_process)}] Обработка файла: {file.name}")

            # Выбираем загрузчик в зависимости от типа файла
            if file.suffix.lower() == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif file.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix.lower() == ".docx":
                loader = Docx2txtLoader(str(file))
            elif file.suffix.lower() == ".html":
                loader = UnstructuredHTMLLoader(str(file))
            else:
                print(f"  Пропуск неподдерживаемого формата: {file.suffix}")
                continue

            # Загружаем документ
            pages = loader.load()
            print(f"  Загружено страниц: {len(pages)}")

            # Добавляем метаданные
            for page in pages:
                source_title = extract_title(page.page_content, file.name)
                page.metadata["source"] = source_title
                all_docs.append(page)

            print(f"  Документ успешно обработан")

        except Exception as e:
            print(f"  ОШИБКА при обработке {file.name}: {e}")
            error_files.append((file.name, str(e)))
            continue

    print(f"\nОбработка файлов завершена. Успешно: {len(all_docs)} страниц, ошибок: {len(error_files)}")

    if error_files:
        print("\nФайлы с ошибками:")
        for filename, error in error_files:
            print(f"- {filename}: {error}")

    # Если нет документов, выходим
    if not all_docs:
        print("Нет документов для индексации")
        return None

    # Разбиваем документы на чанки
    print("\nРазбиваем документы на чанки...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    texts = splitter.split_documents(all_docs)
    print(f"Создано {len(texts)} чанков")

    # Создаем FAISS индекс
    print("Создаем FAISS индекс...")
    db = FAISS.from_documents(texts, embeddings)

    print("FAISS индекс успешно создан!")

    return {
        "vectorstore": db,
        "chunk_store": chunk_store,
        "document_count": len(all_docs),
        "chunk_count": len(texts),
        "error_files": error_files
    }


def save_index_to_directory(index_data, output_dir):
    """Сохраняет индекс и связанные данные в указанную директорию"""
    print(f"Сохранение индекса в {output_dir}...")

    # Создаем директорию если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем FAISS индекс
    index_data["vectorstore"].save_local(output_dir)
    print("Индекс FAISS сохранен")

    # Сохраняем chunk_store
    chunk_store_path = os.path.join(output_dir, "chunk_store.json")
    with open(chunk_store_path, 'w', encoding='utf-8') as f:
        json.dump(index_data["chunk_store"], f, ensure_ascii=False, indent=2)
    print(f"Сохранен chunk_store с {len(index_data['chunk_store'])} чанками")

    # Сохраняем метаданные индекса
    metadata_path = os.path.join(output_dir, "index_metadata.json")
    metadata = {
        "created_at": datetime.now().isoformat(),
        "document_count": index_data["document_count"],
        "chunk_count": index_data["chunk_count"],
        "error_count": len(index_data["error_files"]),
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("Метаданные индекса сохранены")

    # Сохраняем информацию о файлах с ошибками
    if index_data["error_files"]:
        errors_path = os.path.join(output_dir, "processing_errors.json")
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(index_data["error_files"], f, ensure_ascii=False, indent=2)
        print(f"Сохранена информация о {len(index_data['error_files'])} файлах с ошибками")

    # Создаем файл с датой обновления
    last_updated_path = os.path.join(output_dir, "last_updated.txt")
    with open(last_updated_path, 'w', encoding='utf-8') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (индекс создан локально)")

    # Создаем README файл
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(f"""# FAISS индекс для RAG чат-бота

Индекс создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Статистика
- Всего документов: {index_data["document_count"]}
- Всего чанков: {index_data["chunk_count"]}
- Файлов с ошибками: {len(index_data["error_files"])}

Этот индекс создан автоматически с помощью скрипта `build_index_local.py`.
""")

    print("Индекс и все связанные файлы успешно сохранены")
    return True


def copy_index_to_render(local_index_dir, render_index_dir):
    """Копирует индекс из локальной директории в директорию Render"""
    print(f"Копирование индекса из {local_index_dir} в {render_index_dir}...")

    # Проверяем доступность директории Render
    if not os.path.exists(render_index_dir):
        try:
            os.makedirs(render_index_dir, exist_ok=True)
            print(f"Создана директория {render_index_dir} для хранения индекса")
        except Exception as e:
            print(f"Ошибка при создании директории {render_index_dir}: {e}")
            print("Вероятно, скрипт запущен не на сервере Render или без необходимых прав")
            return False

    try:
        # Проверяем наличие индекса
        if not os.path.exists(os.path.join(local_index_dir, "index.faiss")):
            print(f"Ошибка: Индекс не найден в {local_index_dir}")
            return False

        # Копируем все файлы
        for item in os.listdir(local_index_dir):
            source = os.path.join(local_index_dir, item)
            destination = os.path.join(render_index_dir, item)

            if os.path.isfile(source):
                shutil.copy2(source, destination)
                print(f"Скопирован файл: {item}")
            elif os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
                print(f"Скопирована директория: {item}")

        # Создаем дополнительные служебные файлы
        with open(os.path.join(render_index_dir, "copied_at.txt"), "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        with open(os.path.join(render_index_dir, "index_copied_flag.txt"), "w", encoding="utf-8") as f:
            f.write(f"Индекс скопирован из локальной директории в {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"Индекс успешно скопирован в {render_index_dir}")
        return True

    except Exception as e:
        print(f"Ошибка при копировании индекса в {render_index_dir}: {e}")
        return False


def main():
    """Основная функция скрипта"""
    # Засекаем время начала выполнения
    start_time = time.time()

    # Получаем аргументы командной строки
    args = parse_arguments()

    # Установка API ключа OpenAI, если указан
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    # Проверяем наличие API ключа OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        print("Ошибка: не найден API ключ OpenAI")
        print("Укажите ключ через аргумент --openai-api-key или переменную окружения OPENAI_API_KEY")
        return 1

    print(f"Директория с документами: {args.docs_dir}")
    print(f"Директория для сохранения индекса: {INDEX_DIR}")

    # Проверяем запущен ли скрипт на Render
    is_render = os.environ.get("RENDER") == "true"
    print(f"Запуск на Render: {'Да' if is_render else 'Нет'}")

    # Обнаружение среды запуска для определения стратегии сохранения
    if args.direct_copy or is_render:
        print(f"Режим прямого копирования. Индекс будет сохранен в {RENDER_INDEX_DIR}")
    else:
        print(f"Стандартный режим. Индекс будет сохранен в локальную директорию {INDEX_DIR}")

    # Строим индекс из локальной директории документов
    index_data = build_index(args.docs_dir, args.max_docs)
    if not index_data:
        print("Ошибка: не удалось создать индекс")
        return 1

    # Сохраняем индекс в локальную директорию проекта
    if not save_index_to_directory(index_data, INDEX_DIR):
        print("Ошибка: не удалось сохранить индекс")
        return 1

    # Если запущен в режиме прямого копирования или на Render,
    # дополнительно копируем индекс в директорию Render
    if args.direct_copy or is_render:
        if os.path.exists(RENDER_INDEX_DIR) or args.direct_copy:
            print("Копирование индекса в директорию Render...")
            if copy_index_to_render(INDEX_DIR, RENDER_INDEX_DIR):
                print(f"Индекс успешно скопирован в {RENDER_INDEX_DIR}")
            else:
                print(f"Не удалось скопировать индекс в {RENDER_INDEX_DIR}")
                print("Индекс сохранен только в локальной директории проекта.")
                print("Вы можете скопировать его в persistent storage вручную или через API.")
        else:
            print(f"Директория {RENDER_INDEX_DIR} не найдена. Копирование на Render пропущено.")
            print("Индекс сохранен только в локальной директории проекта.")

    # Вычисляем общее время выполнения
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"\nГотово! Общее время выполнения: {int(minutes)} мин {int(seconds)} сек")
    print(f"Создан индекс из {index_data['document_count']} документов и {index_data['chunk_count']} чанков")
    print(f"Индекс сохранен в директории: {INDEX_DIR}")

    if args.direct_copy or is_render:
        print(f"Индекс также скопирован в директорию Render: {RENDER_INDEX_DIR}")
    else:
        print("\nДля копирования индекса на Render вы можете:")
        print("1. Запустить этот скрипт с флагом --direct-copy на сервере Render")
        print("2. Использовать API эндпоинт /update-index с соответствующим токеном")
        print("3. Вручную скопировать содержимое папки индекса в persistent storage")

    return 0


if __name__ == "__main__":
    sys.exit(main())