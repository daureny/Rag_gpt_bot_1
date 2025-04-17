#!/usr/bin/env python3
"""
Скрипт для локальной индексации документов и загрузки результатов на GitHub.
Этот скрипт нужно запускать на локальной машине с достаточными ресурсами.

Использование:
    python build_index_local.py [--docs-repo URL] [--index-repo URL] [--openai-api-key KEY] [--github-token TOKEN]

По умолчанию скрипт:
1. Клонирует репозиторий с документами
2. Обрабатывает все документы и создает FAISS индекс
3. Загружает готовый индекс в репозиторий
"""

import os
import sys
import time
import json
import shutil
import tempfile
import argparse
import subprocess
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

# Константы по умолчанию
DEFAULT_DOCS_REPO = "https://github.com/daureny/rag-chatbot-documents.git"
DEFAULT_INDEX_REPO = "https://github.com/daureny/rag-chatbot-index.git"


def parse_arguments():
    """Обработка аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Локальная индексация документов и загрузка на GitHub.')
    parser.add_argument('--docs-repo', default=DEFAULT_DOCS_REPO,
                        help=f'URL репозитория с документами (по умолчанию: {DEFAULT_DOCS_REPO})')
    parser.add_argument('--index-repo', default=DEFAULT_INDEX_REPO,
                        help=f'URL репозитория для сохранения индекса (по умолчанию: {DEFAULT_INDEX_REPO})')
    parser.add_argument('--openai-api-key',
                        help='API ключ OpenAI (по умолчанию берется из переменной OPENAI_API_KEY)')
    parser.add_argument('--github-token',
                        help='Токен GitHub для аутентификации (по умолчанию берется из переменной GITHUB_TOKEN)')
    parser.add_argument('--skip-update', action='store_true',
                        help='Пропустить обновление репозитория индекса (использовать только для тестирования)')
    parser.add_argument('--max-docs', type=int, default=0,
                        help='Максимальное количество документов для обработки (0 = все документы)')

    return parser.parse_args()


def clone_repository(repo_url, target_dir, github_token=None):
    """Клонирует репозиторий из GitHub с поддержкой аутентификации"""
    print(f"Клонирование репозитория {repo_url} в {target_dir}...")

    # Если указан токен, добавляем его в URL
    if github_token:
        # Проверяем формат URL (https:// или git@)
        if repo_url.startswith("https://"):
            # Для HTTPS URL добавляем токен в формате https://token@github.com/...
            repo_parts = repo_url.split("//")
            auth_url = f"{repo_parts[0]}//{github_token}@{repo_parts[1]}"
            print("Используется HTTPS аутентификация с токеном GitHub")
        else:
            # Для SSH URL не нужно модифицировать
            auth_url = repo_url
            print("Используется SSH URL, токен не требуется")
    else:
        auth_url = repo_url

    try:
        # Проверяем, существует ли директория
        if os.path.exists(target_dir):
            print(f"Директория {target_dir} уже существует, выполняется git pull...")
            # Если директория существует, обновляем репозиторий
            if github_token:
                # Сначала настраиваем credentials для этого репозитория
                set_git_credentials_cmd = [
                    "git", "-C", target_dir, "config", "credential.helper",
                    f"store --file={os.path.join(tempfile.gettempdir(), 'git_credentials')}"
                ]
                subprocess.run(set_git_credentials_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Создаем временный файл с учетными данными
                cred_path = os.path.join(tempfile.gettempdir(), 'git_credentials')
                repo_host = auth_url.split("@")[1].split("/")[0] if "@" in auth_url else \
                auth_url.split("//")[1].split("/")[0]
                with open(cred_path, 'w') as f:
                    f.write(f"https://{github_token}@{repo_host}\n")

            # Затем делаем pull
            subprocess.run(
                ["git", "-C", target_dir, "pull"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        else:
            # Если директории нет, клонируем репозиторий
            subprocess.run(
                ["git", "clone", auth_url, target_dir],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        print(f"Репозиторий {repo_url} успешно клонирован/обновлен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при клонировании репозитория: {e}")
        print(f"stderr: {e.stderr.decode() if e.stderr else 'Нет вывода'}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка при клонировании репозитория: {e}")
        return False


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
        print("Нет файлов для индексации")
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

    # Добавляем ID для каждого чанка и сохраняем их в словаре
    print("Добавляем идентификаторы к чанкам...")
    import uuid

    # for doc in texts:
    #     # doc.metadata["id"] = str(uuid.uuid4())
    #     chunk_store[doc.metadata["id"]] = doc.page_content

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


def update_index_repository(index_repo_dir, github_token=None):
    """Отправляет изменения в GitHub репозиторий с поддержкой аутентификации"""
    print(f"Отправка индекса в GitHub репозиторий {index_repo_dir}...")

    try:
        # Настраиваем git для этого репозитория, если указан токен
        if github_token:
            # Получаем удаленный URL
            get_remote_cmd = ["git", "-C", index_repo_dir, "config", "--get", "remote.origin.url"]
            remote_url = subprocess.check_output(get_remote_cmd).decode().strip()

            # Определяем хост
            if remote_url.startswith("https://"):
                repo_host = remote_url.split("//")[1].split("/")[0]

                # Настраиваем credential helper
                set_git_credentials_cmd = [
                    "git", "-C", index_repo_dir, "config", "credential.helper",
                    f"store --file={os.path.join(tempfile.gettempdir(), 'git_credentials')}"
                ]
                subprocess.run(set_git_credentials_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Создаем временный файл с учетными данными
                cred_path = os.path.join(tempfile.gettempdir(), 'git_credentials')
                with open(cred_path, 'w') as f:
                    f.write(f"https://{github_token}@{repo_host}\n")

                print(f"Настроены учетные данные для {repo_host}")

        # Настраиваем имя и email для коммита, если не настроены
        try:
            subprocess.check_output(["git", "-C", index_repo_dir, "config", "user.name"])
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "-C", index_repo_dir, "config", "user.name", "RAG Bot Indexer"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        try:
            subprocess.check_output(["git", "-C", index_repo_dir, "config", "user.email"])
        except subprocess.CalledProcessError:
            subprocess.run(
                ["git", "-C", index_repo_dir, "config", "user.email", "ragbot@example.com"],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

        # Добавляем все изменения
        subprocess.run(
            ["git", "-C", index_repo_dir, "add", "."],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Создаем коммит
        commit_message = f"Обновление индекса {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(
            ["git", "-C", index_repo_dir, "commit", "-m", commit_message],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        # Отправляем в GitHub
        push_cmd = ["git", "-C", index_repo_dir, "push"]
        result = subprocess.run(push_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("Индекс успешно отправлен в GitHub")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при отправке индекса в GitHub: {e}")
        print(f"stderr: {e.stderr.decode() if e.stderr else 'Нет вывода'}")
        return False
    except Exception as e:
        print(f"Неожиданная ошибка при отправке индекса в GitHub: {e}")
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

    # Получаем токен GitHub из аргументов или переменной окружения
    github_token = args.github_token or os.environ.get("GITHUB_TOKEN")

    # Проверяем наличие API ключа OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        print("Ошибка: не найден API ключ OpenAI")
        print("Укажите ключ через аргумент --openai-api-key или переменную окружения OPENAI_API_KEY")
        return 1

    # Создаем временные директории
    temp_docs_dir = os.path.join(tempfile.gettempdir(), "rag_bot_docs")
    temp_index_dir = os.path.join(tempfile.gettempdir(), "rag_bot_index")

    print(f"Временные директории:")
    print(f"- Документы: {temp_docs_dir}")
    print(f"- Индекс: {temp_index_dir}")

    # Клонируем репозиторий с документами
    if not clone_repository(args.docs_repo, temp_docs_dir, github_token):
        print("Ошибка: не удалось клонировать репозиторий с документами")
        return 1

    # Если в репозитории есть поддиректория docs, используем её
    docs_subdir = os.path.join(temp_docs_dir, "docs")
    if os.path.exists(docs_subdir) and os.path.isdir(docs_subdir):
        print(f"Найдена поддиректория docs, используем её")
        docs_dir = docs_subdir
    else:
        docs_dir = temp_docs_dir

    # Строим индекс
    index_data = build_index(docs_dir, args.max_docs)
    if not index_data:
        print("Ошибка: не удалось создать индекс")
        return 1

    # Клонируем репозиторий для индекса, если не указано пропустить обновление
    if not args.skip_update:
        if not clone_repository(args.index_repo, temp_index_dir, github_token):
            print("Ошибка: не удалось клонировать репозиторий для индекса")
            return 1

    # Сохраняем индекс в директорию
    if not save_index_to_directory(index_data, temp_index_dir):
        print("Ошибка: не удалось сохранить индекс")
        return 1

    # Отправляем изменения в GitHub, если не указано пропустить обновление
    if not args.skip_update:
        if not update_index_repository(temp_index_dir, github_token):
            print("Ошибка: не удалось отправить индекс в GitHub")
            return 1
    else:
        print("Пропуск отправки индекса в GitHub (указан флаг --skip-update)")
        print(f"Индекс сохранен локально в {temp_index_dir}")

    # Вычисляем общее время выполнения
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"\nГотово! Общее время выполнения: {int(minutes)} мин {int(seconds)} сек")
    print(f"Создан индекс из {index_data['document_count']} документов и {index_data['chunk_count']} чанков")

    if not args.skip_update:
        print(f"Индекс успешно отправлен в GitHub: {args.index_repo}")

    return 0


if __name__ == "__main__":
    sys.exit(main())