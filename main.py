from fastapi import FastAPI, Form, Request, Cookie, Response, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import os
import uuid
import html
import time
import hashlib
import subprocess
import tempfile

from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Создаем приложение FastAPI с подробными логами
app = FastAPI(
    title="RAG Chat Bot",
    description="Чат-бот с использованием Retrieval-Augmented Generation",
    version="0.3.0",
    debug=True
)

# Настройка CORS для разрешения запросов из разных источников
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://2f93-2a03-32c0-2d-d051-716a-650e-df98-8a9f.ngrok-free.app",
    "https://standardbusiness.online",
    "https://*.standardbusiness.online",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Проверка директории static перед монтированием
static_dir = "."
if not os.path.exists(static_dir):
    print(f"ВНИМАНИЕ: Директория {static_dir} не существует!")

app.mount("/static", StaticFiles(directory=static_dir), name="static")
print(f"Статические файлы монтированы из директории: {static_dir}")

INDEX_PATH = "/data/faiss_index"  # Постоянный диск на Render
LAST_UPDATED_FILE = "/data/last_updated.txt"
LOG_FILE = "/data/rebuild_log.txt"
INDEX_LOCK_FILE = "/data/index_building.lock"
INDEX_VERSION_FILE = "/data/index_version.txt"
chunk_store = {}

# Словарь для хранения истории диалогов
session_memories = {}
session_last_activity = {}
SESSION_MAX_AGE = 86400


# Обновленная функция для проверки, строится ли индекс в данный момент
def is_index_building():
    """Проверяет, идет ли в данный момент процесс построения индекса"""
    return os.path.exists(INDEX_LOCK_FILE)


# Обновленная функция для создания и удаления блокировки индекса
def create_index_lock():
    """Создает файл блокировки, указывающий что идет построение индекса"""
    with open(INDEX_LOCK_FILE, 'w') as f:
        f.write(f"Index building started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def remove_index_lock():
    """Удаляет файл блокировки после завершения построения индекса"""
    if os.path.exists(INDEX_LOCK_FILE):
        os.remove(INDEX_LOCK_FILE)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_github_with_retry(repo_url, temp_dir):
    """Загружает документы из репозитория GitHub с автоматическими повторными попытками"""
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, temp_dir],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
    )
    return True


def download_documents_from_github():
    """Загружает документы из репозитория GitHub с обработкой ошибок"""
    # URL репозитория с документами
    GITHUB_REPO = "https://github.com/daureny/rag-chatbot-documents.git"

    # Для приватного репозитория используем токен из переменных окружения
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        GITHUB_REPO = f"https://{github_token}@github.com/ваш_пользователь/rag-chatbot-documents.git"

    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()

    try:
        print(f"Клонирование репозитория с документами в {temp_dir}...")

        # Используем функцию с повторными попытками
        download_github_with_retry(GITHUB_REPO, temp_dir)

        print("Репозиторий с документами успешно клонирован")
        return temp_dir  # Возвращаем путь к временной директории
    except Exception as e:
        print(f"Ошибка при клонировании репозитория: {e}")
        return None


def extract_title(text: str, filename: str) -> str:
    try:
        # Сначала пытаемся получить осмысленный заголовок из первых строк
        lines = text.splitlines()[:10]  # Проверяем больше строк

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


# Обновленная функция save_last_updated для сохранения на постоянном диске
def save_last_updated(message=""):
    """Сохраняет информацию о последнем обновлении в нескольких местах"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    update_text = timestamp
    if message:
        update_text += f" ({message})"

    # Сохраняем в корне постоянного диска
    try:
        with open(LAST_UPDATED_FILE, "w", encoding="utf-8") as f:
            f.write(update_text)
        print(f"Информация о последнем обновлении сохранена в {LAST_UPDATED_FILE}")
    except Exception as e:
        print(f"Ошибка при сохранении информации об обновлении в {LAST_UPDATED_FILE}: {e}")

    # Создаем файл с дополнительной информацией о сборке
    try:
        build_info_file = os.path.join(INDEX_PATH, "build_info.txt")
        with open(build_info_file, "w", encoding="utf-8") as f:
            f.write(f"Дата сборки: {timestamp}\n")
            f.write(f"Сервер: {os.environ.get('RENDER_SERVICE_NAME', 'local')}\n")
            f.write(f"Дополнительная информация: {message}\n")
        print(f"Информация о сборке сохранена в {build_info_file}")
    except Exception as e:
        print(f"Ошибка при сохранении информации о сборке: {e}")

    # Обновляем версию индекса для кэширования
    try:
        with open(INDEX_VERSION_FILE, "w", encoding="utf-8") as f:
            f.write(str(int(time.time())))
        print(f"Версия индекса обновлена в {INDEX_VERSION_FILE}")
    except Exception as e:
        print(f"Ошибка при обновлении версии индекса: {e}")


# Модифицированная функция построения индекса
def build_combined_txt(force=False):
    """Собирает индекс из документов, с асинхронной обработкой и прогрессом"""
    # Проверяем, не строится ли индекс уже
    if is_index_building() and not force:
        print("Построение индекса уже выполняется. Пропускаем запрос.")
        return {"status": "already_running", "message": "Индексация уже выполняется"}

    # Проверяем существование индекса
    if os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH) and not force:
        # Если индекс существует и не требуется пересоздание, просто возвращаем успех
        faiss_files = [f for f in os.listdir(INDEX_PATH) if f.endswith('.faiss')]
        if faiss_files:
            print("Индекс уже существует и не требует пересоздания")
            return {"status": "exists", "message": "Индекс уже существует и не требует пересоздания"}

    global chunk_store
    chunk_store = {}

    # Создаем директорию для индекса и временных файлов
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH, exist_ok=True)

    temp_index_path = os.path.join(INDEX_PATH, "temp_batches")
    if not os.path.exists(temp_index_path):
        os.makedirs(temp_index_path, exist_ok=True)

    # Создаем блокировку, чтобы показать что индекс строится
    create_index_lock()

    # Запись начального прогресса
    with open(os.path.join(INDEX_PATH, "progress.txt"), "w") as f:
        f.write("0,Начало индексации")

    # Запускаем индексацию в отдельном потоке
    import threading
    thread = threading.Thread(target=_run_indexing_process)
    thread.daemon = True
    thread.start()

    return {"status": "started", "message": "Индексация запущена в фоновом режиме"}


# Дополнительная функция для повторных попыток сохранения временного индекса
def save_temp_index_with_retry(temp_index_path, batch_index, texts, embeddings, max_retries=3):
    """Сохраняет временный индекс с повторными попытками при ошибках API"""
    for attempt in range(max_retries):
        try:
            batch_db = FAISS.from_documents(texts, embeddings)
            batch_db.save_local(os.path.join(temp_index_path, f"batch_{batch_index}"))
            print(f"Временный индекс для батча {batch_index} сохранен")
            return True
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 30 * (attempt + 1)  # Увеличиваем время ожидания с каждой попыткой
                print(f"Ограничение API при сохранении batch_{batch_index}, ожидание {wait_time} сек...")
                time.sleep(wait_time)
            else:
                print(f"Ошибка при сохранении временного индекса batch_{batch_index}: {e}")
                if attempt == max_retries - 1:
                    raise  # Последняя попытка, выбрасываем ошибку
    return False


# Обновленная функция выполнения индексации
def _run_indexing_process():
    """Выполняет индексацию в фоновом режиме с улучшенной обработкой ошибок"""
    import threading

    def heartbeat():
        while True:
            print(f"[Хартбит] Индексация жива: {datetime.now().isoformat()}", flush=True)
            time.sleep(60)

    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        # Обновляем прогресс
        def update_progress(percent, message):
            with open(os.path.join(INDEX_PATH, "progress.txt"), "w") as f:
                f.write(f"{percent},{message}")
            print(f"Прогресс индексации: {percent}% - {message}", flush=True)

        update_progress(5, "Загрузка документов из GitHub")

        # Загружаем документы из GitHub
        github_docs_path = download_documents_from_github()
        if not github_docs_path:
            print("ERROR: GitHub документы не загружены. Проверяем локальную папку docs...")
            # Если не удалось загрузить из GitHub, используем локальную папку
            if os.path.exists("docs") and os.path.isdir("docs"):
                github_docs_path = "docs"
                print("Найдена локальная папка docs, используем её.")
            else:
                update_progress(100, "Ошибка: не удалось загрузить документы")
                _create_empty_index()
                remove_index_lock()  # Удаляем блокировку при ошибке
                return

        # Определяем путь к документам
        print(f"Проверка структуры пути: {github_docs_path}")
        if os.path.exists(github_docs_path):
            print(f"Путь существует: {github_docs_path}")
            if os.path.isdir(github_docs_path):
                print(f"Путь является директорией")
                contents = os.listdir(github_docs_path)
                print(f"Содержимое директории: {contents}")
            else:
                print(f"Путь НЕ является директорией!")
        else:
            print(f"Путь НЕ существует: {github_docs_path}")
            update_progress(100, "Ошибка: указанный путь не существует")
            remove_index_lock()  # Удаляем блокировку при ошибке
            return

        repo_docs_path = os.path.join(github_docs_path, "docs")
        if os.path.exists(repo_docs_path) and os.path.isdir(repo_docs_path):
            docs_path = Path(repo_docs_path)
            print(f"Используем путь к документам: {docs_path}")
        else:
            docs_path = Path(github_docs_path)
            print(f"Используем корневой путь для документов: {docs_path}")

        update_progress(15, "Сканирование файлов")

        # Получаем все файлы с подробным логированием
        try:
            all_files = list(docs_path.glob("*.*"))
            print(f"Найдено {len(all_files)} файлов: {[f.name for f in all_files]}")
        except Exception as e:
            print(f"Ошибка при сканировании файлов: {e}")
            all_files = []

        supported_extensions = [".pdf", ".docx", ".txt", ".html"]
        files_to_process = [f for f in all_files if f.suffix.lower() in supported_extensions]
        print(f"Файлы для обработки: {[f.name for f in files_to_process]}", flush=True)

        # Если нет файлов, создаем пустой индекс
        if not files_to_process:
            update_progress(100, "Нет файлов для индексации")
            _create_empty_index()
            remove_index_lock()  # Удаляем блокировку
            return

        # Разбиваем файлы на группы для пакетной обработки
        batch_size = 5  # Обрабатываем по 5 файлов за раз
        file_batches = [files_to_process[i:i + batch_size] for i in range(0, len(files_to_process), batch_size)]

        # Создаем векторайзер для эмбеддингов
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        temp_index_path = os.path.join(INDEX_PATH, "temp_batches")
        all_docs = []

        # Обрабатываем батчи файлов с защитой от ошибок API и ограничением частоты запросов
        for batch_index, batch in enumerate(file_batches):
            update_progress(
                20 + (60 * batch_index / len(file_batches)),
                f"Обработка батча {batch_index + 1} из {len(file_batches)}"
            )

            # Добавляем задержку между батчами, чтобы избежать ограничений API
            if batch_index > 0:
                time.sleep(2)  # 2 секунды задержки между батчами

            batch_docs = []
            for file in batch:
                try:
                    print(f"Обработка файла: {file.name}")
                    # Логика загрузки файлов
                    if file.suffix.lower() == ".txt":
                        loader = TextLoader(str(file), encoding="utf-8")
                    elif file.suffix.lower() == ".pdf":
                        loader = PyPDFLoader(str(file))
                    elif file.suffix.lower() == ".docx":
                        loader = Docx2txtLoader(str(file))
                    elif file.suffix.lower() == ".html":
                        loader = UnstructuredHTMLLoader(str(file))
                    else:
                        print(f"Пропуск неподдерживаемого формата: {file.suffix}")
                        continue

                    pages = loader.load()
                    print(f"Загружено страниц: {len(pages)}")
                    for page in pages:
                        source_title = extract_title(page.page_content, file.name)
                        page.metadata["source"] = source_title
                        batch_docs.append(page)
                        all_docs.append(page)
                    print(f"Документ успешно обработан: {file.name}")

                except Exception as e:
                    print(f"Ошибка при обработке {file.name}: {e}")
                    continue

            # Если в батче есть документы, создаем временный индекс с повторными попытками
            if batch_docs:
                print(f"Создание временного индекса для батча {batch_index} ({len(batch_docs)} документов)")
                # Разбиваем на чанки
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )

                texts = splitter.split_documents(batch_docs)
                print(f"Разбито на {len(texts)} чанков")

                # Сохраняем во временный индекс с повторными попытками
                save_temp_index_with_retry(temp_index_path, batch_index, texts, embeddings)

        # Объединяем все временные индексы
        update_progress(85, "Объединение индексов")

        if not all_docs:
            update_progress(100, "Нет документов для индексации")
            _create_empty_index()
            remove_index_lock()  # Удаляем блокировку
            return

        print(f"Финальная обработка, всего документов: {len(all_docs)}")
        # Финальная обработка и создание индекса
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        texts = splitter.split_documents(all_docs)
        print(f"Финальное разбиение на {len(texts)} чанков")
        for doc in texts:
            doc.metadata["id"] = str(uuid.uuid4())
            chunk_store[doc.metadata["id"]] = doc.page_content

        print("Создание итогового FAISS индекса")

        # Создаем индекс с заботой о потенциальных ошибках OpenAI API
        try:
            db = FAISS.from_documents(texts, embeddings)
            db.save_local(INDEX_PATH)
            print(f"Индекс сохранен в {INDEX_PATH}")
        except Exception as e:
            if "rate_limit" in str(e).lower():
                print(f"Обнаружено ограничение API OpenAI, ждем 60 секунд перед повторной попыткой")
                time.sleep(60)  # Ждем 1 минуту при превышении лимита
                # Повторная попытка создания индекса
                db = FAISS.from_documents(texts, embeddings)
                db.save_local(INDEX_PATH)
                print(f"Индекс успешно сохранен в {INDEX_PATH} после повторной попытки")
            else:
                raise  # Пробрасываем ошибку, если это не ограничение API

        # Очистка временных файлов
        update_progress(95, "Очистка временных файлов")
        import shutil
        shutil.rmtree(temp_index_path, ignore_errors=True)

        if github_docs_path and github_docs_path != "docs":
            shutil.rmtree(github_docs_path, ignore_errors=True)

        # Сохраняем информацию о последнем обновлении
        save_last_updated("индекс успешно создан")

        update_progress(100, "Индексация завершена")

    except Exception as e:
        error_msg = f"Ошибка в фоновой индексации: {e}"
        print(error_msg)
        try:
            save_last_updated(f"ошибка: {str(e)}")
        except:
            pass
        with open(os.path.join(INDEX_PATH, "progress.txt"), "w") as f:
            f.write(f"100,Ошибка: {str(e)}")
    finally:
        # Важно: всегда удаляем блокировку, даже при ошибках
        remove_index_lock()


# Обновленная функция для создания пустого индекса
def _create_empty_index():
    """Создает пустой индекс в случае отсутствия документов"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        from langchain.schema.document import Document

        # Создаем более информативный документ вместо пустого
        empty_doc = Document(
            page_content="Это пустой индекс. Пожалуйста, добавьте документы в репозиторий или выполните пересборку базы.",
            metadata={
                "source": "Системное сообщение",
                "id": str(uuid.uuid4())
            }
        )

        db = FAISS.from_documents([empty_doc], embeddings)
        db.save_local(INDEX_PATH)

        # Сохраняем информацию о последнем обновлении
        save_last_updated("пустой индекс создан")

        print("Создан пустой индекс")
    except Exception as e:
        print(f"Ошибка при создании пустого индекса: {e}")


# Обновленная функция загрузки векторного хранилища
def load_vectorstore():
    """Загружает векторное хранилище с постоянного диска Render"""
    print("Попытка загрузки векторного хранилища...")

    # Проверка API ключа OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Проверьте .env файл.")

    # Проверка существования индекса
    if not os.path.exists(INDEX_PATH):
        print(f"Директория индекса {INDEX_PATH} не существует. Создаем...")
        os.makedirs(INDEX_PATH, exist_ok=True)

    # Проверка наличия файлов индекса
    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        # Если индекс не существует и не строится в данный момент
        if not is_index_building():
            print("Индекс не найден и не строится. Начинаем построение...")
            build_combined_txt()

        # Возвращаем временный индекс для текущего запроса
        print("Возвращаем временный индекс, пока основной индекс строится...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        temp_doc = {"page_content": "Индекс в процессе построения", "metadata": {"source": "Системное сообщение"}}
        return FAISS.from_texts([temp_doc["page_content"]], embeddings, metadatas=[temp_doc["metadata"]])

    try:
        print("Загрузка векторного хранилища из постоянного диска...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # Пробуем разные варианты загрузки в зависимости от версии библиотеки
        try:
            # Сначала пробуем новый метод
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except TypeError:
            # Если не поддерживается, используем старый метод
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings)

        print("Векторное хранилище успешно загружено")
        return vectorstore
    except Exception as e:
        print(f"Ошибка при загрузке индекса: {e}")

        # Если индекс поврежден и еще не строится
        if not is_index_building():
            print("Индекс поврежден. Запускаем процесс пересоздания...")
            build_combined_txt()

        # Создаем минимальный рабочий индекс для текущего запроса
        print("Возвращаем временный индекс для текущего запроса...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        temp_texts = [{"page_content": "Индекс в процессе обновления", "metadata": {"source": "Системное сообщение"}}]
        return FAISS.from_texts([t["page_content"] for t in temp_texts], embeddings,
                                metadatas=[t["metadata"] for t in temp_texts])


def clean_old_sessions():
    """Очищает старые сессии для экономии памяти"""
    current_time = time.time()
    expired_sessions = []

    for session_id, last_active in session_last_activity.items():
        if current_time - last_active > SESSION_MAX_AGE:
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        if session_id in session_memories:
            del session_memories[session_id]
        if session_id in session_last_activity:
            del session_last_activity[session_id]


@app.on_event("startup")
async def startup_event():
    """Инициализирует необходимые директории и проверяет индекс при запуске"""
    print("Запуск приложения...")

    # Создаем все необходимые директории
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH, exist_ok=True)
        print(f"Создана директория для индекса: {INDEX_PATH}")

    # Проверка и очистка устаревших блокировок
    if is_index_building():
        # Если найдена блокировка, проверяем её возраст
        try:
            # Получаем время создания файла блокировки
            lock_time = os.path.getmtime(INDEX_LOCK_FILE)
            current_time = time.time()

            # Если блокировка старше 2 часов, считаем её зависшей и удаляем
            if current_time - lock_time > 7200:  # 2 часа в секундах
                print("Обнаружена устаревшая блокировка индекса. Удаляем...")
                remove_index_lock()

                # Проверяем статус индекса
                if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
                    print("Индекс не найден после удаления зависшей блокировки. Запускаем построение...")
                    build_combined_txt()
                else:
                    print("Индекс существует, но была обнаружена зависшая блокировка. Индекс готов к использованию.")
            else:
                print("Обнаружена активная блокировка индекса. Индекс в процессе построения.")
        except Exception as e:
            print(f"Ошибка при проверке блокировки индекса: {e}")
            remove_index_lock()  # На всякий случай удаляем блокировку

    # Проверка индекса
    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("Индекс не найден. Проверяем, строится ли он сейчас...")
        if not is_index_building():
            print("Индекс не строится. Запускаем построение...")
            build_combined_txt()
        else:
            print("Индекс в процессе построения.")
    else:
        print("Индекс найден и готов к использованию.")

    # Записываем информацию о запуске сервиса
    try:
        startup_log = os.path.join(INDEX_PATH, "service_startups.log")
        with open(startup_log, "a", encoding="utf-8") as f:
            f.write(f"Сервис запущен: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    except Exception as e:
        print(f"Не удалось записать лог запуска: {e}")

    print("Приложение запущено и готово к работе!")


@app.post("/get-last-updated")
def get_last_updated():
    """Возвращает информацию о последнем обновлении базы знаний"""
    try:
        # Проверяем несколько возможных путей к файлу
        locations = [
            LAST_UPDATED_FILE,
            os.path.join(INDEX_PATH, "last_updated.txt"),
            "/data/last_updated.txt"
        ]

        for location in locations:
            if os.path.exists(location):
                with open(location, "r", encoding="utf-8") as f:
                    last_updated = f.read().strip()
                    return {
                        "status": "success",
                        "last_updated": last_updated,
                        "source": location
                    }

        # Проверяем статус индексации
        if is_index_building():
            try:
                progress_path = os.path.join(INDEX_PATH, "progress.txt")
                if os.path.exists(progress_path):
                    with open(progress_path, "r") as f:
                        progress_data = f.read().strip().split(",", 1)
                        if len(progress_data) == 2:
                            percent, message = progress_data
                            return {
                                "status": "indexing",
                                "last_updated": f"Выполняется индексация ({percent}%): {message}",
                                "source": "progress"
                            }
            except:
                pass

            return {
                "status": "indexing",
                "last_updated": "Индексация в процессе",
                "source": "lock_file"
            }

        # Если нигде не нашли, создаем новый файл
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            save_last_updated("файл создан автоматически")
            return {
                "status": "success",
                "last_updated": timestamp + " (файл создан автоматически)",
                "source": "generated"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Не удалось найти или создать файл с информацией: {str(e)}",
                "last_updated": timestamp + " (восстановлено из системного времени)"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка при получении информации: {str(e)}"
        }


@app.post("/github-webhook")
async def github_webhook(request: Request):
    """Обрабатывает вебхуки от GitHub для автоматического обновления базы знаний"""
    try:
        # Получаем данные запроса для логирования
        payload = await request.json()
        repository = payload.get("repository", {}).get("full_name", "Unknown")

        print(f"Получен вебхук от GitHub репозитория: {repository}")

        # Проверяем, что это push событие в нужный репозиторий
        if "rag-chatbot-documents" not in repository.lower():
            return {"status": "skipped", "message": "Вебхук не относится к репозиторию с документами"}

        # Проверяем, не строится ли индекс уже
        if is_index_building():
            return {"status": "info", "message": "Индексация уже выполняется"}

        # Запускаем обновление индекса в фоновом режиме без ожидания завершения
        import threading
        thread = threading.Thread(target=build_combined_txt)
        thread.daemon = True  # Важно! Позволяет завершить поток при выходе из приложения
        thread.start()

        # Записываем в лог информацию о вебхуке
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"=== Получен GitHub вебхук от {repository} в {timestamp} ===\n")
            log.write("Запущено автоматическое обновление базы знаний в фоновом режиме\n\n")

        return {"status": "success", "message": "Начато обновление базы знаний"}

    except Exception as e:
        error_msg = f"Ошибка при обработке GitHub вебхука: {str(e)}"
        print(error_msg)

        # Логируем ошибку
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"=== Ошибка обработки GitHub вебхука в {timestamp} ===\n")
            log.write(f"Ошибка: {str(e)}\n\n")

        return {"status": "error", "message": error_msg}


@app.post("/rebuild")
async def rebuild_index(admin_token: str = Header(None)):
    """Пересоздает индекс документов с проверкой пароля администратора"""
    # Получаем пароль из переменных окружения
    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_password:
        return JSONResponse({
            "status": "error",
            "message": "Пароль администратора не задан в конфигурации сервера"
        }, status_code=500)

    # Проверяем переданный токен с ожидаемым значением
    expected_token = hashlib.sha256(admin_password.encode()).hexdigest()

    if not admin_token or admin_token != expected_token:
        return JSONResponse({
            "status": "error",
            "message": "Доступ запрещен: неверный пароль администратора"
        }, status_code=403)

    # Проверяем, не строится ли индекс уже
    if is_index_building():
        # Проверяем возраст блокировки
        try:
            lock_time = os.path.getmtime(INDEX_LOCK_FILE)
            current_time = time.time()

            # Если блокировка старше 3 часов, считаем ее зависшей и удаляем
            if current_time - lock_time > 10800:  # 3 часа в секундах
                print("Обнаружена устаревшая блокировка индекса. Удаляем...")
                remove_index_lock()
            else:
                return JSONResponse({
                    "status": "info",
                    "message": "Индексация уже выполняется. Пожалуйста, подождите завершения текущего процесса."
                })
        except:
            # При ошибке на всякий случай удаляем блокировку
            remove_index_lock()

    try:
        print("Запрос на пересоздание индекса от администратора...")
        # Принудительное пересоздание индекса
        result = build_combined_txt(force=True)
        print("Запущен процесс пересоздания индекса")
        return JSONResponse({
            "status": "success",
            "message": "Запущен процесс обновления базы знаний. Это может занять некоторое время."
        })
    except Exception as e:
        error_msg = f"Ошибка при запуске пересоздания индекса: {str(e)}"
        print(error_msg)
        return JSONResponse({
            "status": "error",
            "message": error_msg
        }, status_code=500)


@app.post("/clear-session")
def clear_session(session_id: str = Cookie(None), response: Response = None):
    """Очищает историю сессии"""
    if session_id and session_id in session_memories:
        session_memories[session_id] = []
        return {"status": "success", "message": "История диалога очищена"}
    else:
        return {"status": "error", "message": "Сессия не найдена"}


@app.get("/ping")
def ping():
    """Простой эндпоинт для проверки, что сервер работает"""
    return {"status": "ok", "message": "Сервер работает"}


@app.get("/indexing-status")
def indexing_status():
    """Возвращает текущий статус индексации"""
    try:
        # Проверяем, строится ли индекс
        if is_index_building():
            try:
                # Ищем файл с прогрессом
                progress_path = os.path.join(INDEX_PATH, "progress.txt")
                if os.path.exists(progress_path):
                    with open(progress_path, "r") as f:
                        progress_data = f.read().strip().split(",", 1)
                        if len(progress_data) == 2:
                            percent, message = progress_data
                            return {
                                "status": "in_progress",
                                "percent": int(percent),
                                "message": message
                            }

                # Если нет файла прогресса, но есть блокировка
                return {
                    "status": "in_progress",
                    "percent": 0,
                    "message": "Индексация начата"
                }
            except Exception as e:
                return {
                    "status": "in_progress",
                    "percent": 0,
                    "message": f"Индексация выполняется, но не удалось получить детали: {str(e)}"
                }

        # Если индекс не строится, проверяем его наличие
        if os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
            # Проверяем информацию о последнем обновлении
            if os.path.exists(LAST_UPDATED_FILE):
                with open(LAST_UPDATED_FILE, "r", encoding="utf-8") as f:
                    last_updated = f.read().strip()
                return {
                    "status": "completed",
                    "percent": 100,
                    "message": f"Индексация завершена. Последнее обновление: {last_updated}"
                }

            return {
                "status": "completed",
                "percent": 100,
                "message": "Индексация завершена"
            }

        # Если индекс не найден и не строится
        return {
            "status": "not_started",
            "percent": 0,
            "message": "Индекс не найден и не строится"
        }

    except Exception as e:
        return {
            "status": "error",
            "percent": 0,
            "message": f"Ошибка при получении статуса индексации: {str(e)}"
        }



@app.get("/test-openai")
async def test_openai():
    """Тестирует подключение к API OpenAI"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"status": "error", "message": "API ключ не найден в .env"}

        # Тестовый вызов API
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        result = llm.invoke("Привет! Это тестовое сообщение.")

        return {
            "status": "success",
            "message": "API OpenAI работает корректно",
            "api_response": str(result)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка при вызове OpenAI API: {str(e)}"
        }


@app.post("/test-search")
async def test_search(q: str = Form(...)):
    """Тестирует поиск документов по запросу"""
    try:
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4})
        docs = retriever.get_relevant_documents(q)

        results = []
        for i, doc in enumerate(docs):
            results.append({
                "index": i,
                "content": doc.page_content[:300] + "...",
                "source": doc.metadata.get("source", "Unknown")
            })

        return {
            "status": "success",
            "query": q,
            "results": results
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Ошибка при тестировании поиска: {str(e)}"
        }


@app.get("/config")
def check_config():
    """Проверяет базовую конфигурацию сервера"""
    config = {
        "app_running": True,
        "static_files": os.path.exists(static_dir),
        "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "data_dir_exists": os.path.exists("/data"),
        "index_dir_exists": os.path.exists(INDEX_PATH),
        "index_exists": os.path.exists(os.path.join(INDEX_PATH, "index.faiss")),
        "is_indexing": is_index_building(),
        "last_updated_file_exists": os.path.exists(LAST_UPDATED_FILE),
        "documents_dir_exists": os.path.exists("docs"),
        "documents_count": len(list(Path("docs").glob("*"))) if os.path.exists("docs") else 0,
        "active_sessions": len(session_memories)
    }

    # Добавляем содержимое директории индекса, если она существует
    if os.path.exists(INDEX_PATH):
        config["index_dir_contents"] = os.listdir(INDEX_PATH)

    # Добавляем информацию о последнем обновлении, если файл существует
    if os.path.exists(LAST_UPDATED_FILE):
        try:
            with open(LAST_UPDATED_FILE, "r", encoding="utf-8") as f:
                config["last_updated"] = f.read().strip()
        except:
            config["last_updated"] = "Невозможно прочитать файл"

    return config


@app.get("/debug-pdf-loading")
def debug_pdf_loading():
    """Детальная диагностика загрузки PDF-файлов"""
    docs_path = Path("docs")
    pdf_diagnostics = []

    for file in docs_path.iterdir():
        if file.suffix.lower() == ".pdf":
            try:
                # Проверка с новым PdfReader
                with open(str(file), 'rb') as f:
                    pdf_reader = PdfReader(f)
                    pages_count = len(pdf_reader.pages)

                    # Попытка извлечь текст
                    text_samples = []
                    for i, page in enumerate(pdf_reader.pages[:3], 1):
                        page_text = page.extract_text()
                        text_samples.append({
                            'page': i,
                            'text_length': len(page_text),
                            'first_100_chars': page_text[:100]
                        })

                # Загрузка через PyPDFLoader
                loader = PyPDFLoader(str(file))
                pages = loader.load()

                pdf_diagnostics.append({
                    "filename": file.name,
                    "total_pages": pages_count,
                    "text_samples": text_samples,
                    "page_lengths": [len(page.page_content) for page in pages],
                    "first_page_sample": pages[0].page_content[:500] if pages else "Пустая страница",
                    "is_text_extractable": all(len(page.page_content.strip()) > 0 for page in pages)
                })
            except Exception as e:
                pdf_diagnostics.append({
                    "filename": file.name,
                    "error": str(e)
                })

    return pdf_diagnostics


@app.get("/diagnose-vectorization")
def diagnose_vectorization():
    """Диагностика процесса векторизации документов"""
    try:
        vectorstore = load_vectorstore()

        # Выбираем случайный запрос для тестирования
        test_queries = [
            "Что такое запасы?",
            "Как определяется себестоимость?",
            "Методы оценки запасов"
        ]

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        results = {}
        for query in test_queries:
            try:
                docs = retriever.get_relevant_documents(query)
                results[query] = {
                    "documents_found": len(docs),
                    "document_sources": [doc.metadata.get("source", "Unknown") for doc in docs],
                    "document_lengths": [len(doc.page_content) for doc in docs]
                }
            except Exception as e:
                results[query] = {"error": str(e)}

        return {
            "total_indexed_documents": len(vectorstore.index_to_docstore_id) if hasattr(vectorstore,
                                                                                        'index_to_docstore_id') else "неизвестно",
            "retrieval_test_results": results
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/data-directory")
def list_data_directory():
    """Показывает содержимое директории данных (только для отладки)"""
    try:
        data_dir = "/data"
        if not os.path.exists(data_dir):
            return {"status": "error", "message": "Директория /data не существует"}

        directories = []
        files = []

        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                try:
                    dir_content = os.listdir(item_path)
                    directories.append({
                        "name": item,
                        "path": item_path,
                        "items_count": len(dir_content),
                        "items": dir_content[:20] if len(dir_content) <= 20 else dir_content[:20] + [
                            "...и еще " + str(len(dir_content) - 20) + " элементов"]
                    })
                except Exception as e:
                    directories.append({
                        "name": item,
                        "path": item_path,
                        "error": str(e)
                    })
            else:
                try:
                    size = os.path.getsize(item_path)
                    mtime = datetime.fromtimestamp(os.path.getmtime(item_path)).strftime('%Y-%m-%d %H:%M:%S')

                    # Для текстовых файлов пытаемся получить первые строки
                    content_preview = None
                    if item.endswith(('.txt', '.log')) and size < 10000:
                        try:
                            with open(item_path, 'r', encoding='utf-8') as f:
                                content_preview = f.read(1000)
                        except:
                            content_preview = "Невозможно прочитать содержимое"

                    files.append({
                        "name": item,
                        "path": item_path,
                        "size": size,
                        "modified": mtime,
                        "preview": content_preview
                    })
                except Exception as e:
                    files.append({
                        "name": item,
                        "path": item_path,
                        "error": str(e)
                    })

        return {
            "status": "success",
            "data_dir": data_dir,
            "directories": directories,
            "files": files
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/ask")
async def ask(q: str = Form(...), session_id: str = Cookie(None), response: Response = None):
    print(f"Получен запрос: {q[:50]}...")

    # Проверяем, есть ли текст в запросе
    if not q or len(q.strip()) == 0:
        return JSONResponse({
            "answer": "Пожалуйста, введите ваш вопрос.",
            "sources": ""
        })

    try:
        # Очищаем старые сессии периодически
        clean_old_sessions()

        # Создаем новый ID сессии, если его нет или устанавливаем существующий
        if not session_id:
            session_id = str(uuid.uuid4())
            if response:
                response.set_cookie(key="session_id", value=session_id, max_age=SESSION_MAX_AGE)
            print(f"Создана новая сессия: {session_id}")
        else:
            print(f"Использована существующая сессия: {session_id}")
            # Обновляем cookie, чтобы продлить срок жизни
            if response:
                response.set_cookie(key="session_id", value=session_id, max_age=SESSION_MAX_AGE)

        # Получаем или создаем историю чата для текущей сессии
        if session_id not in session_memories:
            session_memories[session_id] = []
            print(f"Создана новая история для сессии: {session_id}")

        # Обновляем время последней активности
        session_last_activity[session_id] = time.time()

        chat_history = session_memories[session_id]

        # Логируем текущую историю чата
        print(f"История диалога для сессии {session_id} (всего {len(chat_history)} обменов):")
        for i, (question, answer) in enumerate(chat_history):
            print(f"  {i + 1}. Вопрос: {question[:50]}...")
            print(f"     Ответ: {answer[:50]}...")

        print("Загружаем векторное хранилище...")
        vectorstore = load_vectorstore()

        print("Инициализируем модель LLM...")
        if not os.getenv("OPENAI_API_KEY"):
            return JSONResponse({
                "answer": "Ошибка: Не найден ключ API OpenAI. Пожалуйста, проверьте настройки .env файла.",
                "sources": ""
            }, status_code=500)

        # Создаем улучшенный системный промпт с инструкциями по контексту и форматированию
        system_prompt = """
                Ты ассистент с доступом к базе знаний. Используй информацию из базы знаний для ответа на вопросы.

                ОЧЕНЬ ВАЖНО: При ответе обязательно учитывай историю диалога и предыдущие вопросы пользователя!
                Если пользователь задает вопрос, который связан с предыдущим (например "Как его рассчитать?"), 
                то обязательно восстанови контекст из предыдущих сообщений.

                Если в базе знаний нет достаточной информации для полного ответа, честно признайся, что не знаешь.

                ВАЖНОЕ ТРЕБОВАНИЕ К ФОРМАТИРОВАНИЮ:
                1. Структурируй ответ с использованием АБЗАЦЕВ - каждый новый абзац должен начинаться с новой строки и отделяться ПУСТОЙ строкой.
                2. Для создания абзаца используй ДВОЙНОЙ перенос строки (два символа новой строки).
                3. Избегай длинных параграфов без разбивки - максимум 5-7 строк в одном абзаце.
                4. Для списков используй следующие форматы:
                   - Маркированный список: каждый пункт с новой строки, начиная с символа "•" или "-"
                   - Нумерованный список: с новой строки, начиная с "1.", "2." и т.д.
                5. НИКОГДА не используй HTML-теги (например <br>, <p>, <div> и т.д.)
                6. Выделяй важные концепции с помощью символов * (для выделения) или ** (для сильного выделения)

                ПРИМЕР ПРАВИЛЬНОГО ФОРМАТИРОВАНИЯ:

                Первый абзац с объяснением. Здесь я описываю основную концепцию и даю ключевую информацию.

                Второй абзац с дополнительными деталями. Обрати внимание на пустую строку между абзацами.

                Вот список важных моментов:
                • Первый пункт списка
                • Второй пункт списка
                • Третий пункт списка

                Заключительный абзац с выводами.

                КОНЕЦ ПРИМЕРА

                Твоя задача — отвечать максимально информативно и точно по контексту, сохраняя преемственность диалога и правильное форматирование.

                Если в вопросе есть местоимения ("он", "это", "такой"), используй историю диалога, чтобы понять, о чём речь.

                Если пользователь спрашивает "как рассчитывается" или "как определяется" некий термин, 
                и в базе знаний отсутствует точная формула или численный метод, 
                ты должен:
                - интерпретировать вопрос шире — как просьбу объяснить **как определяется, из чего состоит, какие компоненты, лимиты или методология используются**
                - описать **подходы, параметры и логику**, стоящие за определением или управлением этим понятием
                - НЕ путать такие вопросы с расчётом нормативов капитала или других несвязанных показателей

                Твоя цель — дать экспертный, логичный и понятный ответ, даже если прямых данных нет, используя всё, что тебе доступно.
                """

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

        print("Создаем расширенный запрос с учетом контекста...")

        # Подготовка истории диалога для включения в запрос - берем ВСЮ историю для лучшего контекста
        dialog_context = ""
        if chat_history:
            dialog_context = "История диалога:\n"
            for i, (prev_q, prev_a) in enumerate(chat_history):
                dialog_context += f"Вопрос пользователя: {prev_q}\nТвой ответ: {prev_a}\n\n"

        # Создаем обогащенный запрос, включающий историю диалога
        # Собираем последние 3 пары вопрос-ответ, чтобы добавить больше контекста
        recent_dialogue = " ".join([qa[0] + " " + qa[1] for qa in chat_history[-3:]]) if chat_history else ""
        enhanced_query = f"{recent_dialogue} {q}"

        print(f"Поисковый запрос: {enhanced_query[:200]}...")

        # Получаем релевантные документы - увеличиваем до 6 для большего охвата
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        relevant_docs = retriever.get_relevant_documents(enhanced_query)

        if len(relevant_docs) == 0:
            context = "Документов не найдено. Постарайся ответить, используя только историю диалога, если это возможно."
        else:
            # Создаем контекст из релевантных документов
            context = ""
            for i, doc in enumerate(relevant_docs):
                context += f"Документ {i + 1}: {doc.page_content}\n\n"

        print(f"Найдено {len(relevant_docs)} релевантных документов")

        # Создаем полный промпт для LLM
        full_prompt = f"""
                {system_prompt}

                {dialog_context}

                Контекст из базы знаний:
                {context}

                Текущий вопрос пользователя: {q}

                Дай подробный, содержательный ответ на основе предоставленной информации и с учётом предыдущего диалога.
                Если вопрос связан с предыдущими вопросами, обязательно учти это в ответе.
                Не используй HTML-теги в ответе.
                """

        print("Отправляем запрос в LLM...")
        result = llm.invoke(full_prompt)
        answer = result.content
        print(f"Получен ответ от LLM: {answer[:100]}...")

        # Сохраняем пару вопрос-ответ в историю сессии
        session_memories[session_id].append((q, answer))

        # Ограничиваем длину истории, чтобы избежать переполнения
        if len(session_memories[session_id]) > 15:  # Увеличили до 15 для лучшего контекста
            session_memories[session_id] = session_memories[session_id][-15:]

        # Формируем источники для отображения
        source_links = ""
        used_titles = set()
        for doc in relevant_docs:
            title = doc.metadata.get("source", "Источник неизвестен")
            if title not in used_titles:
                content = html.escape(doc.page_content[:3000])
                source_links += f"<details><summary>📄 {title}</summary><pre style='white-space:pre-wrap;text-align:left'>{content}</pre></details>"
                used_titles.add(title)

        print("Возвращаем ответ клиенту")
        # Заменяем любые случайно оставшиеся HTML-теги
        clean_answer = answer.replace("<br>", "\n").replace("<p>", "").replace("</p>", "\n")

        return JSONResponse({"answer": clean_answer, "sources": source_links})

    except Exception as e:
        error_message = f"Ошибка при обработке запроса: {str(e)}"
        print(error_message)
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"=== Ошибка запроса от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(f"Вопрос: {q}\n")
            log.write(f"Ошибка: {error_message}\n\n")

        return JSONResponse({
            "answer": f"Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже или обратитесь к администратору.",
            "sources": ""
        }, status_code=500)


# Код для запуска приложения
if __name__ == "__main__":
    import uvicorn

    print("Запуск сервера FastAPI...")
    print("Для доступа откройте в браузере: http://127.0.0.1:8000")
    print("НЕ используйте адрес 0.0.0.0:8000 в браузере!")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")