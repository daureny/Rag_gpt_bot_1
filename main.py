from fastapi import FastAPI, Form, Request, Cookie, Response
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

from pypdf import PdfReader  # Обновленный импорт

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

import hashlib
from fastapi import Header

import os
import subprocess
import tempfile
from pathlib import Path

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

INDEX_PATH = "faiss_index"
LAST_UPDATED_FILE = "last_updated.txt"
LOG_FILE = "rebuild_log.txt"
chunk_store = {}

# Словарь для хранения истории диалогов
session_memories = {}
session_last_activity = {}
SESSION_MAX_AGE = 86400


def download_documents_from_github():
    """Загружает документы из репозитория GitHub"""

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
        # Клонируем только последнюю версию для экономии времени и места
        subprocess.run(
            ["git", "clone", "--depth", "1", GITHUB_REPO, temp_dir],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print("Репозиторий с документами успешно клонирован")
        return temp_dir  # Возвращаем путь к временной директории
    except Exception as e:
        print(f"Ошибка при клонировании репозитория: {e}")
        return None




def extract_title(text: str, filename: str) -> str:
    lines = text.splitlines()[:5]
    for line in lines:
        if len(line.strip()) > 10 and any(
                kw in line.upper() for kw in ["ЗАКОН", "ПРАВИЛ", "ПОСТАНОВЛ", "МСФО", "КОДЕКС", "РЕГУЛИРОВАНИЕ"]):
            return f"{line.strip()} ({filename})"
    return filename


def build_combined_txt():
    global chunk_store
    chunk_store = {}
    log_lines = []
    start_time = time.time()  # Замер времени начала

    print("Начало индексации: загружаем документы из GitHub...")

    # Создаем директорию для индекса, если её нет
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH)

    # Загружаем документы из GitHub
    github_docs_path = download_documents_from_github()
    if not github_docs_path:
        log_lines.append("❌ Не удалось загрузить документы из GitHub")
        # Проверяем локальную директорию docs как запасной вариант
        docs_path = Path("docs")
        if not docs_path.exists():
            docs_path.mkdir(exist_ok=True)
            log_lines.append("⚠️ Создана пустая директория docs")
    else:
        # Успешно загружены документы из GitHub
        # Проверяем, есть ли директория docs в репозитории
        repo_docs_path = os.path.join(github_docs_path, "docs")
        if os.path.exists(repo_docs_path) and os.path.isdir(repo_docs_path):
            # Если есть директория docs, используем её
            docs_path = Path(repo_docs_path)
            log_lines.append("✅ Успешно загружены документы из GitHub (директория docs)")
        else:
            # Иначе используем корень репозитория
            docs_path = Path(github_docs_path)
            log_lines.append("✅ Успешно загружены документы из GitHub (корневая директория)")

    # Выводим отладочную информацию о директории с документами
    print(f"Путь к документам: {docs_path}")
    if docs_path.exists():
        files_list = list(docs_path.glob("*.*"))
        print(f"Файлы в директории: {[f.name for f in files_list]}")
        print(f"PDF файлы: {list(docs_path.glob('*.pdf'))}")
        print(f"TXT файлы: {list(docs_path.glob('*.txt'))}")
    else:
        print(f"Директория {docs_path} не существует")

    print(f"Начало обработки документов: обнаружено {sum(1 for _ in docs_path.glob('*.pdf'))} PDF-файлов")

    all_docs = []
    processed_files = 0
    failed_files = 0

    for file in docs_path.iterdir():
        try:
            if file.name == "combined.txt":
                continue

            # Специальная проверка для PDF
            if file.suffix == ".pdf":
                try:
                    # Проверка возможности чтения
                    with open(str(file), 'rb') as f:
                        pdf_reader = PdfReader(f)
                        # Дополнительная проверка доступности текста
                        has_text = any(page.extract_text().strip() for page in pdf_reader.pages)

                    if not has_text:
                        log_lines.append(f"⚠️ PDF {file.name} не содержит извлекаемого текста")
                        failed_files += 1
                        continue
                except Exception as e:
                    log_lines.append(f"❌ Ошибка при проверке PDF {file.name}: {e}")
                    failed_files += 1
                    continue

            # Выбор загрузчика
            if file.suffix == ".txt":
                loader = TextLoader(str(file), encoding="utf-8")
            elif file.suffix == ".pdf":
                loader = PyPDFLoader(str(file))
            elif file.suffix == ".docx":
                loader = Docx2txtLoader(str(file))
            elif file.suffix == ".html":
                loader = UnstructuredHTMLLoader(str(file))
            else:
                continue

            pages = loader.load()
            for page in pages:
                source_title = extract_title(page.page_content, file.name)
                page.metadata["source"] = source_title
                all_docs.append(page)

            processed_files += 1
            log_lines.append(f"✅ Загружен файл: {file.name}")

            # Прогресс-бар каждые 5 файлов
            if processed_files % 5 == 0:
                print(f"Обработано файлов: {processed_files}")

        except Exception as e:
            log_lines.append(f"❌ Ошибка при обработке {file.name}: {e}")
            failed_files += 1

    # Проверяем, есть ли документы для индексации
    if not all_docs:
        log_lines.append("⚠️ Нет документов для индексации")
        # Создаем пустой индекс
        with open(LAST_UPDATED_FILE, "w", encoding="utf-8") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " (пустой индекс)")

        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"=== Пересборка от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write("\n".join(log_lines) + "\n\n")

        # Создаем пустой индекс безопасным способом
        try:
            print("Создаем пустой индекс...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            from langchain.schema.document import Document
            # Создаем документ непосредственно, без использования словаря
            empty_doc = Document(page_content="Empty index", metadata={"source": "Empty", "id": str(uuid.uuid4())})
            db = FAISS.from_documents([empty_doc], embeddings)
            db.save_local(INDEX_PATH)
            print("Пустой индекс успешно создан")
        except Exception as e:
            print(f"Ошибка при создании пустого индекса: {e}")
            # Альтернативный способ создания индекса
            try:
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                db = FAISS.from_texts(["Empty index"], embeddings, metadatas=[{"source": "Empty"}])
                db.save_local(INDEX_PATH)
                print("Пустой индекс создан альтернативным способом")
            except Exception as e2:
                print(f"Вторая ошибка при создании индекса: {e2}")

        # Очищаем временную директорию с GitHub документами, если она существует
        if github_docs_path:
            try:
                import shutil
                shutil.rmtree(github_docs_path)
                print(f"Удалена временная директория GitHub: {github_docs_path}")
            except Exception as e:
                print(f"Ошибка при удалении временной директории: {e}")

        return

    # Используем улучшенный сплиттер
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    texts = splitter.split_documents(all_docs)
    for doc in texts:
        doc.metadata["id"] = str(uuid.uuid4())
        chunk_store[doc.metadata["id"]] = doc.page_content

    try:
        print("Начало векторизации документов...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        db = FAISS.from_documents(texts, embeddings)
        db.save_local(INDEX_PATH)

        # Финальная статистика
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Индексация завершена за {total_time:.2f} секунд")
        print(f"Всего файлов: {processed_files + failed_files}")
        print(f"Успешно обработано: {processed_files}")
        print(f"Не удалось обработать: {failed_files}")

    except Exception as e:
        log_lines.append(f"❌ Ошибка при создании индекса: {e}")
        # Записываем ошибку в лог
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(f"=== Ошибка пересборки от {timestamp} ===\n")
            log.write("\n".join(log_lines) + "\n")
            log.write(f"Ошибка: {e}\n\n")

        # Очищаем временную директорию с GitHub документами, если она существует
        if github_docs_path:
            try:
                import shutil
                shutil.rmtree(github_docs_path)
                print(f"Удалена временная директория GitHub: {github_docs_path}")
            except Exception as e2:
                print(f"Ошибка при удалении временной директории: {e2}")

        raise

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LAST_UPDATED_FILE, "w", encoding="utf-8") as f:
        f.write(timestamp)

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(f"=== Пересборка от {timestamp} ===\n")
        log.write("\n".join(log_lines) + "\n\n")

    # Очищаем временную директорию с GitHub документами, если она существует
    if github_docs_path:
        try:
            import shutil
            shutil.rmtree(github_docs_path)
            print(f"Удалена временная директория GitHub: {github_docs_path}")
        except Exception as e:
            print(f"Ошибка при удалении временной директории: {e}")


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

        # Запускаем обновление индекса в фоновом режиме
        import threading
        thread = threading.Thread(target=build_combined_txt)
        thread.start()

        print(f"Запущено фоновое обновление базы знаний из репозитория {repository}")

        # Записываем в лог информацию о вебхуке
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.write(f"=== Получен GitHub вебхук от {repository} в {timestamp} ===\n")
            log.write("Запущено автоматическое обновление базы знаний\n\n")

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

def load_vectorstore():
    """Загружает векторное хранилище, создавая его при необходимости"""
    print("Попытка загрузки векторного хранилища...")

    # Проверка API ключа OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY не найден в переменных окружения. Проверьте .env файл.")

    # Проверка существования индекса
    if not os.path.exists(INDEX_PATH):
        print(f"Директория индекса {INDEX_PATH} не существует. Создаем...")
        os.makedirs(INDEX_PATH, exist_ok=True)

    if not os.listdir(INDEX_PATH):
        print("Индекс пуст. Создаем новый индекс...")
        build_combined_txt()

    try:
        print("Загрузка векторного хранилища...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("Векторное хранилище успешно загружено")
        return vectorstore
    except Exception as e:
        print(f"Ошибка при загрузке индекса: {e}")
        print("Пересоздаем индекс...")
        try:
            # Пересоздаем индекс при ошибке
            build_combined_txt()
            # Повторная попытка загрузки
            print("Повторная попытка загрузки индекса...")
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            print("Векторное хранилище успешно загружено после пересоздания")
            return vectorstore
        except Exception as e2:
            # Если повторная попытка не удалась, создаем пустой индекс
            print(f"Вторая ошибка при работе с индексом: {e2}")
            print("Создаем минимальный рабочий индекс...")
            # Создаем минимальный индекс с одним документом
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            empty_texts = [{"page_content": "Индекс пуст или поврежден", "metadata": {"source": "Empty"}}]
            db = FAISS.from_texts([t["page_content"] for t in empty_texts], embeddings,
                                  metadatas=[t["metadata"] for t in empty_texts])
            db.save_local(INDEX_PATH)
            return db

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
    """Инициализирует индекс при запуске, если его нет"""
    print("Запуск приложения...")
    # Не делаем тяжелую инициализацию при запуске, чтобы приложение стартовало быстро
    # Проверяем только наличие необходимых директорий
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH, exist_ok=True)
        print(f"Создана директория для индекса: {INDEX_PATH}")

    docs_path = Path("docs")
    if not docs_path.exists():
        docs_path.mkdir(exist_ok=True)
        print("Создана директория для документов: docs")

    print("Приложение запущено и готово к работе!")

@app.get("/", response_class=HTMLResponse)

def chat_ui():
    try:
        print("Запрос к главной странице...")
        last_updated = "Неизвестно"
        if os.path.exists(LAST_UPDATED_FILE):
            with open(LAST_UPDATED_FILE, "r", encoding="utf-8") as f:
                last_updated = f.read().strip()

        # Проверка наличия HTML шаблона
        html_path = "static/index_chat.html"
        if not os.path.exists(html_path):
            return HTMLResponse(
                content="<html><body><h1>Ошибка: файл index_chat.html не найден</h1><p>Убедитесь, что файл существует в директории static.</p></body></html>"
            )

        with open(html_path, "r", encoding="utf-8") as f:
            html_template = f.read()

        print("Главная страница успешно загружена")
        return HTMLResponse(content=html_template.replace("{{last_updated}}", last_updated))
    except Exception as e:
        error_msg = f"Ошибка при загрузке главной страницы: {str(e)}"
        print(error_msg)
        return HTMLResponse(
            content=f"<html><body><h1>Ошибка</h1><p>{error_msg}</p></body></html>",
            status_code=500
        )

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

@app.get("/ping")
def ping():
    """Простой эндпоинт для проверки, что сервер работает"""
    return {"status": "ok", "message": "Сервер работает"}

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
        "index_exists": os.path.exists(INDEX_PATH) and os.listdir(INDEX_PATH),
        "documents_dir_exists": os.path.exists("docs"),
        "documents_count": len(list(Path("docs").glob("*"))) if os.path.exists("docs") else 0,
        "active_sessions": len(session_memories)
    }
    return config


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

    try:
        print("Запрос на пересоздание индекса...")
        build_combined_txt()
        print("Индекс успешно пересоздан")
        return JSONResponse({"status": "success", "message": "Индекс успешно пересоздан"})
    except Exception as e:
        error_msg = f"Ошибка при пересоздании индекса: {str(e)}"
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


@app.get("/debug-pdf-loading")
def debug_pdf_loading():
    """Детальная диагностика загрузки PDF-файлов"""
    docs_path = Path("docs")
    pdf_diagnostics = []

    for file in docs_path.iterdir():
        if file.suffix == ".pdf":
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
            "total_indexed_documents": len(vectorstore.index_to_docstore_id),
            "retrieval_test_results": results
        }
    except Exception as e:
        return {"error": str(e)}


# Код для запуска приложения
if __name__ == "__main__":
    import uvicorn

    print("Запуск сервера FastAPI...")
    print("Для доступа откройте в браузере: http://127.0.0.1:8000")
    print("НЕ используйте адрес 0.0.0.0:8000 в браузере!")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")