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
import json
import shutil

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import traceback

load_dotenv()

# Создаем приложение FastAPI
app = FastAPI(
    title="RAG Chat Bot",
    description="Чат-бот с использованием Retrieval-Augmented Generation",
    version="1.0.0"
)

# Настройка CORS
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
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
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Константы
INDEX_PATH = "/data/faiss_index"  # Постоянный диск на Render
GITHUB_INDEX_REPO = "https://github.com/daureny/rag-chatbot-index.git"

# Хранение сессий
session_memories = {}
session_last_activity = {}
SESSION_MAX_AGE = 86400  # 24 часа


# Загрузка индекса из GitHub
def download_index_from_github(force=False):
    """Загружает готовый индекс из репозитория GitHub"""
    # Проверяем наличие индекса, если не требуется принудительное обновление
    if not force and os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("Индекс уже существует и не требует загрузки")
        return True

    # Создаем директорию для индекса, если её нет
    os.makedirs(INDEX_PATH, exist_ok=True)

    temp_dir = tempfile.mkdtemp()

    try:
        print(f"Клонирование репозитория с индексом в {temp_dir}...")

        # Добавляем токен для приватного репозитория, если есть
        github_token = os.environ.get("GITHUB_TOKEN")
        repo_url = GITHUB_INDEX_REPO
        if github_token:
            repo_parts = GITHUB_INDEX_REPO.split("//")
            repo_url = f"{repo_parts[0]}//{github_token}@{repo_parts[1]}"

        # Клонируем репозиторий
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120
        )

        print("Репозиторий с индексом успешно клонирован")

        # Проверяем наличие файлов индекса
        if os.path.exists(os.path.join(temp_dir, "index.faiss")):
            # Копируем файлы индекса
            for file in os.listdir(temp_dir):
                if file.endswith('.faiss') or file == 'docstore.json':
                    shutil.copy(
                        os.path.join(temp_dir, file),
                        os.path.join(INDEX_PATH, file)
                    )

            # Сохраняем дату обновления
            with open(os.path.join(INDEX_PATH, "last_updated.txt"), "w") as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

            print(f"Индекс успешно загружен в {INDEX_PATH}")
            return True
        else:
            print(f"Ошибка: файлы индекса не найдены в репозитории")
            return False

    except Exception as e:
        print(f"Ошибка при загрузке индекса: {e}")
        return False
    finally:
        # Очищаем временную директорию
        try:
            shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            print(f"Ошибка при очистке временной директории: {cleanup_error}")


# Загружаем векторное хранилище
def load_vectorstore():
    print("Загрузка векторного хранилища...")
    index_file = os.path.join(INDEX_PATH, "index.faiss")

    if not os.path.exists(index_file):
        print("Индекс не найден. Попытка загрузки из GitHub...")
        try:
            download_index_from_github()
        except Exception as e:
            print("Ошибка при загрузке индекса из GitHub:", e)
            traceback.print_exc()
            raise RuntimeError("Индекс не загружен. Причина: ошибка при загрузке с GitHub.")

    try:
        print("Попытка загрузки индекса из:", INDEX_PATH)
        vectorstore = FAISS.load_local(INDEX_PATH, OpenAIEmbeddings())
        print("Векторное хранилище успешно загружено")
        return vectorstore
    except Exception as e:
        print("Ошибка при загрузке индекса:", e)
        traceback.print_exc()
        raise RuntimeError("Индекс найден, но не удалось загрузить. Подробности выше.")

# Очистка старых сессий
def clean_old_sessions():
    """Очищает старые сессии для экономии памяти"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, last_active in session_last_activity.items()
        if current_time - last_active > SESSION_MAX_AGE
    ]

    for session_id in expired_sessions:
        if session_id in session_memories:
            del session_memories[session_id]
        if session_id in session_last_activity:
            del session_last_activity[session_id]


# События приложения
@app.on_event("startup")
async def startup_event():
    """При запуске приложения проверяем наличие индекса"""
    print("Запуск приложения...")
    if not os.path.exists(INDEX_PATH):
        os.makedirs(INDEX_PATH, exist_ok=True)

    if not os.path.exists(os.path.join(INDEX_PATH, "index.faiss")):
        print("Индекс не найден. Загружаем из GitHub...")
        download_index_from_github()
    else:
        print("Индекс найден и готов к использованию")

    print("Приложение запущено и готово к работе!")


# Эндпоинты
@app.get("/ping")
def ping():
    """Проверка работы сервера"""
    return {"status": "ok", "message": "Сервер работает"}


@app.post("/rebuild")
async def rebuild_index(admin_token: str = Header(None)):
    """Обновление индекса из GitHub"""
    # Проверка пароля администратора
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password:
        return JSONResponse({
            "status": "error",
            "message": "Пароль администратора не задан в конфигурации сервера"
        }, status_code=500)

    expected_token = hashlib.sha256(admin_password.encode()).hexdigest()
    if not admin_token or admin_token != expected_token:
        return JSONResponse({
            "status": "error",
            "message": "Доступ запрещен: неверный пароль администратора"
        }, status_code=403)

    # Принудительное обновление индекса
    try:
        print("Запрос на обновление индекса от администратора...")
        success = download_index_from_github(force=True)

        if success:
            return JSONResponse({
                "status": "success",
                "message": "Индекс успешно обновлен из GitHub"
            })
        else:
            return JSONResponse({
                "status": "error",
                "message": "Не удалось обновить индекс из GitHub"
            }, status_code=500)
    except Exception as e:
        error_msg = f"Ошибка при обновлении индекса: {str(e)}"
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


@app.post("/ask")
async def ask(q: str = Form(...), session_id: str = Cookie(None), response: Response = None):
    """Основной эндпоинт для вопросов к чат-боту"""
    print(f"Получен запрос: {q[:50]}...")

    # Проверяем, есть ли текст в запросе
    if not q or len(q.strip()) == 0:
        return JSONResponse({
            "answer": "Пожалуйста, введите ваш вопрос.",
            "sources": ""
        })

    try:
        # Очищаем старые сессии
        clean_old_sessions()

        # Управление сессией
        if not session_id:
            session_id = str(uuid.uuid4())
            if response:
                response.set_cookie(key="session_id", value=session_id, max_age=SESSION_MAX_AGE)
            print(f"Создана новая сессия: {session_id}")
        else:
            print(f"Использована существующая сессия: {session_id}")
            if response:
                response.set_cookie(key="session_id", value=session_id, max_age=SESSION_MAX_AGE)

        # История диалога
        if session_id not in session_memories:
            session_memories[session_id] = []

        session_last_activity[session_id] = time.time()
        chat_history = session_memories[session_id]

        # Загружаем индекс
        vectorstore = load_vectorstore()

        # Проверяем API ключ
        if not os.getenv("OPENAI_API_KEY"):
            return JSONResponse({
                "answer": "Ошибка: Не найден ключ API OpenAI. Пожалуйста, проверьте настройки .env файла.",
                "sources": ""
            }, status_code=500)

        # Настройка LLM
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

        # Подготовка контекста из истории диалога
        dialog_context = ""
        if chat_history:
            dialog_context = "История диалога:\n"
            for i, (prev_q, prev_a) in enumerate(chat_history):
                dialog_context += f"Вопрос пользователя: {prev_q}\nТвой ответ: {prev_a}\n\n"

        # Обогащенный запрос с контекстом
        recent_dialogue = " ".join([qa[0] + " " + qa[1] for qa in chat_history[-3:]]) if chat_history else ""
        enhanced_query = f"{recent_dialogue} {q}"

        # Получаем релевантные документы
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        relevant_docs = retriever.get_relevant_documents(enhanced_query)

        # Готовим контекст для LLM
        if len(relevant_docs) == 0:
            context = "Документов не найдено. Постарайся ответить, используя только историю диалога, если это возможно."
        else:
            context = ""
            for i, doc in enumerate(relevant_docs):
                context += f"Документ {i + 1}: {doc.page_content}\n\n"

        print(f"Найдено {len(relevant_docs)} релевантных документов")

        # Системный промпт
        system_prompt = """
        Ты ассистент с доступом к базе знаний. Используй информацию из базы знаний для ответа на вопросы.

        ОЧЕНЬ ВАЖНО: При ответе обязательно учитывай историю диалога и предыдущие вопросы пользователя!
        Если пользователь задает вопрос, который связан с предыдущим (например "Как его рассчитать?"), 
        то обязательно восстанови контекст из предыдущих сообщений.

        Если в базе знаний нет достаточной информации для полного ответа, честно признайся, что не знаешь.

        Структурируй ответ с абзацами для лучшей читаемости. Используй маркированные списки где уместно.
        Избегай длинных параграфов без разбивки - максимум 5-7 строк в одном абзаце.

        Твоя цель — дать экспертный, логичный и понятный ответ, даже если прямых данных нет, используя всё, что тебе доступно.
        """

        # Полный промпт для LLM
        full_prompt = f"""
        {system_prompt}

        {dialog_context}

        Контекст из базы знаний:
        {context}

        Текущий вопрос пользователя: {q}

        Дай подробный, содержательный ответ на основе предоставленной информации и с учётом предыдущего диалога.
        Если вопрос связан с предыдущими вопросами, обязательно учти это в ответе.
        """

        # Запрос к LLM
        result = llm.invoke(full_prompt)
        answer = result.content

        # Сохраняем в историю диалога
        session_memories[session_id].append((q, answer))
        if len(session_memories[session_id]) > 15:
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

        # Возвращаем ответ
        clean_answer = answer.replace("<br>", "\n").replace("<p>", "").replace("</p>", "\n")
        return JSONResponse({"answer": clean_answer, "sources": source_links})

    except Exception as e:
        error_message = f"Ошибка при обработке запроса: {str(e)}"
        print(error_message)

        # Запись ошибки в лог
        log_dir = "/data"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "error.log")

        with open(log_file, "a", encoding="utf-8") as log:
            log.write(f"=== Ошибка запроса от {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log.write(f"Вопрос: {q}\n")
            log.write(f"Ошибка: {error_message}\n\n")

        return JSONResponse({
            "answer": f"Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже или обратитесь к администратору.",
            "sources": ""
        }, status_code=500)


@app.get("/last-updated")
def get_last_updated():
    """Возвращает информацию о последнем обновлении индекса"""
    last_updated_file = os.path.join(INDEX_PATH, "last_updated.txt")

    if os.path.exists(last_updated_file):
        try:
            with open(last_updated_file, "r", encoding="utf-8") as f:
                last_updated = f.read().strip()
                return {"status": "success", "last_updated": last_updated}
        except Exception as e:
            return {"status": "error", "message": f"Ошибка чтения файла: {str(e)}"}
    else:
        return {"status": "info", "message": "Информация о последнем обновлении отсутствует"}


# Запуск сервера
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)