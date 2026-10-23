"""Телеграм-бот, который хранит факты в Pinecone и отвечает с учётом памяти."""

from __future__ import annotations

import os
import uuid

import telebot
from dotenv import load_dotenv
from openai import OpenAI
from pinecone.errors.exceptions import ApiError

from pine import PineconeClient
from pine import TSearchMatch

# Сколько воспоминаний подмешивать в каждый ответ.
TOP_MEMORIES = 10
# Индекс, в котором уже лежит схема документов.
INDEX_NAME = "vpg02data-storage"
# Модель ответов через тот же прокси, что и эмбеддинги.
CHAT_MODEL = "gpt-4o-mini"
OPENAI_BASE_URL = "https://api.proxyapi.ru/openai/v1"
# Лимит одного сообщения Telegram чуть ниже официальных 4096 символов.
MAX_TELEGRAM_LENGTH = 4000
EMPTY_MEMORY = "NONE"
MEMORY_PREFIX = "MEMORY:"
REMEMBER_COMMAND = "запомни"
ENV_OPENAI_KEY = "OPENAI_KEY"
ENV_TELEGRAM_TOKEN = "TELEGRAM_BOT_TOKEN"

load_dotenv()


def _require_env(name: str) -> str:
    """Читает обязательную переменную окружения.

    Входные значения:
        name: имя переменной.

    Выходные значения:
        str: непустое значение без крайних пробелов.
    """
    value = (os.getenv(name) or "").strip()
    if value == "":
        raise ValueError(f"Не задана переменная окружения {name}.")
    return value


pinecone_client = PineconeClient(index_name=INDEX_NAME, namespace="memories")
openai_client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=_require_env(ENV_OPENAI_KEY),
)
bot = telebot.TeleBot(_require_env(ENV_TELEGRAM_TOKEN))


def _user_namespace(chat_id: int) -> str:
    """Возвращает пространство имён воспоминаний одного чата.

    Входные значения:
        chat_id: идентификатор чата Telegram.

    Выходные значения:
        str: имя пространства, например memories-123.
    """
    return f"memories-{chat_id}"


def _memory_text(match: TSearchMatch) -> str:
    """Достаёт текст воспоминания из результата поиска.

    Входные значения:
        match: совпадение из PineconeClient.search.

    Выходные значения:
        str: текст факта или пустая строка.
    """
    metadata = match.get("metadata") or {}
    phrase = metadata.get("phrase")
    if isinstance(phrase, str) and phrase.strip() != "":
        return phrase.strip()
    chunk_text = metadata.get("chunk_text")
    if isinstance(chunk_text, str):
        return chunk_text.strip()
    return ""


def SearchVectors(
    query: str,
    namespace: str,
    top_k: int = TOP_MEMORIES,
) -> list[str]:
    """Ищет ближайшие воспоминания в Pinecone.

    Входные значения:
        query: текст, по которому ищутся воспоминания.
        namespace: пространство имён пользователя.
        top_k: сколько результатов вернуть.

    Выходные значения:
        list[str]: тексты от более подходящего к менее подходящему.
    """
    cleaned = query.strip()
    if cleaned == "":
        return []
    try:
        matches = pinecone_client.search(
            query_text=cleaned,
            top_k=top_k,
            namespace=namespace,
        )
    except ApiError as error:
        # Пустое пространство имён ещё не создано: воспоминаний нет.
        if error.status_code == 404:
            return []
        raise
    texts: list[str] = []
    for match in matches:
        text = _memory_text(match)
        if text != "":
            texts.append(text)
    return texts


def remember(text: str, namespace: str) -> str:
    """Сохраняет один факт в Pinecone.

    Входные значения:
        text: текст факта.
        namespace: пространство имён пользователя.

    Выходные значения:
        str: идентификатор записанного документа.
    """
    memory_id = uuid.uuid4().hex
    pinecone_client.write(
        [
            {
                "id": memory_id,
                "metadata": {"phrase": text.strip()},
            }
        ],
        namespace=namespace,
    )
    return memory_id


def _build_prompt(memories: list[str]) -> str:
    """Собирает системную инструкцию с найденными воспоминаниями.

    Входные значения:
        memories: тексты из SearchVectors.

    Выходные значения:
        str: текст системного сообщения для модели.
    """
    if len(memories) == 0:
        remembered = "Пока нет сохранённых воспоминаний."
    else:
        lines = [f"{index}. {text}" for index, text in enumerate(memories, start=1)]
        remembered = "\n".join(lines)
    return (
        "Ты внимательный помощник. Отвечай по-русски, коротко и по делу. "
        "Опирайся на воспоминания, только если они относятся к вопросу.\n"
        f"Воспоминания:\n{remembered}\n"
        "Последней строкой ответа напиши MEMORY: и один новый факт, "
        "который стоит запомнить, либо NONE, если запоминать нечего. "
        "Факт — короткое утверждение, а не вопрос пользователя."
    )


def _split_reply(raw: str) -> tuple[str, str]:
    """Отделяет ответ пользователю от строки MEMORY.

    Входные значения:
        raw: полный текст модели.

    Выходные значения:
        tuple[str, str]: ответ и факт. Факт пустой, если запоминать нечего.
    """
    memory = ""
    body: list[str] = []
    for line in raw.strip().splitlines():
        if line.strip().upper().startswith(MEMORY_PREFIX):
            memory = line.split(":", 1)[1].strip()
            continue
        body.append(line)
    reply = "\n".join(body).strip()
    if reply == "":
        reply = raw.strip()
    if memory.upper() == EMPTY_MEMORY:
        memory = ""
    return reply, memory


def _explicit_memory(text: str) -> str:
    """Выделяет факт из фразы, которая начинается с «запомни».

    Входные значения:
        text: сообщение пользователя.

    Выходные значения:
        str: факт для записи или пустая строка.
    """
    stripped = text.strip()
    lower = stripped.lower()
    if not lower.startswith(REMEMBER_COMMAND):
        return ""
    fact = stripped[len(REMEMBER_COMMAND):].strip(" :,-")
    if fact == "":
        return stripped
    return fact


def compose_answer(user_text: str, memories: list[str]) -> tuple[str, str]:
    """Получает ответ модели и факт, который нужно сохранить.

    Входные значения:
        user_text: сообщение пользователя.
        memories: тексты из SearchVectors.

    Выходные значения:
        tuple[str, str]: ответ пользователю и факт. Факт может быть пустым.
    """
    completion = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": _build_prompt(memories)},
            {"role": "user", "content": user_text},
        ],
    )
    content = completion.choices[0].message.content or ""
    reply, memory = _split_reply(content)
    explicit = _explicit_memory(user_text)
    if explicit != "":
        memory = explicit
    return reply[:MAX_TELEGRAM_LENGTH], memory


@bot.message_handler(commands=["start", "help"])
def send_welcome(message: telebot.types.Message) -> None:
    """Отправляет короткую справку по боту."""
    bot.reply_to(
        message,
        "Я запоминаю полезные факты в Pinecone. "
        "Напишите факт или задайте вопрос. Перед ответом я беру "
        "10 ближайших воспоминаний. Явная запись: «запомни ...».",
    )


@bot.message_handler(content_types=["text"])
def handle_text(message: telebot.types.Message) -> None:
    """Отвечает на текст: сначала ищет память, затем сохраняет новый факт."""
    text = (message.text or "").strip()
    if text == "":
        return
    namespace = _user_namespace(message.chat.id)
    try:
        memories = SearchVectors(text, namespace=namespace, top_k=TOP_MEMORIES)
        reply, memory = compose_answer(text, memories)
        if memory != "":
            remember(memory, namespace)
        if reply == "":
            reply = "Не получилось сформулировать ответ."
        bot.reply_to(message, reply)
    except Exception as error:
        print(f"Ошибка обработки сообщения: {error}")
        bot.reply_to(message, "Не удалось ответить. Попробуйте ещё раз чуть позже.")


def main() -> None:
    """Подключает Pinecone и запускает длительный опрос Telegram."""
    pinecone_client.connect()
    print("Бот запущен.")
    bot.infinity_polling(skip_pending=True)


if __name__ == "__main__":
    main()
