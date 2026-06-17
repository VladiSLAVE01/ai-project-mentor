#!/usr/bin/env python3
"""
Telegram бот с интеграцией RAG системы
Запуск: python telegram_rag_bot.py
"""

import telebot
from telebot import types
from openai import OpenAI
import os
import sys
from dotenv import load_dotenv
import logging

# Добавляем путь к проекту
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импортируем наш RAG пайплайн
from src.rag.pipeline import RAGPipeline

# Настройка логирования
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not TELEGRAM_TOKEN:
	raise ValueError("TELEGRAM_TOKEN не найден в .env файле")
if not OPENAI_API_KEY:
	raise ValueError("OPENAI_API_KEY не найден в .env файле")

# Инициализация OpenAI клиента (для общих вопросов)
openai_client = OpenAI(
	api_key=OPENAI_API_KEY,
	base_url="https://openrouter.ai/api/v1",
)

# Инициализация RAG пайплайна (для вопросов по базе знаний)
try:
	rag_pipeline = RAGPipeline(use_llm=True)
	logger.info(" RAG пайплайн успешно загружен")
except Exception as e:
	logger.error(f"Ошибка загрузки RAG: {e}")
	rag_pipeline = None

# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_TOKEN)

# Хранение режимов работы для каждого пользователя
user_modes = {}  # user_id: 'rag' or 'general'


def is_project_question(question: str) -> bool:
	"""
	Проверяет, относится ли вопрос к управлению проектами
	"""
	project_keywords = [
		'agile', 'scrum', 'kanban', 'devops', 'ci/cd', 'waterfall',
		'проект', 'управление проектами', 'техническое задание', 'тз',
		'роли', 'спринт', 'проджект', 'pm', 'продукт', 'бэклог',
		'ретроспектива', 'daily', 'планирование', 'команда',
		'git', 'docker', 'jenkins', 'kubernetes', 'мониторинг'
	]

	question_lower = question.lower()
	return any(keyword in question_lower for keyword in project_keywords)


def get_answer_from_rag(question: str) -> str:
	"""
	Получение ответа из RAG системы
	"""
	if rag_pipeline is None:
		return "База знаний временно недоступна. Попробуйте позже."

	try:
		result = rag_pipeline.answer(question, k=3)

		if result['num_sources'] == 0:
			return "**Информация не найдена в базе знаний**\n\n" \
			       "Попробуйте переформулировать вопрос или задайте общий вопрос " \
			       "(я переключусь на общий режим)."

		# Форматируем ответ с источниками
		answer = f"📖 **На основе базы знаний:**\n\n"
		answer += result['answer']


		return answer

	except Exception as e:
		logger.error(f"RAG ошибка: {e}")
		return f"❌ Ошибка при поиске в базе знаний: {e}"


def get_answer_from_general(question: str) -> str:
	"""
	Получение ответа от общей LLM
	"""
	try:
		response = openai_client.chat.completions.create(
			messages=[
				{"role": "system", "content": "Ты полезный ИИ-ассистент, который помогает с любыми вопросами."},
				{"role": "user", "content": question}
			],
			model="gpt-3.5-turbo",  # используем более дешёвую модель
			max_tokens=1000,
			temperature=0.7
		)

		return response.choices[0].message.content

	except Exception as e:
		logger.error(f"OpenAI ошибка: {e}")
		return f"❌ Ошибка при запросе к ИИ: {e}"


@bot.message_handler(commands=['start'])
def start(message):
	"""Обработчик команды /start"""
	user_id = message.chat.id

	# Клавиатура с режимами
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
	btn_rag = types.KeyboardButton("📚 Режим: База знаний")
	btn_general = types.KeyboardButton("🌐 Режим: Общий ИИ")
	btn_help = types.KeyboardButton("❓ Помощь")
	markup.add(btn_rag, btn_general, btn_help)

	welcome_text = """
🤖 **Привет! Я ИИ-ассистент по управлению проектами!**

У меня есть два режима работы:

📚 **Режим базы знаний** - отвечаю только на основе загруженных материалов (Scrum_guide, Scrum, DevOps, ТЗ)

🌐 **Общий режим** - отвечаю на любые вопросы как обычный ИИ

**Как использовать:**
- Просто задавайте вопросы
- Используйте кнопки для переключения режимов
- Спрашивайте про Scrum, DevOps, Scrum_guide, техническое задание и т.д.

**Примеры вопросов для базы знаний:**
• Что такое DevOps?
• Какие роли в Scrum?
• Что такое CI/CD?

Выберите режим работы 👇
    """

	bot.send_message(
		message.chat.id,
		welcome_text,
		parse_mode='Markdown',
		reply_markup=markup
	)

	# По умолчанию используем RAG режим
	user_modes[user_id] = 'rag'
	logger.info(f"Пользователь {user_id} запустил бота")


@bot.message_handler(commands=['mode'])
def set_mode(message):
	"""Переключение режима через команду"""
	markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
	btn_rag = types.KeyboardButton("📚 Режим: База знаний")
	btn_general = types.KeyboardButton("🌐 Режим: Общий ИИ")
	markup.add(btn_rag, btn_general)

	bot.send_message(
		message.chat.id,
		"Выберите режим работы:",
		reply_markup=markup
	)


@bot.message_handler(func=lambda message: message.text == "📚 Режим: База знаний")
def set_rag_mode(message):
	"""Установка режима базы знаний"""
	user_id = message.chat.id
	user_modes[user_id] = 'rag'
	bot.send_message(
		user_id,
		"✅ **Режим базы знаний активирован!**\n\n"
		"Теперь я буду отвечать только на основе загруженных материалов "
		"по управлению проектами.",
		parse_mode='Markdown'
	)
	logger.info(f"Пользователь {user_id} переключился в RAG режим")


@bot.message_handler(func=lambda message: message.text == "🌐 Режим: Общий ИИ")
def set_general_mode(message):
	"""Установка общего режима"""
	user_id = message.chat.id
	user_modes[user_id] = 'general'
	bot.send_message(
		user_id,
		"✅ **Общий режим активирован!**\n\n"
		"Теперь я буду отвечать на любые вопросы как обычный ИИ.",
		parse_mode='Markdown'
	)
	logger.info(f"Пользователь {user_id} переключился в общий режим")


@bot.message_handler(func=lambda message: message.text == "❓ Помощь")
def help_command(message):
	"""Помощь"""
	help_text = """
📚 **Помощь по использованию бота**

**Режимы работы:**
• 📚 База знаний - ответы из ваших материалов по управлению проектами
• 🌐 Общий ИИ - GPT отвечает на любые вопросы

**Примеры вопросов для базы знаний:**
• Что такое DevOps?
• Какие роли в Scrum?
• Как написать техническое задание?
• Что входит в CI/CD?

**Команды:**
• /start - перезапустить бота
• /mode - сменить режим
• /stats - статистика базы знаний

**Советы:**
• Вопросы по управлению проектами лучше задавать в режиме "База знаний"
• Для общих вопросов используйте "Общий ИИ"
• Бот автоматически определит тему вопроса
    """
	bot.send_message(message.chat.id, help_text, parse_mode='Markdown')


@bot.message_handler(commands=['stats'])
def show_stats(message):
	"""Показать статистику базы знаний"""
	if rag_pipeline:
		try:
			stats = rag_pipeline.vector_store.get_collection_stats()
			bot.send_message(
				message.chat.id,
				f"📊 **Статистика базы знаний**\n\n"
				f"• Документов в БД: {stats['total_documents']}\n"
				f"• Режим: {'Активен' if stats['total_documents'] > 0 else 'Пустая БД'}\n"
				f"• Коллекция: {stats['collection_name']}",
				parse_mode='Markdown'
			)
		except Exception as e:
			bot.send_message(message.chat.id, f"❌ Ошибка получения статистики: {e}")
	else:
		bot.send_message(message.chat.id, "❌ База знаний не загружена")


@bot.message_handler(content_types=['text'])
def handle_message(message):
	"""Обработка текстовых сообщений"""
	user_text = message.text.strip()
	user_id = message.chat.id

	# Пропускаем команды и кнопки
	if user_text in ["📚 Режим: База знаний", "🌐 Режим: Общий ИИ", "❓ Помощь"]:
		return

	# Показываем, что бот печатает
	bot.send_chat_action(user_id, 'typing')

	# Определяем режим пользователя
	current_mode = user_modes.get(user_id, 'rag')

	# Автоматическое переключение на RAG если вопрос по проектам
	if current_mode == 'general' and is_project_question(user_text):
		bot.send_message(
			user_id,
			"🔍 Я заметил, что ваш вопрос похож на тему управления проектами.\n"
			"Возможно, лучше использовать режим 'База знаний'.\n"
			"Нажмите кнопку '📚 Режим: База знаний' для переключения."
		)

	# Получаем ответ в зависимости от режима
	if current_mode == 'rag':
		# Проверяем наличие базы знаний
		if rag_pipeline:
			answer = get_answer_from_rag(user_text)
		else:
			answer = "❌ База знаний не загружена. Запустите python scripts/ingest_knowledge.py"
	else:
		answer = get_answer_from_general(user_text)

	# Отправляем ответ
	try:
		# Разбиваем длинные сообщения на части
		if len(answer) > 4096:
			for i in range(0, len(answer), 4000):
				bot.send_message(user_id, answer[i:i + 4000], parse_mode='Markdown')
		else:
			bot.send_message(user_id, answer, parse_mode='Markdown')
	except Exception as e:
		# Если ошибка с Markdown, отправляем без форматирования
		bot.send_message(user_id, answer)

	logger.info(f"Пользователь {user_id} ({current_mode}): {user_text[:50]}...")


@bot.message_handler(commands=['clear'])
def clear_context(message):
	"""Очистка контекста (для будущих версий с памятью)"""
	bot.send_message(message.chat.id, "🔄 Контекст диалога очищен!")


# Запуск бота
if __name__ == '__main__':
	print("=" * 50)
	print("🤖 TELEGRAM БОТ С RAG СИСТЕМОЙ")
	print("=" * 50)
	print(f"Токен бота: {TELEGRAM_TOKEN[:10]}...")
	print("База знаний: " + ("✅ загружена" if rag_pipeline else "❌ не загружена"))
	print("=" * 50)
	print("Бот запущен... Нажмите Ctrl+C для остановки")

	try:
		bot.polling(none_stop=True, interval=1, timeout=60)
	except KeyboardInterrupt:
		print("\n👋 Бот остановлен")
	except Exception as e:
		print(f"❌ Ошибка: {e}")