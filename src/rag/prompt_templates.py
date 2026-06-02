"""
Шаблоны промптов для RAG системы
"""

from typing import List, Dict


class PromptEngineer:
	"""Класс для работы с промптами"""

	SYSTEM_PROMPT = """Ты — ИИ-ассистент для студентов по управлению IT-проектами.

Твои правила:
1. Отвечай ТОЛЬКО на основе предоставленного контекста из базы знаний
2. Если информации нет в контексте, честно скажи: "В базе знаний нет информации по этому вопросу" и больше ничего не выводи
3. Не выдумывай факты, не указанные в контексте
4. Будь полезным, точным и дружелюбным
5. Отвечай на русском языке
6. Структурируй ответ: используй списки, заголовки, выделение важного
7. Если контекст содержит несколько источников, обобщи информацию
8. В конце ответа укажи источники информации

Помни: ты помогаешь студентам, поэтому отвечай понятно и по делу."""

	def build_context(self, search_results: List[Dict]) -> str:
		"""Формирует контекст из результатов поиска"""
		if not search_results:
			return "Информация не найдена."

		context_parts = []
		for i, result in enumerate(search_results, 1):
			category = result['metadata'].get('category', 'Общее')
			source = result['metadata'].get('filename', 'Неизвестный источник')
			relevance = result.get('score', 0)

			part = f"[ИСТОЧНИК {i}] (релевантность: {relevance:.2f})\n"
			part += f"Категория: {category}\n"
			part += f"Файл: {source}\n"
			part += f"Содержание:\n{result['content']}\n"
			context_parts.append(part)

		return "\n---\n".join(context_parts)

	def build_prompt(self, question: str, context: str) -> str:
		"""Собирает полный промпт для LLM"""
		return f"""{self.SYSTEM_PROMPT}

=== КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
{question}

=== ТВОЙ ОТВЕТ (на основе только контекста) ===
"""

	def build_simple_prompt(self, question: str, context: str) -> str:
		"""Упрощённый промпт для тестирования"""
		return f"""
Контекст:
{context}

Вопрос: {question}

Ответ на основе контекста:
"""