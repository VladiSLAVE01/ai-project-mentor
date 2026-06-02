# diagnostic.py
def diagnose_rag_pipeline(assistant, test_questions):
	"""Диагностика RAG пайплайна"""
	for question in test_questions:
		print(f"\n{'=' * 50}")
		print(f"Вопрос: {question}")

		# 1. Проверяем, какие документы нашлись
		docs = assistant.retriever.get_relevant_documents(question)
		print(f"Найдено документов: {len(docs)}")
		for i, doc in enumerate(docs):
			print(f"  Документ {i + 1}: {doc.page_content[:100]}...")

		# 2. Проверяем, какой контекст подаётся в LLM
		context = "\n\n".join([d.page_content for d in docs])
		print(f"Контекст: {context[:200]}...")

		# 3. Проверяем ответ LLM
		answer = assistant.llm.invoke(assistant.prompt_template.format(
			question=question, context=context
		))
		print(f"Ответ: {answer[:100]}...")


# Запускаем диагностику
test_questions = [
	"Что такое Scrum?",
	"Как работает DevOps?",
	"Что такое техническое задание?"
]
diagnose_rag_pipeline(assistant, test_questions)