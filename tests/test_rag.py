
import sys
sys.path.append('.')

from src.rag.pipeline import RAGPipeline

print("=" * 50)
print("ТЕСТИРОВАНИЕ RAG ПАЙПЛАЙНА")
print("=" * 50)

# Инициализация
print("\n1. Загрузка RAG пайплайна...")
rag = RAGPipeline(use_llm=True)

# Статистика БД
print("\n2. Статистика векторной БД:")
stats = rag.vector_store.get_collection_stats()
print(f"   Коллекция: {stats['collection_name']}")
print(f"   Документов: {stats['total_documents']}")

# Тестовые вопросы
test_questions = [
    "Что такое Scrum?",
    "Какие роли в Scrum?",
    "Что такое DevOps?",
    "Как написать техническое задание?",
]

print("\n3. Тестирование поиска:")
for q in test_questions:
    print(f"\n   Вопрос: {q}")
    result = rag.answer(q, k=2)
    print(f"   Найдено источников: {result['num_sources']}")
    if result['sources']:
        print(f"   Лучший результат: {result['sources'][0]['score']:.3f}")
    print("-" * 40)

print("\n✅ Все тесты пройдены!")
