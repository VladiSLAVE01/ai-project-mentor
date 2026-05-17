"""
Скрипт для загрузки всей базы знаний в векторную БД
Запуск: python scripts/ingest_knowledge.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import TextSplitter
from src.core.embeddings import EmbeddingProvider
from src.core.vector_store import VectorStore


def main():
	print("ЗАГРУЗКА БАЗЫ ЗНАНИЙ В ВЕКТОРНУЮ БД")

	# 1. Загружаем документы
	print("\n Шаг 1: Загрузка документов...")
	loader = DocumentLoader("data/raw")
	documents = loader.load_all_documents()
	stats = loader.get_statistics(documents)
	print(f"   Категории: {stats['categories']}")

	# 2. Разбиваем на чанки
	print("\n✂  Шаг 2: Разбивка на чанки...")
	splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = splitter.split_documents(documents)

	# 3. Создаём эмбеддинги
	print("\n Шаг 3: Создание эмбеддингов...")
	embedder = EmbeddingProvider()
	embeddings = embedder.encode_batch([chunk.content for chunk in chunks])

	# 4. Сохраняем в векторную БД
	print("\n Шаг 4: Сохранение в векторную БД...")
	vector_store = VectorStore("data/vectors")
	vector_store.add_documents(chunks, embeddings)

	# 5. Статистика
	print("\n Шаг 5: Итоговая статистика:")
	stats = vector_store.get_collection_stats()
	print(f"   Коллекция: {stats['collection_name']}")
	print(f"   Документов в БД: {stats['total_documents']}")

	print("\n Загрузка базы знаний завершена!")


if __name__ == "__main__":
	main()