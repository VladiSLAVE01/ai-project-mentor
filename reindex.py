# reindex.py
import sys
import os
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.embeddings import EmbeddingProvider
from src.core.vector_store import VectorStore
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import TextSplitter


def main():
	print("=" * 60)
	print("ПЕРЕИНДЕКСАЦИЯ БАЗЫ ЗНАНИЙ")
	print("=" * 60)

	# 1. Загружаем документы
	print("\nЗагрузка документов из data/raw...")
	loader = DocumentLoader("data/raw")
	documents = loader.load_all_documents()

	# Выводим список загруженных файлов
	print("\nНайденные файлы:")
	for doc in documents:
		print(f"   - {doc.metadata['filename']} (категория: {doc.metadata['category']}) - {len(doc.content)} символов")

	if not documents:
		print("Нет документов для индексации!")
		return

	# 2. Разбиваем на чанки
	print("\nРазбивка на чанки...")
	splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = splitter.split_documents(documents)
	print(f"   Создано {len(chunks)} чанков")

	# 3. Очищаем старую БД (опционально)
	if os.path.exists("data/vectors"):
		print("\n Удаляем старую векторную БД...")
		shutil.rmtree("data/vectors")

	# 4. Создаём новую векторную БД
	print("\nСоздание эмбеддингов и сохранение...")
	embedder = EmbeddingProvider()
	vector_store = VectorStore("data/vectors")
	vector_store.create_from_documents(chunks, embedder)

	print("\n" + "=" * 60)
	print("ИНДЕКСАЦИЯ ЗАВЕРШЕНА!")
	print("=" * 60)


if __name__ == "__main__":
	main()