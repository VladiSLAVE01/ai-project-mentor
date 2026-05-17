
"""
Модуль для разбивки документов на чанки
"""

import re
from typing import List
from src.ingestion.document_loader import Document


class TextSplitter:
	"""Класс для разбивки текста на фрагменты"""

	def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
		"""
		Args:
			chunk_size: максимальный размер чанка в символах
			chunk_overlap: перекрытие между чанками
		"""
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def split_by_paragraphs(self, text: str) -> List[str]:
		"""Разбивка по параграфам"""
		# Разделяем по пустым строкам или заголовкам
		paragraphs = re.split(r'\n\s*\n', text)
		return [p.strip() for p in paragraphs if p.strip()]

	def split_by_sentences(self, text: str) -> List[str]:
		"""Разбивка по предложениям (простая версия)"""
		# Простое разделение по точкам
		sentences = re.split(r'[.!?]+', text)
		return [s.strip() for s in sentences if s.strip()]

	def split_recursive(self, text: str) -> List[str]:
		"""Рекурсивная разбивка с перекрытием"""
		chunks = []

		# Простая реализация: режем по размеру с перекрытием
		start = 0
		while start < len(text):
			end = start + self.chunk_size
			chunk = text[start:end]
			chunks.append(chunk)
			start += (self.chunk_size - self.chunk_overlap)

		return chunks

	def split_document(self, document: Document) -> List[Document]:
		"""Разбиваем один документ на несколько чанков"""
		chunks = self.split_recursive(document.content)

		chunk_documents = []
		for i, chunk_content in enumerate(chunks):
			metadata = document.metadata.copy()
			metadata['chunk_id'] = i
			metadata['chunk_total'] = len(chunks)
			chunk_documents.append(Document(chunk_content, metadata))

		return chunk_documents

	def split_documents(self, documents: List[Document]) -> List[Document]:
		"""Разбиваем все документы на чанки"""
		all_chunks = []
		for doc in documents:
			chunks = self.split_document(doc)
			all_chunks.extend(chunks)

		print(f"Разбито {len(documents)} документов на {len(all_chunks)} чанков")
		return all_chunks


class SemanticTextSplitter:
	"""Более умное разбиение текста с перекрытием для сохранения контекста"""

	def __init__(self, chunk_size: int = 400, chunk_overlap: int = 100):
		self.chunk_size = chunk_size
		self.chunk_overlap = chunk_overlap

	def split_documents(self, documents: List) -> List:
		"""Разбивает документы с сохранением контекста"""
		all_chunks = []

		for doc in documents:
			# Разбиваем по абзацам и склеиваем до размера чанка
			paragraphs = doc.content.split('\n\n')
			current_chunk = []
			current_size = 0

			for para in paragraphs:
				para_size = len(para)
				if current_size + para_size > self.chunk_size and current_chunk:
					# Сохраняем текущий чанк
					chunk_text = '\n\n'.join(current_chunk)
					all_chunks.append(Document(chunk_text, doc.metadata.copy()))
					# Перекрытие: берём последние предложения
					overlap_text = self._get_overlap(current_chunk)
					current_chunk = [overlap_text] if overlap_text else []
					current_size = len(overlap_text)

				current_chunk.append(para)
				current_size += para_size

			# Последний чанк
			if current_chunk:
				chunk_text = '\n\n'.join(current_chunk)
				all_chunks.append(Document(chunk_text, doc.metadata.copy()))

		return all_chunks

	def _get_overlap(self, chunks: List[str]) -> str:
		"""Получает текст для перекрытия"""
		if not chunks:
			return ""
		# Берём последние 2 предложения
		text = ' '.join(chunks[-2:]) if len(chunks) >= 2 else chunks[0]
		sentences = text.split('. ')
		overlap = '. '.join(sentences[-2:]) if len(sentences) >= 2 else sentences[0]
		return overlap

# Тестирование
if __name__ == "__main__":
	from document_loader import DocumentLoader

	loader = DocumentLoader("data/raw")
	docs = loader.load_all_documents()

	splitter = TextSplitter(chunk_size=500, chunk_overlap=50)
	chunks = splitter.split_documents(docs)

	print(f"\n📄 Первый чанк:")
	print(f"  Содержание: {chunks[0].content[:100]}...")
	print(f"  Метаданные: {chunks[0].metadata}")