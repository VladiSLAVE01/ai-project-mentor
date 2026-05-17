
"""
Модуль для работы с эмбеддингами (векторными представлениями текста)
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingProvider:
	"""Провайдер эмбеддингов для преобразования текста в векторы"""

	def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
		"""
		Args:
			model_name: название модели для эмбеддингов
						multilingual-e5-small - хороша для русского языка
		"""
		print(f"🔧 Загружаем модель эмбеддингов: {model_name}")
		self.model = SentenceTransformer(model_name)
		self.dimension = self.model.get_sentence_embedding_dimension()
		print(f"Модель загружена. Размерность вектора: {self.dimension}")

	def encode(self, text: str) -> np.ndarray:
		"""Преобразует один текст в вектор"""
		return self.model.encode(text)

	def encode_batch(self, texts: List[str]) -> np.ndarray:
		"""Преобразует список текстов в векторы (быстрее для многих)"""
		return self.model.encode(texts)

	def encode_documents(self, documents: List) -> List[np.ndarray]:
		"""Преобразует список объектов Document в векторы"""
		texts = [doc.content for doc in documents]
		return self.encode_batch(texts)


# Тестирование
if __name__ == "__main__":
	provider = EmbeddingProvider()

	# Тестовый текст
	test_text = "Scrum - это фреймворк для управления проектами"
	vector = provider.encode(test_text)

	print(f"\nТест эмбеддингов:")
	print(f"  Текст: {test_text}")
	print(f"  Размер вектора: {len(vector)}")
	print(f"  Первые 5 значений: {vector[:5]}")