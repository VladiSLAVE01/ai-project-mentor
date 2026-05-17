
"""
Модуль для загрузки документов из разных форматов
"""

import os
from typing import List, Dict, Any
from pathlib import Path


class Document:
	"""Класс для представления документа"""

	def __init__(self, content: str, metadata: Dict[str, Any]):
		self.content = content
		self.metadata = metadata

	def __repr__(self):
		return f"Document(content={self.content[:50]}..., metadata={self.metadata})"


class DocumentLoader:
	"""Загрузчик документов из разных источников"""

	def __init__(self, data_path: str = "data/raw"):
		self.data_path = Path(data_path)

	def load_text_file(self, file_path: Path) -> Document:
		"""Загрузка текстового файла"""
		with open(file_path, 'r', encoding='utf-8') as f:
			content = f.read()

		metadata = {
			'source': str(file_path),
			'type': 'text',
			'category': file_path.parent.name,
			'filename': file_path.name
		}

		return Document(content, metadata)

	def load_all_documents(self) -> List[Document]:
		"""Загрузка всех документов из папки data/raw"""
		documents = []

		# Рекурсивно обходим все .txt файлы
		for file_path in self.data_path.rglob("*.txt"):
			print(f"Загружаем: {file_path}")
			doc = self.load_text_file(file_path)
			documents.append(doc)

		print(f"\nЗагружено {len(documents)} документов")
		return documents

	def get_statistics(self, documents: List[Document]) -> Dict:
		"""Статистика по загруженным документам"""
		stats = {
			'total_documents': len(documents),
			'total_chars': sum(len(doc.content) for doc in documents),
			'categories': {}
		}

		for doc in documents:
			cat = doc.metadata.get('category', 'unknown')
			stats['categories'][cat] = stats['categories'].get(cat, 0) + 1

		return stats


# Тестирование
if __name__ == "__main__":
	loader = DocumentLoader("data/raw")
	docs = loader.load_all_documents()
	print("\n Статистика:")
	stats = loader.get_statistics(docs)
	for key, value in stats.items():
		print(f"  {key}: {value}")