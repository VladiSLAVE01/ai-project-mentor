"""
Модуль для работы с векторной базой данных (ChromaDB)
"""

import os
import chromadb
from typing import List, Dict, Optional
import numpy as np


class VectorStore:
    """Обёртка для работы с ChromaDB"""

    def __init__(self, persist_directory: str = "data/vectors", relevance_threshold: float = 0.7):
        """
        Args:
            persist_directory: директория для хранения векторной БД
            relevance_threshold: порог релевантности (0-1), ниже которого результаты отфильтровываются
        """
        self.persist_directory = persist_directory
        self.relevance_threshold = relevance_threshold
        os.makedirs(persist_directory, exist_ok=True)

        # Подключаемся к ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Создаём или получаем коллекцию
        self.collection_name = "project_knowledge"

        # Проверяем, существует ли коллекция
        existing_collections = [c.name for c in self.client.list_collections()]
        if self.collection_name in existing_collections:
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Загружена существующая коллекция: {self.collection_name}")
        else:
            self.collection = self.client.create_collection(self.collection_name)
            print(f"Создана новая коллекция: {self.collection_name}")

    def create_from_documents(self, documents: List, embedder, batch_size: int = 100):
        """
        Создание векторной БД из списка документов

        Args:
            documents: список объектов Document (из text_splitter)
            embedder: объект EmbeddingProvider
            batch_size: размер батча для добавления
        """
        if not documents:
            print("⚠Нет документов для индексации!")
            return

        print(f"Индексация {len(documents)} документов...")

        # Удаляем старую коллекцию, если есть
        try:
            self.client.delete_collection(self.collection_name)
            print("Удалена старая коллекция")
        except:
            pass

        # Создаём новую коллекцию
        self.collection = self.client.create_collection(self.collection_name)
        print(f"Создана новая коллекция: {self.collection_name}")

        # Подготавливаем данные
        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i:06d}"
            ids.append(doc_id)
            texts.append(doc.content)

            # Добавляем метаданные
            metadata = doc.metadata.copy()
            metadata['doc_id'] = doc_id
            metadatas.append(metadata)

        # Создаём эмбеддинги для всех документов
        print(f"Создание эмбеддингов для {len(texts)} чанков...")
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = [embedder.encode(text) for text in batch]
            embeddings.extend(batch_embeddings)
            print(f"      Прогресс: {min(i+batch_size, len(texts))}/{len(texts)}")

        # Добавляем в коллекцию
        print(f"Сохранение в базу данных...")
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        print(f"Индексация завершена! Сохранено {len(ids)} векторов")

        # Выводим статистику по категориям
        categories = {}
        for metadata in metadatas:
            cat = metadata.get('category', 'unknown')
            categories[cat] = categories.get(cat, 0) + 1

        print(f"\nСтатистика по категориям:")
        for cat, count in categories.items():
            print(f"   - {cat}: {count} чанков")

    def search(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """
        Поиск похожих документов с фильтрацией по релевантности
        """
        try:
            # Запрашиваем больше результатов, так как часть может отфильтроваться
            fetch_k = k * 3 if self.relevance_threshold > 0 else k

            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=fetch_k,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    # Преобразуем расстояние в релевантность (чем ближе к 1, тем лучше)
                    # ChromaDB использует L2 расстояние, максимальное ~2, минимальное 0
                    # Конвертируем: relevance = 1 - (distance / max_distance)
                    distance = results['distances'][0][i]
                    # Нормализуем расстояние в релевантность (приблизительно)
                    relevance = max(0, min(1, 1 - distance / 2))

                    # Фильтруем по порогу релевантности
                    if relevance >= self.relevance_threshold:
                        formatted_results.append({
                            'content': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'score': relevance,
                            'distance': distance
                        })

            # Ограничиваем количество результатов
            formatted_results = formatted_results[:k]

            # Логируем информацию о фильтрации
            if len(results['documents'][0]) > 0:
                original_count = len(results['documents'][0])
                filtered_count = len(formatted_results)
                if filtered_count < original_count:
                    print(f"Фильтрация: {original_count} → {filtered_count} (порог: {self.relevance_threshold})")

            return formatted_results

        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

    def search_without_filter(self, query_vector: List[float], k: int = 5) -> List[Dict]:
        """
        Поиск похожих документов БЕЗ фильтрации (для отладки)

        Args:
            query_vector: вектор запроса
            k: количество результатов

        Returns:
            список словарей с результатами поиска
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    distance = results['distances'][0][i]
                    relevance = max(0, min(1, 1 - distance / 2))
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': relevance,
                        'distance': distance
                    })

            return formatted_results

        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

    def add_document(self, doc_id: str, text: str, metadata: Dict, embedding: List[float]):
        """Добавление одного документа"""
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata],
            embeddings=[embedding]
        )

    def get_collection_info(self) -> Dict:
        """Информация о коллекции"""
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'count': count,
                'persist_directory': self.persist_directory,
                'relevance_threshold': self.relevance_threshold
            }
        except:
            return {'error': 'Коллекция не найдена'}


# Тестирование
if __name__ == "__main__":
    # Простое тестирование
    store = VectorStore("data/vectors", relevance_threshold=0.7)
    info = store.get_collection_info()
    print(f"Коллекция: {info}")