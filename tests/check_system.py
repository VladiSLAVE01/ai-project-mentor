"""
Полная проверка системы
Запуск: python check_system.py
"""

import sys
import os


def print_section(title):
	print("\n" + "=" * 50)
	print(f"  {title}")
	print("=" * 50)


def check_imports():
	"""Проверка импорта библиотек"""
	print_section("ПРОВЕРКА ИМПОРТОВ")
	libraries = [
		'langchain', 'chromadb', 'sentence_transformers',
		'streamlit', 'numpy', 'pandas'
	]

	all_ok = True
	for lib in libraries:
		try:
			__import__(lib)
			print(f"   {lib}")
		except ImportError as e:
			print(f"   {lib}: {e}")
			all_ok = False
	return all_ok


def check_project_structure():
	"""Проверка структуры проекта"""
	print_section("ПРОВЕРКА СТРУКТУРЫ ПРОЕКТА")

	required_dirs = [
		'src/core',
		'src/ingestion',
		'src/rag',
		'src/interfaces',
		'data/raw',
		'scripts'
	]

	all_ok = True
	for dir_path in required_dirs:
		if os.path.exists(dir_path):
			print(f"  ✅ {dir_path}/")
		else:
			print(f"  ❌ {dir_path}/ - не найдена")
			all_ok = False
	return all_ok


def check_data_files():
	"""Проверка наличия данных"""
	print_section("ПРОВЕРКА ДАННЫХ")

	data_files = []
	for root, dirs, files in os.walk('data/raw'):
		for file in files:
			if file.endswith('.txt'):
				data_files.append(os.path.join(root, file))

	if data_files:
		print(f"  ✅ Найдено {len(data_files)} текстовых файлов:")
		for f in data_files:
			print(f"     - {f}")
		return True
	else:
		print("  ❌ Нет текстовых файлов в data/raw/")
		return False


def check_vector_db():
	"""Проверка векторной БД"""
	print_section("ПРОВЕРКА ВЕКТОРНОЙ БД")

	try:
		sys.path.append('.')
		from src.core.vector_store import VectorStore

		store = VectorStore('data/vectors')
		stats = store.get_collection_stats()

		print(f"  ✅ Коллекция: {stats['collection_name']}")
		print(f"  ✅ Документов: {stats['total_documents']}")

		if stats['total_documents'] > 0:
			return True
		else:
			print("  ⚠️  БД пуста. Запустите: python scripts/ingest_knowledge.py")
			return False
	except Exception as e:
		print(f"  ❌ Ошибка: {e}")
		return False


def check_rag():
	"""Проверка RAG пайплайна"""
	print_section("ПРОВЕРКА RAG ПАЙПЛАЙНА")

	try:
		sys.path.append('.')
		from src.rag.pipeline import RAGPipeline

		rag = RAGPipeline(use_llm=True)
		result = rag.answer("Что такое Scrum?", k=1)

		print(f"  ✅ RAG пайплайн работает")
		print(f"  ✅ Вопрос: {result['question']}")
		print(f"  ✅ Найдено источников: {result['num_sources']}")

		return True
	except Exception as e:
		print(f"  ❌ Ошибка: {e}")
		return False


def main():
	print("\n" + "🔍 " * 20)
	print("     ПРОВЕРКА СИСТЕМЫ AI-PROJECT-MENTOR")
	print("🔍 " * 20)

	results = {
		'imports': check_imports(),
		'structure': check_project_structure(),
		'data': check_data_files(),
		'vector_db': check_vector_db(),
		'rag': check_rag()
	}

	print_section("ИТОГОВЫЙ РЕЗУЛЬТАТ")

	for component, status in results.items():
		icon = "✅" if status else "❌"
		print(f"  {icon} {component.upper()}: {'ОК' if status else 'НЕ ПРОЙДЕН'}")

	if all(results.values()):
		print("\n🎉 ПОЗДРАВЛЯЮ! ВСЕ КОМПОНЕНТЫ РАБОТАЮТ!")
		print("\nЧто дальше?")
		print("  1. Запустите веб-интерфейс: streamlit run src/interfaces/streamlit_app.py")
		print(
			"  2. Или используйте консоль: python -c 'from src.rag.pipeline import RAGPipeline; RAGPipeline().ask_interactive()'")
	else:
		print("\n⚠️ Некоторые компоненты не работают. Проверьте ошибки выше.")
		print("\nРекомендации:")
		if not results['data']:
			print("  - Создайте .txt файлы в папке data/raw/")
		if not results['vector_db']:
			print("  - Запустите: python scripts/ingest_knowledge.py")
		if not results['imports']:
			print("  - Установите зависимости: pip install -r requirements.txt")


if __name__ == "__main__":
	main()