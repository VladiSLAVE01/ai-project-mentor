"""
Основной RAG пайплайн: поиск + генерация ответа через LLM
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embeddings import EmbeddingProvider
from src.core.vector_store import VectorStore
from src.core.llm_client import LLMClient
from src.rag.prompt_templates import PromptEngineer


class RAGPipeline:
    """Основной класс RAG-пайплайна"""

    def __init__(self, use_llm: bool = True, relevance_threshold: float = 0.7):
        """
        Args:
            use_llm: использовать ли LLM для генерации
            relevance_threshold: порог релевантности (0-1)
        """
        print("=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ RAG ПАЙПЛАЙНА")
        print("=" * 60)

        print("\nЗагрузка компонентов...")
        self.embedder = EmbeddingProvider()
        self.vector_store = VectorStore("data/vectors", relevance_threshold=relevance_threshold)
        self.prompt_engineer = PromptEngineer()

        self.use_llm = use_llm
        self.relevance_threshold = relevance_threshold

        if use_llm:
            print("\nИнициализация LLM...")
            try:
                self.llm = LLMClient(temperature=0.5)
                print(" LLM готова к работе")
            except Exception as e:
                print(f" Не удалось инициализировать LLM: {e}")
                print(" Буду работать в режиме без LLM (только поиск)")
                self.use_llm = True
        else:
            self.llm = None
            print("Режим без LLM (только поиск)")

        print("\nRAG пайплайн готов!\n")

    def answer(self, question: str, k: int = 5) -> dict:
        """
        Получение ответа на вопрос
        """
        # 1. Преобразуем вопрос в вектор
        query_vector = self.embedder.encode(question)

        # 2. Ищем релевантные документы
        results = self.vector_store.search(query_vector, k=k)

        # 3. Формируем контекст
        context = self.prompt_engineer.build_context(results)

        # 4. Формируем ответ
        if self.use_llm and results:
            # 🔥 Генерация ответа через LLM
            prompt = self.prompt_engineer.build_prompt(question, context)

            print("LLM генерирует ответ...")
            try:
                response = self.llm.generate(prompt)

                # Добавляем информацию об источниках в конец ответа
                sources_text = self._format_sources(results)
                #response = f"{response}\n\n{sources_text}"
                response = response
            except Exception as e:
                print(f" Ошибка LLM: {e}")
                response = self._format_search_results(question, results)
        else:
            # Без LLM - просто показываем найденные документы
            response = self._format_search_results(question, results)

        return {
            'question': question,
            'answer': response,
            'sources': results,
            'num_sources': len(results),
            'threshold': self.relevance_threshold,
            'used_llm': self.use_llm and bool(results)
        }

    def _format_sources(self, results: list) -> str:
        """Форматирует список источников для добавления в конец ответа"""
        if not results:
            return ""

        sources_text = "\n---\n **Источники информации:**\n"
        for i, result in enumerate(results, 1):
            filename = result['metadata'].get('filename', 'unknown')
            category = result['metadata'].get('category', 'unknown')
            score = result['score']
            sources_text += f"{i}. {filename} (категория: {category}, релевантность: {score:.2f})\n"

        return sources_text

    def _format_search_results(self, question: str, results: list) -> str:
        """Форматирует результаты поиска (режим без LLM)"""
        if not results:
            return f""" **Не найдено релевантных источников**

Ваш вопрос: "{question}"

Порог релевантности: {self.relevance_threshold}

💡 **Возможные причины:**
- В базе знаний нет информации по этому вопросу
- Порог релевантности слишком высокий
- Попробуйте переформулировать вопрос"""

        answer = f"**Вопрос:** {question}\n\n"
        answer += f" **Найдено {len(results)} релевантных источников** (порог: {self.relevance_threshold})\n\n"
        answer += "=" * 50 + "\n\n"

        for i, result in enumerate(results, 1):
            score = result['score']
            # Визуальная шкала релевантности
            if score >= 0.9:
                indicator = "🟢🔥"
            elif score >= 0.8:
                indicator = "🟢"
            elif score >= self.relevance_threshold:
                indicator = "🟡"
            else:
                indicator = "🔴"

            category = result['metadata'].get('category', 'unknown')
            filename = result['metadata'].get('filename', 'unknown')
            content = result['content']

            answer += f"### {indicator} **Источник {i}** (релевантность: {score:.2f})\n"
            answer += f"**Категория:** {category}\n"
            answer += f"**Файл:** {filename}\n\n"
            answer += f"**Содержание:**\n{content}\n\n"
            answer += "-" * 40 + "\n\n"

        return answer

    def ask_interactive(self):
        """Интерактивный режим вопросов-ответов"""
        print("\n" + "=" * 60)
        print("🤖 ИИ-АССИСТЕНТ ПО УПРАВЛЕНИЮ ПРОЕКТАМИ")
        print("=" * 60)
        print(f"🎯 Порог релевантности: {self.relevance_threshold}")
        print(f"🧠 LLM режим: {'Включен' if self.use_llm else 'Выключен'}")
        print("\n🟢 >0.8 - высокая релевантность")
        print("🟡 0.7-0.8 - средняя релевантность")
        print("🔴 <0.7 - отфильтровано")
        print("\n❓ Введите 'exit' для выхода")
        print("💡 Введите 'threshold 0.5' чтобы изменить порог релевантности")
        print("💡 Введите 'llm on' или 'llm off' для включения/выключения LLM")
        print("=" * 60)

        while True:
            question = input("\n💭 Ваш вопрос: ").strip()

            if question.lower() in ['exit', 'quit', 'выход']:
                print("\n👋 До свидания!")
                break

            if question.lower() == 'llm on':
                self.use_llm = True
                print("LLM режим ВКЛЮЧЁН")
                continue

            if question.lower() == 'llm off':
                self.use_llm = False
                print("LLM режим ВЫКЛЮЧЁН")
                continue

            if question.lower().startswith('threshold '):
                try:
                    new_threshold = float(question.split()[1])
                    if 0 <= new_threshold <= 1:
                        self.relevance_threshold = new_threshold
                        self.vector_store.relevance_threshold = new_threshold
                        print(f" Порог релевантности изменён на: {new_threshold}")
                    else:
                        print(" Порог должен быть от 0 до 1")
                except:
                    print(" Используйте формат: threshold 0.7")
                continue

            if not question:
                continue

            print("\n Поиск в базе знаний...")
            result = self.answer(question)
            print("\n" + "=" * 60)
            print(result['answer'])
            print("=" * 60)


if __name__ == "__main__":
    import sys

    # Парсинг аргументов командной строки
    use_llm = '--no-llm' not in sys.argv
    threshold = 0.7

    for arg in sys.argv:
        if arg.startswith('--threshold='):
            threshold = float(arg.split('=')[1])

    # Создаём пайплайн
    rag = RAGPipeline(use_llm=True, relevance_threshold=threshold)
    rag.ask_interactive()