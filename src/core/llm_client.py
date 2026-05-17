"""
Модуль для работы с LLM (OpenAI или локальная Ollama)
"""

import os
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()


class LLMClient:
    """Клиент для работы с LLM"""

    def __init__(self, provider: str = None, temperature: float = 0.5):
        """
        Args:
            provider: 'openai' или 'ollama'
            temperature: креативность ответов (0-1)
        """
        self.provider = provider or os.getenv('LLM_PROVIDER', 'openai')
        self.temperature = temperature

        print(f"🔧 Инициализация LLM клиента (provider={self.provider})")

        if self.provider == 'openai':
            self._init_openai()
        elif self.provider == 'ollama':
            self._init_ollama()
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _init_openai(self):
	    """Инициализация OpenAI или OpenRouter"""
	    try:
		    from openai import OpenAI

		    api_key = os.getenv('OPENAI_API_KEY')
		    base_url = os.getenv('OPENAI_BASE_URL', None)  # Добавляем поддержку кастомного URL

		    if not api_key:
			    raise ValueError("OPENAI_API_KEY not found in .env file!")

		    # Если используется OpenRouter
		    if api_key.startswith('sk-or-v1'):
			    base_url = base_url or "https://openrouter.ai/api/v1"
			    print(f"   🔗 Используем OpenRouter API")

		    self.client = OpenAI(
			    api_key=api_key,
			    base_url=base_url
		    )
		    self.model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')

		    # Для OpenRouter нужно указать другую модель
		    if 'openrouter' in str(base_url):
			    self.model = os.getenv('OPENAI_MODEL', 'openai/gpt-3.5-turbo')

		    print(f"API инициализирован. Модель: {self.model}")

	    except Exception as e:
		    print(f"Ошибка: {e}")
		    raise

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Генерация ответа от LLM

        Args:
            prompt: пользовательский запрос
            system_prompt: системная инструкция

        Returns:
            сгенерированный ответ
        """
        try:
            if self.provider == 'openai':
                return self._generate_openai(prompt, system_prompt)
            elif self.provider == 'ollama':
                return self._generate_ollama(prompt, system_prompt)
            else:
                return f" Неизвестный провайдер: {self.provider}"
        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return f"Извините, произошла ошибка при генерации ответа. Попробуйте позже или переключитесь в режим 'llm off'."

    def _generate_openai(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Генерация через OpenAI"""
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            print(f"Отправка запроса в OpenAI...")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=1000,
                timeout=30
            )

            result = response.choices[0].message.content
            print(f" Ответ получен ({len(result)} символов)")
            return result

        except Exception as e:
            print(f" Ошибка OpenAI: {e}")
            raise

    def _generate_ollama(self, prompt: str, system_prompt: Optional[str]) -> str:
        """Генерация через Ollama"""
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }

        if system_prompt:
            payload["system"] = system_prompt

        response = requests.post(
            f"{self.ollama_base}/api/generate",
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get('response', '')
        else:
            raise Exception(f"Ollama error: {response.status_code}")


# Тестирование
if __name__ == "__main__":
    print("=" * 50)
    print("Тестирование LLM клиента")
    print("=" * 50)

    try:
        llm = LLMClient(provider='openai', temperature=0.5)
        response = llm.generate("Скажи 'Привет, я работаю!'")
        print(f"\nОтвет: {response}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")