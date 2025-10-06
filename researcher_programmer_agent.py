import asyncio
import os
import subprocess

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.tools.wikipedia import WikipediaToolSpec

print("Ініціалізація LLM (qwen3:8b)...")
llm = Ollama(model="qwen3:8b", request_timeout=300.0)

print("Створення інструментів...")

wiki_spec = WikipediaToolSpec()
wiki_tools = wiki_spec.to_tool_list()
print(f"Завантажено інструменти Вікіпедії: {[t.metadata.name for t in wiki_tools]}")

code_output_path = "./code_output"
os.makedirs(code_output_path, exist_ok=True)


def write_file(file_path: str, content: str) -> str:
    """
    Записує текстовий вміст у файл. Працює тільки в директорії './code_output'.
    Якщо файл існує, він буде перезаписаний.
    file_path: відносна назва файлу, наприклад 'my_script.py' або 'report.md'.
    content: Текст, який потрібно записати у файл.
    """
    safe_path = os.path.join(code_output_path, os.path.basename(file_path))
    try:
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Файл '{safe_path}' успішно записано."
    except Exception as e:
        return f"Помилка при записі файлу '{safe_path}': {e}"


def run_python_script(file_path: str) -> str:
    """
    Виконує Python-скрипт і повертає його вивід (stdout).
    Працює тільки з файлами в директорії './code_output'.
    file_path: відносна назва файлу, наприклад 'my_script.py'.
    """
    safe_path = os.path.join(code_output_path, os.path.basename(file_path))
    if not os.path.exists(safe_path):
        return f"Помилка: Файл '{safe_path}' не знайдено."

    try:
        result = subprocess.run(
            ['python', safe_path],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30
        )
        if result.returncode == 0:
            return f"Результат виконання '{safe_path}':\n{result.stdout}"
        else:
            return f"Помилка виконання '{safe_path}':\n{result.stderr}"
    except Exception as e:
        return f"Помилка при запуску скрипта '{safe_path}': {e}"


writer_tool = FunctionTool.from_defaults(fn=write_file)
runner_tool = FunctionTool.from_defaults(fn=run_python_script)

python_tools = [writer_tool, runner_tool]
print(f"Завантажено кастомні інструменти для файлів: {[t.metadata.name for t in python_tools]}")

all_tools = wiki_tools + python_tools

print("Створення агента 'Дослідник-Програміст'...")
system_prompt = """
Ти — AI-агент "Дослідник-Програміст". Твоя мета — відповідати на складні питання, комбінуючи пошук інформації та написання коду.

Твій робочий процес:
1.  **Дослідження:** Використай інструменти Вікіпедії (`wikipedia_search` або `wikipedia_load_data`), щоб знайти факти (дати, імена, числа).
2.  **Програмування:** Якщо потрібні розрахунки, використай інструмент `write_file`, щоб написати Python-скрипт. Вказуй лише назву файлу (напр., 'script.py').
3.  **Виконання:** Використай інструмент `run_python_script`, щоб виконати написаний скрипт і отримати результат з його виводу.
4.  **Синтез:** На основі знайденої інформації та результатів виконання коду, сформулюй фінальну відповідь.
5.  **Звіт:** Якщо користувач просить, збережи фінальний звіт у файл за допомогою `write_file`.

Ти дієш автономно і послідовно. Завжди відповідай українською.
"""

agent = FunctionAgent(
    tools=all_tools,
    llm=llm,
    system_prompt=system_prompt,
    verbose=True
)


async def main():
    task = (
        "Скільки повних днів пройшло між датою висадки 'Аполлон-11' на Місяць "
        "і датою першого запуску шатлу 'Колумбія' (місія STS-1)? "
        "1. Знайди обидві дати на Вікіпедії. "
        "2. Напиши Python-скрипт 'date_calculator.py' для розрахунку різниці в днях. "
        "3. Виконай скрипт. "
        "4. Напиши фінальний звіт 'mission_report.md', де вкажи обидві дати та отриману кількість днів."
    )

    print(f"\n🚀 Запускаю агента із завданням: '{task}'\n")
    response = await agent.run(user_msg=task)

    print("\n✅ Завдання виконано!")
    print(f"Фінальна відповідь агента: {response}")

    print("\n--- Перевірка створених файлів ---")
    report_path = os.path.join(code_output_path, "mission_report.md")
    if os.path.exists(report_path):
        print(f"\nЗвіт знайдено в '{report_path}'. Його вміст:")
        with open(report_path, 'r', encoding='utf-8') as f:
            print("---")
            print(f.read().strip())
            print("---")
    else:
        print("Звітний файл не було створено.")


if __name__ == "__main__":
    asyncio.run(main())