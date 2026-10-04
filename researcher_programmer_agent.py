"""
researcher_programmer_agent.py — агент «Дослідник-Програміст».

Приклад студентської роботи: агент шукає факти у Вікіпедії, пише Python-скрипт,
ВИКОНУЄ його і зводить результат у звіт.

[!] УВАГА, ЦЕ НЕБЕЗПЕЧНИЙ ПРИКЛАД — і саме тому він тут.
    Інструмент run_python_script запускає код, який щойно написала модель.
    os.path.basename() утримує ФАЙЛ усередині code_output/, але не обмежує
    нічого у ВМІСТІ: згенерований скрипт виконується з усіма правами вашого
    користувача і має доступ до мережі та файлової системи.
    Для навчання на власній машині це прийнятно. У продакшені — ніколи:
    потрібна пісочниця (Docker, E2B, Modal). Див. 09_security_mcp.py
    з матеріалів Лекції 5.

ЯК ЗАПУСТИТИ
    python researcher_programmer_agent.py
    Потрібен Ollama з qwen3:8b.
"""
import asyncio
import os
import subprocess
import sys

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent

from llama_index.tools.wikipedia import WikipediaToolSpec


def _setup_console() -> None:
    """Windows-консоль стартує у cp1251 і падає на емодзі, щойно вивід
    перенаправляють у файл. Викликаємо ПЕРЕД будь-яким print()."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


_setup_console()


print("Ініціалізація LLM (qwen3:8b)...")
llm = Ollama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    thinking=False,        # qwen3 інакше пише <think>-трасу перед кожним кроком
    context_window=8192,   # LlamaIndex шле це в Ollama як num_ctx: 6.3 ГБ замість 11
    request_timeout=300.0,
)

print("Створення інструментів...")

# ──────────────────────────────────────────────────────────────────
# WikipediaToolSpec «з коробки» НЕ ПРАЦЮЄ: пакет wikipedia ходить до
# Wikimedia з дефолтним User-Agent, і API блокує такі запити, віддаючи
# HTML замість JSON. Будь-який виклик падає з JSONDecodeError, причому
# для БУДЬ-ЯКОЇ сторінки. Лікується одним рядком — представтеся.
# Правила Wikimedia вимагають контакт у User-Agent.
# ──────────────────────────────────────────────────────────────────
import wikipedia

wikipedia.set_user_agent(
    "AppliedAI-MLOps-Course/1.0 (https://your-university.edu; you@example.edu)"
)

wiki_spec = WikipediaToolSpec()
# ──────────────────────────────────────────────────────────────────
# Друга пастка ToolSpec: load_data віддає СТАТТЮ ЦІЛКОМ. Для "Apollo 11"
# це ~81 000 символів, тобто ~20 000 токенів — при num_ctx=8192 модель
# просто губить завдання і починає переказувати статтю. Обгортаємо
# інструмент власним і обрізаємо до розумного розміру.
# ──────────────────────────────────────────────────────────────────
WIKI_CHARS = 2500


def wiki_page(page: str, lang: str = "en") -> str:
    """Loads a Wikipedia article and returns its beginning, where the key facts are.

    Args:
        page: Exact article title, e.g. 'Apollo 11'
        lang: Language code, 'en' by default
    """
    try:
        docs = wiki_spec.load_data(page=page, lang=lang)
    except Exception as exc:                      # noqa: BLE001
        return f"Wikipedia failed for {page!r} ({type(exc).__name__}). Try another title, once."
    text = " ".join(str(d) for d in docs)
    return text[:WIKI_CHARS] + ("…" if len(text) > WIKI_CHARS else "")

wiki_tools = [FunctionTool.from_defaults(fn=wiki_page)]
print(f"Інструменти Вікіпедії: {[t.metadata.name for t in wiki_tools]}")

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
            [sys.executable, safe_path],
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
1.  **Дослідження:** Використай `wiki_page(page=..., lang='en')`, щоб узяти факти (дати, імена, числа).
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
    # Дефолт — 20 ітерацій, і ніде не видно: агент, що заплутався,
    # падає з WorkflowRuntimeError. "generate" дає деградовану,
    # але збережену відповідь замість винятку.
    early_stopping_method="generate",
)



async def _run_with_steps(agent, task: str) -> str:
    """Друкує кожен виклик інструмента.

    `verbose=True` цього НЕ дає: це не поле FunctionAgent, аргумент
    провалюється у Workflow-рушій і друкує сирий лог подій.
    """
    from llama_index.core.agent.workflow import ToolCall, ToolCallResult

    handler = agent.run(user_msg=task, max_iterations=12)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            args = {k: str(v)[:60] for k, v in ev.tool_kwargs.items()}
            print(f"   [tool] {ev.tool_name}({args})")
        elif isinstance(ev, ToolCallResult):
            print(f"   [out ] {str(ev.tool_output).replace(chr(10), ' ')[:140]}…")
    return str(await handler)


async def main():
    task = (
        "Скільки повних днів минуло між висадкою Apollo 11 на Місяць "
        "і першим запуском шатла Columbia (місія STS-1)? "
        "1. Візьми дати зі сторінок англійської Вікіпедії 'Apollo 11' і 'STS-1' "
        "(wiki_page з lang='en'). Не шукай українські сторінки. "
        "2. Напиши Python-скрипт 'date_calculator.py' для розрахунку різниці в днях. "
        "3. Виконай скрипт. "
        "4. Напиши фінальний звіт 'mission_report.md', де вкажи обидві дати та отриману кількість днів."
    )

    print(f"\n🚀 Запускаю агента із завданням: '{task}'\n")
    response = await _run_with_steps(agent, task)

    print("\nOK: Завдання виконано!")
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