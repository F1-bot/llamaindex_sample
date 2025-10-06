import asyncio
import os
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.core.tools import FunctionTool

print("Ініціалізація LLM (qwen3:8b)...")
llm = Ollama(model="qwen3:8b", request_timeout=300.0)

print("Створення інструментів...")
arxiv_spec = ArxivToolSpec()

def write_report(filename: str, content: str) -> str:
    """Записує фінальний звіт у Markdown файл."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Звіт успішно збережено у файл {filename}"
    except Exception as e:
        return f"Помилка при збереженні звіту: {e}"

all_tools = arxiv_spec.to_tool_list() + [FunctionTool.from_defaults(fn=write_report)]

print("Створення агента 'Науковий Дайджест'...")
system_prompt = """
Ти – AI-асистент "Науковий Дайджест". Твоя задача – шукати наукові статті на Arxiv.org та створювати структурований звіт.

Твій план дій:
1.  **Пошук:** Використай інструмент `arxiv_search` для пошуку статей за ключовими словами.
2.  **Аналіз:** Для кожної знайденої статті уважно витягни наступну інформацію:
    *   Назва статті (Title)
    *   Список авторів (Authors)
    *   Повна анотація (Abstract/Summary)
    *   Посилання на статтю (URL)
3.  **Формування звіту:** Створи звіт у форматі Markdown. Дотримуйся **дуже чітко** наступного шаблону для кожної статті:

    ---
    ### **Назва статті**
    **Автори:** *Ім'я Автора 1, Ім'я Автора 2, ...*
    **Посилання:** [Читати на Arxiv](URL_статті)

    **Анотація:**
    > Повний текст анотації...
    ---

4.  **Збереження:** Використай інструмент `write_report`, щоб зберегти фінальний звіт у файл.
"""

agent = FunctionAgent(
    tools=all_tools,
    llm=llm,
    system_prompt=system_prompt,
    verbose=True
)

async def main():
    task = (
        "Знайди 2 найновіші статті на Arxiv за темою 'Multimodal Large Language Models'. "
        "Створи детальний звіт з їх назвами, авторами, анотаціями та посиланнями. "
        "Збережи звіт у файл 'multimodal_llm_digest.md'."
    )

    print(f"\n🚀 Запускаю агента із завданням: '{task}'\n")
    response = await agent.run(user_msg=task)

    print("\n✅ Завдання виконано!")
    print(f"Фінальна відповідь агента: {response}")

    report_path = 'multimodal_llm_digest.md'
    if os.path.exists(report_path):
        print(f"\nЗвіт знайдено в '{report_path}'. Його вміст:")
        with open(report_path, 'r', encoding='utf-8') as f:
            print("---")
            print(f.read().strip())
            print("---")
    else:
        print(f"Звітний файл '{report_path}' не було створено.")

if __name__ == "__main__":
    asyncio.run(main())