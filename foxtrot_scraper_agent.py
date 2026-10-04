"""
foxtrot_scraper_agent.py — агент-скрапер магазину.

Приклад студентської роботи: агент шукає товари на сайті foxtrot.com.ua,
розбирає HTML через BeautifulSoup і зводить результат у звіт.

[!] ЦЕЙ ПРИКЛАД КРИХКИЙ ЗА ПРИРОДОЮ.
    Він тримається на конкретній HTML-розмітці чужого сайту. Розмітка
    змінюється без попередження, і тоді скрапер мовчки повертає порожній
    результат — не помилку, а просто «нічого не знайдено». Станом на
    жовтень 2026 пошук уже не повертає товарів: селектори застаріли.
    Для лабораторної надійніше брати документоване API або ToolSpec
    з каталогу інтеграцій. Якщо все ж скрапите — поважайте robots.txt
    і умови використання сайту, і не кладіть такий інструмент у продакшен
    без узгодження з власником даних.

ЯК ЗАПУСТИТИ
    python foxtrot_scraper_agent.py
    Потрібен Ollama з qwen3:8b та інтернет.
"""
import asyncio
import requests
from bs4 import BeautifulSoup
import urllib.parse

from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import FunctionAgent


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



def search_foxtrot(query: str) -> str:
    """
    Шукає товари на сайті foxtrot.com.ua. Витягує назви, ціни, рейтинг та посилання на товари.
    """
    print(f"--- Пошук на Foxtrot: '{query}' ---")
    try:
        base_url = "https://www.foxtrot.com.ua"
        encoded_query = urllib.parse.quote_plus(query)
        search_url = f"{base_url}/uk/search?query={encoded_query}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        }
        response = requests.get(search_url, headers=headers, timeout=20)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        product_cards = soup.find_all('div', class_='card', limit=5)

        if not product_cards:
            return f"Товарів за запитом '{query}' на Foxtrot не знайдено."

        results = []
        for card in product_cards:
            card_head = card.find('div', class_='card__head')
            title_tag = card.find('a', class_='card__title')

            title = title_tag.get_text(strip=True) if title_tag else "Назва не знайдена"

            price_text = "Ціна не вказана"
            if card_head and 'data-price' in card_head.attrs:
                price_value = card_head['data-price']
                if price_value and price_value.isdigit():
                    price_text = f"{price_value} грн"

            rating_tag = card.find('div', class_='star-gradient-number')
            rating_text = "Рейтинг відсутній"
            if rating_tag:
                rating_text = rating_tag.get_text(strip=True)

            link_text = "Посилання не знайдено"
            if title_tag and 'href' in title_tag.attrs:
                link_text = base_url + title_tag['href']

            results.append(f"- Назва: {title}, Ціна: {price_text}, Рейтинг: {rating_text}, Посилання: {link_text}")

        return "\n".join(results)

    except Exception as e:
        return f"Помилка під час пошуку на Foxtrot: {e}"


def save_report_to_file(filename: str, report_content: str) -> str:
    """Зберігає звіт (текст у форматі Markdown) у файл."""
    print(f"--- Збереження звіту у файл: {filename} ---")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return f"Звіт успішно збережено у файл '{filename}'."
    except Exception as e:
        return f"Помилка при збереженні файлу: {e}"


print("Ініціалізація LLM (qwen3:8b)...")
llm = Ollama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    thinking=False,        # qwen3 інакше пише <think>-трасу перед кожним кроком
    context_window=8192,   # LlamaIndex шле це в Ollama як num_ctx: 6.3 ГБ замість 11
    request_timeout=300.0,
)

print("Створення інструментів...")
foxtrot_tool = FunctionTool.from_defaults(fn=search_foxtrot)
file_write_tool = FunctionTool.from_defaults(fn=save_report_to_file)

all_tools = [foxtrot_tool, file_write_tool]

print("Створення агента 'Foxtrot Scraper'...")
system_prompt = """
Ти — AI-асистент, що спеціалізується на пошуку товарів на сайті Foxtrot.com.ua.
Твоя мета — знаходити товари за запитом користувача, витягувати ключову інформацію (назву, ціну, рейтинг, посилання) та створювати звіт.

Твій план дій:
1.  **Пошук:** Використовуй інструмент `search_foxtrot` з ключовим словом від користувача.
2.  **Аналіз:** Проаналізуй отримані результати.
3.  **Формування звіту:** Створи звіт у форматі Markdown з таблицею результатів. Включи колонки: Назва, Ціна, Рейтинг, Посилання. Посилання має бути клікабельним у Markdown форматі.
4.  **Збереження:** Використовуй `save_report_to_file`, щоб зберегти звіт у файл з назвою, яку вказав користувач.

Працюй автономно до повного виконання завдання. Завжди відповідай українською.
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
        "Знайди на Foxtrot товари, пов'язані з 'капібара', "
        "і збережи звіт з їх цінами, рейтингами та посиланнями у файл 'foxtrot_capybara_full_report.md'."
    )
    print(f"\n🚀 Запускаю агента із завданням: '{task}'\n")

    response = await _run_with_steps(agent, task)

    print("\nOK: Завдання виконано!")
    print(f"Фінальна відповідь агента: {response}")
    print("\nПеревірте файл 'foxtrot_capybara_full_report.md' у вашій директорії.")


if __name__ == "__main__":
    asyncio.run(main())