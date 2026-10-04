# Лабораторна робота №2 — AI-агенти на LlamaIndex

**[UA]** Код до лабораторної роботи курсу «Прикладний штучний інтелект та MLOps
процесу розробки програмного забезпечення» (Лекція 5). Агенти працюють на
**локальній** моделі через Ollama: без API-ключів і без хмарних витрат.

**[EN]** Lab code for an applied AI & MLOps university course. The agents run on a
local LLM via Ollama — no API keys, no cloud costs.

---

## ⚠️ Важливо: LlamaHub більше не окремий сайт

Торішня версія роботи починалася словами «відвідайте llamahub.ai → розділ Agent
Tools». **Так більше не працює.** `llamahub.ai` тепер віддає заглушку:

> LlamaHub has moved. Browse LlamaIndex integrations at developers.llamaindex.ai

Де шукати інструменти тепер:

| Що потрібно | Де дивитися |
|---|---|
| Огляд категорій інтеграцій | [developers.llamaindex.ai → Community → Integrations](https://developers.llamaindex.ai/python/framework/community/integrations) |
| **Повний перелік ToolSpec-ів** | [каталог `llama-index-integrations/tools`](https://github.com/run-llama/llama_index/tree/main/llama-index-integrations/tools) — один підкаталог на пакет |
| Як узагалі влаштовані інструменти | [tools guide](https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools) |

Станом на жовтень 2026 у каталозі **67 пакетів** інструментів. Назва пакета
завжди має вигляд `llama-index-tools-<назва>`, встановлюється через pip.

### І головне правило, якого не було торік

**ToolSpec, який встановлюється й імпортується, — це ще не ToolSpec, який
працює.** Перед тим як будувати лабораторну навколо інструмента, викличте його
один раз голяка, без агента. Два приклади з цього ж репозиторію:

- `llama-index-tools-duckduckgo` — імпортується нормально, але **будь-який**
  виклик падає з `primp.BuilderError: Invalid impersonate`: пакет прибитий до
  знятого з підтримки `duckduckgo-search`. Робочий шлях — пакет `ddgs` і власний
  `FunctionTool` на 12 рядків.
- `llama-index-tools-wikipedia` — `load_data` падає з `JSONDecodeError` для
  **будь-якої** сторінки, бо Wikimedia блокує дефолтний User-Agent. Лікується
  одним рядком: `wikipedia.set_user_agent("Назва/1.0 (контакт)")`.

---

## Швидкий старт

```bash
# 1. Ollama (один раз). Модель має підтримувати tool calling.
ollama pull qwen3:8b

# 2. Середовище
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # Linux / macOS
pip install -r requirements.txt

# 3. Запуск
python main.py
```

---

## Файли

| Файл | Що робить | Прогін |
|---|---|---|
| `main.py` | **«Науковий Дайджест»** — еталонний агент роботи: шукає статті на arXiv і зводить їх у звіт за шаблоном | ~10 c |
| `researcher_programmer_agent.py` | «Дослідник-Програміст»: Вікіпедія → пише Python-скрипт → **виконує його** → звіт | ~11 c |
| `foxtrot_scraper_agent.py` | скрапер магазину (приклад студентської роботи) | ~15 c |
| `tests/test_tools.py` | 37 тестів інструментів **без виклику моделі й без мережі** | ~4 c |

```bash
python main.py --topic "RAG evaluation" --count 3
python main.py --show-toolspec    # що насправді віддає ArxivToolSpec
python main.py --help
pytest -q
```

Звіти агента потрапляють у `reports/` (ця тека в `.gitignore`).

---

## Чому в `main.py` є власний `search_arxiv`

Лабораторна вимагає звіт із авторами. `ArxivToolSpec` віддає **один** інструмент —
`arxiv_query` (не `arxiv_search`, як пишуть у старих прикладах), і в його
відповіді **авторів немає** — лише URL, назва й анотація одним рядком.

Торішній приклад звіту це й показував: у двох різних статей був виписаний один і
той самий список авторів. Модель його просто вигадала, бо інструмент авторів не
давав, а шаблон їх вимагав.

Тому ToolSpec тут показано (`--show-toolspec`), а для самого звіту написано
власний інструмент поверх бібліотеки `arxiv`: він повертає назву, авторів, дату,
URL і анотацію. Запустіть `--show-toolspec` і порівняйте — це найкоротший шлях
зрозуміти, навіщо взагалі писати власні обгортки.

Там же зашита ще одна пастка arXiv: із сортуванням за датою запит **без префікса
поля** ігнорується, і повертаються просто найсвіжіші сабміти з усіх розділів —
квантова фізика замість вашої теми. Лікується префіксом `all:"..."`.

---

## Три деталі підключення до Ollama

```python
Ollama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    thinking=False,         # 1
    context_window=8192,    # 2
    request_timeout=300.0,  # 3
)
```

1. **`thinking=False`.** `qwen3` — гібридна reasoning-модель, і LlamaIndex лишає
   `thinking=None`, тобто трасу вмикає сама модель. Без цього прапорця вона пише
   довгий `<think>` перед кожним кроком.
2. **`context_window` доходить до сервера.** LlamaIndex надсилає це значення в
   Ollama як `num_ctx`. Якщо не задати, він лишає `-1`, питає повне вікно моделі
   і просить його цілком: для `qwen3:8b` це 40960 токенів і ~11 ГБ VRAM замість
   6.3 ГБ. Перевіряти треба **на холодному старті** — завантажена модель тримає
   своє вікно до кінця `keep_alive`.
3. **`request_timeout`.** Дефолт LlamaIndex — 30 секунд, для агента замало.

---

## `verbose=True` не показує кроків агента

Це **не поле** `FunctionAgent`: аргумент провалюється у Workflow-рушій і друкує
сирий технічний лог (`[tick] add: AgentInput(...)`), а не те, що зробив агент.
Читабельні кроки дає стрімінг подій:

```python
from llama_index.core.agent.workflow import ToolCall, ToolCallResult

handler = agent.run(user_msg=task, max_iterations=10)
async for ev in handler.stream_events():
    if isinstance(ev, ToolCall):
        print(ev.tool_name, ev.tool_kwargs)
    elif isinstance(ev, ToolCallResult):
        print(ev.tool_output)
response = await handler
```

Скріншот для звіту робіть саме з цього виводу.

---

## Безпека

**Ім'я файлу приходить від моделі** — це недовірений ввід. Усі інструменти
запису йдуть через `safe_output_path()`: він зрізає теки й літери диска,
відкидає `..`, порожні імена та зарезервовані імена Windows. Перевірте своєю
реалізацією запит на `../../pwned.md` — інструмент має відмовити.

**`researcher_programmer_agent.py` виконує код, який написала модель.**
Для навчання на власній машині це прийнятно; у продакшені — ніколи. Потрібна
пісочниця (Docker, E2B, Modal). Деталі — у матеріалах Лекції 5,
`09_security_mcp.py`.

---

## Траблшутинг

| Симптом | Причина і що робити |
|---|---|
| `ConnectionError` на `localhost:11434` | Ollama не запущений. Windows/macOS — відкрити застосунок; Linux — `ollama serve` |
| Агент відповідає прозою і не кличе інструменти | модель без capability `tools`. Перевірте: `ollama show qwen3:8b` |
| `JSONDecodeError` у Вікіпедії | заблоковано дефолтний User-Agent → `wikipedia.set_user_agent("Назва/1.0 (контакт)")` |
| `primp.BuilderError: Invalid impersonate` | ви взяли `llama-index-tools-duckduckgo`. Він мертвий — беріть `ddgs` |
| `WorkflowRuntimeError: Max iterations of 20 reached` | агент зациклився. `early_stopping_method="generate"` + `max_iterations` у `run()` |
| Агент «забув» завдання після виклику інструмента | інструмент повернув забагато тексту (Вікіпедія віддає статтю цілком — 80+ тис. символів). Обріжте вихід |
| Модель вигадує авторів / дати | інструмент їх не повертає. Перевірте, що саме приходить, і додайте в промпт «використовуй ЛИШЕ поля, які повернув інструмент» |
| `ReadTimeout` | `.env` → `REQUEST_TIMEOUT=600` |
| Не вистачає VRAM | `.env` → `LLM_MODEL=qwen3:4b`, `NUM_CTX=4096` |
| Кракозябри / `UnicodeEncodeError` | консоль у cp1251. Усі файли викликають `setup_console()`; у своєму коді робіть так само |

---

## Версії

Зафіксовано в `requirements.txt`, перевірено запуском: Windows 10, Python 3.10.6,
Ollama 0.32.0, `llama-index-core` 0.14.25, `llama-index-llms-ollama` 0.11.0,
ToolSpec-и 0.6.0, `arxiv` 2.4.1.
