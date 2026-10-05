"""
main.py — Лабораторна робота №2: AI-агент «Науковий Дайджест» на LlamaIndex.

Курс «Прикладний штучний інтелект та MLOps процесу розробки програмного
забезпечення», Лекція 5 (LlamaIndex + Ollama).

ЩО ТУТ Є
    1. Демонстрація ToolSpec з каталогу інтеграцій LlamaIndex (ArxivToolSpec) —
       і чесна перевірка того, що саме він повертає.
    2. Власний інструмент search_arxiv: ToolSpec НЕ віддає авторів, а шаблон
       звіту їх вимагає. Обгортка над бібліотекою arxiv закриває цю прогалину.
    3. Інструмент write_report із санітизацією шляху: ім'я файлу приходить від
       моделі, тобто це недовірений ввід.
    4. FunctionAgent зі стрімінгом подій — видно кожен виклик інструмента.

ЯК ЗАПУСТИТИ
    python main.py                      # тема за замовчуванням, 2 статті
    python main.py --topic "RAG evaluation" --count 3
    python main.py --show-toolspec      # що насправді повертає ArxivToolSpec
    python main.py --help

    Потрібен запущений Ollama з моделлю qwen3:8b.
    Якщо чогось бракує — скрипт пояснює, що саме, і виходить без трейсбеку.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# 1. КОНСОЛЬ І НАЛАШТУВАННЯ
# ══════════════════════════════════════════════════════════════════
def setup_console() -> None:
    """Windows-консоль стартує у cp1251 і падає на кирилиці, щойно вивід
    перенаправляють у файл. Викликаємо ПЕРЕД будь-яким print()."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


setup_console()

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

DEFAULT_TOPIC = "Multimodal Large Language Models"
REPORT_NAME = "digest.md"
ABSTRACT_CHARS = 900


# ══════════════════════════════════════════════════════════════════
# 2. БЕЗПЕКА: ім'я файлу приходить від МОДЕЛІ
# ══════════════════════════════════════════════════════════════════
_RESERVED = {"con", "prn", "aux", "nul",
             *(f"com{i}" for i in range(1, 10)),
             *(f"lpt{i}" for i in range(1, 10))}


def safe_output_path(filename: str, base_dir: Path | None = None) -> Path:
    """
    Перетворює довільне ім'я файлу на шлях усередині робочої теки.

    Модель може попросити записати у "../../.env" або "C:/Windows/x.dll".
    Це такий самий недовірений ввід, як рядок із веб-форми.
    """
    base = (base_dir or REPORTS_DIR).resolve()
    base.mkdir(parents=True, exist_ok=True)

    raw = str(filename).strip()
    if not raw or "\x00" in raw:
        raise ValueError("Порожнє або некоректне ім'я файлу")
    name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        raise ValueError(f"Некоректне ім'я файлу: {filename!r}")
    if name.split(".")[0].lower() in _RESERVED:
        raise ValueError(f"Зарезервоване ім'я Windows: {name!r}")
    if not name.endswith(".md"):
        name += ".md"

    target = (base / name).resolve()
    if not target.is_relative_to(base):          # страхувальний пояс
        raise ValueError(f"Шлях виходить за межі теки: {filename!r}")
    return target


# ══════════════════════════════════════════════════════════════════
# 3. ПЕРЕДПОЛЬОТНА ПЕРЕВІРКА
# ══════════════════════════════════════════════════════════════════
def preflight(model_name: str) -> list[str]:
    """Повертає список проблем. Порожній список — усе гаразд."""
    import json
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=8) as r:
            models = [m["name"] for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return [f"Ollama не відповідає на {OLLAMA_BASE_URL} ({type(exc).__name__}).",
                "   Windows/macOS — відкрийте застосунок; Linux — `ollama serve`"]

    stem = model_name.split(":")[0]
    if not any(m == model_name or m.split(":")[0] == stem for m in models):
        return [f"Модель '{model_name}' не знайдена в Ollama.",
                f"   Виправити:  ollama pull {model_name}"]
    return []


def build_llm(model_name: str, num_ctx: int):
    """
    LLM для агента. Три параметри, яких немає в типових прикладах:

      thinking=False    qwen3 — гібридна reasoning-модель і за замовчуванням
                        пише <think>-трасу перед кожним кроком. Відповіді ті
                        самі, час — у рази більший.
      context_window    LlamaIndex надсилає це значення в Ollama як num_ctx.
                        Якщо не задати, він попросить ПОВНЕ вікно моделі:
                        для qwen3:8b це 40960 токенів і ~11 ГБ VRAM замість 6.3.
      request_timeout   дефолт LlamaIndex — 30 с. Для агента з кількома
                        викликами інструментів цього мало.
    """
    from llama_index.llms.ollama import Ollama

    return Ollama(
        model=model_name,
        base_url=OLLAMA_BASE_URL,
        thinking=False,
        context_window=num_ctx,
        request_timeout=REQUEST_TIMEOUT,
    )


# ══════════════════════════════════════════════════════════════════
# 4. ToolSpec З КАТАЛОГУ ІНТЕГРАЦІЙ — і чому його мало
# ══════════════════════════════════════════════════════════════════
def show_toolspec() -> None:
    """
    Крок 1 лабораторної: подивитися, що насправді дає готовий ToolSpec.

    ArxivToolSpec віддає РІВНО ОДИН інструмент — `arxiv_query` (не
    `arxiv_search`, як пишуть у багатьох прикладах). А головне: у його
    відповіді немає авторів — лише URL, назва й анотація одним рядком.
    Шаблон звіту авторів вимагає, тож модель їх просто вигадає.

    Це і є причина написати власний інструмент: не тому, що ToolSpec
    «поганий», а тому, що його вихід не покриває вимогу задачі.
    """
    from llama_index.tools.arxiv import ArxivToolSpec

    spec = ArxivToolSpec()
    tools = spec.to_tool_list()
    print(f"\nArxivToolSpec віддає {len(tools)} інструмент(ів):")
    for t in tools:
        params = t.metadata.fn_schema.model_json_schema().get("properties", {})
        print(f"  • {t.metadata.name}({', '.join(params)})")

    print("\nЩо повертає arxiv_query (перші 300 символів першого результату):")
    # Ще одна вада готового ToolSpec: він просить у arXiv одразу 100
    # результатів і не має ретраїв. На повторних запусках arXiv відповідає
    # HTTP 429 (Too Many Requests), і виклик просто падає. Наш власний
    # search_arxiv цього не має: там arxiv.Client(num_retries=3).
    docs = None
    for attempt in range(3):
        try:
            docs = spec.arxiv_query(query='all:"Multimodal Large Language Models"')
            break
        except Exception as exc:                  # noqa: BLE001
            if "429" in str(exc) and attempt < 2:
                print(f"  arXiv відповів 429 (ліміт запитів), пауза {5 * (attempt + 1)} c…")
                time.sleep(5 * (attempt + 1))
                continue
            print(f"  Не вдалося: {type(exc).__name__}: {str(exc)[:120]}")
            return
    sample = str(docs[0])[:300].replace("\n", " ") if docs else "(порожньо)"
    print(f"  {sample}…")
    print("\n  ↑ Авторів тут немає. Саме тому нижче ми пишемо власний "
          "search_arxiv.\n")


# ══════════════════════════════════════════════════════════════════
# 5. ІНСТРУМЕНТИ АГЕНТА
# ══════════════════════════════════════════════════════════════════
def search_arxiv(query: str, max_results: int = 2, recent: bool = True) -> str:
    """Searches arXiv and returns full records: title, authors, date, URL, abstract.

    Args:
        query: Search phrase in English, e.g. 'Multimodal Large Language Models'
        max_results: How many papers to return (1-5)
        recent: True for newest submissions first, False for best relevance
    """
    import arxiv

    n = max(1, min(int(max_results), 5))
    # Пастка arXiv: із сортуванням за датою запит БЕЗ префікса поля
    # ігнорується, і повертаються просто найсвіжіші сабміти з усіх розділів.
    # Префікс all:"..." прив'язує пошук до слів запиту.
    raw = query.strip()
    if ":" in raw:
        expr = raw          # запит уже з префіксом поля (ti:, au:, cat:) — не чіпаємо
    else:
        expr = f'all:"{raw.strip(chr(34))}"'
    order = (arxiv.SortCriterion.SubmittedDate if recent
             else arxiv.SortCriterion.Relevance)
    try:
        client = arxiv.Client(page_size=n, delay_seconds=3.0, num_retries=3)
        results = list(client.results(arxiv.Search(query=expr, max_results=n,
                                                   sort_by=order)))
    except Exception as exc:                      # noqa: BLE001 — мережа
        return (f"arXiv search failed ({type(exc).__name__}). "
                f"Do not retry more than once.")
    if not results:
        return "No papers found. Try a broader query, once."

    out = []
    for i, r in enumerate(results, 1):
        authors = ", ".join(a.name for a in r.authors) or "n/a"
        abstract = " ".join(r.summary.split())[:ABSTRACT_CHARS]
        out.append(f"[{i}]\nTITLE: {r.title.strip()}\nAUTHORS: {authors}\n"
                   f"PUBLISHED: {r.published.date()}\nURL: {r.entry_id}\n"
                   f"ABSTRACT: {abstract}")
    return "\n\n".join(out)


def write_report(filename: str, content: str) -> str:
    """Saves the finished Markdown report. Call this LAST, exactly once.

    Args:
        filename: File name ending in .md, e.g. 'digest.md'
        content: Full report in Markdown
    """
    try:
        target = safe_output_path(filename)
    except ValueError as exc:
        return f"Refused to write: {exc}. Pass a plain file name like 'digest.md'."
    target.write_text(content, encoding="utf-8")
    words = len(content.split())
    return (f"Saved to {target.name} ({words} words). Task complete — "
            f"call final answer now.")


# ══════════════════════════════════════════════════════════════════
# 6. СИСТЕМНИЙ ПРОМПТ
# ══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are "Науковий Дайджест", an agent that builds a report
on recent arXiv papers.

WORKFLOW — follow it exactly, once:
1. Call search_arxiv ONCE with the user's topic and the requested count.
2. Use ONLY the fields the tool returned. Never invent authors, dates or URLs.
   If a field is missing, write "n/a".
3. Build a Markdown report. Use this template for EVERY paper, verbatim.
   Do NOT add your own numbering, headings or intro - the report starts
   straight with the first "---":

---
### **<TITLE>**
**Автори:** *<AUTHORS>*
**Опубліковано:** <PUBLISHED>
**Посилання:** [Читати на arXiv](<URL>)

**Анотація:**
> <ABSTRACT>
---

4. Call write_report(filename, content) ONCE with the whole report.
   The task is NOT done until the file is saved.
5. Then give a one-sentence final answer in Ukrainian.

Your only tools are search_arxiv and write_report."""


def build_agent(llm, verbose: bool = False):
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import FunctionTool

    return FunctionAgent(
        tools=[FunctionTool.from_defaults(fn=search_arxiv),
               FunctionTool.from_defaults(fn=write_report)],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        # Дефолт — 20 ітерацій. "generate" дає деградовану відповідь
        # замість WorkflowRuntimeError, якщо агент заплутався.
        early_stopping_method="generate",
        # verbose НЕ є полем FunctionAgent: аргумент провалюється у
        # Workflow-рушій і друкує сирий лог подій, а не кроки агента.
        # Читабельні кроки дає стрімінг — див. run_agent().
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════════
# 7. ЗАПУСК ЗІ СТРІМІНГОМ ПОДІЙ
# ══════════════════════════════════════════════════════════════════
async def run_agent(agent, task: str) -> str:
    """Запускає агента і друкує кожен виклик інструмента."""
    from llama_index.core.agent.workflow import ToolCall, ToolCallResult

    handler = agent.run(user_msg=task, max_iterations=10)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            args = {k: str(v)[:60] for k, v in ev.tool_kwargs.items()}
            print(f"   [tool] {ev.tool_name}({args})")
        elif isinstance(ev, ToolCallResult):
            head = str(ev.tool_output).replace("\n", " ")[:150]
            print(f"   [out ] {head}…")
    return str(await handler)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="ЛР №2: агент «Науковий Дайджест» на LlamaIndex + Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Приклади:\n"
               "  python main.py\n"
               "  python main.py --topic \"RAG evaluation\" --count 3\n"
               "  python main.py --show-toolspec\n")
    ap.add_argument("--topic", default=DEFAULT_TOPIC, help="тема пошуку")
    ap.add_argument("--count", type=int, default=2, help="скільки статей (1-5)")
    ap.add_argument("--model", default=LLM_MODEL, help=f"модель (типово {LLM_MODEL})")
    ap.add_argument("--num-ctx", type=int, default=NUM_CTX, help="вікно контексту")
    ap.add_argument("--show-toolspec", action="store_true",
                    help="показати, що віддає ArxivToolSpec, і вийти")
    ap.add_argument("--verbose", action="store_true",
                    help="додати технічний лог Workflow-рушія")
    args = ap.parse_args()

    print("=" * 62)
    print("  ЛР №2 — «Науковий Дайджест» на LlamaIndex")
    print("=" * 62)

    if args.show_toolspec:
        try:
            show_toolspec()
        except Exception as exc:                  # noqa: BLE001
            print(f"Не вдалося опитати ArxivToolSpec: {type(exc).__name__}: {exc}")
        return 0

    problems = preflight(args.model)
    if problems:
        print("\n".join(f"❌ {p}" if i == 0 else p for i, p in enumerate(problems)))
        return 0

    llm = build_llm(args.model, args.num_ctx)
    print(f"   [agent] {args.model}, num_ctx={args.num_ctx}, think=off")
    agent = build_agent(llm, verbose=args.verbose)

    task = (f"Знайди {args.count} найновіші статті на arXiv за темою "
            f"'{args.topic}'. Склади звіт за шаблоном і збережи його "
            f"у файл '{REPORT_NAME}'.")
    print(f"\n> Завдання: {task}\n" + "─" * 62)

    started = time.time()
    try:
        answer = asyncio.run(run_agent(agent, task))
    except KeyboardInterrupt:
        print("\nПерервано.")
        return 130
    except Exception as exc:                      # noqa: BLE001
        print(f"\n❌ {type(exc).__name__}: {exc}")
        return 0

    print("─" * 62)
    print(f"OK: Готово за {time.time() - started:.0f} c")
    print(f"   Відповідь агента: {str(answer)[:300]}")

    report = REPORTS_DIR / REPORT_NAME
    if report.exists():
        text = report.read_text(encoding="utf-8")
        papers = len(re.findall(r"\*\*Автори:\*\*", text))
        print(f"\n[file] {report} — {report.stat().st_size} байт, статей: {papers}")
        print("─" * 62)
        print(text[:1200] + ("\n… [далі у файлі]" if len(text) > 1200 else ""))
    else:
        print("\n[!] Звіт не створено: модель не викликала write_report.")
        print("    Посильте вимогу в SYSTEM_PROMPT.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
