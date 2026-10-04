"""
Регресійні тести до ЛР №2 — БЕЗ виклику моделі та без мережі.

Інструмент агента — звичайний детермінований код, і покривати його треба як
звичайний код. Саму LLM у тестах не ганяють: вона повільна й недетермінована.
Мережу до arXiv підміняємо фейковим клієнтом, тому pytest минає за секунди
і працює без інтернету.

    pytest -q
"""
from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main  # noqa: E402


# ══════════════════════════════════════════════════════════════════
# 1. safe_output_path — ім'я файлу приходить ВІД МОДЕЛІ
# ══════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("hostile", [
    "../../../etc/passwd",
    "..\\..\\Windows\\System32\\drivers\\etc\\hosts",
    "/absolute/unix/path.md",
    "C:\\Windows\\System32\\evil.md",
    "sub/dir/nested.md",
    "....//....//escape.md",
])
def test_safe_path_contains_hostile_input(tmp_path, hostile):
    result = main.safe_output_path(hostile, base_dir=tmp_path)
    assert result.parent == tmp_path.resolve(), f"вийшли за межі теки: {result}"
    assert result.suffix == ".md"


def test_safe_path_keeps_plain_name(tmp_path):
    assert main.safe_output_path("digest.md", base_dir=tmp_path).name == "digest.md"


def test_safe_path_adds_md_extension(tmp_path):
    assert main.safe_output_path("digest", base_dir=tmp_path).name == "digest.md"


@pytest.mark.parametrize("bad", ["", "   ", ".", "..", "\x00x"])
def test_safe_path_rejects_degenerate_names(tmp_path, bad):
    with pytest.raises(ValueError):
        main.safe_output_path(bad, base_dir=tmp_path)


@pytest.mark.parametrize("reserved", ["CON", "nul.md", "com1.md", "LPT9"])
def test_safe_path_rejects_windows_device_names(tmp_path, reserved):
    with pytest.raises(ValueError):
        main.safe_output_path(reserved, base_dir=tmp_path)


def test_safe_path_creates_base_dir(tmp_path):
    base = tmp_path / "does" / "not" / "exist"
    main.safe_output_path("x.md", base_dir=base)
    assert base.is_dir()


# ══════════════════════════════════════════════════════════════════
# 2. write_report
# ══════════════════════════════════════════════════════════════════
def test_write_report_saves_and_reports_words(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path)
    out = main.write_report("digest.md", "# Звіт\n\nодин два три")
    assert "Saved to digest.md" in out
    assert "5 words" in out            # '#', 'Звіт' і три слова тексту
    assert (tmp_path / "digest.md").read_text(encoding="utf-8").startswith("# Звіт")


def test_write_report_refuses_escape_instead_of_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path)
    out = main.write_report("..", "x")
    assert out.startswith("Refused to write")
    # інструмент повертає текст помилки, а не кидає виняток:
    # для агента це Observation, за яким він виправляється


def test_write_report_is_utf8(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "REPORTS_DIR", tmp_path)
    main.write_report("u.md", "Анотація українською — тире й лапки «»")
    assert "українською" in (tmp_path / "u.md").read_text(encoding="utf-8")


# ══════════════════════════════════════════════════════════════════
# 3. search_arxiv — мережу підміняємо
# ══════════════════════════════════════════════════════════════════
class FakeAuthor:
    def __init__(self, name):
        self.name = name


class FakeResult:
    def __init__(self, title, authors, summary, url, published):
        self.title = title
        self.authors = [FakeAuthor(a) for a in authors]
        self.summary = summary
        self.entry_id = url
        self.published = published


def fake_arxiv(results, captured):
    """Підставний модуль arxiv: записує побудований запит у captured."""
    mod = types.SimpleNamespace()

    class Search:
        def __init__(self, query, max_results, sort_by):
            captured["query"] = query
            captured["max_results"] = max_results
            captured["sort_by"] = sort_by

    class Client:
        def __init__(self, **kw):
            pass

        def results(self, search):
            return iter(results)

    mod.Search = Search
    mod.Client = Client
    mod.SortCriterion = types.SimpleNamespace(SubmittedDate="date", Relevance="rel")
    return mod


@pytest.fixture
def one_paper():
    return [FakeResult("Test Title", ["Ада Лавлейс", "Alan Turing"],
                       "Рядок\nз   переносами", "http://arxiv.org/abs/1", datetime(2026, 10, 1))]


def test_search_returns_all_template_fields(monkeypatch, one_paper):
    cap = {}
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv(one_paper, cap))
    out = main.search_arxiv("Multimodal LLM", max_results=1)
    for field in ("TITLE:", "AUTHORS:", "PUBLISHED:", "URL:", "ABSTRACT:"):
        assert field in out, f"у відповіді немає {field}"
    assert "Ада Лавлейс, Alan Turing" in out
    assert "2026-10-01" in out


def test_search_wraps_query_in_field_prefix(monkeypatch, one_paper):
    """
    Пастка arXiv: із сортуванням за датою запит без префікса поля
    ігнорується і повертаються просто найсвіжіші сабміти з усіх розділів.
    """
    cap = {}
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv(one_paper, cap))
    main.search_arxiv("Multimodal LLM", max_results=1)
    assert cap["query"] == 'all:"Multimodal LLM"'


def test_search_keeps_explicit_field_query(monkeypatch, one_paper):
    cap = {}
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv(one_paper, cap))
    main.search_arxiv('ti:"Attention Is All You Need"', max_results=1)
    assert cap["query"] == 'ti:"Attention Is All You Need"'


@pytest.mark.parametrize("asked,expected", [(0, 1), (1, 1), (3, 3), (99, 5)])
def test_search_clamps_result_count(monkeypatch, one_paper, asked, expected):
    cap = {}
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv(one_paper, cap))
    main.search_arxiv("x", max_results=asked)
    assert cap["max_results"] == expected


def test_search_switches_sort_order(monkeypatch, one_paper):
    cap = {}
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv(one_paper, cap))
    main.search_arxiv("x", recent=True)
    assert cap["sort_by"] == "date"
    main.search_arxiv("x", recent=False)
    assert cap["sort_by"] == "rel"


def test_search_collapses_whitespace_in_abstract(monkeypatch, one_paper):
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv(one_paper, {}))
    out = main.search_arxiv("x")
    assert "Рядок з переносами" in out


def test_search_reports_empty_result_without_crashing(monkeypatch):
    monkeypatch.setitem(sys.modules, "arxiv", fake_arxiv([], {}))
    assert "No papers found" in main.search_arxiv("нісенітниця")


def test_search_survives_network_error(monkeypatch):
    class Boom:
        def __init__(self, **kw):
            raise ConnectionError("no network")

    mod = types.SimpleNamespace(Search=lambda **kw: None, Client=Boom,
                                SortCriterion=types.SimpleNamespace(
                                    SubmittedDate="d", Relevance="r"))
    monkeypatch.setitem(sys.modules, "arxiv", mod)
    out = main.search_arxiv("x")
    assert out.startswith("arXiv search failed")   # текст помилки, не виняток


# ══════════════════════════════════════════════════════════════════
# 4. Передпольотна перевірка
# ══════════════════════════════════════════════════════════════════
def test_preflight_reports_unreachable_ollama(monkeypatch):
    monkeypatch.setattr(main, "OLLAMA_BASE_URL", "http://127.0.0.1:1")
    problems = main.preflight("qwen3:8b")
    assert problems and "не відповідає" in problems[0]


# ══════════════════════════════════════════════════════════════════
# 5. Факти про API, на яких тримається лабораторна
# ══════════════════════════════════════════════════════════════════
def test_arxiv_toolspec_tool_is_named_arxiv_query():
    """Методички роками писали `arxiv_search` — такого інструмента немає."""
    from llama_index.tools.arxiv import ArxivToolSpec

    names = [t.metadata.name for t in ArxivToolSpec().to_tool_list()]
    assert names == ["arxiv_query"], f"назви змінились: {names}"


def test_function_agent_has_no_verbose_field():
    """
    `verbose=True` не є полем FunctionAgent: аргумент провалюється у
    Workflow-рушій і друкує сирий лог подій, а не кроки агента.
    """
    from llama_index.core.agent.workflow import FunctionAgent

    assert "verbose" not in FunctionAgent.model_fields


def test_tool_call_result_is_not_a_tool_call():
    """Від цього залежить коректність if/elif у стрімінгу подій."""
    from llama_index.core.agent.workflow import ToolCall, ToolCallResult

    assert not issubclass(ToolCallResult, ToolCall)


def test_module_import_is_free_of_network_calls():
    """Імпорт main не має нічого будувати: інакше тести не запустити."""
    import importlib
    import time

    started = time.time()
    importlib.reload(main)
    assert time.time() - started < 5.0
