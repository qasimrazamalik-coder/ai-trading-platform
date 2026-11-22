from collections.abc import Sequence
from pathlib import Path

import pytest

from quant_os_core.knowledge import rag
from quant_os_core.knowledge.rag import (
    answer_multilingual,
    detect_language,
    glossary_translate,
    load_knowledge_base,
)

DOCS = {
    "halal.md": "# Halal rules\n## Futures\nFutures are blocked because both counter-values are "
    "postponed (IIFA Resolution 63).\n## Riba\nInterest (riba) is forbidden; halal accounts "
    "hold no interest.\n",
    "zakat.md": "# Zakat\n## Zakat basics\nZakat is 2.5% of zakatable wealth above the nisab "
    "held for one lunar year. Not a fatwa.\n",
    "risk.md": "# Risk\n## Drawdown\nA drawdown is the fall from the equity peak; a 50% drawdown "
    "needs a 100% gain to recover.\n",
}

UR_FUTURES = "کیا فیوچرز حلال ہیں؟"
AR_FUTURES = "هل العقود الآجلة حلال؟"
UR_ZAKAT = "زکوٰۃ کا حساب کیسے کریں؟"
AR_RISK = "ما هو التراجع في المخاطر؟"


@pytest.fixture
def docs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("QUANT_OS_OLLAMA_MODEL", raising=False)
    for name, text in DOCS.items():
        (tmp_path / name).write_text(text, "utf-8")
    return tmp_path


def test_detects_script_and_tells_urdu_from_arabic() -> None:
    assert detect_language("what is a drawdown?") == "en"
    assert detect_language(UR_FUTURES) == "ur"
    assert detect_language(UR_ZAKAT) == "ur"
    assert detect_language(AR_FUTURES) == "ar"
    assert detect_language(AR_RISK) == "ar"


def test_glossary_builds_an_english_query() -> None:
    assert set(glossary_translate(UR_FUTURES).split()) == {"futures", "halal"}
    assert "futures" in glossary_translate(AR_FUTURES).split()
    assert "zakat" in glossary_translate(UR_ZAKAT).split()
    assert {"drawdown", "risk"} <= set(glossary_translate(AR_RISK).split())
    assert "ETF" in glossary_translate("SPUS ETF حلال").split()  # Latin words kept


@pytest.mark.parametrize(
    ("question", "lang", "source"),
    [
        (UR_FUTURES, "ur", "halal.md"),
        (AR_FUTURES, "ar", "halal.md"),
        (UR_ZAKAT, "ur", "zakat.md"),
        (AR_RISK, "ar", "risk.md"),
    ],
)
def test_without_a_model_returns_english_passages_with_a_note(
    docs: Path, question: str, lang: str, source: str
) -> None:
    reply = answer_multilingual(question, docs=docs)
    assert reply["lang"] == lang
    assert source in {s["source"] for s in reply["sources"]}
    note = rag._NO_MODEL_NOTE[lang]
    assert reply["answer"].startswith(note) and "Relevant passages" in reply["answer"]


def test_english_path_and_explicit_language(docs: Path) -> None:
    english = answer_multilingual("are futures halal?", docs=docs)
    assert english["lang"] == "en" and english["sources"][0]["source"] == "halal.md"
    forced = answer_multilingual("are futures halal?", "ur", docs=docs)
    assert forced["lang"] == "ur" and forced["answer"].startswith(rag._NO_MODEL_NOTE["ur"])
    with pytest.raises(ValueError):
        answer_multilingual("x", "fr", docs=docs)


def test_refuses_trade_calls_and_unknown_topics_in_the_users_language(docs: Path) -> None:
    buy = answer_multilingual("کیا میں بٹ کوائن خریدوں؟", docs=docs)
    assert buy == {"answer": rag._NO_SIGNALS["ur"], "sources": [], "lang": "ur"}
    sell = answer_multilingual("هل أبيع الذهب الآن؟", docs=docs)
    assert sell["answer"] == rag._NO_SIGNALS["ar"] and not sell["sources"]
    unknown = answer_multilingual("یہ کیا ہے؟", docs=docs)
    assert unknown["answer"] == rag._NOT_COVERED["ur"] and not unknown["sources"]


def _model(translation_in: str, answer: str, translation_out: str):  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def generate(messages: Sequence[dict[str, str]]) -> str:
        system = messages[0]["content"]
        calls.append(system)
        if system.startswith("Translate the user's"):
            return translation_in
        if system.startswith("Translate the text"):
            return translation_out
        return answer

    return generate, calls


def test_with_a_model_translates_in_and_out_and_keeps_citations(docs: Path) -> None:
    generate, calls = _model(
        "Are futures halal?", "<think>x</think>Futures are blocked [1].", "فیوچرز ممنوع ہیں [1]"
    )
    reply = answer_multilingual(UR_FUTURES, docs=docs, llm=generate)
    assert reply == {
        "answer": "فیوچرز ممنوع ہیں [1]",
        "sources": reply["sources"],
        "lang": "ur",
    }
    assert reply["sources"][0]["source"] == "halal.md"
    assert len(calls) == 3  # translate in, answer, translate out


def test_translation_that_drops_citations_falls_back_to_english(docs: Path) -> None:
    generate, _ = _model("Are futures halal?", "Futures are blocked [1].", "ممنوع")
    reply = answer_multilingual(AR_FUTURES, docs=docs, llm=generate)
    assert reply["answer"].startswith(rag._NO_MODEL_NOTE["ar"])
    assert "Futures are blocked [1]." in reply["answer"]


def test_offline_model_falls_back_to_the_glossary(docs: Path) -> None:
    def offline(_m: Sequence[dict[str, str]]) -> str:
        raise ConnectionError("ollama down")

    reply = answer_multilingual(UR_ZAKAT, docs=docs, llm=offline)
    assert reply["sources"][0]["source"] == "zakat.md"
    assert reply["answer"].startswith(rag._NO_MODEL_NOTE["ur"])


def test_env_model_is_used_only_when_set_and_allowed(
    docs: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_ollama(model: str = "x", **_kw: object):  # type: ignore[no-untyped-def]
        seen.append(model)

        def generate(_m: Sequence[dict[str, str]]) -> str:
            raise ConnectionError("no network in tests")

        return generate

    monkeypatch.setattr(rag, "ollama_generate", fake_ollama)
    monkeypatch.setenv("QUANT_OS_OLLAMA_MODEL", "qwen3:8b")
    answer_multilingual(UR_FUTURES, docs=docs)
    assert seen == ["qwen3:8b"]
    answer_multilingual(UR_FUTURES, docs=docs, env_model=False)
    assert seen == ["qwen3:8b"]


def test_knowledge_base_is_cached_per_path_and_rebuilt_on_change(docs: Path) -> None:
    first = load_knowledge_base(docs)
    assert load_knowledge_base(docs) is first
    (docs / "extra.md").write_text("# Extra\n## Sukuk\nSukuk are asset-backed.\n", "utf-8")
    second = load_knowledge_base(docs)
    assert second is not first and "extra.md" in {c.source for c in second.chunks}
    with pytest.raises(FileNotFoundError):
        load_knowledge_base(docs / "missing")
