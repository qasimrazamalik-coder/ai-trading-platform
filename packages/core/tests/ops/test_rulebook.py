import re
from pathlib import Path

import pytest

from quant_os_core.ops import rulebook

REPO = Path(__file__).resolve().parents[4]
ARABIC_SCRIPT = re.compile(r"[؀-ۿ]")


def test_every_rule_points_at_code_that_exists() -> None:
    for rule in rulebook.RULES:
        assert rule.enforced_by, rule.id
        for ref in rule.enforced_by:
            assert rulebook.resolve(ref) is not None, ref
            assert (REPO / rulebook.file_of(ref)).is_file(), ref


def test_rules_are_complete_and_translated() -> None:
    ids = [r.id for r in rulebook.RULES]
    assert len(ids) == len(set(ids))
    for rule in rulebook.RULES:
        assert rule.group in {"shariah", "risk", "ai", "operations"}
        assert set(rule.text) == {"en", "ur", "ar"} and all(rule.text.values())
        assert ARABIC_SCRIPT.search(rule.text["ur"]) and ARABIC_SCRIPT.search(rule.text["ar"])
        assert rule.sources and all(s.startswith(rulebook.SOURCES) for s in rule.sources)


def test_guardrail_shariah_codes_are_all_covered() -> None:
    text = " ".join(r.text["en"] for r in rulebook.RULES)
    for code in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R10", "R13", "R14"):
        assert re.search(rf"\b{code}\b", text), code


def test_resolve_rejects_bad_references() -> None:
    with pytest.raises(ValueError):
        rulebook.resolve("quant_os_core.shariah.rules")
    with pytest.raises(AttributeError):
        rulebook.resolve("quant_os_core.shariah.rules:NoSuchThing")


def test_json_is_read_only_and_localised() -> None:
    data = rulebook.as_json("ur-PK")
    assert data["lang"] == "ur" and data["editable"] is False
    first = data["rules"][0]
    assert first["label"] == rulebook.RULES[0].text["ur"]
    assert first["enforced_by"][0]["file"].endswith("shariah/rules.py")
    assert rulebook.as_json("xx")["lang"] == "en"


def test_docs_rulebook_matches_the_code() -> None:
    on_disk = (REPO / "docs" / "RULEBOOK.md").read_text("utf-8").replace("\r\n", "\n")
    assert on_disk == rulebook.markdown(), "regenerate docs/RULEBOOK.md from rulebook.markdown()"
