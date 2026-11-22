"""The ``python -m quant_os_core.knowledge`` command, run offline with ``--no-llm``."""

from pathlib import Path

import pytest

from quant_os_core.knowledge.__main__ import main

DOCS = Path(__file__).resolve().parents[4] / "docs" / "knowledge"


def test_cli_answers_from_docs_and_lists_sources(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--docs", str(DOCS), "--no-llm", "are futures halal IIFA"]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    assert "[1] 05-halal-rules.md" in out


def test_cli_rejects_missing_docs_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "nope"
    assert main(["--docs", str(missing), "--no-llm", "anything"]) == 2
    assert "docs folder not found" in capsys.readouterr().err
