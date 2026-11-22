"""Ask the local knowledge base from a terminal.

python -m quant_os_core.knowledge --docs docs "how do I size a position?"
python -m quant_os_core.knowledge --docs docs --no-llm "what is deflated sharpe?"
python -m quant_os_core.knowledge --docs docs/knowledge --lang ur "zakat kaise nikalein?"

Urdu or Arabic questions (detected from the script, or forced with --lang) go through
``answer_multilingual``: a local model translates when one is available, otherwise a
glossary builds the English query and the English passages come back with a note.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_os_core.knowledge.rag import (
    KnowledgeBase,
    KnowledgeChatbot,
    answer_multilingual,
    detect_language,
    ollama_generate,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS local knowledge chatbot")
    parser.add_argument("question")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--no-llm", action="store_true", help="quote passages only")
    parser.add_argument("--lang", choices=["auto", "en", "ur", "ar"], default="auto")
    args = parser.parse_args(argv)
    if not args.docs.is_dir():
        print(f"docs folder not found: {args.docs}", file=sys.stderr)
        return 2
    lang = detect_language(args.question) if args.lang == "auto" else args.lang
    if lang != "en":
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if reconfigure is not None:  # Windows consoles default to a code page without Urdu
            reconfigure(encoding="utf-8")
        llm = None if args.no_llm else ollama_generate(args.model)
        reply = answer_multilingual(
            args.question, lang, docs=args.docs, llm=llm, env_model=not args.no_llm
        )
        print(reply["answer"])
        for i, source in enumerate(reply["sources"], start=1):
            print(f"  [{i}] {source['source']} - {source['heading']}")
        return 0
    bot = KnowledgeChatbot(
        KnowledgeBase.from_directory(args.docs),
        None if args.no_llm else ollama_generate(args.model),
    )
    answer = bot.ask(args.question)
    print(answer.text)
    for i, chunk in enumerate(answer.sources, start=1):
        print(f"  [{i}] {chunk.source} - {chunk.heading}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
