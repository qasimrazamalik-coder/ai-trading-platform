"""Ask the local knowledge base from a terminal.

python -m quant_os_core.knowledge --docs docs "how do I size a position?"
python -m quant_os_core.knowledge --docs docs --no-llm "what is deflated sharpe?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_os_core.knowledge.rag import KnowledgeBase, KnowledgeChatbot, ollama_generate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quant OS local knowledge chatbot")
    parser.add_argument("question")
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--no-llm", action="store_true", help="quote passages only")
    args = parser.parse_args(argv)
    if not args.docs.is_dir():
        print(f"docs folder not found: {args.docs}", file=sys.stderr)
        return 2
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
