"""Local retrieval-augmented chatbot over the markdown knowledge base.

* Retrieval is BM25 over heading-sized chunks, standard library only, so it runs offline.
* Generation goes through an injectable ``Generate`` callable (Ollama by default).
* Guardrails: no answer without retrieved sources; requests for personal buy/sell calls are
  redirected to the trade pipeline; if the model is offline or uncited, the bot falls back to
  quoting the retrieved passages instead of inventing an answer.
"""

from __future__ import annotations

import json
import math
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

Generate = Callable[[Sequence[dict[str, str]]], str]

_TOKEN = re.compile(r"[a-z0-9]+")
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_CITATION = re.compile(r"\[(\d+)\]")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # reasoning models such as Qwen3
_SIGNAL_REQUEST = re.compile(
    r"\b(should i|shall i|tell me to|which (stock|coin|crypto) to)\b.*\b(buy|sell|enter|exit)\b"
    r"|\b(buy|sell) (now|today|tomorrow)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "when",
        "which",
        "why",
        "with",
    ]
)

SYSTEM_PROMPT = (
    "You are the Quant OS knowledge assistant. Answer only from the numbered sources. "
    "Cite every claim with its source number in square brackets, e.g. [1]. If the sources do "
    "not answer the question, say so. Never give personal buy or sell instructions, never "
    "promise returns, and never present a ruling as a fatwa. Keep answers short and plain."
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True, slots=True)
class Chunk:
    source: str
    heading: str
    text: str


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


def chunk_markdown(source: str, text: str, max_chars: int = 1500) -> list[Chunk]:
    """Split on headings (#, ##, ###); long sections are split on blank lines."""

    chunks: list[Chunk] = []
    heading = source
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        buffer.clear()
        if not body:
            return
        part: list[str] = []
        size = 0
        for paragraph in body.split("\n\n"):
            if part and size + len(paragraph) > max_chars:
                chunks.append(Chunk(source, heading, "\n\n".join(part)))
                part, size = [], 0
            part.append(paragraph)
            size += len(paragraph)
        if part:
            chunks.append(Chunk(source, heading, "\n\n".join(part)))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()
    return chunks


class KnowledgeBase:
    """BM25 index (k1=1.5, b=0.75) over markdown chunks."""

    def __init__(self, chunks: Iterable[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("knowledge base is empty")
        self.k1, self.b = k1, b
        self._docs = [Counter(tokenize(f"{c.heading} {c.text}")) for c in self.chunks]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg = sum(self._lengths) / len(self._lengths)
        frequency: Counter[str] = Counter()
        for doc in self._docs:
            frequency.update(doc.keys())
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in frequency.items()}

    @classmethod
    def from_directory(cls, root: Path, pattern: str = "**/*.md") -> KnowledgeBase:
        files = sorted(p for p in root.glob(pattern) if p.is_file())
        chunks = [
            chunk
            for path in files
            for chunk in chunk_markdown(path.relative_to(root).as_posix(), path.read_text("utf-8"))
        ]
        return cls(chunks)

    def coverage(self, query: str) -> float:
        """Share of the query's distinct terms that appear anywhere in the knowledge base."""

        terms = set(tokenize(query))
        return sum(1 for t in terms if t in self._idf) / len(terms) if terms else 0.0

    def search(self, query: str, k: int = 4) -> list[Hit]:
        terms = tokenize(query)
        hits: list[Hit] = []
        for chunk, doc, length in zip(self.chunks, self._docs, self._lengths, strict=True):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg)
                    score += self._idf[term] * tf * (self.k1 + 1) / norm
            if score > 0:
                hits.append(Hit(chunk, score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[Chunk, ...]
    grounded: bool  # True when the model answered with valid citations
    refused: bool = False


def _context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.source} - {h.chunk.heading})\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    )


def _extractive(hits: Sequence[Hit], note: str) -> str:
    lines = [note]
    for i, hit in enumerate(hits, start=1):
        excerpt = hit.chunk.text.strip().replace("\n", " ")
        lines.append(f"[{i}] {hit.chunk.heading}: {excerpt[:400]}")
    return "\n".join(lines)


class KnowledgeChatbot:
    def __init__(
        self,
        kb: KnowledgeBase,
        generate: Generate | None = None,
        *,
        k: int = 4,
        min_score: float = 1.0,
        min_coverage: float = 0.5,
    ) -> None:
        self.kb, self.generate, self.k = kb, generate, k
        self.min_score, self.min_coverage = min_score, min_coverage

    def ask(self, question: str) -> Answer:
        if not question.strip():
            return Answer("Please ask a question.", (), False, refused=True)
        if _SIGNAL_REQUEST.search(question):
            return Answer(
                "I explain trading; I do not give personal buy or sell calls. Trades are decided "
                "by the rule pipeline (plan, Shariah gate, protections, risk engine), which logs "
                "its reasons in the audit log.",
                (),
                False,
                refused=True,
            )
        hits = [h for h in self.kb.search(question, self.k) if h.score >= self.min_score]
        if not hits or self.kb.coverage(question) < self.min_coverage:
            return Answer("That is not covered in the knowledge base yet.", (), False, refused=True)
        sources = tuple(h.chunk for h in hits)
        if self.generate is None:
            return Answer(_extractive(hits, "Relevant passages:"), sources, False)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{_context(hits)}\n\nQuestion: {question}"},
        ]
        try:
            reply = _THINK.sub("", self.generate(messages)).strip()
        except Exception:  # model offline or broken: fail to quotes, never to invention
            return Answer(
                _extractive(hits, "Model unavailable; relevant passages:"), sources, False
            )
        cited = {int(n) for n in _CITATION.findall(reply)}
        if not reply or not cited or not cited <= set(range(1, len(hits) + 1)):
            return Answer(
                _extractive(hits, "The model's answer lacked valid citations; passages:"),
                sources,
                False,
            )
        return Answer(reply, sources, True)


def ollama_generate(
    model: str = "qwen3:8b", url: str = "http://localhost:11434/api/chat", timeout: float = 120
) -> Generate:
    """Ollama chat call at temperature 0 (https://github.com/ollama/ollama/blob/main/docs/api.md)."""

    def call(messages: Sequence[dict[str, str]]) -> str:
        body = json.dumps(
            {
                "model": model,
                "messages": list(messages),
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode()
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        return str(payload.get("message", {}).get("content", ""))

    return call
