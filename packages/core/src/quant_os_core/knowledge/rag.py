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
import os
import re
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Generate = Callable[[Sequence[dict[str, str]]], str]
Embed = Callable[[Sequence[str]], list[list[float]]]

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

    def __init__(
        self,
        chunks: Iterable[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        embed: Embed | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embed = embed
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
        self._vectors: list[list[float]] = []
        if embed is not None:
            texts = [f"{c.heading}\n{c.text}" for c in self.chunks]
            self._vectors = [_unit(v) for v in embed(texts)]
            if len(self._vectors) != len(self.chunks):
                raise ValueError("embedder returned the wrong number of vectors")

    @classmethod
    def from_directory(
        cls, root: Path, pattern: str = "**/*.md", embed: Embed | None = None
    ) -> KnowledgeBase:
        files = sorted(p for p in root.glob(pattern) if p.is_file())
        chunks = [
            chunk
            for path in files
            for chunk in chunk_markdown(path.relative_to(root).as_posix(), path.read_text("utf-8"))
        ]
        return cls(chunks, embed=embed)

    def coverage(self, query: str) -> float:
        """Share of the query's distinct terms that appear anywhere in the knowledge base."""

        terms = set(tokenize(query))
        return sum(1 for t in terms if t in self._idf) / len(terms) if terms else 0.0

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """BM25, fused with embedding similarity by reciprocal rank when an embedder is set."""

        lexical = self._bm25(query)
        if not self._vectors or self.embed is None:
            return lexical[:k]
        try:
            q = _unit(self.embed([query])[0])
        except Exception:  # embedder offline: keyword search still works
            return lexical[:k]
        semantic = sorted(
            (
                (sum(a * b for a, b in zip(q, v, strict=True)), i)
                for i, v in enumerate(self._vectors)
            ),
            reverse=True,
        )[: max(20, k)]
        fused: dict[int, float] = {}
        index = {id(c): i for i, c in enumerate(self.chunks)}
        for rank, hit in enumerate(lexical[:20]):
            i = index[id(hit.chunk)]
            fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        for rank, (_, i) in enumerate(semantic):
            fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
        bm25 = {index[id(h.chunk)]: h.score for h in lexical}
        order = sorted(fused, key=lambda i: fused[i], reverse=True)[:k]
        # Keep the BM25 score for the confidence guard; semantic-only hits get the floor score.
        return [Hit(self.chunks[i], max(bm25.get(i, 0.0), 1.0)) for i in order]

    def _bm25(self, query: str) -> list[Hit]:
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
        return hits


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


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def ollama_embed(
    model: str = "nomic-embed-text", url: str = "http://localhost:11434/api/embed"
) -> Embed:
    """Local embeddings through Ollama's ``/api/embed`` (returns ``embeddings``)."""

    def call(texts: Sequence[str]) -> list[list[float]]:
        body = json.dumps({"model": model, "input": list(texts)}).encode()
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read())
        return [list(map(float, v)) for v in payload["embeddings"]]

    return call


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


# ---- multilingual front end (English, Urdu, Arabic) -------------------------------------

LANGUAGES = ("en", "ur", "ar")
_ARABIC_SCRIPT = re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufeff]")
# Letters used in Urdu but not Arabic (tteh, ddal, rreh, noon ghunna, do-chashmi heh,
# heh goal, yeh barree, keheh, gaf, Farsi yeh, peh, tcheh, jeh) and the reverse (teh
# marbuta, Arabic yeh, alef maksura, Arabic kaf, Arabic heh, harakat).
_URDU_ONLY = re.compile(
    r"[\u0679\u0688\u0691\u06ba\u06be\u06c1\u06c3\u06d2\u06d3\u06a9\u06af\u06cc\u067e\u0686\u0698]"
)
_ARABIC_ONLY = re.compile(r"[\u0629\u064a\u0649\u0643\u0647\u064b-\u0652]")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
# "Should I buy / sell" in Urdu and Arabic: refused like the English pattern.
_SIGNAL_REQUEST_UR_AR = re.compile(
    "|".join(
        [
            "خریدوں",
            "بیچوں",
            "خرید لوں",
            "بیچ دوں",
            "هل أشتري",
            "هل أبيع",
            "أشتري الآن",
            "أبيع الآن",
            "هل يجب أن أشتري",
            "هل يجب أن أبيع",
        ]
    )
)

# Small bilingual glossary so retrieval still works without a model. Keys are matched as
# substrings, longest first (Arabic attaches the article and some prepositions to the word).
GLOSSARY: dict[str, str] = {
    # Urdu
    "حلال": "halal",
    "حرام": "haram",
    "سود": "riba interest",
    "زکوٰۃ": "zakat",
    "زکوۃ": "zakat",
    "زکات": "zakat",
    "زکاۃ": "zakat",
    "شیئر": "stock share",
    "حصص": "stock shares",
    "اسٹاک": "stock",
    "رسک": "risk",
    "خطرہ": "risk",
    "خطرے": "risk",
    "ڈرا ڈاؤن": "drawdown",
    "ڈراڈاؤن": "drawdown",
    "منافع": "profit",
    "نقصان": "loss",
    "سونا": "gold",
    "سونے": "gold",
    "کرپٹو": "crypto",
    "بٹ کوائن": "bitcoin crypto",
    "صکوک": "sukuk",
    "شریعت": "shariah",
    "شرعی": "shariah",
    "فیوچرز": "futures",
    "آپشنز": "options",
    "لیوریج": "leverage",
    "مارجن": "margin",
    "شارٹ": "short selling",
    "ٹریڈنگ": "trading",
    "پوزیشن": "position size",
    "اسٹاپ": "stop loss",
    "تطہیر": "purification",
    "فتویٰ": "fatwa",
    "فتوی": "fatwa",
    "جوا": "gambling",
    "غرر": "gharar uncertainty",
    "فنڈ": "fund etf",
    "قرض": "debt",
    "مہنگائی": "inflation",
    "تنوع": "diversification diversified",
    "قیمت": "price",
    "مارکیٹ": "market",
    "بیک ٹیسٹ": "backtest",
    "حکمت عملی": "strategy",
    "نفسیات": "psychology",
    "لالچ": "greed",
    "ملین": "million",
    # Arabic
    "ربا": "riba interest",
    "زكاة": "zakat",
    "فائدة": "interest riba",
    "فوائد": "interest riba",
    "أسهم": "stock shares",
    "اسهم": "stock shares",
    "سهم": "stock share",
    "مخاطر": "risk",
    "تراجع": "drawdown",
    "انخفاض": "drawdown",
    "ربح": "profit",
    "أرباح": "profit",
    "خسارة": "loss",
    "خسائر": "loss",
    "ذهب": "gold",
    "عملات مشفرة": "crypto",
    "العملات الرقمية": "crypto",
    "عملة رقمية": "crypto",
    "كريبتو": "crypto",
    "بيتكوين": "bitcoin crypto",
    "صكوك": "sukuk",
    "شريعة": "shariah",
    "شرعي": "shariah",
    "العقود الآجلة": "futures",
    "عقود مستقبلية": "futures",
    "خيارات": "options",
    "رافعة": "leverage",
    "هامش": "margin",
    "البيع على المكشوف": "short selling",
    "تداول": "trading",
    "حجم المركز": "position size",
    "وقف الخسارة": "stop loss",
    "تطهير": "purification",
    "فتوى": "fatwa",
    "قمار": "gambling",
    "ميسر": "gambling",
    "صندوق": "fund etf",
    "صناديق": "fund etf",
    "ديون": "debt",
    "تضخم": "inflation",
    "تنويع": "diversification diversified",
    "سعر": "price",
    "سوق": "market",
    "اختبار رجعي": "backtest",
    "استراتيجية": "strategy",
    "خوف": "fear",
    "جشع": "greed",
    "مليون": "million",
}

_NO_MODEL_NOTE = {
    "ur": "مکمل ترجمے کے لیے مقامی ماڈل (Ollama) درکار ہے\u06d4 "
    "نیچے انگریزی اقتباسات دیے گئے ہیں\u06d4",
    "ar": "الترجمة الكاملة تتطلب نموذجًا محليًا (Ollama). فيما يلي المقتطفات باللغة الإنجليزية.",
}
_NOT_COVERED = {
    "en": "That is not covered in the knowledge base yet.",
    "ur": "یہ موضوع ابھی نالج بیس میں شامل نہیں ہے\u06d4",
    "ar": "هذا الموضوع غير مشمول في قاعدة المعرفة بعد.",
}
_NO_SIGNALS = {
    "ur": "میں ٹریڈنگ سمجھاتا ہوں؛ ذاتی خرید و فروخت کا مشورہ نہیں دیتا\u06d4 فیصلے قواعد پر مبنی "
    "پائپ لائن (شرعی گیٹ، رسک انجن) کرتی ہے\u06d4",
    "ar": "أنا أشرح التداول ولا أقدم توصيات شخصية بالشراء أو البيع. القرارات يتخذها خط "
    "القواعد (البوابة الشرعية ومحرك المخاطر).",
}
_LANG_NAMES = {"ur": "Urdu", "ar": "Arabic", "en": "English"}

_KB_CACHE: dict[Path, tuple[tuple[tuple[str, float], ...], KnowledgeBase]] = {}


def load_knowledge_base(docs: Path) -> KnowledgeBase:
    """KnowledgeBase for ``docs``, cached per path and rebuilt when a markdown file changes."""

    root = docs.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"knowledge folder not found: {docs}")
    signature = tuple(
        (p.relative_to(root).as_posix(), p.stat().st_mtime)
        for p in sorted(root.glob("**/*.md"))
        if p.is_file()
    )
    cached = _KB_CACHE.get(root)
    if cached is not None and cached[0] == signature:
        return cached[1]
    kb = KnowledgeBase.from_directory(root)
    _KB_CACHE[root] = (signature, kb)
    return kb


def detect_language(text: str) -> str:
    """'ur', 'ar' or 'en' from the script; Urdu and Arabic are told apart by their letters."""

    if not _ARABIC_SCRIPT.search(text):
        return "en"
    urdu = len(_URDU_ONLY.findall(text))
    arabic = len(_ARABIC_ONLY.findall(text))
    return "ur" if urdu > 0 and urdu >= arabic else "ar"


def glossary_translate(question: str) -> str:
    """English retrieval query from glossary terms (longest first) plus any Latin words."""

    terms: list[str] = []
    remaining = question
    for key in sorted(GLOSSARY, key=len, reverse=True):
        if key in remaining:
            terms.append(GLOSSARY[key])
            remaining = remaining.replace(key, " ")
    terms.extend(_LATIN_WORD.findall(question))
    return " ".join(dict.fromkeys(" ".join(terms).split()))


def _env_llm() -> Generate | None:
    model = os.environ.get("QUANT_OS_OLLAMA_MODEL", "").strip()
    return ollama_generate(model) if model else None


def _llm_text(llm: Generate, system: str, user: str) -> str:
    reply = llm([{"role": "system", "content": system}, {"role": "user", "content": user}])
    return _THINK.sub("", reply).strip()


def _source_list(answer: Answer) -> list[dict[str, str]]:
    return [{"source": c.source, "heading": c.heading} for c in answer.sources]


def _with_note(target: str, answer: Answer) -> dict[str, Any]:
    text = f"{_NO_MODEL_NOTE[target]}\n{answer.text}"
    return {"answer": text, "sources": _source_list(answer), "lang": target}


def answer_multilingual(
    question: str,
    lang: str = "en",
    *,
    docs: Path = Path("docs/knowledge"),
    llm: Generate | None = None,
    env_model: bool = True,
) -> dict[str, Any]:
    """Answer in English, Urdu or Arabic from the knowledge base.

    Retrieval is always in English. For Urdu or Arabic, a local model (``llm``, else Ollama
    when ``QUANT_OS_OLLAMA_MODEL`` is set) translates the question in and the cited answer
    back; without a model the glossary builds the query and the English passages are
    returned with a note in the user's language. The answer language is ``lang`` if it is
    'ur' or 'ar', otherwise the question's script. ``env_model=False`` never reads the
    environment (quote-only mode). Returns ``{"answer", "sources", "lang"}``.
    """

    if lang not in LANGUAGES:
        raise ValueError(f"lang must be one of {LANGUAGES}")
    detected = detect_language(question)
    target = lang if lang != "en" else detected
    model = llm if llm is not None else (_env_llm() if env_model else None)
    kb = load_knowledge_base(docs)

    if target == "en":
        answer = KnowledgeChatbot(kb, model).ask(question)
        return {"answer": answer.text, "sources": _source_list(answer), "lang": "en"}
    if _SIGNAL_REQUEST_UR_AR.search(question) or _SIGNAL_REQUEST.search(question):
        return {"answer": _NO_SIGNALS[target], "sources": [], "lang": target}

    english = question if detected == "en" else ""
    if not english and model is not None:
        try:
            english = _llm_text(
                model,
                f"Translate the user's {_LANG_NAMES[detected]} question into English. "
                "Reply with the English question only.",
                question,
            )
        except Exception:  # model offline: the glossary still gives a usable query
            english = ""
    if not english:
        english = glossary_translate(question)
    if not english.strip():
        return {"answer": _NOT_COVERED[target], "sources": [], "lang": target}

    answer = KnowledgeChatbot(kb, model).ask(english)
    if answer.refused:
        return {"answer": _NOT_COVERED[target], "sources": [], "lang": target}
    if model is None or not answer.grounded:
        return _with_note(target, answer)
    try:
        translated = _llm_text(
            model,
            f"Translate the text into {_LANG_NAMES[target]}. Keep every citation such as [1] "
            "exactly as written. Do not add advice, numbers or claims.",
            answer.text,
        )
    except Exception:  # translation failed: English answer, flagged in the user's language
        translated = ""
    if not translated or set(_CITATION.findall(translated)) != set(_CITATION.findall(answer.text)):
        return _with_note(target, answer)
    return {"answer": translated, "sources": _source_list(answer), "lang": target}
