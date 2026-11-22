"""Read trade signals from one Discord channel and hand them to the paper signal intake.

Discord, not Telegram: the owner cannot use Telegram. The bot token comes from the
``QUANT_OS_DISCORD_BOT_TOKEN`` environment variable (never a file) and is sent only in the
``Authorization`` header to the Discord REST API. The reader:

* polls ``GET /channels/{id}/messages?after=<last id>`` through an injected HTTP callable;
* dedupes by message id (Discord snowflakes) and keeps the newest id as a cursor;
* rate-limits itself (minimum poll interval, capped messages per poll) and honours HTTP 429
  ``retry_after`` with a back-off;
* never places an order: every message goes to :class:`SignalIntake`, which runs the full
  entry pipeline in paper mode.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from quant_os_core.agents.marketplace import ProviderKind
from quant_os_core.agents.signal_intake import ExternalSignal, IntakeResult, SignalIntake
from quant_os_core.ops.alpaca_paper import Http, urllib_http

DISCORD_API = "https://discord.com/api/v10"
TOKEN_ENV = "QUANT_OS_DISCORD_BOT_TOKEN"
MAX_PAGE = 100
_SEEN_MEMORY = 5_000


def _utc_now() -> datetime:
    return datetime.now(UTC)


class DiscordIntakeError(RuntimeError):
    """Configuration or transport failure; the message never contains the token."""


@dataclass(frozen=True, slots=True)
class PollResult:
    fetched: int
    processed: tuple[IntakeResult, ...]
    skipped_duplicates: int
    throttled: bool = False
    error: str | None = None


@dataclass(slots=True)
class DiscordChannelReader:
    token: str
    channel_id: str
    intake: SignalIntake
    http: Http = urllib_http
    clock: Callable[[], datetime] = _utc_now
    min_interval: timedelta = timedelta(seconds=30)
    max_messages_per_poll: int = 20
    allowed_author_ids: frozenset[str] | None = None  # None: anyone in the channel
    _after: str | None = None
    _seen: set[str] = field(default_factory=set)
    _seen_order: deque[str] = field(default_factory=deque)
    _next_allowed: datetime | None = None

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise DiscordIntakeError(f"Discord bot token missing; set {TOKEN_ENV}")
        if not self.channel_id.isdigit():
            raise DiscordIntakeError("channel_id must be a numeric Discord snowflake")
        if not 1 <= self.max_messages_per_poll <= MAX_PAGE:
            raise DiscordIntakeError(f"max_messages_per_poll must be in [1, {MAX_PAGE}]")
        if self.min_interval < timedelta(seconds=1):
            raise DiscordIntakeError("min_interval must be at least one second")

    @classmethod
    def from_env(
        cls,
        channel_id: str,
        intake: SignalIntake,
        *,
        env: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> DiscordChannelReader:
        source = os.environ if env is None else env
        return cls(token=source.get(TOKEN_ENV, ""), channel_id=channel_id, intake=intake, **kwargs)

    @property
    def provider_id(self) -> str:
        return f"discord:{self.channel_id}"

    def _remember(self, message_id: str) -> None:
        self._seen.add(message_id)
        self._seen_order.append(message_id)
        while len(self._seen_order) > _SEEN_MEMORY:
            self._seen.discard(self._seen_order.popleft())

    def _url(self) -> str:
        url = (
            f"{DISCORD_API}/channels/{self.channel_id}/messages?limit={self.max_messages_per_poll}"
        )
        return f"{url}&after={self._after}" if self._after else url

    def poll(self) -> PollResult:
        now = self.clock()
        if self._next_allowed is not None and now < self._next_allowed:
            return PollResult(0, (), 0, throttled=True)
        self._next_allowed = now + self.min_interval
        headers = {
            "Authorization": f"Bot {self.token}",
            "User-Agent": "DiscordBot (quant-os, 1.0)",
            "Accept": "application/json",
        }
        try:
            status, body = self.http("GET", self._url(), headers, None)
        except Exception as exc:  # network trouble: no signals, try again later
            return PollResult(0, (), 0, error=f"transport error: {type(exc).__name__}")
        if status == 429:
            retry_after = 60.0
            with contextlib.suppress(ValueError, TypeError, AttributeError):
                retry_after = float(json.loads(body or b"{}").get("retry_after", 60.0))
            self._next_allowed = now + timedelta(seconds=max(retry_after, 1.0))
            return PollResult(0, (), 0, throttled=True, error="rate limited by Discord")
        if status != 200:
            return PollResult(0, (), 0, error=f"Discord HTTP {status}")
        try:
            messages = json.loads(body)
        except ValueError:
            return PollResult(0, (), 0, error="Discord returned invalid JSON")
        if not isinstance(messages, list):
            return PollResult(0, (), 0, error="unexpected Discord payload")

        valid = [m for m in messages if isinstance(m, dict) and str(m.get("id", "")).isdigit()]
        valid.sort(key=lambda m: int(m["id"]))
        processed: list[IntakeResult] = []
        duplicates = 0
        for message in valid[: self.max_messages_per_poll]:
            message_id = str(message["id"])
            if self._after is None or int(message_id) > int(self._after):
                self._after = message_id
            if message_id in self._seen:
                duplicates += 1
                continue
            self._remember(message_id)
            raw_author = message.get("author")
            author_id = str(raw_author.get("id")) if isinstance(raw_author, dict) else ""
            if self.allowed_author_ids is not None and author_id not in self.allowed_author_ids:
                continue
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            signal = ExternalSignal(
                provider_id=self.provider_id,
                message_id=message_id,
                text=content,
                received_at=now,
                provider_kind=ProviderKind.DISCORD_CHANNEL,
            )
            processed.append(self.intake.process(signal))
        return PollResult(len(valid), tuple(processed), duplicates)
