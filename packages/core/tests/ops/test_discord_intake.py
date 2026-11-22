import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from quant_os_core.agents.signal_intake import ExternalSignal, IntakeResult, SignalIntake
from quant_os_core.ops.discord_intake import (
    DISCORD_API,
    TOKEN_ENV,
    DiscordChannelReader,
    DiscordIntakeError,
)

TOKEN = "test-token-not-real"
CHANNEL = "123456789"


@dataclass
class FakeHttp:
    responses: list[tuple[int, bytes]]
    calls: list[tuple[str, str, dict[str, str]]] = field(default_factory=list)
    error: Exception | None = None

    def __call__(
        self, method: str, url: str, headers: Mapping[str, str], body: bytes | None
    ) -> tuple[int, bytes]:
        self.calls.append((method, url, dict(headers)))
        if self.error is not None:
            raise self.error
        return self.responses.pop(0)


@dataclass
class FakeIntake:
    seen: list[ExternalSignal] = field(default_factory=list)

    def process(self, signal: ExternalSignal) -> IntakeResult:
        self.seen.append(signal)
        return IntakeResult(signal.provider_id, signal.message_id, True, "ok")


@dataclass
class Clock:
    now: datetime = datetime(2026, 11, 22, 12, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _msg(message_id: str, content: str, author: str = "7") -> dict[str, object]:
    return {"id": message_id, "content": content, "author": {"id": author, "username": "x"}}


def _ok(messages: object) -> tuple[int, bytes]:
    return 200, json.dumps(messages).encode()


def _reader(http: FakeHttp, intake: FakeIntake, clock: Clock, **kw: object) -> DiscordChannelReader:
    return DiscordChannelReader(
        token=TOKEN,
        channel_id=CHANNEL,
        intake=intake,  # type: ignore[arg-type]
        http=http,
        clock=clock,
        **kw,  # type: ignore[arg-type]
    )


def test_poll_reads_in_order_dedupes_and_advances_cursor() -> None:
    http = FakeHttp(
        [
            _ok([_msg("12", "BUY AAPL 100 SL 95 TP 110"), _msg("11", "gm"), _msg("x", "bad")]),
            _ok([_msg("12", "dup"), _msg("13", "BUY MSFT 300 SL 290 TP 330")]),
        ]
    )
    intake, clock = FakeIntake(), Clock()
    reader = _reader(http, intake, clock)
    first = reader.poll()
    assert first.fetched == 2
    assert [s.message_id for s in intake.seen] == ["11", "12"]
    assert intake.seen[0].provider_id == f"discord:{CHANNEL}"
    method, url, headers = http.calls[0]
    assert method == "GET"
    assert url == f"{DISCORD_API}/channels/{CHANNEL}/messages?limit=20"
    assert headers["Authorization"] == f"Bot {TOKEN}"
    clock.advance(31)
    second = reader.poll()
    assert http.calls[1][1].endswith("&after=12")
    assert second.skipped_duplicates == 1
    assert [s.message_id for s in intake.seen] == ["11", "12", "13"]


def test_rate_limit_and_429_backoff() -> None:
    http = FakeHttp([_ok([]), (429, b'{"retry_after": 5.5}'), (429, b"oops"), _ok([])])
    intake, clock = FakeIntake(), Clock()
    reader = _reader(http, intake, clock)
    assert not reader.poll().throttled
    assert reader.poll().throttled  # inside min_interval: no HTTP call
    assert len(http.calls) == 1
    clock.advance(30)
    limited = reader.poll()
    assert limited.throttled and limited.error == "rate limited by Discord"
    clock.advance(3)
    assert reader.poll().throttled
    clock.advance(3)
    limited = reader.poll()  # unparsable 429 body: default 60 s back-off
    assert limited.throttled
    clock.advance(59)
    assert reader.poll().throttled
    clock.advance(2)
    assert not reader.poll().throttled


@pytest.mark.parametrize(
    ("response", "error"),
    [
        ((500, b"{}"), "Discord HTTP 500"),
        ((200, b"not json"), "Discord returned invalid JSON"),
        ((200, b'{"message": "x"}'), "unexpected Discord payload"),
    ],
)
def test_bad_responses_yield_no_signals(response: tuple[int, bytes], error: str) -> None:
    intake = FakeIntake()
    result = _reader(FakeHttp([response]), intake, Clock()).poll()
    assert result.error == error and result.processed == ()
    assert intake.seen == []


def test_transport_error_hides_details() -> None:
    http = FakeHttp([], error=OSError(f"failed with {TOKEN}"))
    result = _reader(http, FakeIntake(), Clock()).poll()
    assert result.error == "transport error: OSError"
    assert TOKEN not in (result.error or "")


def test_author_filter_empty_content_and_cap() -> None:
    messages = [_msg(str(i), f"BUY AAPL 100 SL 95 TP 110 #{i}") for i in range(10, 20)]
    messages.append(_msg("30", "BUY AAPL 100 SL 95 TP 110", author="666"))
    messages.append({"id": "31", "content": "", "author": "not-a-dict"})
    intake = FakeIntake()
    reader = _reader(
        FakeHttp([_ok(messages)]),
        intake,
        Clock(),
        max_messages_per_poll=12,
        allowed_author_ids=frozenset({"7"}),
    )
    result = reader.poll()
    assert result.fetched == 12
    assert len(intake.seen) == 10
    assert all(s.message_id != "30" for s in intake.seen)


def test_configuration_errors() -> None:
    intake = FakeIntake()
    with pytest.raises(DiscordIntakeError, match=TOKEN_ENV):
        DiscordChannelReader.from_env(CHANNEL, intake, env={})  # type: ignore[arg-type]
    with pytest.raises(DiscordIntakeError, match="snowflake"):
        DiscordChannelReader(TOKEN, "../guilds", intake)  # type: ignore[arg-type]
    with pytest.raises(DiscordIntakeError, match="max_messages_per_poll"):
        DiscordChannelReader(TOKEN, CHANNEL, intake, max_messages_per_poll=0)  # type: ignore[arg-type]
    with pytest.raises(DiscordIntakeError, match="min_interval"):
        DiscordChannelReader(
            TOKEN,
            CHANNEL,
            intake,
            min_interval=timedelta(0),  # type: ignore[arg-type]
        )
    reader = DiscordChannelReader.from_env(
        CHANNEL,
        intake,
        env={TOKEN_ENV: TOKEN},  # type: ignore[arg-type]
    )
    assert reader.provider_id == f"discord:{CHANNEL}"


def test_seen_memory_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import quant_os_core.ops.discord_intake as module

    monkeypatch.setattr(module, "_SEEN_MEMORY", 2)
    reader = _reader(
        FakeHttp([_ok([_msg("1", "a"), _msg("2", "b"), _msg("3", "c")])]), FakeIntake(), Clock()
    )
    reader.poll()
    assert len(reader._seen) == 2


def test_real_intake_never_trades_outside_pipeline() -> None:
    class NoPipeline:
        def decide(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("must not be called for a haram signal")

    intake = SignalIntake(NoPipeline(), lambda plan: None)  # type: ignore[arg-type]
    reader = _reader(
        FakeHttp([_ok([_msg("5", "BUY BTC PERP 65000 SL 62000 TP 72000")])]),
        intake,  # type: ignore[arg-type]
        Clock(),
    )
    (result,) = reader.poll().processed
    assert not result.accepted and "perpetual" in result.reason
