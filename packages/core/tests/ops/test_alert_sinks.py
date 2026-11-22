"""Email and HTTP alert transports, exercised with fakes (no network, no SMTP server)."""

import urllib.request
from email.message import EmailMessage
from typing import Any, ClassVar

import pytest

from quant_os_core.ops import alerts
from quant_os_core.ops.alerts import EmailSink, MultiSink


class _FakeSmtp:
    instances: ClassVar[list["_FakeSmtp"]] = []

    def __init__(self, host: str, port: int) -> None:
        self.host, self.port = host, port
        self.logged_in: tuple[str, str] | None = None
        self.sent: list[EmailMessage] = []
        self.closed = False
        _FakeSmtp.instances.append(self)

    def __enter__(self) -> "_FakeSmtp":
        return self

    def __exit__(self, *exc: object) -> None:
        self.closed = True

    def login(self, username: str, password: str) -> None:
        self.logged_in = (username, password)

    def send_message(self, mail: EmailMessage) -> None:
        self.sent.append(mail)


def test_email_sink_logs_in_and_sends_one_message() -> None:
    _FakeSmtp.instances.clear()
    sink = EmailSink(
        host="smtp.example.invalid",
        port=465,
        username="bot@example.invalid",
        password="app-password",
        to="owner@example.invalid",
        smtp=_FakeSmtp,  # type: ignore[arg-type]
    )
    sink.send("critical", "kill switch", "daily loss limit hit")

    (client,) = _FakeSmtp.instances
    assert (client.host, client.port) == ("smtp.example.invalid", 465)
    assert client.logged_in == ("bot@example.invalid", "app-password")
    assert client.closed
    (mail,) = client.sent
    assert mail["Subject"] == "[Quant OS critical] kill switch"
    assert mail["From"] == "bot@example.invalid"
    assert mail["To"] == "owner@example.invalid"
    assert mail.get_content().strip() == "daily loss limit hit"


def test_urllib_post_sends_a_post_request(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *exc: object) -> None:
            seen["closed"] = True

        def read(self) -> bytes:
            seen["read"] = True
            return b"ok"

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        seen.update(
            url=request.full_url,
            method=request.get_method(),
            body=request.data,
            title=request.get_header("Title"),
            timeout=timeout,
        )
        return _Response()

    monkeypatch.setattr(alerts.urllib.request, "urlopen", fake_urlopen)
    alerts.urllib_post("https://ntfy.example.invalid/topic", b"hello", {"Title": "t"})
    assert seen == {
        "url": "https://ntfy.example.invalid/topic",
        "method": "POST",
        "body": b"hello",
        "title": "t",
        "timeout": 10,
        "read": True,
        "closed": True,
    }


def test_multisink_drops_alerts_below_min_level() -> None:
    received: list[str] = []

    class _Sink:
        def send(self, level: str, title: str, message: str) -> None:
            received.append(level)

    sink = MultiSink([_Sink()], min_level="warning")
    sink.send("info", "noise", "ignored")
    sink.send("warning", "heads up", "kept")
    sink.send("critical", "stop", "kept")
    assert received == ["warning", "critical"]
