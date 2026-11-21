"""Alert sinks: phone push (ntfy), Discord/Slack-style webhooks, email, and the state log.

Telegram is deliberately not the default: it is not usable for the owner. ntfy
(https://ntfy.sh, free, open source, Android/iOS apps) needs no account: pick a long, private
topic name and subscribe to it in the app. A failing sink never stops the trading loop.
"""

from __future__ import annotations

import json
import smtplib
import sys
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

from quant_os_core.ops.state import StateStore

Post = Callable[[str, bytes, Mapping[str, str]], None]
LEVELS = ("info", "warning", "critical")


def urllib_post(url: str, body: bytes, headers: Mapping[str, str]) -> None:
    request = urllib.request.Request(url, data=body, headers=dict(headers), method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        response.read()


class AlertSink(Protocol):
    def send(self, level: str, title: str, message: str) -> None: ...


@dataclass(frozen=True, slots=True)
class NtfySink:
    topic: str
    server: str = "https://ntfy.sh"
    post: Post = urllib_post

    def __post_init__(self) -> None:
        if len(self.topic) < 16:
            raise ValueError("use a long, unguessable ntfy topic (16+ characters)")

    def send(self, level: str, title: str, message: str) -> None:
        priority = {"info": "3", "warning": "4", "critical": "5"}.get(level, "3")
        headers = {"Title": title[:120], "Priority": priority, "Tags": level}
        self.post(f"{self.server.rstrip('/')}/{self.topic}", message.encode(), headers)


@dataclass(frozen=True, slots=True)
class WebhookSink:
    """Discord reads ``content``, Slack reads ``text``; both are sent."""

    url: str
    post: Post = urllib_post

    def send(self, level: str, title: str, message: str) -> None:
        text = f"[{level.upper()}] {title}\n{message}"[:1900]
        body = json.dumps({"content": text, "text": text}).encode()
        self.post(self.url, body, {"Content-Type": "application/json"})


@dataclass(frozen=True, slots=True)
class EmailSink:
    """SMTP over TLS, e.g. Gmail with an app password (smtp.gmail.com:465)."""

    host: str
    port: int
    username: str
    password: str
    to: str
    smtp: Callable[[str, int], smtplib.SMTP_SSL] = smtplib.SMTP_SSL

    def send(self, level: str, title: str, message: str) -> None:
        mail = EmailMessage()
        mail["Subject"] = f"[Quant OS {level}] {title}"
        mail["From"] = self.username
        mail["To"] = self.to
        mail.set_content(message)
        with self.smtp(self.host, self.port) as client:
            client.login(self.username, self.password)
            client.send_message(mail)


@dataclass(frozen=True, slots=True)
class StoreSink:
    store: StateStore

    def send(self, level: str, title: str, message: str) -> None:
        self.store.log(level, "alert", f"{title}: {message}")


@dataclass(frozen=True, slots=True)
class MultiSink:
    sinks: Sequence[AlertSink]
    min_level: str = "info"

    def send(self, level: str, title: str, message: str) -> None:
        if LEVELS.index(level) < LEVELS.index(self.min_level):
            return
        for sink in self.sinks:
            try:
                sink.send(level, title, message)
            except Exception as exc:  # alerts must never break trading
                print(f"quant-os alert sink {type(sink).__name__} failed: {exc!r}", file=sys.stderr)
