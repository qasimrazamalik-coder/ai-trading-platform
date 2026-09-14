#!/usr/bin/env python3
"""Dependency-light static release checks for Quant OS."""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (ROOT / "packages/core/src", ROOT / "apps/api/src", ROOT / "scripts")
PRIVATE_KEY_MARKERS = (
    "BEGIN RSA PRIVATE KEY",
    "BEGIN OPENSSH PRIVATE KEY",
    "BEGIN EC PRIVATE KEY",
    "BEGIN PRIVATE KEY",
)
DANGEROUS_BUILTINS = {"eval", "exec"}


def python_files() -> list[Path]:
    files: list[Path] = []
    for root in PYTHON_ROOTS:
        files.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(files)


def check_python_sources(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax: {path.relative_to(ROOT)}: {exc}")
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if len(line) > 100:
                errors.append(f"line-length: {path.relative_to(ROOT)}:{number}:{len(line)}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in DANGEROUS_BUILTINS:
                    errors.append(
                        f"dangerous-call: {path.relative_to(ROOT)}:{node.lineno}:{node.func.id}"
                    )
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "system"
                ):
                    errors.append(f"os.system: {path.relative_to(ROOT)}:{node.lineno}")
    return errors


def check_repository_secrets() -> list[str]:
    errors: list[str] = []
    candidates = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.resolve() != Path(__file__).resolve()
        and ".git" not in path.parts
        and "node_modules" not in path.parts
        and path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".zip", ".pyc"}
    ]
    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(marker in text for marker in PRIVATE_KEY_MARKERS):
            errors.append(f"private-key-material: {path.relative_to(ROOT)}")
    if (ROOT / ".env").exists():
        errors.append("repository contains .env")
    return errors


def check_runtime_dependencies() -> list[str]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    errors: list[str] = []
    for raw in data["project"]["dependencies"]:
        requirement = Requirement(raw)
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            errors.append(f"missing-runtime-dependency: {requirement.name}")
            continue
        if requirement.specifier and installed not in requirement.specifier:
            errors.append(
                f"runtime-version: {requirement.name}=={installed} not in {requirement.specifier}"
            )
    return errors


def check_safety_defaults() -> list[str]:
    sys.path.insert(0, str(ROOT / "packages/core/src"))
    from quant_os_core.execution import default_adapter_catalog
    from quant_os_core.settings import Settings

    errors: list[str] = []
    settings = Settings(_env_file=None, live_trading_enabled=False)
    if settings.live_trading_enabled:
        errors.append("live trading default is enabled")
    if any(adapter.live_trading for adapter in default_adapter_catalog()):
        errors.append("one or more shipped adapters advertise live trading")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    if not re.search(r"^QUANT_OS_LIVE_TRADING_ENABLED=false$", env_example, re.MULTILINE):
        errors.append(".env.example does not explicitly disable live trading")
    return errors


def main() -> int:
    files = python_files()
    errors = [
        *check_python_sources(files),
        *check_repository_secrets(),
        *check_runtime_dependencies(),
        *check_safety_defaults(),
    ]
    if errors:
        print("RELEASE STATIC CHECK: FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"RELEASE STATIC CHECK: PASS ({len(files)} Python files checked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
