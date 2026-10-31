"""The claim gate, written as a document renderer.

Every published document (README, RESULTS, the briefing, the model cards, the validation reports,
the release notes) is a Jinja template rendered against results/manifest.json. In check mode the gate
renders each template again and compares the whole file with what is on disk, so a number cannot be
edited into a document by hand and cannot appear in one unless a query produced it today. A second
check reads the templates and rejects digits typed outside a template expression, apart from a short
committed list of literals (a version, a year in a citation). No template function reads a completion:
a model never writes a sentence about a number.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

from goodhart_core.formats import format_value
from goodhart_core.manifest import Manifest
from goodhart_core.statements import STATEMENT, statement_markdown
from jinja2 import Environment, FileSystemLoader, StrictUndefined


@dataclass(frozen=True)
class Target:
    template: str
    output: str


@dataclass(frozen=True)
class Drift:
    output: str
    diff: str


class ClaimGateError(AssertionError):
    pass


_EXPRESSION = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|\{#.*?#\}", re.S)
_DIGIT_RUN = re.compile(r"\d[\d,.]*")


def table_html(manifest: Manifest, key: str) -> str:
    """A manifest table as an HTML table, numbers right aligned in tabular figures."""
    entry = manifest.tables.get(key)
    if entry is None:
        raise KeyError(f"manifest has no table {key!r}")
    head = "".join(
        f'<th class="{"num" if f != "text" else "txt"}">{escape(c)}</th>'
        for c, f in zip(entry.columns, entry.formats, strict=True)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{"num" if f != "text" else "txt"}">{escape(format_value(v, f))}</td>'
            for v, f in zip(row, entry.formats, strict=True)
        )
        + "</tr>"
        for row in entry.rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _missing(section: str) -> str:
    raise ClaimGateError(f"a required section was not written: {section}")


def _rows(manifest: Manifest, key: str) -> list[list[Any]]:
    if key not in manifest.tables:
        raise KeyError(f"manifest has no table {key!r}")
    return manifest.tables[key].rows


def environment(template_roots: Sequence[Path], manifest: Manifest, root: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader([str(r) for r in template_roots]),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.globals.update(
        v=manifest.text,
        raw=manifest.raw,
        has=manifest.has,
        table=manifest.table_markdown,
        table_html=lambda key: table_html(manifest, key),
        rows=lambda key: _rows(manifest, key),
        fmt=format_value,
        statement=statement_markdown,
        statement_text=STATEMENT,
        manifest=manifest,
        missing=_missing,
        inline=lambda rel: (root / rel).read_text(encoding="utf-8").rstrip("\n"),
        done=lambda: sum(1 for r in _rows(manifest, "checklist") if r[1] == "done"),
        open_items=lambda: [r for r in _rows(manifest, "checklist") if r[1] != "done"],
    )
    return env


class Renderer:
    def __init__(
        self, root: Path, manifest: Manifest, targets: Sequence[Target], template_roots: Sequence[Path]
    ) -> None:
        if not targets:
            raise ClaimGateError("the claim gate was given no documents to render")
        if not (manifest.values or manifest.tables):
            raise ClaimGateError("the claim gate was given an empty manifest")
        self.root = root
        self.manifest = manifest
        self.targets = list(targets)
        self.env = environment(template_roots, manifest, root)

    def render(self, target: Target) -> str:
        return self.env.get_template(target.template).render()

    def write_all(self) -> list[Path]:
        written: list[Path] = []
        for target in self.targets:
            out = self.root / target.output
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.render(target), encoding="utf-8")
            written.append(out)
        return written

    def check_all(self) -> list[Drift]:
        drifts: list[Drift] = []
        for target in self.targets:
            out = self.root / target.output
            expected = self.render(target)
            actual = out.read_text(encoding="utf-8") if out.exists() else ""
            if expected != actual:
                diff = "".join(
                    difflib.unified_diff(
                        actual.splitlines(keepends=True),
                        expected.splitlines(keepends=True),
                        fromfile=f"{target.output} (on disk)",
                        tofile=f"{target.output} (rendered from manifest)",
                        n=1,
                    )
                )
                drifts.append(Drift(target.output, diff or "file missing"))
        return drifts


def hand_typed_numbers(template_text: str, allowed: Iterable[str]) -> list[str]:
    """Digit runs outside template expressions that are not on the allow list."""
    prose = _EXPRESSION.sub(" ", template_text)
    for literal in sorted(allowed, key=len, reverse=True):
        prose = prose.replace(literal, " ")
    return [m.group(0) for m in _DIGIT_RUN.finditer(prose)]


def load_allowed_literals(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]
