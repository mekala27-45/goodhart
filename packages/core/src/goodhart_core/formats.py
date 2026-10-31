"""Display formats for the manifest, the documents and the site.

A value is stored raw in the manifest with a named format, so a number in a document and the same
number on a page cannot be printed two ways. The codes follow readback's table; goodhart adds the
ones its figures need (points of accuracy, GPU hours, megabytes, seconds).
"""

from __future__ import annotations

from collections.abc import Callable

FormatFn = Callable[[float | int | str | None], str]


def _num(value: float | int | str | None) -> float:
    if value is None or isinstance(value, str):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _is_zero(text: str) -> bool:
    digits = text.replace(",", "").lstrip("+-")
    return float(digits) == 0


def _plain(value: float, decimals: int) -> str:
    text = f"{value:,.{decimals}f}"
    return text[1:] if text.startswith("-") and _is_zero(text) else text


def _signed(value: float, decimals: int) -> str:
    text = _plain(value, decimals)
    return f"+{text}" if value > 0 and not _is_zero(text) else text


def _fixed(decimals: int) -> FormatFn:
    return lambda value: _plain(_num(value), decimals)


def _signed_fixed(decimals: int) -> FormatFn:
    return lambda value: _signed(_num(value), decimals)


def _pct(decimals: int) -> FormatFn:
    return lambda value: f"{_plain(_num(value) * 100.0, decimals)}%"


def _points(decimals: int) -> FormatFn:
    """A difference of two shares, stored as a fraction, printed in signed percentage points."""
    return lambda value: f"{_signed(_num(value) * 100.0, decimals)} pts"


def _plain_points(decimals: int) -> FormatFn:
    return lambda value: f"{_plain(_num(value) * 100.0, decimals)} pts"


def _suffix(decimals: int, unit: str) -> FormatFn:
    return lambda value: f"{_plain(_num(value), decimals)} {unit}"


def fmt_int(value: float | int | str | None) -> str:
    return f"{round(_num(value)):,}"


def fmt_text(value: float | int | str | None) -> str:
    return "not available" if value is None else str(value)


def fmt_usd0(value: float | int | str | None) -> str:
    v = _num(value)
    text = f"${abs(v):,.0f}"
    return f"-{text}" if v < 0 and round(v) != 0 else text


def fmt_usd2(value: float | int | str | None) -> str:
    v = _num(value)
    text = f"${abs(v):,.2f}"
    return f"-{text}" if v < 0 and not _is_zero(text[1:]) else text


def fmt_usd4(value: float | int | str | None) -> str:
    v = _num(value)
    text = f"${abs(v):,.4f}"
    return f"-{text}" if v < 0 and not _is_zero(text[1:]) else text


def fmt_mb1(value: float | int | str | None) -> str:
    return f"{_plain(_num(value) / 1_000_000.0, 1)} MB"


def fmt_gb1(value: float | int | str | None) -> str:
    return f"{_plain(_num(value) / 1_000_000_000.0, 1)} GB"


def fmt_millions1(value: float | int | str | None) -> str:
    return f"{_plain(_num(value) / 1_000_000.0, 1)} million"


def fmt_bool(value: float | int | str | None) -> str:
    if isinstance(value, str):
        return value
    return "yes" if value else "no"


FORMATS: dict[str, FormatFn] = {
    "int": fmt_int,
    "float1": _fixed(1),
    "float2": _fixed(2),
    "float3": _fixed(3),
    "sfloat2": _signed_fixed(2),
    "sfloat3": _signed_fixed(3),
    "pct0": _pct(0),
    "pct1": _pct(1),
    "pct2": _pct(2),
    "pts1": _points(1),
    "apts1": _plain_points(1),
    "usd0": fmt_usd0,
    "usd2": fmt_usd2,
    "usd4": fmt_usd4,
    "hours1": _suffix(1, "h"),
    "hours2": _suffix(2, "h"),
    "min1": _suffix(1, "min"),
    "sec1": _suffix(1, "s"),
    "sec2": _suffix(2, "s"),
    "ms0": lambda value: f"{round(_num(value)):,} ms",
    "mb1": fmt_mb1,
    "gb1": fmt_gb1,
    "millions1": fmt_millions1,
    "bool": fmt_bool,
    "text": fmt_text,
}


def format_value(value: float | int | str | None, fmt: str) -> str:
    if fmt not in FORMATS:
        raise KeyError(f"unknown format {fmt!r}")
    if value is None and fmt != "text":
        return "not available"
    return FORMATS[fmt](value)
