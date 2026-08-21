from __future__ import annotations

from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
import re

# Fallback for models that write e.g. "boxed: 42" rather than LaTeX \boxed{42}.
PLAIN_BOXED_RE = re.compile(
    r"\bboxed\s*[:=]?\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
    re.I,
)

FINAL_PATTERNS = [
    re.compile(
        r"(?:final\s+answer|answer\s+is|therefore|thus)\s*[:=]?\s*\$?\s*"
        r"([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)",
        re.I,
    ),
    re.compile(r"####\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?)"),
]

NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")

def _clean_token(token: str) -> str:
    return token.strip().replace("$", "").replace(",", "").rstrip(".,;:!?")


def _to_number(token):
    token = _clean_token(token)
    if not token:
        return None
    try:
        if "/" in token and re.fullmatch(r"[+-]?\d+(?:\.\d+)?/\d+(?:\.\d+)?", token):
            a, b = token.split("/", 1)
            return float(Fraction(Decimal(a)) / Fraction(Decimal(b)))
        return float(Decimal(token))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def _balanced_brace_content(text: str, open_brace: int):
    """Return the content and end index for a balanced {...} block."""
    if open_brace >= len(text) or text[open_brace] != "{":
        return None

    depth = 0
    for i in range(open_brace, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : i], i + 1
    return None


def _boxed_contents(text: str):
    """Yield balanced contents from every LaTeX \boxed{...} occurrence."""
    for match in re.finditer(r"\\boxed\s*", text):
        i = match.end()
        if i < len(text) and text[i] == "{":
            parsed = _balanced_brace_content(text, i)
            if parsed is not None:
                content, _ = parsed
                yield content


def _number_from_boxed_content(content: str):
    """Parse one numeric answer from a LaTeX boxed expression.

    The scorer is intentionally conservative: it accepts common GSM8K-style
    formatting (currency, commas, units, percentages, simple fractions), but
    refuses boxed expressions containing multiple distinct numeric values.
    """
    s = content.strip()

    # Formatting that should not change the numeric value.
    s = s.replace(r"\$", "$")
    s = s.replace(r"\%", "%")
    s = re.sub(r"\\(?:,|!|;|:|quad|qquad)", "", s)
    s = re.sub(r"\\(?:text|textrm|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r" \1 ", s)

    # Accept one simple LaTeX fraction even when it is followed by a unit.
    # Reject the box if another numeric value is present as well.
    frac_match = re.search(
        r"\\(?:d|t)?frac\s*\{\s*([+-]?(?:\d+(?:\.\d+)?))\s*\}"
        r"\s*\{\s*([+-]?(?:\d+(?:\.\d+)?))\s*\}",
        s,
    )
    if frac_match:
        remainder = s[: frac_match.start()] + s[frac_match.end() :]
        if NUMBER_RE.search(remainder):
            return None
        a, b = frac_match.groups()
        try:
            return float(Fraction(Decimal(a)) / Fraction(Decimal(b)))
        except (InvalidOperation, ValueError, ZeroDivisionError):
            return None

    # A single numeric candidate is safe even when surrounded by units, e.g.
    # "16 hours" or "$15,400". Multiple numbers are treated as ambiguous.
    candidates = NUMBER_RE.findall(s)
    if len(candidates) != 1:
        return None
    return _to_number(candidates[0])


def extract_numeric_answer(text):
    # Prefer the final LaTeX boxed answer. Balanced-brace parsing is necessary
    # for outputs such as \boxed{16 \text{ hours}}.
    boxed = list(_boxed_contents(text))
    for content in reversed(boxed):
        val = _number_from_boxed_content(content)
        if val is not None:
            return val, content, "boxed"

    # Non-LaTeX fallback: "boxed: 42".
    plain = PLAIN_BOXED_RE.findall(text)
    if plain:
        val = _to_number(plain[-1])
        if val is not None:
            return val, plain[-1], "boxed"

    for pattern in FINAL_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            val = _to_number(matches[-1])
            if val is not None:
                return val, matches[-1], "final_phrase"

    matches = NUMBER_RE.findall(text)
    if matches:
        val = _to_number(matches[-1])
        if val is not None:
            return val, matches[-1], "last_number"

    return None, None, "none"


def numeric_equal(pred, gold, atol=1e-6, rtol=1e-6):
    return (
        pred is not None
        and gold is not None
        and math.isclose(float(pred), float(gold), abs_tol=atol, rel_tol=rtol)
    )
