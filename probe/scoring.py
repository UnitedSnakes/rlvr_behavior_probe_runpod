from __future__ import annotations
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math, re

BOXED_PATTERNS=[
    re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}"),
    re.compile(r"boxed\s*[:=]?\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?)", re.I),
]
FINAL_PATTERNS=[
    re.compile(r"(?:final\s+answer|answer\s+is|therefore|thus)\s*[:=]?\s*\$?\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?)", re.I),
    re.compile(r"####\s*([+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?)"),
]
NUMBER_RE=re.compile(r"[+-]?(?:\d[\d,]*)(?:\.\d+)?(?:/\d+)?")

def _clean_token(token):
    return token.strip().replace("$","").replace(",","").rstrip(".,;:!?")

def _to_number(token):
    token=_clean_token(token)
    if not token: return None
    try:
        if "/" in token and re.fullmatch(r"[+-]?\d+(?:\.\d+)?/\d+(?:\.\d+)?", token):
            a,b=token.split("/",1)
            return float(Fraction(Decimal(a))/Fraction(Decimal(b)))
        return float(Decimal(token))
    except (InvalidOperation,ValueError,ZeroDivisionError):
        return None

def extract_numeric_answer(text):
    for p in BOXED_PATTERNS:
        m=p.findall(text)
        if m:
            val=_to_number(m[-1])
            if val is not None: return val,m[-1],"boxed"
    for p in FINAL_PATTERNS:
        m=p.findall(text)
        if m:
            val=_to_number(m[-1])
            if val is not None: return val,m[-1],"final_phrase"
    m=NUMBER_RE.findall(text)
    if m:
        val=_to_number(m[-1])
        if val is not None: return val,m[-1],"last_number"
    return None,None,"none"

def numeric_equal(pred,gold,atol=1e-6,rtol=1e-6):
    return pred is not None and gold is not None and math.isclose(float(pred),float(gold),abs_tol=atol,rel_tol=rtol)
