"""Replace capitalised names in a generated story that the evidence does not contain.

The story model invents people, places and breeds ("Thomas", "Sarah", "Willowbrook")
even when the prompt forbids it. This guard finds capitalised words that are neither
in the evidence context nor in a list of ordinary English words, and swaps each one
for a neutral phrase. It is deterministic and does not call a model.
"""

from __future__ import annotations

import re

# Ordinary capitalised words that can start a sentence or name a time, not a person or place.
_COMMON_WORDS = frozenset(
    """
    a an the i it its we you he she they them his her their our your my me him us
    this that these those there here some many most all both each every any no
    one two three once later soon then when while after before but and so yet or nor
    if with without from into onto under over across through near beyond above below
    inside outside together meanwhile finally suddenly also just still even only
    as at by for in of on to up down out off again away back around about
    yes not hello morning afternoon evening night day days week today tonight
    monday tuesday wednesday thursday friday saturday sunday
    """.split()
)

_WORD = re.compile(r"\b[A-Z][a-z]+\b")
_SENTENCE_START = re.compile(r"(?:^|[.!?]\s+)$")
REPLACEMENT = "the figure"


def _context_words(context: str) -> frozenset[str]:
    return frozenset(w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*", context))


def guard_invented_names(text: str, context: str) -> tuple[str, list[str]]:
    """Return the story with invented capitalised names replaced, and the names removed.

    A name counts as supported if any of its letters-only form appears in the context,
    compared case-insensitively. Words in ``_COMMON_WORDS`` are never replaced.
    """
    if not text or not context:
        return text, []

    supported = _context_words(context)
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        lowered = word.lower()
        if lowered in _COMMON_WORDS or lowered in supported:
            return word
        removed.append(word)
        before = text[: match.start()]
        at_sentence_start = bool(_SENTENCE_START.search(before))
        if at_sentence_start:
            return "The figure"
        return REPLACEMENT

    return _WORD.sub(replace, text), removed
