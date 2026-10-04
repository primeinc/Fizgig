"""Small helpers for the Captions tab's Find & Replace preview."""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Pattern


def compile_find_pattern(find_text: str, whole_word: bool = True) -> Pattern[str]:
    """Case-insensitive literal search, optionally limited to word boundaries.

    Lookarounds are used instead of wrapping in \\b: phrases and punctuation stay literal,
    while a word like "her" does not match "together".
    """
    expr = re.escape(find_text)
    if whole_word:
        # Only add a boundary on an edge that is itself a word character. This keeps
        # punctuation/phrase searches literal and even preserves the old " her " workaround.
        if find_text and re.match(r"\w", find_text[0]):
            expr = rf"(?<!\w){expr}"
        if find_text and re.match(r"\w", find_text[-1]):
            expr = rf"{expr}(?!\w)"
    return re.compile(expr, re.IGNORECASE)


def highlighted_segments(
    text: str,
    pattern: Pattern[str],
    replacement: str | None = None,
) -> Iterator[tuple[str, bool]]:
    """Yield (text, highlighted) chunks.

    With no replacement, matched source spans are highlighted. With a replacement, the
    output is the replaced text and only text introduced by a match is highlighted.
    """
    pos = 0
    for match in pattern.finditer(text):
        if match.start() > pos:
            yield text[pos:match.start()], False
        if replacement is None:
            yield match.group(0), True
        elif replacement:
            yield replacement, True
        pos = match.end()
    if pos < len(text):
        yield text[pos:], False
