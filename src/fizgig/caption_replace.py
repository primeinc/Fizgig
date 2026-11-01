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

def parse_recaption_rules(spec: str):
    """Ordered literal whole-word rules, one 'find => replacement' per line.

    This is job configuration, not a global gender correction. Rule text is
    interpreted literally, never as user-supplied regular expressions.
    """
    if not isinstance(spec, str) or len(spec) > 4096:
        raise ValueError("Auto-recaption replacements must be text, 4096 characters or fewer.")
    rules = []
    for line_number, raw in enumerate(spec.splitlines(), start=1):
        if not raw.strip():
            continue
        if "=>" not in raw:
            raise ValueError(f"Auto-recaption replacement line {line_number}: use find => replacement.")
        source, replacement = (part.strip() for part in raw.split("=>", 1))
        if not source or len(source) > 120 or len(replacement) > 240:
            raise ValueError(f"Auto-recaption replacement line {line_number} has an invalid length.")
        rules.append((compile_find_pattern(source, whole_word=True), replacement))
        if len(rules) > 30:
            raise ValueError("Auto-recaption supports at most 30 replacement rules per job.")
    return tuple(rules)


def apply_recaption_rules(caption: str, rules) -> str:
    """Fix one newly generated caption before it is written and re-encoded."""
    for pattern, replacement in rules:
        caption = pattern.sub(lambda _match: replacement, caption)
    return caption
