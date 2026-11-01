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

def compile_saved_rules(items):
    """Ordered per-folder rules for both caption workers and training auto-recaption.

    Use the same case-insensitive *literal* matcher as manual Find & Replace.
    Whole-word is retained for each rule; disabled rules do not run.
    """
    if not isinstance(items, list) or len(items) > 30:
        raise ValueError("Use a list of at most 30 caption replacement rules.")
    compiled = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Caption rule {index} must be an object.")
        find, replacement = item.get("find"), item.get("replace")
        if not isinstance(find, str) or not find.strip() or len(find) > 120:
            raise ValueError(f"Caption rule {index} needs text to find (1-120 characters).")
        if not isinstance(replacement, str) or len(replacement) > 240:
            raise ValueError(f"Caption rule {index} replacement exceeds 240 characters.")
        if not isinstance(item.get("whole_word", True), bool) or not isinstance(item.get("enabled", True), bool):
            raise ValueError(f"Caption rule {index} has invalid matching options.")
        if item.get("enabled", True):
            compiled.append((compile_find_pattern(find, item.get("whole_word", True)), replacement))
    return tuple(compiled)


def normalized_saved_rules(items):
    """Validate and return the explicit safe fields; no arbitrary regex or code."""
    compile_saved_rules(items)
    return [{"find": item["find"].strip(), "replace": item["replace"],
             "whole_word": item.get("whole_word", True), "enabled": item.get("enabled", True)}
            for item in items]


def parse_recaption_rules(spec: str):
    """Ordered literal whole-word rules, one 'find => replacement' per line.

    This is job configuration, not a global gender correction. Rule text is
    interpreted literally, never as user-supplied regular expressions.
    """
    if not isinstance(spec, str) or len(spec) > 24576:
        raise ValueError("Auto-recaption replacement bundle exceeds 24576 characters.")
    # Structured folder rules share the exact matcher with Caption All; the
    # legacy 'find => replace' Training text stays fully backwards-compatible.
    if spec.startswith('{"saved_rules":'):
        import json
        try:
            bundle = json.loads(spec)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid saved caption replacement rules.") from exc
        if not isinstance(bundle, dict) or not isinstance(bundle.get("legacy", ""), str):
            raise ValueError("Invalid saved caption replacement rule configuration.")
        legacy = parse_recaption_rules(bundle.get("legacy", ""))
        saved = compile_saved_rules(bundle.get("saved_rules"))
        if len(legacy) + len(saved) > 60:
            raise ValueError("Too many combined caption replacement rules.")
        return legacy + saved
    if len(spec) > 4096:
        raise ValueError("Training caption replacement text exceeds 4096 characters.")
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
