"""Find & Replace matching semantics: tiny, headless, no GPU."""

from fizgig.caption_replace import compile_find_pattern, highlighted_segments


def _matches(text, needle, whole=True):
    return [m.group(0) for m in compile_find_pattern(needle, whole).finditer(text)]


def test_whole_word_avoids_substrings():
    assert _matches("her dog, together with HER cat", "her") == ["her", "HER"]
    assert _matches("get it together; GET moving", "get") == ["get", "GET"]


def test_substring_mode_is_still_available():
    assert _matches("together", "get", whole=False) == ["get"]


def test_phrase_and_case_insensitive_matching():
    assert _matches("Red Hair, red hairpin, RED HAIR", "red hair") == ["Red Hair", "RED HAIR"]


def test_preview_segments_mark_only_changed_text():
    pattern = compile_find_pattern("her")
    before = list(highlighted_segments("her dog together", pattern))
    after = list(highlighted_segments("her dog together", pattern, "his"))
    assert before == [("her", True), (" dog together", False)]
    assert after == [("his", True), (" dog together", False)]
