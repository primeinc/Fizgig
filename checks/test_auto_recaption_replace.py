"""Per-job auto-recaption repair: only generated captions, never existing data in bulk."""
import sys
import types

import pytest

from fizgig.caption_replace import apply_recaption_rules, parse_recaption_rules


def test_literal_whole_word_replacements_fix_hannah_example():
    rules = parse_recaption_rules("boy => girl\nmale => female\nman => woman")
    caption = "A BOY with a male jacket beside a man, with a boyfriend in the background."
    result = apply_recaption_rules(caption, rules)
    assert result == "A girl with a female jacket beside a woman, with a boyfriend in the background."


def test_no_rules_leave_generated_caption_alone():
    caption = "A boy is wearing a jersey"
    assert apply_recaption_rules(caption, parse_recaption_rules("")) == caption


def test_multiline_rules_do_not_match_word_fragments():
    rules = parse_recaption_rules("male => female\nman => woman\nboy => girl")
    assert apply_recaption_rules("female, woman, boys, boyfriend, human, male", rules) == (
        "female, woman, boys, boyfriend, human, female")


def test_rules_are_literal_not_regex():
    rules = parse_recaption_rules("a+b => matching")
    assert apply_recaption_rules("a+b and aaab", rules) == "matching and aaab"


@pytest.mark.parametrize("rules", [
    "boy girl",
    "=> girl",
    "a" * 121 + " => girl",
    "boy => " + "x" * 241,
    "x => y\n" * 31,
])
def test_invalid_rules_rejected_before_training(rules):
    with pytest.raises(ValueError):
        parse_recaption_rules(rules)


def test_apply_before_file_write_and_training_reencoding(tmp_path, monkeypatch):
    """Exercise loss-watch's real _recaption method with a fake captioner, not just the parser."""
    from fizgig.families.loss_watch import Watch

    fake_embedder = types.ModuleType("fizgig.krea2.embedder")
    fake_embedder.generate_caption = lambda *args, **kwargs: "A young BOY wearing a male shirt stands by a man."
    fake_utils = types.ModuleType("fizgig.krea2.utils")

    class FakeEncoder:
        def to(self, _device):
            return self
    fake_utils.load_krea2_text_encoder = lambda *args, **kwargs: FakeEncoder()
    monkeypatch.setitem(sys.modules, "fizgig.krea2.embedder", fake_embedder)
    monkeypatch.setitem(sys.modules, "fizgig.krea2.utils", fake_utils)

    watch = Watch.__new__(Watch)
    watch.instr = "Describe the subject"
    watch.instr_detailed = ""
    watch.captioner_path = "unused_fake_model.safetensors"
    watch.caption_rules = parse_recaption_rules("boy => girl\nmale => female\nman => woman")
    watch.caption_ext = ".txt"
    watch.triggers = {}
    watch.trigger_word = "hannah_subject"
    watch.trigger_position = "start"
    watch.watch = types.SimpleNamespace(where=lambda _: ("", str(tmp_path), "photo"))
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fake image")
    item = object()
    rows = watch._recaption([("case", str(image), 1)], {"case": item}, "cpu")
    expected = "hannah_subject, A young girl wearing a female shirt stands by a woman."
    assert rows == [("case", item, expected, 1)]
    assert (tmp_path / "photo.txt").read_text() == expected
