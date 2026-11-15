"""Real caption-job construction, without model downloads or GPU allocation.

Exercises the same ai_captions() entrypoint used by the server. The pre-fix
fork fails on unexpected tag kwargs and on its undefined QWEN_MODEL name.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fizgig.web import captions


@pytest.mark.parametrize("kind,instruction", [
    ("tags", False),
    ("sentences", True),
])
def test_ai_caption_job_keeps_upstream_options_and_fork_rules(
        tmp_path, monkeypatch, kind, instruction):
    picture = tmp_path / "example.jpg"
    picture.write_bytes(b"caption test never opens the picture")
    job_dir = tmp_path / "jobs"
    desc = SimpleNamespace(
        key="stub_" + kind,
        label="Stub " + kind,
        kind=kind,
        files=object(),
        instruction=instruction,
        tasks=(),
        default_task="",
    )
    monkeypatch.setattr(captions._captioners, "get", lambda key: desc if key == desc.key else None)
    monkeypatch.setattr(captions, "_busy", lambda: False)
    monkeypatch.setattr(captions, "JOB_DIR", str(job_dir))
    monkeypatch.setattr(captions, "with_crops", lambda _folder: [str(tmp_path)])
    monkeypatch.setattr(captions, "media_files", lambda *_args, **_kwargs: [str(picture)])
    monkeypatch.setattr(captions, "_crop_images", lambda images, _overwrite, _ext: (images, []))
    monkeypatch.setattr(captions, "_st", {**captions._st, "proc": None, "state": "idle", "log": []})
    monkeypatch.setattr(captions, "_popen", lambda *_args, **_kwargs: SimpleNamespace(poll=lambda: None))
    class InertThread:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
    monkeypatch.setattr(captions.threading, "Thread", InertThread)

    rule = {"find": "boy", "replace": "girl", "whole_word": True, "enabled": True}
    result = captions.ai_captions(
        {}, "", str(tmp_path), desc.key, "", "describe this image", 200, "",
        True, False, photos=True, ext=".txt",
        threshold=0.42, leave_out="watermark, blurry", underscores=True,
        replacement_rules=[rule],
    )
    assert result == {"files": 1}
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    assert job["replacement_rules"] == [rule]
    assert job["max_new_tokens"] == 200
    if kind == "tags":
        assert job["threshold"] == pytest.approx(0.42)
        assert job["leave_out"] == ["watermark", "blurry"]
        assert job["underscores"] is True
        assert "instruction_file" not in job
    else:
        assert Path(job["instruction_file"]).read_text(encoding="utf-8") == "describe this image"
        assert "threshold" not in job


def test_instruction_choice_follows_metadata_not_qwen_name():
    from fizgig import captioners
    instructions = [d for d in captioners.CAPTIONERS if d.instruction]
    assert instructions, "No instructional captioners in live registry"
    assert all(d.kind != "tags" for d in instructions)
    assert any(d.key != "qwen3vl" for d in instructions), (
        "A future registered instruction model must not need a Qwen-named code branch"
    )
