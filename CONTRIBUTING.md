# Contributing to Fizgig

Pull requests are welcome — bug fixes, features, docs, pod/RunPod improvements, all of it.

## What you can expect

- **Your authorship is preserved.** PRs are merged with a real GitHub merge: your name on the
  commit, the green Merged badge, and contribution-graph credit. Accepted work is also
  credited by @handle in the release notes, and substantial contributions go in
  [CONTRIBUTORS.md](CONTRIBUTORS.md).
- **A real review.** PRs here get read properly — including adversarial testing where it's
  warranted — and feedback comes as review comments or fixups to your branch, never as a
  silent rewrite of your work.
- **Fast turnaround.** This project ships often; good PRs tend to land in the next release.

## What helps a PR land quickly

- One concern per PR where possible.
- Say what you tested and on what hardware (GPU/VRAM matters a lot here).
- Fizgig targets Flux 2 Klein 9B and Krea 2 only — see CLAUDE.md for the codebase map.
- Match the surrounding code's style, and prefer editing `COLORS[...]`/shared helpers over
  hardcoding UI values.

## Model discovery checks

Install development tools with `python -m pip install -r requirements-dev.txt`.
Run `python -m pytest` (or `xvfb-run -a python -m pytest` on headless Linux to include
the Tk Preferences seam tests), and `python -m compileall -q src lora_trainer_gui.py`.
For the new discovery module and focused tests, run:

```sh
python -m ruff check src/fizgig/model_sources.py tests/test_model_discovery.py tests/test_model_preferences.py
python -m ruff format --check src/fizgig/model_sources.py tests/test_model_discovery.py tests/test_model_preferences.py
```

The legacy GUI/downloader currently have existing lint/format failures; these focused checks
do not establish a repository-wide clean baseline. Discovery tests use temporary model roots
and a local HF-cache fixture, with downloads stubbed; no GPU or real model weights are needed.

## Bugs and ideas

Open an issue with the console output and your setup (GPU, OS, desktop or RunPod). Detailed
reports here have repeatedly gone from filed to fixed-and-released inside a day.
