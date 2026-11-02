"""Exercise the actual Model library JS row, not just string markers."""
from pathlib import Path
import subprocess


def test_model_verify_row_runs_in_javascript_without_reference_errors():
    app_path = Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js"
    node = r"""
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const source = fs.readFileSync(process.argv[1], "utf8");
const start = source.indexOf("function mpRow(f, key, m) {");
const end = source.indexOf("\nfunction mpOptionalPanel(", start);
assert(start >= 0 && end > start);
const nodes = [];
function el(tag, attrs = {}, ...children) {
  const events = {};
  const node = {tag, attrs, children, events, dataset: {}, style: {}, isConnected: true,
    textContent: attrs.text || "", title: attrs.title || "", hidden: !!attrs.hidden,
    disabled: !!attrs.disabled, value: attrs.value || "",
    append(...added) { this.children.push(...added); },
    addEventListener(name, cb) { events[name] = cb; }};
  nodes.push(node);
  return node;
}
const S = {prefs: {dit: "/models/test.safetensors"}, mpRows: {}, mpVerifyTasks: {}};
const api = async (path) => {
  if (path === "/api/models/verify") return {state: "running", stage: "inspect", done: 0, total: 0};
  if (path.startsWith("/api/models/verify/status?")) return {
    state: "layout_match", message: "Tensor layout matches the expected architecture."};
  throw Error("Unexpected API: " + path);
};
const mpRow = vm.runInNewContext("(" + source.slice(start, end) + ")", {
  el, S, api, Promise, encodeURIComponent, setTimeout: (f) => setImmediate(f),
  mpStat: () => ({exists: true, directory: false, bytes: 1048576}),
  mpChip: (label) => ({tag: "chip", label}),
  mpSize: () => "1 MiB", mpHelp: () => ({tag: "help"}),
  MP_ROLES: {dit: "DiT"}, pick() {},
});
(async () => {
  const row = mpRow({key: "family"}, "dit", {
    label: "Training model", required: true, role: "dit", size_gb: 1,
    repo: "official/model", path: "weight.safetensors"
  });
  assert.equal(row.tag, "article");
  const button = nodes.find((n) => n.tag === "button" && n.textContent === "Test model");
  assert(button && button.events.click, "Test model button must be wired");
  assert.equal(button.disabled, false);
  const result = await S.mpVerifyTasks.dit.run();
  assert.equal(result.state, "layout_match");
  const status = nodes.find((n) => n.tag === "span" &&
    n.attrs.title === "Path existence does not verify model identity.");
  assert(status.textContent.startsWith("Architecture matches"), status.textContent);
  assert.equal(status.dataset.result, "layout_match");
  assert.equal(button.disabled, false);
})().catch((err) => { console.error(err); process.exitCode = 1; });
"""
    proc = subprocess.run(["node", "-e", node, str(app_path)], capture_output=True,
                          text=True, timeout=15)
    assert proc.returncode == 0, proc.stdout + proc.stderr
