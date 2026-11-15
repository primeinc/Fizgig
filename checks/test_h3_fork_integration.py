"""H3 Image integration contract across upstream family and fork-owned model UX.

CPU-only. Tests the actual registry/schema, fork downloader, local model-library
layouts, file reuse, and web UI renderer without fetching checkpoint weights.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from fizgig.families import form
from fizgig.families.registry import get, training_families
from fizgig.scripts import fetch_models
from fizgig.web.model_inventory import inventory
from fizgig.web.schema import get_schema
from fizgig import model_destinations


KEYS = {
    "minimax_image_dit",
    "minimax_image_vae",
    "minimax_image_text_encoder",
    "minimax_image_training_adapter",
}
ROOT = Path(__file__).resolve().parents[1]


def _h3():
    desc = get("minimax_image")
    assert desc is not None
    return desc


def _weights():
    return {
        weight.pref_key: weight
        for weight in fetch_models.model_candidates(("minimax_image",), include_optional=False)
    }


def test_upstream_h3_image_surfaces_through_actual_web_schema():
    desc = _h3()
    assert desc in training_families()
    assert desc.gui_label == "MiniMax H3 Image"
    assert desc.driver == "fizgig.minimax_image.driver:MiniMaxH3ImageDriver"
    h3 = next(f for f in get_schema()["families"] if f["key"] == "minimax_image")
    assert h3["display_name"] == "MiniMax H3 Image"
    assert not h3.get("prefs_skip", False)
    assert KEYS <= set(h3["model_keys"])
    assert KEYS <= {m["pref_key"] for m in h3["model_files"]}
    assert h3["model_keys"] == form.model_keys(desc)
    assert next(m for m in h3["model_files"] if m["pref_key"] == "minimax_image_vae")["path"].endswith(
        "minimax_h3_image_vae_fp16.safetensors"
    )


def test_custom_catalog_discovers_all_h3_files_and_downloads_training_adapter_by_default():
    weights = _weights()
    candidates = {
        w.pref_key: w for w in fetch_models.discovery_candidates(("minimax_image",))
    }
    assert KEYS <= set(weights)
    assert KEYS <= set(candidates)
    assert weights["minimax_image_training_adapter"].optional is False
    assert weights["minimax_image_text_encoder"].same_as == "minimax_text_encoder"
    assert weights["minimax_image_dit"].filename == "minimax_h3_image_pruned_int8_convrot.safetensors"
    assert weights["minimax_image_vae"].filename == "minimax_h3_image_vae_fp16.safetensors"
    assert weights["minimax_image_dit"].alt.filename == "minimax_h3_image_pruned_bf16.safetensors"
    assert weights["minimax_image_text_encoder"].alt.filename == "qwen3vl_32b_minimax_h3_bf16.safetensors"


@pytest.mark.parametrize("layout,base_folder,expected_parts", [
    ("fizgig", "", {
        "minimax_image_dit": "own", "minimax_image_vae": "own",
        "minimax_image_text_encoder": "own", "minimax_image_training_adapter": "own",
    }),
    ("comfyui", "models", {
        "minimax_image_dit": "diffusion_models", "minimax_image_vae": "vae",
        "minimax_image_text_encoder": "text_encoders",
        "minimax_image_training_adapter": "fizgig_adapters",
    }),
    ("swarmui", "Models", {
        "minimax_image_dit": "diffusion_models", "minimax_image_vae": "VAE",
        "minimax_image_text_encoder": "text_encoders",
        "minimax_image_training_adapter": "fizgig_adapters",
    }),
])
def test_custom_destination_places_h3_models_in_correct_library(
        tmp_path, layout, base_folder, expected_parts):
    own = tmp_path / "own"
    own.mkdir()
    target = tmp_path / "app"
    (target / base_folder).mkdir(parents=True, exist_ok=True) if base_folder else None
    prefs = {"model_download_layout": layout, "model_download_root": str(target)}
    for key, weight in _weights().items():
        if key not in KEYS:
            continue
        parent, dest = model_destinations.destination(weight, str(own), prefs)
        assert Path(parent).name == expected_parts[key]
        assert Path(dest).parent == Path(parent)
        assert Path(dest).name == weight.filename
        assert not Path(dest).exists()  # Planning paths must not download or write weights.


def test_h3_image_shared_encoder_reuses_prior_h3_download(tmp_path, monkeypatch):
    original = tmp_path / "already-chosen-h3-qwen.safetensors"
    original.write_bytes(b"fake local model; must not be opened or downloaded")
    prefs = {"minimax_text_encoder": str(original)}
    model = _weights()["minimax_image_text_encoder"]

    def no_search(*_args, **_kwargs):
        raise AssertionError("shared encoder should be linked before any filesystem search")

    monkeypatch.setattr(fetch_models, "find_weight", no_search)
    assert fetch_models.fetch_weight(model, str(tmp_path / "new"), prefs, log=lambda _: None)
    assert prefs["minimax_image_text_encoder"] == str(original)

    # Negative control: removing the real alias must make the same assertion path fail.
    broken = copy.copy(model)
    broken.same_as = ""
    with pytest.raises(AssertionError, match="linked before"):
        fetch_models.fetch_weight(broken, str(tmp_path / "new"), {
            "minimax_text_encoder": str(original)}, log=lambda _: None, dry_run=True)


def test_h3_model_inventory_uses_canonical_file_list(tmp_path):
    data = inventory({}, [_h3()], str(tmp_path), key_resolver=form.model_keys)
    assert KEYS <= set(data["files"])
    for key in KEYS:
        assert data["files"][key]["configured"] is False
        assert data["files"][key]["expected_bytes"] > 0
    assert data["summary"]["remaining_required_bytes"] > 0


_NODE_TEST = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const family = JSON.parse(fs.readFileSync(0, "utf8"));
const source = fs.readFileSync("src/fizgig/web/static/app.js", "utf8");
const start = source.indexOf("// ---- Model library: one family at a time");
const end = source.indexOf("// Search libraries on the Fizgig computer", start);
assert(start >= 0 && end > start, "Model library renderer must be present");

function el(tag, attrs, ...kids) {
  const isAttrs = attrs && typeof attrs === "object" && !Array.isArray(attrs);
  const properties = isAttrs ? attrs : {};
  const children = isAttrs ? kids : [attrs, ...kids].filter(x => x !== undefined);
  const node = {
    tag, props: properties, children: [],
    append(...xs) { this.children.push(...xs.filter(x => x !== null && x !== undefined)); },
    addEventListener() {},
    querySelector() { return this; },
    scrollIntoView() {}
  };
  node.append(...children);
  return node;
}

const context = {
  S: {prefs:{}, modelInventory:{files:{}}, mpRows:{}, fetchRows:{},
      mpFilter:"all", mpQuery:""},
  el, fetchRow: (_f) => el("section", {class:"fetch-row"}),
  document: {querySelector:()=>null, querySelectorAll:()=>[]}
};
vm.createContext(context);
vm.runInContext(source.slice(start, end), context, {timeout: 10000});

function walk(root, predicate, results=[]) {
  if (!root || typeof root !== "object") return results;
  if (predicate(root)) results.push(root);
  for (const child of root.children || []) walk(child, predicate, results);
  return results;
}
const required = [
  "minimax_image_dit", "minimax_image_vae",
  "minimax_image_text_encoder", "minimax_image_training_adapter"
];
function rowsFor(f) {
  const card = context.mpFamilyCard(f);
  return walk(card, n => n.props && n.props["data-key"])
    .map(n => n.props["data-key"]);
}
const rows = rowsFor(family);
assert(required.every(k => rows.includes(k)), "Real H3 web card dropped a model row");
const card = context.mpFamilyCard(family);
const links = walk(card, n => n.tag === "a")
  .map(n => n.props.href || "");
assert(links.some(x => x.includes("minimax_h3_image_pruned_int8_convrot.safetensors")));
assert(links.some(x => x.includes("minimax_h3_image_pruned_bf16.safetensors")));
assert(links.some(x => x.includes("qwen3vl_32b_minimax_h3_bf16.safetensors")));
assert(links.some(x => x.includes("fizgig_minimax_h3_image_training_adapter.safetensors")));
const toolbar = context.mpToolbar([family], family);
const options = walk(toolbar, n => n.tag === "option").map(n => n.props.value);
assert(options.includes("minimax_image"), "H3 family absent from fork selector");

// Negative control: the same rendered-row gate MUST detect a missing schema field.
const broken = {...family, model_keys:family.model_keys.filter(k => k !== "minimax_image_vae")};
assert(!required.every(k => rowsFor(broken).includes(k)), "Gate failed to detect a missing VAE row");
console.log("H3 custom model UI rendered all model rows, primary and alternative links; negative control detected omission");
"""


def test_real_custom_ui_renderer_displays_h3_rows_and_alternates():
    family = next(f for f in get_schema()["families"] if f["key"] == "minimax_image")
    result = subprocess.run(
        ["node", "-e", _NODE_TEST],
        cwd=ROOT,
        input=json.dumps(family),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "negative control detected omission" in result.stdout
