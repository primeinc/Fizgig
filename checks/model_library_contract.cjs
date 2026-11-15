// Exercise the real upstream Preferences renderer with the fork extension loaded.
// This is a DOM contract test, not a source-string checklist. The synthetic family
// proves the model library is driven by upstream schema, not by model names.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const h3 = JSON.parse(fs.readFileSync(0, "utf8"));
const app = fs.readFileSync("src/fizgig/web/static/app.js", "utf8");
const extension = fs.readFileSync("src/fizgig/web/static/fork/model-library.js", "utf8");
const start = app.indexOf("// ---- Preferences: every family's model files");
const end = app.indexOf("// Preferences' Captioning models card", start);
assert(start >= 0 && end > start, "Cannot find the upstream Preferences renderer seam");
assert(!app.includes("function mpRow("), "Fork model-library implementation leaked back into upstream app.js");
assert(!app.includes("function modelRootsCard("), "Fork path controls leaked back into upstream app.js");

function el(tag, attrs, ...children) {
  const props = attrs && typeof attrs === "object" && !Array.isArray(attrs) ? attrs : {};
  const content = props === attrs ? children : [attrs, ...children];
  const node = {
    tag, props, children: [], style: {}, value: props.value || "", textContent: props.text || "",
    append(...items) {
      this.children.push(...items.flat(Infinity).filter(item => item !== undefined && item !== null));
    },
    prepend(...items) {
      this.children.unshift(...items.flat(Infinity).filter(item => item !== undefined && item !== null));
    },
    replaceChildren(...items) { this.children = []; this.append(...items); },
    addEventListener() {},
    scrollIntoView() {},
    querySelector(selector) {
      if (!selector.startsWith(".")) throw Error("Unsupported test selector " + selector);
      const cls = selector.slice(1);
      return walk(this, n => String(n.props?.class || "").split(/\s+/).includes(cls))[0] || null;
    },
    classList: { toggle() {}, contains() { return false; } }
  };
  Object.defineProperty(node, "innerHTML", {set() {node.children = [];}, get() {return "";}});
  node.append(...content);
  return node;
}

function walk(node, test, out=[]) {
  if (!node || typeof node !== "object") return out;
  if (test(node)) out.push(node);
  for (const child of node.children || []) walk(child, test, out);
  return out;
}
function rows(node) {
  return walk(node, n => !!n.props?.["data-key"]).map(n => n.props["data-key"]);
}
function hrefs(node) {
  return walk(node, n => n.tag === "a").map(n => n.props.href || "");
}
function classCount(node, cls) {
  return walk(node, n => String(n.props?.class || "").split(/\s+/).includes(cls)).length;
}

const host = el("form", {id:"prefsForm"});
const S = {
  schema: {families:[h3]}, family:h3.key, prefs:{}, modelInventory:{files:{}},
  modelRoots:[], modelDownloadLayout:"fizgig", modelDownloadRoot:"",
  fetchRows:{}, mpRows:{}, mpQuery:"", mpFilter:"all", mpFamily:h3.key,
  pod:false
};
const mock = {
  S, window:{}, el, $:() => host,
  document: {querySelector:()=>null, querySelectorAll:()=>[]},
  api:async()=>({models:{},model_inventory:{files:{}},model_search_roots:[]}),
  pick:()=>{}, fetchRow:()=>el("div", {class:"fetch-row"}),
  paintFetch:()=>{}, checkModels:()=>{}, refreshStateSoon:()=>{}, captionModelsChanged:()=>{},
  podCard:()=>el("div"), qtGpuCard:()=>null,
  webPortCard:()=>el("div", {class:"upstream-global web-port"}),
  networkCard:()=>el("div", {class:"upstream-global network"}),
  dataFolderCard:()=>el("div", {class:"upstream-global data-folder"}),
  outputFolderCard:()=>el("div", {class:"upstream-global output-folder"}),
  foldersCard:()=>el("div", {class:"upstream-global folders"}),
  inferenceCard:()=>el("div", {class:"upstream-global inference"}),
  captionerCard:()=>el("div", {class:"upstream-global captioner"}),
  gpuPickCard:()=>null
};
vm.createContext(mock);
vm.runInContext(extension, mock, {timeout: 10000});
vm.runInContext(app.slice(start, end), mock, {timeout: 10000});
assert(mock.window.FizgigForkModelLibrary?.mount, "Fork extension did not register a renderer seam");

mock.window.FizgigForkModelLibrary.initialize({
  model_inventory:{files:{}},
  model_search_roots:["/models/reference-library"],
  model_download_layout:"comfyui",
  model_download_root:"/models"
});
mock.renderPrefs();
const required = [
  "minimax_image_dit", "minimax_image_vae",
  "minimax_image_text_encoder", "minimax_image_training_adapter"
];
assert(required.every(k => rows(host).includes(k)), "H3 Image lost its model-library rows");
const links = hrefs(host);
for (const filename of [
  "minimax_h3_image_pruned_int8_convrot.safetensors",
  "minimax_h3_image_pruned_bf16.safetensors",
  "qwen3vl_32b_minimax_h3_bf16.safetensors",
  "fizgig_minimax_h3_image_training_adapter.safetensors"
]) assert(links.some(link => link.includes(filename)), "Missing official H3 file link " + filename);
assert(classCount(host, "mp-workspace") === 1);
assert(classCount(host, "mp-library-card") === 1);
assert(classCount(host, "upstream-global") === 7, "Upstream global preferences were not preserved");
assert(classCount(host, "pref-file") === 0, "Duplicated upstream per-family model cards");

// An entirely unrecognized future model family is rendered without editing fork code.
const future = {
  ...h3, key:"future-generated-family", display_name:"Future Model",
  model_keys:["future_model_dit"],
  model_files:[{pref_key:"future_model_dit",label:"Future diffusion model", required:true,
    size_gb:3.25, role:"dit", repo:"ExampleOrg/NewModel",
    path:"diffusion_models/new_model.safetensors"}]
};
S.schema.families.push(future);
S.family = future.key;
mock.window.FizgigForkModelLibrary.onFamilyChanged(future.key);
mock.renderPrefs();
assert(rows(host).includes("future_model_dit"), "Future family is absent from schema-driven library");
assert(hrefs(host).some(s => s.includes("ExampleOrg/NewModel")), "Future model metadata not rendered");
assert(classCount(host, "mp-library-card") === 1);

// Ready is the server's identity verdict: a file that exists but holds another model's weights is not Ready.
const rowOf = key => walk(host, n => n.props?.["data-key"] === key)[0];
const chipOf = key => walk(rowOf(key), n => String(n.props?.class || "").split(/\s+/).includes("mp-chip"))[0];
S.prefs.future_model_dit = "/models/new_model.safetensors";
const onDisk = {exists:true, bytes:10, expected_bytes:10, path:"/models/new_model.safetensors", required:true};
S.modelInventory = {files:{future_model_dit:{...onDisk, ready:false, wrong_file:true}}};
mock.renderPrefs();
assert.equal(rowOf("future_model_dit").props["data-status"], "wrong", "Another model's file is shown as Ready");
assert.equal(chipOf("future_model_dit").textContent, "Wrong file");
S.modelInventory = {files:{future_model_dit:{...onDisk, ready:true, wrong_file:false}}};
mock.renderPrefs();
assert.equal(rowOf("future_model_dit").props["data-status"], "ready");
assert.equal(chipOf("future_model_dit").textContent, "Ready");
S.prefs.future_model_dit = "";
S.modelInventory = {files:{}};

// Negative control: remove a required model row and verify the same contract flags it.
const originalKeys = future.model_keys;
future.model_keys = [];
mock.renderPrefs();
assert(!rows(host).includes("future_model_dit"), "Negative control failed: missing model row undetected");
future.model_keys = originalKeys;

// Negative control: disable the extension and prove native upstream Preferences still work.
mock.window.FizgigForkModelLibrary = null;
mock.renderPrefs();
assert(classCount(host, "mp-workspace") === 0, "Fallback still depends on extension");
assert(classCount(host, "pref-file") > 0, "Upstream built-in model cards do not render in fallback");
assert(classCount(host, "upstream-global") === 7, "Native upstream global controls missing");
console.log("PASS: H3, future family, shared upstream controls, fallback and negative controls");
