// Fork-owned model library. Loaded before app.js as a classic script; all upstream
// controls come through the existing schema and UI helpers at the render seam.
// No per-model catalog, model family branching, or replacement of upstream global cards.
// ---- Model library: one family at a time, built from canonical FamilyDescription metadata ------------------
const mpGiB = (bytes) => Math.max(0, Number(bytes) || 0) / (1024 ** 3);
const mpSize = (bytes, approx = false) => (approx ? "≈" : "") + mpGiB(bytes).toFixed(2) + " GB";
const MP_ROLES = {dit: "DiT", preview_dit: "Preview", vae: "VAE", text_encoder: "Text encoder",
  speed_lora: "LoRA", look_lora: "LoRA", training_adapter: "Adapter"};
function mpChip(value, style = "") {
  return el("span", {class: "mp-chip" + (style ? " " + style : ""), text: value});
}
function mpModels(f) {
  const descriptions = new Map((f.model_files || []).map((m) => [m.pref_key, m]));
  return (f.model_keys || []).map((key) => [key, descriptions.get(key) ||
    {pref_key: key, label: key, note: "Model used by this training family.", required: false}]);
}
function mpStat(key) {
  return S.modelInventory?.files?.[key] || {exists: false, ready: false, wrong_file: false, bytes: 0,
    expected_bytes: 0, configured: !!S.prefs[key], required: false};
}
function mpHelp(description, model = null) {
  const content = el("div", {class: "mp-help-content"},
    el("p", {text: description || "See the model's official card for more information."}));
  if (model?.repo && model.path) content.append(
    el("div", {class: "mp-help-source", text: model.repo + " / " + model.path}),
    el("a", {target: "_blank", rel: "noopener", href: "https://huggingface.co/" +
      model.repo + "/blob/main/" + model.path, text: "Official model file ↗"}));
  if (model?.alt_repo && model.alt_path) content.append(
    el("div", {class: "mp-help-source", text: "Alternative: " + model.alt_repo + " / " + model.alt_path}),
    el("a", {target: "_blank", rel: "noopener", href: "https://huggingface.co/" +
      model.alt_repo + "/blob/main/" + model.alt_path, text: "Alternative build ↗"}));
  return el("details", {class: "mp-help"},
    el("summary", {class: "mp-help-button", title: "Model information (hover or click)",
      "aria-label": "Model information"}, "?"), content);
}
async function mpRefresh() {
  const pr = await api("/api/prefs");
  S.prefs = pr.models || {};
  S.modelInventory = pr.model_inventory || {files: {}, summary: {}};
  renderPrefs();
  checkModels();
  refreshStateSoon();
  captionModelsChanged();
}
function mpRow(f, key, m) {
  const st = mpStat(key), raw = String(S.prefs[key] || "");
  // Ready is the server's identity verdict (model_inventory): existence alone is not enough.
  const state = st.ready ? "ready" : st.wrong_file ? "wrong" : raw.trim() ? "stale" : "missing";
  const label = m.label || key;
  const input = el("input", {type: "text", class: "w-path mp-path-input", value: raw,
    "aria-label": label + " model path", title: raw || "No model path selected",
    placeholder: "No model selected", spellcheck: "false"});
  const message = el("span", {class: "mp-save-error", role: "status"});
  let pending = Promise.resolve();
  const clear = el("button", {class: "btn btn-sm", text: "Clear", disabled: !raw.trim(),
    title: "Clear the saved path. The model file stays on disk.", onclick: () => savePath("")});
  const savePath = (value) => {
    const next = String(value || "").trim();
    input.value = next; input.title = next || "No model selected";
    clear.disabled = !next; S.prefs[key] = next;
    pending = pending.then(async () => {
      try {
        await api("/api/prefs", {changes: {[key]: next}});
        await mpRefresh();
      } catch (error) { message.textContent = "Not saved: " + error.message; }
    });
    return pending;
  };
  input.addEventListener("input", () => { clear.disabled = !input.value.trim(); });
  input.addEventListener("change", () => savePath(input.value));
  const statusName = {ready: "Ready", wrong: "Wrong file", stale: "Missing file"}[state] || "Not selected";
  const status = mpChip(statusName, state);
  const role = MP_ROLES[m.role] || "Model";
  const progress = el("span", {class: "mp-row-progress", role: "status", hidden: true});
  const fileNames = [m.local_name || (m.path || "").split("/").pop(),
    m.alt_path ? m.alt_path.split("/").pop() : ""].filter(Boolean);
  (S.mpRows[key] ||= []).push({node: progress, filenames: fileNames});
  const sourceInfo = [m.hint || m.note || "", m.download_note || ""].filter(Boolean).join("\n");
  const name = el("div", {class: "mp-model-heading"},
    el("div", {class: "mp-model-name"}, el("b", {text: label}), mpHelp(sourceInfo, m)),
    el("div", {class: "mp-model-badges"}, status, mpChip(role),
      ...(!m.required ? [mpChip("Optional", "optional")] : [])));
  const diskSize = state === "ready" ?
    (st.directory ? "Model folder ready · size unmeasured" : mpSize(st.bytes) + " on disk") :
    state === "wrong" ? "This file holds another model's weights" :
    state === "stale" ? "Saved path no longer exists" : "No local model selected";
  const catalogSize = m.size_gb ? "Catalog estimate " + Number(m.size_gb).toFixed(2) + " GB" : "";
  const meta = el("div", {class: "mp-model-meta"},
    el("span", {class: state === "ready" ? "mp-size-ready" : "mp-size-missing", text: diskSize}),
    ...(catalogSize ? [el("span", {text: catalogSize})] : []), progress, message);
  const actions = el("div", {class: "mp-model-actions"},
    el("button", {class: "btn btn-sm", text: "Choose…", title: "Choose a local model file", onclick: () =>
      pick(input.value, false, (picked) => savePath(picked))}), clear);
  if (m.repo && m.path) actions.append(el("a", {class: "btn btn-sm",
    target: "_blank", rel: "noopener", href: "https://huggingface.co/" + m.repo + "/blob/main/" + m.path,
    text: "Download ↗", title: "Open the official download page; does not start a background download"}));
  return el("article", {class: "mp-model-row", "data-key": key, "data-status": state,
    "data-optional": m.required ? "0" : "1",
    "data-search": [label, key, raw, m.path || ""].join(" ").toLowerCase()},
    name, el("div", {class: "mp-model-inputs"}, input, actions), meta);
}
function mpOptionalPanel(f) {
  const alternatives = (f.model_files || []).filter((m) => m.alt_repo && m.alt_path);
  if (!alternatives.length) return null;
  const detail = el("details", {class: "mp-alternatives"},
    el("summary", {text: "Alternative model builds (" + alternatives.length + ")"}));
  for (const m of alternatives) detail.append(el("div", {class: "mp-alternative"},
    el("span", {text: m.label + ": " + (m.alt_label || m.alt_path.split("/").pop())}),
    el("a", {href: "https://huggingface.co/" + m.alt_repo + "/blob/main/" + m.alt_path,
      rel: "noopener", target: "_blank", text: "View file ↗"})));
  return detail;
}
function mpFamilySummary(f) {
  const entries = mpModels(f);
  const ready = entries.filter(([key]) => mpStat(key).ready).length;
  const folders = entries.filter(([key]) => mpStat(key).ready && mpStat(key).directory).length;
  const seen = new Set();
  let disk = 0, remaining = 0;
  for (const [key,m] of entries) {
    const st = mpStat(key), same = m.same_as || key;
    if (st.ready) {
      if (!st.directory && st.path && !seen.has(st.path.toLowerCase())) {
        seen.add(st.path.toLowerCase()); disk += st.bytes || 0;
      }
    } else if (m.required && !mpStat(same).ready && !seen.has("missing:" + same)) {
      seen.add("missing:" + same); remaining += st.expected_bytes || 0;
    }
  }
  return {ready, total: entries.length, disk, remaining, folders};
}
function mpToolbar(families, f) {
  const totals = mpFamilySummary(f);
  const searchStatus = el("div", {class: "mp-search-status", role: "status"});
  S.mpSearchStatus = searchStatus;
  const selector = el("select", {class: "mp-family-filter", "aria-label": "Model family",
    onchange: (e) => { S.mpFamily = e.target.value; S.mpFilter = "all"; S.mpQuery = ""; renderPrefs(); }},
    ...families.map((d) => el("option", {value: d.key, text: d.display_name})));
  selector.value = f.key;
  const summary = el("div", {class: "mp-summary"},
    el("div", {class: "mp-metric"}, el("strong", {text: totals.ready + " / " + totals.total}),
      el("small", {text: "models ready"})),
    el("div", {class: "mp-metric"}, el("strong", {text: mpSize(totals.disk)}),
      el("small", {text: totals.folders ?
        "measured file bytes · " + totals.folders + " folder(s) unmeasured" : "measured file bytes"})),
    el("div", {class: "mp-metric"}, el("strong", {text: mpSize(totals.remaining, true)}),
      el("small", {text: "required downloads left"})));
  const top = el("div", {class: "mp-workspace-head"},
    el("div", {class: "mp-title-stack"}, el("h2", {text: "Model library"}),
      el("p", {text: "One family at a time. Reuse local files before downloading anything."})),
    el("div", {class: "mp-toolbar-actions"},
      el("button", {class: "btn", text: "Search locally", title: "Search all saved model directories; no downloads",
        onclick: () => startModelSearch("")}),
      el("button", {class: "btn btn-primary", text: "Download missing…",
        title: "Open the existing family download controls; does not start a download",
        onclick: () => {
          const panel = document.querySelector(".mp-download-panel");
          if (panel) { panel.open = true; panel.scrollIntoView({block: "nearest"}); }
        }})));
  const filterRow = el("div", {class: "mp-filter-row"},
    el("label", {class: "mp-selector-label"}, el("span", {text: "Model family"}), selector),
    el("div", {class: "mp-filter-actions"}));
  const filterHost = filterRow.querySelector(".mp-filter-actions");
  const search = el("input", {type: "search", value: S.mpQuery || "", class: "mp-search",
    placeholder: "Filter models by name or file…", "aria-label": "Filter model list"});
  const applyFilter = () => {
    const query = (S.mpQuery || "").toLowerCase().trim();
    const rows = [...document.querySelectorAll(".mp-model-row")];
    let shown = 0;
    for (const row of rows) {
      const filter = S.mpFilter || "all";
      const visible = (!query || row.dataset.search.includes(query)) &&
        (filter === "all" || (filter === "ready" && row.dataset.status === "ready") ||
          (filter === "missing" && row.dataset.status !== "ready") ||
          (filter === "optional" && row.dataset.optional === "1"));
      row.hidden = !visible;
      if (visible) shown++;
    }
    const empty = document.querySelector(".mp-no-results");
    if (empty) empty.hidden = shown > 0;
  };
  search.addEventListener("input", () => { S.mpQuery = search.value; applyFilter(); });
  const filters = [["all","All"], ["ready","Ready"], ["missing","Missing"], ["optional","Optional"]];
  for (const [key,label] of filters) filterHost.append(el("button", {
    class: "mp-filter" + ((S.mpFilter || "all") === key ? " active" : ""),
    "aria-pressed": (S.mpFilter || "all") === key ? "true" : "false", text: label, onclick: () => {
      S.mpFilter = key; renderPrefs();
    }}));
  filterRow.append(search);
  return el("section", {class: "mp-workspace"}, top, el("div", {class: "mp-toolbar-filter"}, filterRow),
    summary, searchStatus);
}
function mpFamilyCard(f) {
  const body = el("div", {class: "mp-model-list"});
  for (const [key,m] of mpModels(f)) body.append(mpRow(f, key, m));
  body.append(el("div", {class: "mp-no-results", text: "No matching model files.", hidden: true}));
  const fetchBox = fetchRow(f);
  const panel = fetchBox ? el("details", {class: "mp-download-panel"},
    el("summary", {text: "Download missing models · configuration & progress"}),
    fetchBox) : null;
  return el("div", {class: "section-card mp-library-card", "data-model-family": f.key},
    el("div", {class: "mp-library-caption"},
      el("b", {text: f.display_name}),
      el("span", {text: "Choose or clear a file path, or open its official download page. Nothing is deleted here."})),
    body, mpOptionalPanel(f), panel);
}
// Search libraries on the Fizgig computer, not on the browser's host.
function modelRootsCard() {
  const list = el("div"), status = el("div", {class: "field-hint"});
  const field = el("input", {type: "text", class: "w-path",
    placeholder: "Model library or application folder on the Fizgig computer"});
  S.modelSearchMessage = status;
  async function save(next) {
    try {
      await api("/api/prefs", {changes: {model_search_roots: next}});
      S.modelRoots = (await api("/api/prefs")).model_search_roots || [];
      draw();
      status.textContent = "Model search folders saved.";
    } catch (e) { status.textContent = e.message; }
  }
  const add = () => {
    const p = field.value.trim();
    if (!p) return;
    if (!(S.modelRoots || []).includes(p)) save([...(S.modelRoots || []), p]);
    else status.textContent = "This folder is already listed.";
    field.value = "";
  };
  function draw() {
    list.replaceChildren();
    const roots = S.modelRoots || [];
    if (!roots.length) list.append(el("div", {class: "field-hint",
      text: "No custom folders yet. Fizgig also checks the Hugging Face cache, the Stability Matrix library and its own models folder."}));
    roots.forEach((path, i) => {
      const move = (other) => {
        const next = roots.slice();
        [next[i], next[other]] = [next[other], next[i]];
        save(next);
      };
      list.append(el("div", {class: "field-row"},
        el("span", {class: "field-hint", style: "overflow-wrap:anywhere;flex:1", text: path}),
        el("button", {class: "btn btn-sm", text: "↑", title: "Search earlier", disabled: i === 0,
          onclick: () => move(i - 1)}),
        el("button", {class: "btn btn-sm", text: "↓", title: "Search later", disabled: i === roots.length - 1,
          onclick: () => move(i + 1)}),
        el("button", {class: "btn btn-sm", text: "Remove", onclick: () => save(roots.filter((_, n) => n !== i))})));
    });
  }
  draw();
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Model search folders"})),
    el("div", {class: "card-desc", text: "Add ComfyUI, SwarmUI, A1111/Forge, Fooocus, Stability Matrix or any model directory. " +
      "These folders are searched in order. Existing model choices win. Search never downloads."}),
    el("div", {class: "card-content"}, list,
      el("div", {class: "field-row"}, field,
        el("button", {class: "btn", text: "Choose…", onclick: () =>
          pick(field.value, true, (path) => { field.value = path; add(); })}),
        el("button", {class: "btn", text: "Add folder", onclick: add})),
      el("div", {class: "field-row"},
        el("button", {class: "btn btn-accent", text: "Search all models locally",
          onclick: () => startModelSearch("")})), status));
}

function paintModelSearch() {
  const st = S.modelSearch || {};
  const msg = st.state === "running"
    ? "Searching " + st.processed + " of " + st.total + " model files…"
    : st.state === "done" ? "Found " + st.found + ", kept " + st.kept +
        " selections, missing " + st.missing + ", ambiguous " + st.ambiguous +
        ". No model files downloaded."
      : st.state === "failed" ? "Search failed: " + st.error : "";
  if (S.modelSearchMessage) S.modelSearchMessage.textContent = msg;
  if (S.mpSearchStatus) S.mpSearchStatus.textContent = msg;
  for (const [key, row] of Object.entries(S.fetchRows || {})) {
    if (row.searchStatus) row.searchStatus.textContent = st.family === key ? msg : "";
  }
}

async function pollModelSearch() {
  clearTimeout(S.modelSearchTimer);
  const was = S.modelSearch?.state;
  try { S.modelSearch = await api("/api/models/discover"); } catch (e) { return; }
  if (was === "running" && S.modelSearch.state === "done") {
    const pr = await api("/api/prefs");
    S.prefs = pr.models || {};
    S.modelInventory = pr.model_inventory || {files: {}, summary: {}};
    S.modelRoots = pr.model_search_roots || [];
    renderPrefs();
    checkModels();
    refreshStateSoon();
    captionModelsChanged();
  }
  paintModelSearch();
  if (S.modelSearch.state === "running") S.modelSearchTimer = setTimeout(pollModelSearch, 1000);
}

async function startModelSearch(family, choices) {
  try {
    await api("/api/models/discover", {family, choices});
    S.modelSearch = {state: "running", family, processed: 0, total: 0};
    paintModelSearch();
    pollModelSearch();
  } catch (e) {
    S.modelSearch = {state: "failed", family, error: e.message};
    paintModelSearch();
  }
}

// New downloads can live in the model subfolders understood by ComfyUI or SwarmUI.
// Existing path selections are never moved, and the chosen library must be on Fizgig's machine.
function modelDownloadCard() {
  const select = el("select", {class: "w-combo"},
    el("option", {value: "fizgig", text: "Fizgig's own model folder"}),
    el("option", {value: "comfyui", text: "ComfyUI folder names"}),
    el("option", {value: "swarmui", text: "SwarmUI folder names"}));
  select.value = S.modelDownloadLayout || "fizgig";
  const root = el("input", {type: "text", class: "w-path", value: S.modelDownloadRoot || "",
    placeholder: "The ComfyUI / SwarmUI application folder or its models folder"});
  const status = el("div", {class: "field-hint",
    text: "New downloads only. Existing paths stay where they are; no model files will be moved."});
  const rootRow = el("div", {class: "field-row"},
    el("span", {class: "field-label", text: "Library on Fizgig's computer:"}), root,
    el("button", {class: "btn", text: "Choose…",
      onclick: () => pick(root.value, true, (chosen) => { root.value = chosen; })}));
  const show = () => { rootRow.style.display = select.value === "fizgig" ? "none" : ""; };
  select.addEventListener("change", show);
  show();
  const save = async () => {
    try {
      await api("/api/prefs", {changes: {model_download_layout: select.value, model_download_root: root.value.trim()}});
      const next = await api("/api/prefs");
      S.modelDownloadLayout = next.model_download_layout || "fizgig";
      S.modelDownloadRoot = next.model_download_root || "";
      root.value = S.modelDownloadRoot;
      status.textContent = "Saved. The next missing model downloads into this library.";
    } catch (e) { status.textContent = e.message; }
  };
  return el("div", {class: "section-card"},
    el("div", {class: "card-title"}, el("span", {text: "Where to save new models"})),
    el("div", {class: "card-desc", text: "Put missing downloaded model files directly into the appropriate " +
      "diffusion_models, text_encoders, VAE and LoRA subfolders. Select your actual SwarmUI ModelRoot " +
      "if it is customized. Fizgig still prefers already configured model files."}),
    el("div", {class: "card-content"},
      el("div", {class: "field-row"}, el("span", {class: "field-label", text: "Folder layout:"}), select),
      rootRow, el("div", {class: "field-row"},
        el("button", {class: "btn btn-accent", text: "Save download location", onclick: save})), status));
}

// One small registration seam in the upstream Preferences renderer. If this script fails to
// load, Fizgig falls back to its normal built-in model-file cards.
window.FizgigForkModelLibrary = Object.freeze({
  initialize(pr) {
    S.modelInventory = pr.model_inventory || {files: {}, summary: {}};
    S.modelRoots = pr.model_search_roots || [];
    S.modelDownloadLayout = pr.model_download_layout || "fizgig";
    S.modelDownloadRoot = pr.model_download_root || "";
  },
  onFamilyChanged(key) { S.mpFamily = key; },
  mount(host) {
    S.mpRows = {};
    const families = S.schema.families.filter((f) => !f.prefs_skip);
    const selected = families.find((f) => f.key === S.mpFamily) ||
      families.find((f) => f.key === S.family) || families[0];
    if (selected) {
      S.mpFamily = selected.key;
      host.prepend(mpToolbar(families, selected), mpFamilyCard(selected));
    }
    host.append(modelRootsCard(), modelDownloadCard());
    paintModelSearch();
  },
  poll() { return pollModelSearch(); }
});
