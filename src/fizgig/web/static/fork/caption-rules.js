// Fork-owned caption-rule controls. Loaded before app.js; callbacks access upstream's
// S, api, el, $, capFolder, capExt and loadCaptions only after the app has initialized.
// The upstream captioning flow stays available when this extension cannot load.
(() => {
  "use strict";
  function ensureState() {
    if (!S.cap.ruleConfig) S.cap.ruleConfig = {folder: "", enabled: false, rules: []};
  }

// Automatic caption corrections are saved by absolute training-folder identity on the server.
// They are separate from per-family form defaults, so Hannah's rules cannot leak to Amber's folder.
function renderCaptionRules() {
  const folder = capFolder();
  const cfg = S.cap.ruleConfig?.folder === folder ? S.cap.ruleConfig : {enabled: false, rules: []};
  const rules = cfg.rules || [];
  $("capRuleCount").textContent = rules.length + (rules.length === 1 ? " rule" : " rules");
  $("capRulesEnabled").checked = !!cfg.enabled;
  $("capRulesEnabled").disabled = !folder || !rules.length;
  $("frAddRule").disabled = !folder;
  const list = $("capRulesList");
  list.replaceChildren();
  if (!folder || !rules.length) {
    list.append(el("div", {class: "cap-rule-empty", text: !folder ?
      "Choose a training folder on Start to save its corrections." :
      "No rules yet. Find and replace once with “Also save” checked, or save a pair below."}));
    return;
  }
  rules.forEach((rule, index) => {
    const on = el("input", {type: "checkbox", "aria-label": "Enable rule " + (index + 1)});
    on.checked = rule.enabled !== false;
    on.addEventListener("change", () => {
      const next = rules.map((r, i) => i === index ? {...r, enabled: on.checked} : r);
      persistCaptionRules({...cfg, rules: next});
    });
    const edit = el("button", {class: "btn btn-sm", text: "Edit",
      title: "Copy this rule into Find & Replace so you can change it", onclick: () => {
        $("frFind").value = rule.find; $("frReplace").value = rule.replace;
        $("frWholeWord").checked = rule.whole_word !== false;
        $("frFind").focus(); $("frFind").scrollIntoView({block: "nearest"});
      }});
    const move = (delta) => {
      const target = index + delta;
      if (target < 0 || target >= rules.length) return;
      const next = rules.slice();
      [next[index], next[target]] = [next[target], next[index]];
      persistCaptionRules({...cfg, rules: next});
    };
    list.append(el("div", {class: "cap-rule-row"},
      el("label", {class: "cap-rule-enable"}, on,
        el("span", {class: "sr-only", text: "Enable rule"})),
      el("div", {class: "cap-rule-pair"},
        el("code", {text: rule.find}), el("span", {class: "cap-rule-arrow", text: "→"}),
        el("code", {text: rule.replace || "(delete match)"}),
        el("span", {class: "mp-chip", text: rule.whole_word === false ? "Any match" : "Whole words"})),
      el("div", {class: "cap-rule-actions"},
        edit,
        el("button", {class: "btn btn-sm", text: "↑", title: "Move rule earlier",
          "aria-label": "Move rule " + (index + 1) + " earlier", disabled: index === 0, onclick: () => move(-1)}),
        el("button", {class: "btn btn-sm", text: "↓", title: "Move rule later",
          "aria-label": "Move rule " + (index + 1) + " later", disabled: index === rules.length - 1,
          onclick: () => move(1)}),
        el("button", {class: "btn btn-sm", text: "Remove", title: "Remove this automatic rule only",
          onclick: () => persistCaptionRules({...cfg, rules: rules.filter((_, i) => i !== index)})}))));
  });
}

async function loadCaptionRules() {
  const folder = capFolder();
  if (!folder) {
    S.cap.ruleConfig = {folder: "", enabled: false, rules: []};
    renderCaptionRules();
    return;
  }
  try {
    const config = await api("/api/captions/rules?folder=" + encodeURIComponent(folder));
    if (capFolder() !== folder) return;
    S.cap.ruleConfig = {...config, folder};
    $("capRulesStatus").textContent = "";
  } catch (e) {
    if (capFolder() !== folder) return;
    S.cap.ruleConfig = {folder: "", enabled: false, rules: []};
    $("capRulesStatus").textContent = "Could not load rules: " + e.message;
  }
  renderCaptionRules();
}

function persistCaptionRules(next) {
  const folder = capFolder(), family = S.family;
  const status = $("capRulesStatus");
  if (!folder) { status.textContent = "Choose a training folder first."; return Promise.resolve(false); }
  const intended = {...next, folder, enabled: !!next.enabled};
  // Optimistic UI + serialized writes: rapid toggles/reordering cannot clobber
  // one another, and a previous result must not undo a more recent click.
  S.cap.ruleConfig = intended;
  renderCaptionRules();
  status.textContent = "Saving…";
  const write = async () => {
    try {
      const result = await api("/api/captions/rules", {family, folder,
        enabled: intended.enabled, rules: intended.rules});
      if (capFolder() === folder && S.cap.ruleConfig === intended) {
        S.cap.ruleConfig = {...result, folder};
        status.textContent = "Saved for this training folder.";
        renderCaptionRules();
      }
      return true;
    } catch (e) {
      if (capFolder() === folder) status.textContent = "Rules not saved: " + e.message;
      return false;
    }
  };
  S.cap.ruleSave = (S.cap.ruleSave || Promise.resolve()).then(write, write);
  return S.cap.ruleSave;
}

async function rememberCaptionRule() {
  const find = $("frFind").value.trim(), replace = $("frReplace").value;
  const status = $("capRulesStatus");
  if (!find) { status.textContent = "Enter text to find first."; return false; }
  const folder = capFolder();
  if (!folder) { status.textContent = "Choose a training folder first."; return false; }
  if (S.cap.ruleConfig?.folder !== folder) await loadCaptionRules();
  const cfg = S.cap.ruleConfig;
  if (cfg.folder !== folder) return false;
  const whole_word = $("frWholeWord").checked;
  const item = {find, replace, whole_word, enabled: true};
  const rules = (cfg.rules || []).slice();
  const existing = rules.findIndex((r) => r.find.toLowerCase() === find.toLowerCase() &&
    r.whole_word !== false === whole_word);
  if (existing >= 0) rules[existing] = item;
  else rules.push(item);
  const saved = await persistCaptionRules({rules, enabled: cfg.enabled || !cfg.rules.length});
  if (saved) $("frStatus").textContent += " Saved to automatic corrections.";
  return saved;
}

// Show all of a caption, with the same literal match spans that the server will replace.
// Build text nodes and marks, never HTML from user-provided caption contents.
function captionMarkedSegments(segments, matchClass, label) {
  const line = el("div", {class: "field-hint caption-find-line"}, el("strong", {text: label}));
  for (const [part, hit] of segments) {
    line.append(hit ? el("mark", {class: matchClass, text: part}) : document.createTextNode(part));
  }
  return line;
}

async function forkFindReplace(apply) {
  const host = $("frChanges");
  try {
    const r = await api("/api/captions/replace", {family: S.family, folder: capFolder(), find: $("frFind").value,
      replace: $("frReplace").value, whole_word: $("frWholeWord").checked, apply, ext: capExt()});
    host.replaceChildren();
    const n = r.changes.reduce((count, c) => count + (c.match_count || 0), 0);
    $("frStatus").textContent = r.changes.length ? (apply
      ? "Replaced " + n + " matches in " + r.changes.length + " captions."
      : n + " matches across " + r.changes.length + " captions would change:"
    ) : "No caption contains that.";
    if (!apply) for (const c of r.changes.slice(0, 100)) host.append(el("div", {class: "checkpoint-row"},
      el("span", {class: "checkpoint-name", text: c.name}),
      captionMarkedSegments(c.before_segments || [[c.old, false]], "caption-find-before", "Before: "),
      captionMarkedSegments(c.after_segments || [[c.new, false]], "caption-find-after", "After: ")));
    if (apply) {
      loadCaptions();
      if ($("frRemember").checked) await rememberCaptionRule();
    }
  } catch (e) { $("frStatus").textContent = e.message; }
}

  window.FizgigForkCaptionRules = Object.freeze({
    initialize() { ensureState(); renderCaptionRules(); },
    async load() { ensureState(); await loadCaptionRules(); },
    replace: forkFindReplace,
    async beforeCaptioning() {
      return !S.cap.ruleSave || await S.cap.ruleSave;
    },
    wire() {
      ensureState();
      renderCaptionRules();
      $("frAddRule").addEventListener("click", rememberCaptionRule);
      $("capRulesEnabled").addEventListener("change", () => {
        const cfg = S.cap.ruleConfig;
        if (cfg?.folder === capFolder())
          persistCaptionRules({...cfg, enabled: $("capRulesEnabled").checked});
      });
    }
  });
})();
