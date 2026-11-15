// Functional contract for the fork-only caption-rule extension; executes its real script.
// No browser automation or model weights needed. Tests persistence, model-neutral folder
// isolation, replacement preview, save gating and negative controls against wrong folders.
"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const path = "src/fizgig/web/static/fork/caption-rules.js";
const source = fs.readFileSync(path, "utf8");
const elements = new Map();
function el(tag, attrs, ...kids) {
  const props = attrs && typeof attrs === "object" && !Array.isArray(attrs) ? attrs : {};
  const node = {
    tag, props, children: [], value: props.value || "", textContent: props.text || "",
    checked:false, disabled:false, handlers:new Map(),
    append(...xs) { this.children.push(...xs.flat(Infinity).filter(x => x != null)); },
    replaceChildren(...xs) { this.children=[]; this.append(...xs); },
    addEventListener(event, handler) { this.handlers.set(event, handler); },
    focus() {}, scrollIntoView() {}
  };
  node.append(...kids);
  return node;
}
function $(name) {
  if (!elements.has(name)) elements.set(name, el("div", {}));
  return elements.get(name);
}
function walk(n, fn, out=[]) {
  if(!n || typeof n !== "object") return out;
  if(fn(n))out.push(n);
  for(const c of n.children || []) walk(c, fn, out);
  return out;
}
let folder="/dataset/A", failSave=false, captionReloads=0;
const storage = new Map([
  ["/dataset/A", {enabled:true,rules:[{find:"boy",replace:"girl",whole_word:true,enabled:true}]}],
  ["/dataset/B", {enabled:false,rules:[]}]
]);
const calls=[];
async function api(path, body) {
  calls.push({path, body});
  if(path.startsWith("/api/captions/rules?folder=")) {
    const key=decodeURIComponent(path.split("folder=")[1]);
    return JSON.parse(JSON.stringify(storage.get(key) || {enabled:false,rules:[]}));
  }
  if(path==="/api/captions/rules") {
    if(failSave) throw Error("forced save failure");
    storage.set(body.folder, {enabled:body.enabled,rules:body.rules});
    return {enabled:body.enabled,rules:body.rules};
  }
  if(path==="/api/captions/replace") {
    assert.equal(body.folder, folder);
    assert.equal(body.family, "minimax_image");
    assert.equal(body.whole_word, true);
    return {changes:[{
      name:"photo.jpg",old:"a boy",new:"a girl",match_count:1,
      before_segments:[["a ",false],["boy",true]],
      after_segments:[["a ",false],["girl",true]]
    }]};
  }
  throw Error("Unexpected API call "+path);
}
const ctx={
  S:{family:"minimax_image",cap:{page:0,items:[]}},
  window:{}, $, el, api, capFolder:()=>folder, capExt:()=>".txt",
  loadCaptions:()=>{captionReloads++},
  document:{createTextNode:text=>({tag:"text",textContent:text})},
  console
};
vm.createContext(ctx);
vm.runInContext(source,ctx,{timeout:10000});
(async () => {
const extension=ctx.window.FizgigForkCaptionRules;
assert(extension && Object.isFrozen(extension));
extension.wire();
await extension.load();
assert.equal(ctx.S.cap.ruleConfig.folder,"/dataset/A");
assert.equal(ctx.S.cap.ruleConfig.rules.length,1);
assert.equal($("capRuleCount").textContent,"1 rule");
assert.equal($("capRulesEnabled").checked,true);
assert.equal(walk($("capRulesList"), n=>n.props?.class==="cap-rule-row").length,1);

$("frFind").value="cat"; $("frReplace").value="dog";
$("frWholeWord").checked=true;
$("frRemember").checked=true;
await extension.replace(false);
assert.equal($("frChanges").children.length,1);
assert($("frStatus").textContent.includes("1 matches"));
assert.equal(calls.filter(c=>c.path==="/api/captions/rules").length,0);
await extension.replace(true);
assert.equal(captionReloads,1);
assert.equal(storage.get("/dataset/A").rules.length,2);
assert.equal(storage.get("/dataset/A").rules[1].find,"cat");
assert.equal(await extension.beforeCaptioning(),true);

// Dataset switch must never import the first folder's automatic rules.
folder="/dataset/B";
await extension.load();
assert.equal(ctx.S.cap.ruleConfig.folder,folder);
assert.equal(ctx.S.cap.ruleConfig.rules.length,0);
assert.equal($("capRulesEnabled").checked,false);
assert.equal(storage.get("/dataset/A").rules.length,2);
folder="/dataset/A";
await extension.load();
assert.equal(ctx.S.cap.ruleConfig.rules.length,2);

// Negative control: remote persistence failure must block starting AI captioning.
failSave=true;
$("capRulesEnabled").checked=false;
const onChange=$("capRulesEnabled").handlers.get("change");
assert.equal(typeof onChange,"function");
onChange();
assert.equal(await extension.beforeCaptioning(),false);
assert($("capRulesStatus").textContent.includes("Rules not saved"));
assert.equal(storage.get("/dataset/A").enabled,true, "Failed write cannot change server state");

console.log("PASS: caption rules load, save, preview, folder isolation, failing-save gate");
})().catch(e => { console.error(e); process.exitCode=1; });
