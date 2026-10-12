"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  node.append(...children);
  return node;
};
const svg = (tag, attrs = {}) => {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
};

const SCALES = ["famrel", "freetime", "goout", "Dalc", "Walc", "health"];
const GROUP_ORDER = ["Student and family", "Study and support", "Lifestyle and attendance", "Prior grades"];
const state = { meta: null, student: {}, later: false, seq: 0, timer: null, lastCsv: "" };
const xOf = (g) => 20 + g * 18; // ruler: 0–20 mapped onto 20–380

async function api(path, options) {
  const res = await fetch(path, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw Object.assign(new Error("request failed"), { data });
  return data;
}

/* ------------------------------------------------------------ form */
function isYesNo(f) { return f.kind === "choice" && f.options.length === 2 && f.options.some(([c]) => c === "yes"); }

function radioGroup(f, cls, makeSpan) {
  const wrap = el("div", { class: cls, role: "radiogroup", "aria-label": f.label });
  f.options.forEach(([code, text], i) => {
    const input = el("input", { type: "radio", name: f.name, id: `f-${f.name}-${i}`, value: String(i) });
    if (code === state.student[f.name]) input.checked = true;
    input.addEventListener("change", () => { state.student[f.name] = code; schedule(); });
    wrap.append(el("label", { for: input.id, title: text }, input, makeSpan(code, text)));
  });
  return wrap;
}

function control(f) {
  if (f.kind === "choice" && SCALES.includes(f.name)) {
    const dots = radioGroup(f, "dots", (code) => el("span", { text: String(code) }));
    const first = f.options[0][1], last = f.options[f.options.length - 1][1];
    return el("div", { class: "scale" }, dots, el("div", { class: "ends" }, el("span", { text: first }), el("span", { text: last })));
  }
  if (f.kind === "choice") return radioGroup(f, "pills", (code, text) => el("span", { text }));
  if (f.max - f.min <= 4) {
    const numeric = { ...f, options: Array.from({ length: f.max - f.min + 1 }, (_, i) => [f.min + i, String(f.min + i)]) };
    return radioGroup(numeric, "pills", (code, text) => el("span", { text }));
  }
  const input = el("input", { type: "number", id: `f-${f.name}`, min: f.min, max: f.max, step: 1, inputmode: "numeric", "aria-label": f.label });
  input.value = state.student[f.name];
  const set = (v) => {
    const n = Math.round(Number(v));
    const clamped = Number.isFinite(n) ? Math.min(f.max, Math.max(f.min, n)) : f.min;
    input.value = clamped; state.student[f.name] = clamped; schedule();
  };
  input.addEventListener("change", () => set(input.value));
  input.addEventListener("input", () => { if (input.value !== "") { state.student[f.name] = Number(input.value); schedule(); } });
  const minus = el("button", { type: "button", "aria-label": `Decrease ${f.label}`, text: "−" });
  const plus = el("button", { type: "button", "aria-label": `Increase ${f.label}`, text: "+" });
  minus.addEventListener("click", () => set(Number(input.value) - 1));
  plus.addEventListener("click", () => set(Number(input.value) + 1));
  return el("div", {}, el("span", { class: "stepper" }, minus, input, plus), el("span", { class: "range-note", text: `${f.min}–${f.max}` }));
}

function buildForm() {
  const form = $("#form");
  form.replaceChildren();
  const byGroup = new Map(GROUP_ORDER.map((g) => [g, []]));
  state.meta.fields.forEach((f) => byGroup.get(f.group).push(f));
  for (const [group, fields] of byGroup) {
    const set = el("fieldset", { class: "group", id: `group-${group.split(" ")[0].toLowerCase()}` }, el("legend", { class: "eyebrow", text: group }));
    if (group === "Prior grades") set.hidden = !state.later;
    const flags = fields.filter(isYesNo);
    fields.filter((f) => !isYesNo(f)).forEach((f) => {
      set.append(el("div", { class: "row" }, el("span", { class: "label", text: f.label }), control(f)));
    });
    if (flags.length) {
      const wrap = el("div", { class: "flags", role: "group", "aria-label": "Yes or no" });
      flags.forEach((f) => {
        const input = el("input", { type: "checkbox", id: `f-${f.name}` });
        input.checked = state.student[f.name] === "yes";
        input.addEventListener("change", () => { state.student[f.name] = input.checked ? "yes" : "no"; schedule(); });
        wrap.append(el("label", { for: input.id }, input, el("span", { text: f.label })));
      });
      set.append(wrap, el("p", { class: "flags-hint", text: "Highlighted means yes." }));
    }
    form.append(set);
  }
}

/* ------------------------------------------------------------ result */
function drawTicks() {
  const g = $("#ticks");
  for (let v = 0; v <= 20; v++) {
    const major = v % 5 === 0, x = xOf(v);
    g.append(svg("line", { class: `tick${major ? " major" : ""}`, x1: x, x2: x, y1: 40, y2: major ? 48 : 44 }));
    if (major && v !== 10) { const t = svg("text", { class: "num", x, y: 66 }); t.textContent = v; g.append(t); }
  }
}

function currentModel() { return state.meta.models[state.later ? 1 : 0]; }

function render(result) {
  const m = currentModel(), t = m.test, sheet = $(".sheet"), mini = $(".mini");
  const p = result.predicted, risk = result.flag;
  $("#grade").textContent = p.toFixed(2);
  const status = $("#status");
  status.className = `chip ${risk ? "risk" : "ok"}`;
  status.textContent = risk ? "Potentially at risk" : "Not flagged";
  sheet.classList.toggle("is-risk", risk);
  mini.classList.toggle("is-risk", risk);
  $("#mini-grade").textContent = p.toFixed(2);
  $("#mini-status").textContent = risk ? "at risk" : "not flagged";

  const lo = Math.max(0, p - t.MAE), hi = Math.min(20, p + t.MAE);
  $("#band").style.transform = `translateX(${xOf(lo)}px)`;
  $("#band rect").setAttribute("width", ((hi - lo) * 18).toFixed(1));
  $("#marker").style.transform = `translateX(${xOf(Math.min(20, Math.max(0, p)))}px)`;
  $("#ruler").setAttribute("aria-label", `Predicted ${p.toFixed(2)} on a 0 to 20 scale, pass mark 10`);
  $("#band-text").textContent = `±${t.MAE.toFixed(2)}: the model's average error on 79 held-out students. Not a confidence interval.`;

  $("#advice").textContent = risk
    ? `Below ${state.meta.threshold}. Worth a supportive conversation, alongside current classwork and the student's own view.`
    : `${state.meta.threshold} or above, so not flagged. That does not mean the student needs no help.`;
  $("#model-name").textContent = `${state.later ? "Later-stage" : "Main"} · Random Forest`;
  $("#model-recall").textContent = `${t.TP} of ${t.TP + t.FN}`;
  $("#model-extra").textContent = String(t.FP);
  const warn = $("#warn");
  warn.hidden = !result.absencesAboveObserved;
  warn.textContent = `Absences above ${state.meta.observedMaxAbsences} are beyond anything in the training data, so treat this estimate with extra care.`;
}

function renderErrors(errors) {
  $("#status").className = "chip";
  $("#status").textContent = "Check inputs";
  $("#advice").textContent = errors.join(". ") + ".";
}

function schedule() {
  clearTimeout(state.timer);
  state.timer = setTimeout(predict, 120);
}

async function predict() {
  const seq = ++state.seq;
  try {
    const result = await api("/api/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ student: state.student, priorGrades: state.later }),
    });
    if (seq === state.seq) render(result);
  } catch (error) {
    if (seq === state.seq) renderErrors(error.data?.errors || ["The app server is not responding. Restart it and reload this page"]);
  }
}

/* ------------------------------------------------------------ class list */
function csvTemplate() {
  const names = state.meta.fields.map((f) => f.name);
  const example = state.meta.fields.map((f) => (f.group === "Prior grades" ? "" : f.default));
  const later = state.meta.fields.map((f) => (f.name === "G1" ? 8 : f.name === "G2" ? 7 : f.default));
  return [names, example, later].map((r) => r.join(",")).join("\n") + "\n";
}

function download(name, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  const a = el("a", { href: url, download: name });
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function runBatch(file) {
  const summary = $("#class-summary"), table = $("#class-table"), body = $("tbody", table);
  summary.textContent = `Reading ${file.name}…`;
  try {
    const data = await api("/api/batch", { method: "POST", headers: { "Content-Type": "text/csv" }, body: await file.text() });
    body.replaceChildren(...data.rows.map((r) => {
      const chip = el("span", { class: `chip ${r.status === "Potentially at risk" ? "risk" : r.status === "Not flagged" ? "ok" : ""}`, text: r.status });
      return el("tr", {},
        el("td", { class: "num", text: String(r.row) }),
        el("td", { text: r.model.replace(" model", "") }),
        el("td", { class: "num", text: r.predicted_G3 == null ? "–" : r.predicted_G3.toFixed(2) }),
        el("td", {}, chip),
        el("td", { class: "notes", text: r.notes || "" }));
    }));
    const risk = data.rows.filter((r) => r.status === "Potentially at risk").length;
    const invalid = data.rows.filter((r) => r.predicted_G3 == null).length;
    summary.replaceChildren(
      el("strong", { text: `${data.rows.length} students` }), ` · ${risk} potentially at risk · ${invalid} invalid`,
      data.ignoredTarget ? " · the G3 column was ignored, because it is what the model predicts." : "");
    table.hidden = false;
    state.lastCsv = data.csv;
    $("#download").hidden = false;
  } catch (error) {
    summary.textContent = (error.data?.errors || ["Could not read that file"]).join(" ");
    table.hidden = true; $("#download").hidden = true;
  }
}

/* ------------------------------------------------------------ models */
function buildModels() {
  const [a, b] = state.meta.models;
  const nice = (name) => name.replace(/Regressor$/, "").replace(/([a-z])([A-Z])/g, "$1 $2");
  const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);
  const rows = [
    ["Uses G1 and G2", "No", "Yes"],
    ["Estimator", nice(a.estimator), nice(b.estimator)],
    ["Inputs", a.inputs, b.inputs],
    ["Absences", a.logAbsences ? "log1p" : "original", b.logAbsences ? "log1p" : "original"],
    ["Training-CV MAE", a.cvMae.toFixed(3), b.cvMae.toFixed(3)],
    ["Test MAE", a.test.MAE.toFixed(3), b.test.MAE.toFixed(3)],
    ["Test RMSE", a.test.RMSE.toFixed(3), b.test.RMSE.toFixed(3)],
    ["Test R²", a.test.R2.toFixed(3), b.test.R2.toFixed(3)],
    ["Below-10 students found", `${a.test.TP} of ${a.test.TP + a.test.FN} (${pct(a.test.Recall)})`, `${b.test.TP} of ${b.test.TP + b.test.FN} (${pct(b.test.Recall)})`],
    ["Extra flags", a.test.FP, b.test.FP],
  ];
  $("#models-table").replaceChildren(
    el("thead", {}, el("tr", {}, el("th", { text: "" }), el("th", { text: "Main model" }), el("th", { text: "Later-stage model" }))),
    el("tbody", {}, ...rows.map(([k, x, y]) => el("tr", {}, el("td", { text: k }), el("td", { text: String(x) }), el("td", { text: String(y) })))));
}

/* ------------------------------------------------------------ views */
function showView(id) {
  document.querySelectorAll('[role="tab"]').forEach((tab) => {
    const on = tab.getAttribute("aria-controls") === id;
    tab.setAttribute("aria-selected", String(on));
    $(`#${tab.getAttribute("aria-controls")}`).hidden = !on;
  });
  $(".mini").dataset.hidden = String(id !== "view-student");
  window.scrollTo(0, 0);
}

async function init() {
  try {
    state.meta = await api("/api/meta");
  } catch (_) {
    $("#advice").textContent = "Start the app with python app/server.py, then reload this page.";
    return;
  }
  state.meta.fields.forEach((f) => { state.student[f.name] = f.default; });
  drawTicks();
  buildForm();
  buildModels();
  document.querySelectorAll('input[name="stage"]').forEach((r) => r.addEventListener("change", () => {
    state.later = $("#stage-later").checked;
    $("#group-prior").hidden = !state.later;
    predict();
  }));
  document.querySelectorAll('[role="tab"]').forEach((tab) => tab.addEventListener("click", () => showView(tab.getAttribute("aria-controls"))));
  $("#template").addEventListener("click", () => download("edupredict_template.csv", csvTemplate()));
  $("#download").addEventListener("click", () => download("edupredict_predictions.csv", state.lastCsv));
  $("#csv").addEventListener("change", (e) => { const f = e.target.files[0]; if (f) runBatch(f); e.target.value = ""; });
  showView("view-student");
  // The floating grade pill (phones only) shows once the result sheet has scrolled away.
  if ("IntersectionObserver" in window) {
    new IntersectionObserver(([entry]) => $(".mini").classList.toggle("away", !entry.isIntersecting)).observe($(".sheet"));
  }
  predict();
}

init();
