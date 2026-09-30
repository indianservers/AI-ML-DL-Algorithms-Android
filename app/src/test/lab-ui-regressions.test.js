const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// Exercise the production playback controllers with deterministic timers.
function harness() {
  const timers = new Map(), events = {}, elements = {};
  let next = 1;
  for (const id of ['run', 'step', 'reset', 'speed', 'speedValue', 'train'])
    elements[id] = {dataset: {label: id}, disabled: false, textContent: ''};
  const context = vm.createContext({
    window: {}, Gridworld: {actions: []}, matchMedia: () => ({matches: true}),
    document: {hidden: false, getElementById: id => elements[id], addEventListener: (name, fn) => {events[name] = fn;}},
    addEventListener: (name, fn) => {events[name] = fn;},
    setInterval: fn => {const id = next++; timers.set(id, fn); return id;},
    clearInterval: id => timers.delete(id)
  });
  for (const phase of [2, 3, 4]) vm.runInContext(fs.readFileSync(path.join(__dirname, `../main/assets/phase${phase}_labs/common.js`), 'utf8'), context);
  return {context, timers, events, elements};
}
for (const [name, expression] of [['Phase 2', 'PhaseTwoUI.driver'], ['Phase 3', 'PhaseThreeUI.driver'], ['Phase 4', 'PhaseFourUI.controller']]) {
  test(`${name}: one timer, one manual step, speed replacement, reset and page cleanup`, () => {
    const h = harness();
    h.context.create = () => {let n = 0; const snapshot = () => ({n, completed: n, done: n >= 3}); return {snapshot, step: () => {n++; return snapshot();}};};
    h.context.render = () => {};
    const driver = vm.runInContext(`${expression}(create, render)`, h.context);
    driver.bind();
    h.elements.run.onclick(); assert.equal(h.timers.size, 1);
    h.elements.run.onclick(); assert.equal(h.timers.size, 0);
    h.elements.run.onclick(); h.elements.speed.oninput({target: {value: '4'}}); assert.equal(h.timers.size, 1);
    h.elements.step.onclick(); assert.equal(driver.state.n, 1); assert.equal(h.timers.size, 0);
    h.elements.run.onclick(); h.elements.reset.onclick(); assert.equal(driver.state.n, 0); assert.equal(h.timers.size, 0);
    h.elements.run.onclick(); h.events.pagehide(); assert.equal(h.timers.size, 0);
    h.elements.run.onclick(); h.context.document.hidden = true; h.events.visibilitychange(); assert.equal(h.timers.size, 0);
    h.elements.step.onclick(); h.elements.step.onclick(); h.elements.step.onclick();
    assert.equal(driver.state.n, 3); assert.equal(h.elements.run.disabled, true); assert.equal(h.elements.step.disabled, true);
    h.elements.reset.onclick(); assert.equal(h.elements.run.disabled, false); assert.equal(h.elements.step.disabled, false);
  });
}
test('Charts handle empty/constant/nonfinite series and bound hover targets', () => {
  const h = harness(), chart = vm.runInContext('PhaseTwoUI.chart', h.context);
  assert.match(chart([{label: 'Empty', color: '#000', data: []}]), /Chart updates/);
  const svg = chart([{label: 'Empty', color: '#000', data: []}, {label: 'Reward', color: '#00f', data: [2, NaN, 2]}]);
  assert.doesNotMatch(svg, /NaN|undefined/);
  assert.match(svg, /Reward · step 2: 2/);
  const long = chart([{label: 'History', color: '#000', data: Array.from({length: 5000}, (_, i) => i)}]);
  assert.ok((long.match(/class="chart-hit"/g) || []).length <= 81);
});
test('Log chart clamps the plot and labels real iteration coordinates', () => {
  const chart = vm.runInContext('PhaseTwoUI.chart', harness().context);
  const svg = chart([{label: 'Probability', color: '#000', data: [1, 0]}], {log: true, min: 1e-4, max: 1, height: 160, xLabel: 'Iteration', xStep: 2000});
  assert.match(svg, /iteration 2000: 0/);
  for (const [, y] of svg.matchAll(/cy="([^"]+)"/g)) assert.ok(+y >= 19 && +y <= 136);
  assert.match(svg, />2000<\/text>/);
});
test('Keyboard activation dispatches a click on SVG elements without HTMLElement.click', () => {
  const source = fs.readFileSync(path.join(__dirname, '../main/assets/lab-polish.js'), 'utf8');
  const registration = source.split('\n').find(line => line.includes("document.addEventListener('keydown'"));
  let handler, clicks = 0;
  const svgNode = {onclick: () => {}, dispatchEvent: event => {assert.equal(event.type, 'click'); clicks++;}};
  vm.runInNewContext(registration, {document: {addEventListener: (_, fn) => {handler = fn;}}, hideTip: () => {}, MouseEvent: class {constructor(type) {this.type = type;}}});
  for (const key of ['Enter', ' ']) {
    let prevented = false;
    handler({key, target: {matches: () => false, closest: () => svgNode}, preventDefault: () => {prevented = true;}});
    assert.equal(prevented, true);
  }
  assert.equal(clicks, 2);
  handler({key: ' ', defaultPrevented: true});
  assert.equal(clicks, 2, 'existing node handlers are not invoked twice');
});
