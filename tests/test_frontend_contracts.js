// Pure Node VM contract tests. No browser, network, OCR, or mouse APIs run.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'Head', 'Head_web.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Parse the complete shipped script, without executing it.
function functionSource(name) {
  const start = script.indexOf(`function ${name}(`);
  assert.ok(start >= 0, name);
  let open = script.indexOf('{', start), depth = 1, end = open + 1;
  while (depth) {
    if (script[end] === '{') depth++;
    else if (script[end] === '}') depth--;
    end++;
  }
  return script.slice(start, end);
}
function context() {
  const nodes = new Map();
  function node(id) {
    if (!nodes.has(id)) {
      const classes = new Set();
      nodes.set(id, {id, style: {}, disabled: false, innerHTML: '', innerText: '', value: '200',
        classList: {add: (...v) => v.forEach(x => classes.add(x)), remove: (...v) => v.forEach(x => classes.delete(x)), contains: x => classes.has(x)}});
    }
    return nodes.get(id);
  }
  const c = {GameState: {cells: [], rows: 16, cols: 10, totalCells: 160, mode: 'god', execState: 'disabled', bestSolution: null},
    RAW_DATA: null, document: {getElementById: node, querySelectorAll: () => ['classic','omni','god','complete'].map(m => node(`opt-${m}`))},
    showToast: (title, message) => c.toasts.push([title, message]), toasts: [], renderGrid: () => {}, connectBrain: () => {},
    playSolution: () => {}, openEditor: () => {}, triggerHydraEffect: () => {}, checkCaliStatus: () => {}, setTimeout: () => {}, WebSocket: {OPEN: 1}};
  vm.createContext(c);
  for (const name of ['setExecState','initGame','setMode','startSolver','handleServerMsg']) vm.runInContext(functionSource(name), c);
  c.node = node;
  return c;
}
const plain = x => JSON.parse(JSON.stringify(x));
let checks = 0;
{
  const c = context();
  c.initGame('190\n280');
  assert.equal(c.GameState.rows, 2); assert.equal(c.GameState.cols, 3);
  assert.deepEqual(plain(c.GameState.cells.map(v => [v.r,v.c,v.val])), [[0,0,1],[0,1,9],[0,2,0],[1,0,2],[1,1,8],[1,2,0]]);
  const sent = []; c.GameState.socket = {send: text => sent.push(JSON.parse(text)), readyState: 1};
  c.RAW_DATA = '190\n280'; c.node('thread-slider').value = '2'; c.setMode('complete');
  c.GameState.bestSolution = [[9,9,9,9]]; c.startSolver();
  assert.equal(c.GameState.bestSolution, null);
  assert.deepEqual(sent[0], {cmd:'START', rows:2, cols:3, map:[1,1,1,1,1,1], vals:[1,9,0,2,8,0], beamWidth:200, threads:2, mode:'complete'});
  checks++;
}
{
  const c = context(); const raw = '19'.repeat(80);
  c.handleServerMsg({type:'OCR_RESULT',raw_data:raw,matrix:Array.from({length:16},()=>[1,9,1,9,1,9,1,9,1,9])});
  assert.equal(c.GameState.rows,16);assert.equal(c.GameState.cols,10);assert.equal(c.GameState.mode,'god');
  assert.equal(c.GameState.cells[10].r,1);assert.equal(c.GameState.cells[10].c,0);
  checks++;
}
{
  const c = context();c.GameState.bestSolution=[[0,0,0,1]];
  c.handleServerMsg({type:'SOLVER_ERROR',msg:'bad board'});
  c.handleServerMsg({type:'DONE',success:false,msg:'no result'});
  assert.equal(c.GameState.execState,'disabled');assert.equal(c.node('btn-run').disabled,false);
  assert.match(c.toasts.at(-1)[0],/FAILED/);assert.equal(c.node('btn-run').onclick,c.startSolver);
  checks++;
}
{
  const c = context();c.handleServerMsg({type:'BETTER_SOLUTION',score:0,path:[]});c.handleServerMsg({type:'DONE',success:true,optimal:true});
  assert.equal(c.GameState.execState,'disabled');assert.match(c.toasts.at(-1)[0],/NO CLEARING/);
  checks++;
}
{
  const c = context();const rect=[0,1,15,9];
  c.handleServerMsg({type:'BETTER_SOLUTION',score:158,path:[rect],optimal:false});
  assert.equal(c.GameState.execState,'armed');assert.deepEqual(plain(c.GameState.bestSolution),[rect]);
  c.handleServerMsg({type:'DONE',success:true,optimal:false});
  assert.equal(c.GameState.execState,'hot');assert.match(c.toasts.at(-1)[1],/not proven/);
  c.handleServerMsg({type:'DONE',success:true,optimal:true});assert.match(c.toasts.at(-1)[1],/upper bound reached/);
  checks++;
}
{
  const c = context();c.handleServerMsg({type:'BETTER_SOLUTION',score:2,path:[[0,0,0,1]]});
  c.handleServerMsg({type:'DONE',msg:'legacy server'});assert.equal(c.GameState.execState,'hot');assert.match(c.toasts.at(-1)[1],/not proven/);
  checks++;
}
console.log(`${checks} frontend contract scenarios passed; complete script syntax valid`);
