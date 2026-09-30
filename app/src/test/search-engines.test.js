const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../main/assets/search_labs/engines.js');

function complete(engine, limit = 200) {
  let state = engine.snapshot();
  for (let i = 0; !state.done && i < limit; i++) state = engine.step();
  assert.ok(state.done, 'search should terminate');
  return state;
}

test('BFS visits in FIFO order and reconstructs the shortest path', () => {
  const graph = E.adjacency(E.bfsEdges);
  const engine = E.bfs(graph, 'A', 'I');
  assert.deepEqual(engine.snapshot().queue, ['A']);
  assert.deepEqual(engine.step().queue, ['B', 'C', 'D']);
  const result = complete(engine);
  assert.deepEqual(result.path, ['A', 'D', 'I']);
  assert.equal(new Set(result.visited).size, result.visited.length);
});

test('bidirectional shortest meeting agrees with BFS for every pair', () => {
  const graph = E.adjacency(E.biEdges);
  for (const start of Object.keys(graph)) for (const goal of Object.keys(graph)) {
    const expected = complete(E.bfs(graph, start, goal)).path;
    const actual = complete(E.bidirectional(graph, start, goal, 'shortest')).path;
    assert.equal(actual.length, expected.length, `${start} to ${goal}`);
    assert.equal(actual[0], start);
    assert.equal(actual.at(-1), goal);
    for (let i = 1; i < actual.length; i++) assert.ok(graph[actual[i - 1]].includes(actual[i]));
  }
});

test('beam ranks real puzzle successors and enforces width', () => {
  const initial = E.presets.classic;
  const result = E.beam(initial, 2, 'manhattan').step();
  assert.equal(result.candidates.length, E.moves(initial).length);
  assert.equal(result.beam.length, 2);
  assert.deepEqual(result.candidates.map(x => x.h), [...result.candidates.map(x => x.h)].sort((a,b) => a-b));
  assert.ok(result.beam.every(x => E.moves(initial).includes(x.board)));
  assert.ok(result.beam.every(x => x.path[0] === initial && x.path.at(-1) === x.board));
});
