const test = require('node:test');
const assert = require('node:assert/strict');
const E = require('../main/assets/phase2_labs/engines.js');

test('annealing obeys cooling, Metropolis decisions and best-so-far monotonicity', () => {
  const cfg = {limit:40,t0:12,alpha:.9,sigma:.7,restart:true};
  const a = E.annealing(cfg,42), b = E.annealing(cfg,42);
  let sawDownhill=false,sawUphill=false,lastBest=Infinity;
  for(let i=0;i<200;i++){
    const s=a.step(),same=b.step();
    assert.deepEqual(s,same,'a reset with the same seed replays the same steps');
    assert.ok(s.best.energy<=lastBest+1e-9);lastBest=s.best.energy;
    if(s.phase==='update'){
      assert.equal(s.probability,s.delta<=0?1:Math.exp(-s.delta/s.temperature));
      assert.equal(s.accepted,s.draw<s.probability);
      if(s.delta<=0){sawDownhill=true;assert.ok(s.accepted)}else sawUphill=true;
    }
    if(s.phase==='check')assert.ok(s.temperature<=cfg.t0);
  }
  assert.ok(sawDownhill&&sawUphill);
  assert.ok(E.temperature({...cfg,schedule:'linear'},10)<cfg.t0);
  assert.ok(E.temperature({...cfg,schedule:'logarithmic'},10)<cfg.t0);
});

test('genetic cycle uses configured crossover, mutation, elitism and diversity', () => {
  const g=E.genetic({size:8,length:12,crossover:1,mutation:1,elitism:true},17);
  let s=g.snapshot(),initialBest=s.best.bits;
  assert.equal(s.population.length,8);
  assert.ok(s.population.every(x=>x.bits.length===12&&x.fitness===E.fitness(x.bits)));
  s=g.step();assert.equal(s.parents.length,8);
  assert.ok(s.parents.every(x=>s.population.some(p=>p.bits===x.bits)));
  s=g.step();assert.ok(s.pairs.every(x=>x.crossed&&x.point>0&&x.point<12));
  const before=[...s.offspring];
  s=g.step();assert.equal(s.mutations.length,8*12);
  for(let i=0;i<8;i++)for(let j=0;j<12;j++)assert.notEqual(s.offspring[i][j],before[i][j]);
  s=g.step();assert.ok(s.next.some(x=>x.bits===initialBest));
  s=g.step();assert.equal(s.population.length,8);assert.equal(s.generation,1);
  assert.ok(s.diversity>=0&&s.diversity<=1);
  assert.equal(s.history.length,1);
});

test('MCTS expands legal moves, rolls out to terminal and backpropagates root-view reward', () => {
  const m=E.mcts({budget:6,preset:'opening',exploration:Math.SQRT2},4);
  let s=m.snapshot();const initial=s.root.board;
  for(let i=0;i<24;i++)s=m.step();
  assert.equal(s.iteration,6);assert.equal(s.root.visits,6);
  assert.equal(s.results.win+s.results.loss+s.results.draw,6);
  assert.equal(s.history.length,6);
  assert.ok(s.children.every(child=>E.legal(initial).includes(child.action)));
  for(const child of s.children){
    assert.equal(child.board,E.move(initial,child.action));
    assert.equal(child.ucb,child.q+Math.SQRT2*Math.sqrt(Math.log(s.root.visits)/child.visits));
  }
  assert.ok([-1,0,1].includes(s.result));
  assert.ok(E.winner(s.rollout.at(-1))!==null);
  for(let i=1;i<s.rollout.length;i++)assert.equal([...s.rollout[i]].filter((x,j)=>x!==s.rollout[i-1][j]).length,1);
  assert.equal(s.best.visits,Math.max(...s.children.map(x=>x.visits)));
});
