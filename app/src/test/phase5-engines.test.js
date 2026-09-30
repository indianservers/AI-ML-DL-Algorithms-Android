const {test}=require('node:test'),assert=require('node:assert/strict');
const E=require('../main/assets/phase5_labs/search-engines.js'),Q=require('../main/assets/phase5_labs/qlearning-engine.js'),S=require('../main/assets/phase3_labs/sarsa-engine.js'),P=require('../main/assets/phase4_labs/policy-engine.js');
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
function finish(e){let s=e.snapshot(),limit=1000000;while(!s.done&&limit--)s=e.step();assert.ok(s.done);return s}
test('DFS uses actual LIFO frames, visits in order, backtracks, and reconstructs goal path',()=>{
 const e=E.dfs(E.dfsGraph(),'A','G');let old=e.snapshot(),backtracks=0,s=old;
 while(!s.done){s=e.step();if(s.phase==='push'){assert.equal(s.stack.length,old.stack.length+1);assert.equal(s.stack.at(-1).node,s.current)}if(s.phase==='backtrack'){backtracks++;assert.equal(s.stack.length,old.stack.length-1);assert.equal(old.stack.at(-1).node,s.finished.at(-1))}old=s}
 assert.deepEqual(s.visited,['A','B','D','H','I','E','J','K','C','F','L','M','G']);assert.deepEqual(s.path,['A','C','G']);assert.ok(backtracks>0);
 assert.deepEqual(finish(E.dfs(E.dfsGraph(),'A','G',10,'right')).visited,['A','C','G']);
});
test('DFS respects depth limits, handles cycles and disconnected nodes, and resets',()=>{
 const limited=finish(E.dfs(E.dfsGraph(),'A','G',1));assert.equal(limited.found,false);assert.deepEqual(limited.visited,['A','B','C']);
 const cyclic=finish(E.dfs(E.dfsGraph('cyclic'),'A','O'));assert.equal(new Set(cyclic.visited).size,cyclic.visited.length);assert.ok(cyclic.found);
 assert.equal(finish(E.dfs(E.dfsGraph('disconnected'),'A','G')).found,false);
 assert.deepEqual(finish(E.dfs(E.dfsGraph(),'A','A',0)).path,['A']);
 assert.deepEqual(E.dfs(E.dfsGraph()).snapshot().visited,[]);
});
test('Greedy selects minimum h, ignores weights, ranks ties, and reconstructs paths',()=>{
 const graph=E.greedyGraph(),e=E.greedy(graph);let s=e.snapshot();
 while(!s.done){const before=s;s=e.step();if(s.phase==='select')assert.equal(s.h[s.current],Math.min(...before.open.map(n=>s.h[n])))}
 assert.deepEqual(s.path,['A','C','E','H']);assert.equal(s.routeCost,11);assert.deepEqual(s.expanded,['A','C','E']);
 const trap=finish(E.greedy(E.greedyGraph('trap')));assert.deepEqual(trap.path,s.path);assert.equal(trap.routeCost,30); // A-C-E-G-H costs 10: greedy is not optimal.
 const ties={...graph,nodes:graph.nodes.map(n=>({...n,h:1}))};assert.equal(E.greedy(ties).step().current,'A');const te=E.greedy(ties);te.step();te.step();te.step();assert.equal(te.step().current,'B');
 assert.equal(finish(E.greedy(E.greedyGraph('disconnected'))).found,false);
 for(const mode of ['euclidean','manhattan']){const h=E.heuristics(graph,'D',mode);assert.equal(h.D,0);assert.ok(h.A>0);assert.ok(finish(E.greedy(graph,'A','D',mode)).found)}
});
test('Q-learning applies alpha and gamma to max next Q, limits updates to valid pairs, records rewards and terminals',()=>{
 const e=Q.engine({preset:'simple',episodes:80,maxSteps:30,alpha:.3,gamma:.8,epsilon:.4});let s=e.snapshot(),total=0,checkedNonActual=false,terminal=false;
 while(!s.done){const old=s;s=e.step();if(s.phase==='update'){const t=s.transition;total+=t.reward;near(t.nextQ,t.terminal?0:Math.max(...old.q[t.next]));near(t.target,t.reward+.8*t.nextQ);near(t.newQ,t.oldQ+.3*(t.target-t.oldQ));if(!t.terminal&&t.nextQ!==old.q[t.next][t.nextAction])checkedNonActual=true;if(t.terminal){terminal=true;near(t.target,10)}}if(s.phase==='continue'){let diffs=0;for(let i=0;i<s.q.length;i++)for(let a=0;a<4;a++)if(s.q[i][a]!==old.q[i][a]){diffs++;assert.equal(i,s.transition.state);assert.equal(a,s.transition.action);assert.ok(e.env.states.includes(i));assert.ok(!e.env.terminal(i))}assert.ok(diffs<=1)}if(s.completed>old.completed){near(s.history.at(-1).reward,total);assert.ok(s.history.at(-1).steps<=30);total=0}if(old.phase==='episode'){assert.equal(s.state,e.env.start);assert.equal(s.totalReward,0);assert.deepEqual(s.q,old.q)}}
 assert.ok(checkedNonActual);assert.ok(terminal);assert.equal(s.history.length,80);assert.ok(e.env.walls.every(i=>s.q[i].every(v=>v===0)));
});
test('Q-learning epsilon behavior, simple-map learning, reward presets and reset are real',()=>{
 assert.equal(S.choose([0,10,0,0],0,()=>.1).action,1);assert.equal(S.choose([0,10,0,0],1,()=>.6).action,2);
 const e=Q.engine({preset:'simple',episodes:300,epsilon:.2,maxSteps:50}),s=finish(e),policy=s.q.map(row=>row.indexOf(Math.max(...row)));assert.ok(P.path(e.env,policy).goal);assert.ok(s.history.slice(-50).filter(h=>h.success).length>=45);
 const reset=Q.engine({preset:'simple'}).snapshot();assert.equal(reset.completed,0);assert.ok(reset.q.every(row=>row.every(v=>v===0)));
 assert.equal(Q.engine({rewardMode:'risky'}).env.penaltyReward,-10);assert.equal(Q.engine({rewardMode:'sparse'}).env.stepReward,0);assert.deepEqual(Q.engine().env.penalties,[8,21]);
});
