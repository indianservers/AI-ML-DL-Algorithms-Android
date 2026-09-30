const {test}=require('node:test');
const assert=require('node:assert/strict');
const base='../main/assets/phase3_labs/';
const G=require(base+'gridworld.js'),AB=require(base+'alpha-beta-engine.js'),MDP=require(base+'mdp-engine.js'),SARSA=require(base+'sarsa-engine.js');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} != ${b}`);
function finish(e,limit=50000){let s=e.snapshot();while(!s.done&&limit-->0)s=e.step();assert.ok(s.done,'engine terminates');return s}
test('alpha-beta preserves minimax for both root players, depths and move orders',()=>{
 for(const maximizing of [true,false])for(const depth of [2,3,4])for(const order of ['left','right','best']){
  const e=AB.engine({maximizing,depth,order}),s=finish(e);assert.equal(s.value,e.referenceValue());assert.equal(s.evaluated+s.skipped,s.total);
  assert.equal(finish(AB.engine({maximizing,depth,order,pruning:false})).value,s.value);
  for(const n of s.nodes.filter(n=>n.status==='Pruned'))assert.equal(n.value,null);
 }
 const left=finish(AB.engine()),bestFirst=finish(AB.engine({order:'best'}));assert.notEqual(left.evaluated,bestFirst.evaluated);assert.ok(left.skipped>0);
 const unpruned=finish(AB.engine({pruning:false}));assert.equal(unpruned.skipped,0);assert.equal(unpruned.evaluated,unpruned.total);
});
test('Gridworld merges collisions, normalizes slip, and applies entry rewards',()=>{
 for(const preset of ['grid','small','slippery','learning'])for(const slip of [0,.2,1]){const e=G.environment({preset,slip});assert.ok(!e.terminal(e.start));for(const s of e.states)for(let a=0;a<4;a++){const outcomes=e.transitions(s,a);close(outcomes.reduce((n,o)=>n+o.p,0),1);assert.ok(outcomes.every(o=>e.states.includes(o.next)&&o.p>0))}}
 const e=G.environment({slip:.2});assert.deepEqual(e.transitions(11,3),[{next:12,p:.8,reward:-1},{next:6,p:.1,reward:-.04},{next:11,p:.1,reward:-.04}]);
 assert.equal(e.transitions(3,3)[0].reward,1);assert.equal(e.transitions(4,3)[0].reward,0);
});
test('value iteration performs synchronous Bellman updates, converges and extracts policy',()=>{
 const e=MDP.engine({slip:0,gamma:.9});let s=e.step();close(s.values[3],1);close(s.values[2],-.04);s=e.step();close(s.values[2],.86);assert.equal(s.policy[3],3);const end=finish(e);assert.ok(end.delta<.001);assert.equal(end.values[e.env.goal],0);
 const noisy=finish(MDP.engine({slip:.4,gamma:.9}));assert.notEqual(noisy.values[3],end.values[3]);
 const immediate=MDP.engine({gamma:0,slip:0});close(immediate.step().values[2],-.04);
});
test('SARSA uses actual next action, carries it forward, updates numerically and decays epsilon',()=>{
 const e=SARSA.engine({epsilon:.8,alpha:.5,gamma:.9,episodes:6,maxSteps:60});let previous=e.snapshot(),observedNonMax=false,s;
 for(let i=0;i<6000&&!previous.done;i++){s=e.step();const t=s.transition;
  if(s.phase==='update'){close(t.nextQ,t.terminal?0:previous.q[t.next][t.nextAction]);close(t.target,t.reward+.9*t.nextQ);close(t.newQ,t.oldQ+.5*(t.target-t.oldQ));if(!t.terminal&&t.nextQ<Math.max(...previous.q[t.next]))observedNonMax=true}
  if(s.phase==='continue')close(s.q[t.state][t.action],t.newQ);
  if(previous.phase==='continue'&&s.phase==='move')assert.equal(s.action,previous.transition.nextAction);
  previous=s;
 }
 assert.ok(s.done);assert.equal(s.completed,6);assert.ok(s.epsilon<.8);assert.ok(observedNonMax,'on-policy target differs from max next Q');assert.ok(s.history.every(h=>h.steps<=60));
 const reset=SARSA.engine().snapshot();assert.equal(reset.history.length,0);assert.ok(reset.q.flat().every(x=>x===0));
});
test('epsilon-greedy, terminal bootstrap and Q-learning comparison follow their contracts',()=>{
 assert.equal(SARSA.choose([1,9,2,3],0,()=>.3).action,1);assert.equal(SARSA.choose([1,9,2,3],1,()=>.9).action,3);
 const e=SARSA.engine({algorithm:'qlearning',episodes:10,maxSteps:150,epsilon:.7});let prev=e.snapshot(),terminalSeen=false;
 while(!prev.done){const s=e.step();if(s.phase==='update'){const t=s.transition;close(t.nextQ,t.terminal?0:Math.max(...prev.q[t.next]));if(t.terminal){terminalSeen=true;assert.equal(t.nextAction,null);close(t.target,t.reward)}}prev=s}
 assert.ok(terminalSeen,'seeded exploration reaches the goal');
});
