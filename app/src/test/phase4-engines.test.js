const {test}=require('node:test'),assert=require('node:assert/strict');
const H=require('../main/assets/phase4_labs/hmm-engine.js'),P=require('../main/assets/phase4_labs/policy-engine.js'),V=require('../main/assets/phase3_labs/mdp-engine.js'),G=require('../main/assets/phase3_labs/gridworld.js');
const near=(a,b,t=1e-9)=>assert.ok(Math.abs(a-b)<t,`${a} != ${b}`);
function finish(e){let s=e.snapshot(),limit=100000;while(!s.done&&limit--)s=e.step();assert.ok(s.done);return s}
test('HMM validates every probability row and both recurrences agree with exhaustive enumeration',()=>{
 for(const count of [2,3,5])for(const transition of ['persistent','switching','uniform'])for(const emission of ['biased','fair','distinct'])H.validate(H.model({count,transition,emission}));
 const model=H.model(),obs=['H','T','H','T'];let sum=0,best=0,path=[];
 for(let mask=0;mask<16;mask++){const p=obs.map((_,t)=>(mask>>t)&1);let v=model.pi[p[0]];for(let t=0;t<obs.length;t++){v*=model.B[p[t]][model.symbols.indexOf(obs[t])];if(t)v*=model.A[p[t-1]][p[t]]}sum+=v;if(v>best){best=v;path=p}}
 const f=finish(H.engine({model,observations:obs,mode:'forward'})),v=finish(H.engine({model,observations:obs,mode:'viterbi'}));near(Math.exp(f.result),sum);near(Math.exp(v.result),best);assert.deepEqual(v.path,path);assert.ok(f.path.every(x=>x===null));for(let t=1;t<obs.length;t++)assert.equal(v.back[t][v.path[t]],v.path[t-1]);
 assert.throws(()=>H.validate({...model,pi:[.7,.7]}));assert.throws(()=>H.engine({observations:['X']}));
});
test('HMM log domain remains finite for long sequences and handles impossible observations',()=>{
 const e=H.engine({observations:Array(2000).fill('H'),mode:'forward'}),s=finish(e);assert.ok(Number.isFinite(s.result));assert.ok(s.result<0);
 const model=H.model();model.B=[[1,0],[1,0]];for(const mode of ['forward','viterbi']){const x=finish(H.engine({model,observations:['T'],mode}));assert.equal(x.result,-Infinity);assert.ok(x.path.every(p=>p===null))}
});
test('shared Gridworld accepts editable maps without changing Phase 3 defaults',()=>{
 const before=G.environment();assert.equal(before.size,5);assert.deepEqual(before.walls,[7,16]);assert.equal(before.hole,12);
 const e=G.environment({size:4,walls:[5],start:0,goal:3,hole:15,slip:.2});assert.equal(e.move(4,3),4);near(e.transitions(2,3).reduce((s,o)=>s+o.p,0),1);assert.equal(e.reward(2,3),1);assert.throws(()=>G.environment({start:4,goal:4}));
});
test('policy evaluation and improvement converge to the same values as value iteration',()=>{
 for(const slip of [0,.2]){const cfg={size:4,walls:[5],start:0,goal:3,hole:15,slip,gamma:.9,threshold:1e-7};const p=P.engine(cfg);let prev=p.snapshot(),count=0,changed=false;
 while(!prev.done&&count++<100000){const s=p.step();if(s.last?.type==='evaluate'&&s.phase!=='initialize'){const dif=s.values.reduce((n,v,i)=>n+(v!==prev.values[i]?1:0),0);assert.ok(dif<=1)}if(s.last?.type==='improve'&&s.last.oldAction!==s.last.newAction)changed=true;prev=s}
 assert.ok(prev.stable);assert.equal(prev.changes,0);assert.ok(changed);assert.ok(prev.totalSweeps>0);const v=finish(V.engine(cfg));for(const state of p.env.states)near(prev.values[state],v.values[state],2e-6);assert.ok(P.path(p.env,prev.policy).goal);
 }
});
test('value iteration reuses synchronous Bellman sweeps and policy extraction on edited maps',()=>{
 const e=V.engine({size:5,walls:[6,16],start:0,goal:4,hole:24,slip:0,gamma:.9});let s=e.step();near(s.values[3],1);near(s.values[2],-.04);s=e.step();near(s.values[2],.86);s=finish(e);assert.ok(s.delta<.001);const path=P.path(e.env,s.policy);assert.deepEqual(path.route,[0,1,2,3,4]);const reset=V.engine({size:5,walls:[6,16],start:0,goal:4,hole:24}).snapshot();assert.ok(reset.values.every(x=>x===0));
});
