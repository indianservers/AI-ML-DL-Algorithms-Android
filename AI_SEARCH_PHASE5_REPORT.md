# AI Algorithms Virtual Labs — Phase 5

## Delivered

Three offline WebView labs matching the Phase 5 references:

| Lab | Reference | Asset route |
| --- | --- | --- |
| Q Learning | `09 - Q Learning.png` | `phase5_labs/index.html?lab=qlearning` |
| AI Depth First Search | `10 - AI Depth First Search.png` | `phase5_labs/index.html?lab=dfs` |
| Greedy Best First Search | `11 - Greedy Best First Search.png` | `phase5_labs/index.html?lab=greedy` |

Phase 6 was not started.

## Existing-code audit and reuse

- Q-learning already existed in the Phase 3 SARSA comparison engine. The new page reuses that exact update state machine through a small environment/configuration adapter. It does not introduce another Q-update implementation.
- The shared Gridworld now accepts explicit penalty cells. The temporal-difference engine accepts an optional environment configuration. Existing defaults remain unchanged.
- Q-learning, SARSA, MDP Explorer, Policy Iteration, and Value Iteration therefore use the same JavaScript movement, transitions, walls, rewards, and terminal model.
- The existing native Kotlin Q-learning engineering demo remains intact. Its tests pass; the existing **Q-Learning learning-topic route** now opens the upgraded WebView lab.
- No DFS or Greedy Best First engine was present. Their separate state machines reuse Phase 1 adjacency and parent-chain reconstruction helpers. DFS retains explicit stack frames rather than forcing its behavior into BFS.
- Existing styled tables, Gridworld renderer, playback controllers, charts, and local GSAP are reused. No dependency installation or network resource is required.
- No existing A* or UCS simulator was found. The optional comparison is explanatory only; no additional simulator was added.

## Navigation

The AI Algorithms menu aliases the original Q-Learning topic ID as **Q Learning**. Both spellings resolve to the same page. DFS and Greedy each have one new topic. Tests verify fifteen distinct AI lab routes, preserve earlier aliases, and leave unrelated topics on their existing screens.

## Q-learning

The default 5×5 map follows the reference: start at (0,0), goal at (4,4), obstacles at (1,1)/(3,2), and penalty cells at (1,3)/(4,1). A 3×3 preset supports quick experiments.

Every parameter affects the engine: ε, α, γ, episode budget, maximum episode steps, and reward mode. Standard rewards are −1 per step, +10 at the goal, and −5 at penalty cells. Risky changes penalties to −10; Sparse changes ordinary step reward to zero.

Step advances one choose/move/update/episode operation. Run Episode stops at the next episode boundary. Train batches twenty of the same transitions per tick and displays the latest transition. Pause preserves state. Reset clears learning; Reset Environment also restores the reference map.

Directional values and the Q table show real Q estimates. The table uses Up, Right, Down, Left ordering while mapping to the shared action indices. It highlights selected/current states and the updated state-action pair. The TD panel includes old Q, reward, maximum next Q, target, TD error, and new Q, with numeric substitution below.

Terminal targets omit bootstrapping. Time-limit truncation preserves nonterminal bootstrapping. The chart uses completed-episode rewards and a trailing average of up to fifty episodes. Exploration is ε and exploitation is 1−ε. The policy and displayed route are derived from argmax Q; cycles and an unfinished training budget are not called convergence.

GSAP shows actual adjacent transitions, collision feedback, penalties, arrivals, Q updates, changed greedy arrows, and episode resets. Batched training visualizes its latest transition rather than inventing a direct multi-cell move.

## DFS

The default directed tree has A–O with structural depth guides 0–4. Controls select start, goal, tree/cyclic/disconnected preset, depth limit, traversal order, and backtracking visibility.

Each stack frame contains its node, depth, next-child index, and visit status. Step visits a node, pushes one eligible child, or pops/backtracks. The rightmost frame is the real LIFO top. Visited order, finished-frame counts, unvisited counts, adjacency, current path, and next-operation explanations come from the same state.

Search depth starts at the chosen start node; the diagram's guides retain the original tree levels. The default left-first visit order to G is A, B, D, H, I, E, J, K, C, F, L, M, G, with reconstructed path A→C→G. Right-first reaches G through A→C→G immediately. A limit of 1 visits only A, B, C and reports no goal. Cyclic presets cannot loop indefinitely.

Browser QA caught a depth-limit edit that was not being applied immediately. The input now updates on input, validates integer limits, and disables execution for invalid values. Failed searches display “Goal Not Found” rather than claiming goal discovery.

GSAP animates active nodes/edges, stack pushes/pops, backtracking, and goal discovery. Backtracked edges point toward the parent.

## Greedy Best First Search

The reference graph uses fixed heuristic estimates and displayed edge weights. Frontier selection uses **h only**, with alphabetical tie-breaking; weights are used only to report route cost. Steps expose selection, expansion, ranking, goal detection, and parent-chain reconstruction.

The classic example returns A→C→E→H at cost 11, although A→C→E→G→H costs 10. The Expensive Shortcut preset increases E–H to 25; Greedy still selects the same route, now costing 30. This demonstrates its lack of optimality without running a second algorithm.

Euclidean and Manhattan modes compute h from displayed coordinates to the selected goal. Predefined estimates remain fixed for the reference graph, with the selected goal set to zero; this convention is documented in Help. A disconnected preset demonstrates frontier exhaustion.

The graph, sorted frontier, expansion sequence, heuristic table, best pending candidate, last selection comparison, and current/final parent route use real state. Controls include h-label visibility, speed, zoom/fit, node/table selection, help, and contrast theme. GSAP emphasizes candidate selection, frontier ranking, expansion, and the final path.

## Visual review

All three exact PNGs were opened before coding. Browser screenshots were compared against their header layouts, control widths, graph/Gridworld geometry, analytics cards, bottom panels, colors, table styles, legends, and progress indicators. Follow-up passes refined Q-learning branding and spacing, the empty reward chart, stack tiles, graph readability, and failure-state wording.

The implementations begin with genuine initial state; the reference images depict intermediate computations and include inconsistent illustrative values. No screenshot values or training histories were hardcoded. Simplified vector icons, browser fonts, and variable explanations differ from the raster images, so this is not a pixel-identical reproduction.

All three pages were checked at a measured 1448×1086 desktop viewport and approximately 391×844 mobile viewport without page-level horizontal overflow. Large graphs scroll within their own canvas. The layouts stack panels on mobile. Reduced-motion preferences are respected.

## Verification

- **21 JavaScript test groups passed** across Phases 1–5.
- **123 Android unit tests passed**, with zero failures, errors, or skips.
- New tests cover DFS LIFO/order/depth/cycles/backtracking/path/reset; Greedy minimum-h selection, weight independence, ties, heuristics, paths and failure; and Q-learning ε-greedy behavior, α/γ arithmetic, max-next-Q versus the sampled action, terminal handling, valid update pairs, reward history, episode resets, reward presets, and learned goal-reaching behavior.
- Browser tests covered Q updates, training completion, Run Episode boundaries, Pause, reset, environment reset, reward/learning/discount controls, selected Q rows, and mobile help.
- Browser tests covered DFS stack changes, backtracking visibility, right-first traversal, depth limits, and mobile layout; Greedy costly-route selection, heuristics, goals, h visibility, zoom/fit, comparison help, unreachable goals, and mobile layout.
- Browser regression checks loaded and stepped SARSA, MDP Explorer, Value Iteration, BFS, and Bidirectional Search without console errors. Existing algorithm tests also cover the other earlier engines.
- A* regression testing is not applicable: no preexisting A* implementation was present.
- All Phase 5 JavaScript syntax checks passed. The bundle uses JavaScript, not TypeScript.
- `:app:testDebugUnitTest :app:assembleDebug --offline` succeeded.
- The debug APK installed successfully on the connected Android device.
- `git diff --check` passed.

Earlier uncommitted project work was preserved.

## Files

| File | Responsibility |
| --- | --- |
| `phase5_labs/qlearning-engine.js` | Adapter for the existing Q-learning engine and reference map |
| `phase5_labs/search-engines.js` | DFS and Greedy state machines, graph presets and heuristics |
| `phase5_labs/qlearning.js` | Q-learning visualization and controls |
| `phase5_labs/search.js` | DFS and Greedy pages |
| `phase5_labs/common.js` | Shared presentation helpers and help |
| `phase5_labs/labs.css`, `index.html` | Layout, responsive styling and offline dependencies |
| `app/src/test/phase5-engines.test.js` | New algorithm tests |
