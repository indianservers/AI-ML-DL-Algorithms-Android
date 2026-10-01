# AI Algorithms Virtual Labs — Phase 7 report

## Scope and routes

Implemented only Phase 7. Phase 8 has not been started.

| Lab | Offline asset route |
| --- | --- |
| Uniform Cost Search | `phase7_labs/index.html?lab=ucs` |
| Iterative Deepening Depth First Search | `phase7_labs/index.html?lab=iddfs` |
| A Star Search / A* Search | `phase7_labs/index.html?lab=astar` |

All three are integrated into the existing AI Algorithms search catalog and Android WebView dispatcher. There are 21 distinct lab routes; aliases share their intended route. Assets and GSAP are local, with no network dependency.

## Audit and reuse

The repository had BFS, Bidirectional, Beam, DFS, Greedy, Minimax, graph presets, parent-chain reconstruction, tables, progress components, and GSAP playback helpers. It did not have an existing UCS, Dijkstra, IDDFS, or A* engine to upgrade.

- Reused Phase 1 parent-chain reconstruction and Phase 5 tree presets.
- Extended the existing DFS engine with optional path-local cycle checking and explicit depth-cutoff events. Defaults preserve the earlier DFS behavior.
- IDDFS repeatedly invokes this DFS engine rather than implementing another DFS traversal.
- UCS and A* have separate operation generators. A small weighted core owns validation, adjacency, one queue entry per node, costs, parents, relaxation, and snapshots.
- Reused the existing playback controller, table/progress rendering, dialogs, and local GSAP infrastructure.
- New layout styles are scoped to `.phase7`; no earlier lab stylesheet was changed.

## Algorithms and interaction

### Uniform Cost Search

Prioritizes accumulated g, breaking ties by original queue insertion. Strict improvements update g and parent in the single live queue entry. The goal is settled only on removal from the minimum-priority queue, never on discovery. Negative and non-finite costs are rejected; zero-cost and parallel edges are supported by the engine.

Steps expose pop, close, neighbor inspection, tentative cost, queue update, goal, reconstruction, and completion. Queue rows, expanded order, cost table, tentative path, and final cost all come from snapshots. The reference graph completes with A → B → D → G, cost 6.

### Iterative Deepening DFS

Repeats depth-limited DFS from limit 0 through the selected maximum, showing iteration start, visit, descent, cutoff, backtrack, iteration end, limit increase, restart, and goal. Only ancestors on the current path are blocked as cycles, allowing a later shorter route to revisit a node that previously hit a cutoff.

The stack contains actual DFS frames. Iteration history preserves repeated visits and supports inspection of completed visit orders. The default run visits 1, 3, and 7 nodes at unsuccessful limits 0, 1, and 2, then visits 8 nodes and finds A → B → E → K at depth 3 in iteration 4. Maximum depth 0 stops after one visit when the goal is K.

Tree diagram levels are measured from A; stack depths are measured from the selected start. The active bound guide accounts for a different start in tree presets; the cyclic preset avoids presenting a misleading horizontal bound. The educational history stores additional data beyond the working DFS stack.

### A* Search

Prioritizes f=g+h, then lower h, then original queue insertion. Strict improvements update costs/parents and reopen closed nodes when necessary. Steps separately expose tentative g, h, f, and relaxation. The reference graph completes with A → B → D → F → G, cost 8.

All UI heuristic choices are valid lower bounds for the supplied graphs and every selectable goal:

- Consistent predefined values, transformed by absolute difference for another goal.
- Euclidean and Manhattan distances scaled conservatively by the minimum edge-cost/distance ratio.
- Zero heuristic for direct UCS comparison.

Zero mode on the UCS comparison graph produces the same expansion order and cost-6 path as UCS. The engine also supports reopening with an inconsistent admissible heuristic, covered by tests.

### Shared controls and animations

Start/goal/preset/heuristic/depth changes reset the search. Display toggles, inspection, and zoom preserve search state. Run and Step consume the same engine operations. Run/Pause/Reset, speed, zoom, fit/reset zoom, node/edge selection, keyboard selection, tooltips, learning guides, and contrast controls are wired.

State-driven GSAP highlights current nodes and edge relaxation, inserts/reorders queue rows, flashes improved costs, marks dequeues, adds closed-list chips, animates IDDFS restarts/backtracking, and traces completed paths. Reduced-motion preferences are respected. Icons are custom SVGs, not emojis.

## Mockup and responsive QA

Opened and compared all three supplied PNGs. Preserved the controls/canvas/analytics structure, weighted graph geometry, hierarchical tree, semantic colors, compact tables, separate IDDFS summary/stack cards, and bottom progress/explanation/result composition.

The raster examples contain inconsistent intermediate costs, queue order, and depth labels. The implementation uses computed values. In particular, UCS has g(F)=4 on the displayed graph; the A* example path shown in the PNG has cost 10, while the actual optimum is 8. Predefined h(D) is 3 so the preset remains consistent across the D–F edge. IDDFS uses root depth 0 and does not expand beyond its active bound.

This is a close implementation of the compositions, not a pixel-identical raster reproduction: SVG icons, fonts, real intermediate states, and added preset controls differ. Desktop checks used a measured 1448-pixel viewport; mobile checks used approximately 391 pixels. All three pages fit without page-wide horizontal overflow. Large graphs remain readable and scroll inside their canvas on mobile; panels stack vertically.

Browser checks covered full default runs, Step, Run/Pause, reset, speed changes, display toggles, zoom/fit, node inspection, heuristic/goal/preset changes, A* zero-mode comparison, IDDFS maximum depth and historical visits, help dialogs, and mobile layouts. QA found and fixed garbled punctuation and an SVG keyboard-selection error. Subsequent checks produced no new console errors.

## Verification

- **37 JavaScript test groups passed** across Phases 1–7.
- **8 Phase 7 groups passed again** after final engine edits.
- UCS and zero-heuristic A* agree with independent Bellman-Ford across multiple generated weighted graphs and every node pair.
- Tests cover priority order, unique queue entries, decrease-cost updates, parent reassignment, goal-pop termination, reopening, heuristic admissibility/consistency, path reconstruction, invalid/zero/parallel costs, unreachable goals, and start=goal.
- IDDFS tests cover stable traversal, every bound, history/reset behavior, path cycles, later shallower revisits, maximum depth, shortest depth against BFS, and reproducible stepping.
- **125 Android unit tests passed**, with zero failures/errors/skips. Route uniqueness and Phase 7 aliases are tested.
- All 18 Phase 1–6 routes loaded in the browser without new console errors. BFS, Bidirectional, DFS, Greedy, and Minimax were stepped and their resulting state changes verified.
- All Phase 7 JavaScript files pass syntax checks. This bundle uses JavaScript; there is no TypeScript compilation step.
- `:app:testDebugUnitTest :app:assembleDebug --offline` succeeded with final assets.
- `git diff --check` passed.

## Main files

- `app/src/main/assets/phase7_labs/`: weighted core, UCS/A*/IDDFS engines, graph presets, common SVG/UI helpers, three routed pages, responsive styles.
- `app/src/main/assets/phase5_labs/search-engines.js`: optional bounded-DFS event and path-cycle support.
- `LearnCatalog.kt`, `SearchAlgorithmWebLab.kt`, `LearnCatalogTest.kt`: navigation, asset routes, aliases, and route tests.
- `app/src/test/phase7-engines.test.js`: algorithm correctness and regression tests.

The final debug APK installed successfully on emulator-5554 using adb install -r.
