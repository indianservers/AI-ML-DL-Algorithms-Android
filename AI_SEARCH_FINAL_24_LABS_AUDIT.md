# Final audit — 24 AI Algorithm Virtual Labs

## Scope and method

All 24 PNGs in `C:\Indian Servers\ML-Algorithms\Mockup Images\AI Algorithms` were opened and compared with their live pages. The review covered header/title, side controls, central canvas, right and bottom panels, hierarchy, colors, spacing, tables, charts, legends, badges and icons. This is a manual composition audit, not an automated pixel-perfect comparison.

Every route was opened directly and refreshed successfully. Step, Run/Pause and Reset were exercised where applicable. Bayesian Network Construction is an editor: editing a CPT, Apply, Validate and Reset were checked instead. Its updated network validated as a DAG with consistent probability rows. Configuration changes, snapshot-driven tables/charts and algorithm tests supplement the browser checks. No console errors were captured during the final browser audit.

## Reference and route inventory

Each row passed the page/composition and direct-route/refresh review. Prefix numbers map to the exact reference PNG filenames.

| PNG | Lab | Route |
| --- | --- | --- |
| 01 | Uniform Cost Search | `phase7_labs/index.html?lab=ucs` |
| 02 | Iterative Deepening Depth First Search | `phase7_labs/index.html?lab=iddfs` |
| 03 | A Star Search | `phase7_labs/index.html?lab=astar` |
| 04 | Simplified Memory-Bounded A Star Search | `phase8_labs/index.html?lab=sma` |
| 05 | Hill Climbing Search | `phase8_labs/index.html?lab=hill` |
| 06 | Local Beam Search | `phase8_labs/index.html?lab=local-beam` |
| 07 | Policy Iteration | `phase4_labs/index.html?lab=policy` |
| 08 | Value Iteration | `phase4_labs/index.html?lab=value` |
| 09 | Q Learning | `phase5_labs/index.html?lab=qlearning` |
| 10 | AI Depth First Search | `phase5_labs/index.html?lab=dfs` |
| 11 | Greedy Best First Search | `phase5_labs/index.html?lab=greedy` |
| 12 | Minimax Search | `phase6_labs/index.html?lab=minimax` |
| 13 | Construction of Bayesian Network | `phase6_labs/index.html?lab=construction` |
| 14 | Inference from Bayesian Network | `phase6_labs/index.html?lab=inference` |
| 15 | Breadth First Search | `search_labs/index.html?lab=bfs` |
| 16 | Bidirectional Search | `search_labs/index.html?lab=bidirectional` |
| 17 | Beam Search | `search_labs/index.html?lab=beam` |
| 18 | Simulated Annealing | `phase2_labs/index.html?lab=annealing` |
| 19 | Genetic Algorithm | `phase2_labs/index.html?lab=genetic` |
| 20 | Monte Carlo Tree Search | `phase2_labs/index.html?lab=mcts` |
| 21 | Alpha-Beta Pruning | `phase3_labs/index.html?lab=alpha-beta` |
| 22 | Markov Decision Process Explorer | `phase3_labs/index.html?lab=mdp` |
| 23 | SARSA Learning | `phase3_labs/index.html?lab=sarsa` |
| 24 | Hidden Markov Model - Forward and Viterbi Algorithms | `phase4_labs/index.html?lab=hmm` |

The Kotlin catalog test asserts 24 entries and checks intended route mappings, including the SMA* name alias. The alias resolves to the same implementation and does not create a duplicate menu entry. Local-asset tests verify every loader dependency exists; GSAP is bundled for offline WebView use.

## Corrections made during the final audit

- Added document-scoped `lab-runtime.js` to all eight loaders. It converts legacy heading glyphs to semantic SVG icons, clears GSAP motion on reset/settings changes/hidden/pagehide, and applies legacy playback speed to animation timing.
- Added missing pause-on-hidden/pagehide handling to older controllers.
- Added explicit WebView release on Android navigation so leaving a lab destroys its document and timers.
- Corrected Phase 1 graph pulses to animate inner circles rather than translated node groups.
- Added actual parent/candidate/next connections to puzzle Beam Search; preserved source boards during display-only redraws. Connection positions update on resize and strip scrolling.
- Corrected Bidirectional Search's current-step direction to describe the transition just performed.
- Prevented Genetic Algorithm crossover output before offspring/crossover data exist, removing the undefined-position label.
- Corrected HMM progress to count fully evaluated observation columns.
- Corrected the Local Beam replacement animation target and moved SMA* backup labels away from edge-cost labels.

## Architecture and performance

Search algorithms share graph data/adjacency and weighted-search primitives while retaining separate frontier and transition rules. Gridworld transition/reward primitives are shared by MDP, policy/value iteration, Q-learning and SARSA. Game-tree data supports Minimax/Alpha-Beta; Bayesian construction/inference share probability-network primitives; Hill/Local Beam share N-Queens scoring. Presentation and charts consume snapshots rather than driving algorithm decisions. The new shared runtime contains no engine access.

There is one playback timer per controller. Run/Pause is guarded, route teardown pauses playback, reset kills stale motion, and reduced-motion branches preserve deterministic Step behavior. Charts redraw on state or display changes, not an independent animation loop.

Local Beam computes complete candidates but paginates ranking and shows representative boards. Genetic Algorithm caps the configured population at 100 and chromosome length at 64; visible population previews are limited. MCTS caps its UI budget at 5000 iterations and renders a shallow tree overview rather than every retained node. SMA* enforces its resident-node limit independently of bounded audit data. HMM caps the UI at 5 states and 12 observations (60 cells). These are source/interaction checks, not device-wide performance benchmarks.

## Cross-algorithm checks

| Comparison | Evidence |
| --- | --- |
| A* / UCS | Zero-heuristic A* agrees with UCS and Bellman-Ford over weighted graphs and node pairs. |
| SMA* / A* | Sufficient-memory SMA* agrees on tested finite trees; small bounds exercise actual prune/backup/regenerate behavior. |
| Alpha-Beta / Minimax | Root value and optimal move agree across root players, depths and ordering choices. |
| Policy / Value Iteration | Both converge to equivalent optimal values/policies, allowing valid ties. |
| BFS / Bidirectional | Reconstructed shortest-path lengths agree for every tested node pair. |
| Q-learning / SARSA | Both learn goal-reaching policies in a seeded simple MDP; numerical tests distinguish max-next-Q from the actual next-action update. |
| DFS / IDDFS | Shared tree data yields valid paths and appropriately different repeated traversal/depth behavior. |

## Verification results

- JavaScript: **48 test groups passed, 0 failed** (`node --test app/src/test/*test.js`).
- Syntax: **48 non-vendor JavaScript files passed** `node --check`.
- Android unit tests: **126 tests, 0 failures, 0 errors** in 18 suites.
- `git diff --check`: passed (only existing Windows line-ending notices).
- TypeScript: not applicable; these labs use JavaScript and the Android host uses Kotlin. No TypeScript suppression or disabled lint rule was introduced.
- Android lint: **0 errors, 31 warnings** in the fresh September 27 report. Warnings concern dependency updates, existing application cleanup/style items, and `SetJavaScriptEnabled` in the lab host. JavaScript is required for the bundled offline simulation pages; no JavaScript bridge is exposed. No lint rules were disabled.
- Debug and optimized release APK builds: **passed**. The combined lint/test/debug/release command completed successfully (96 tasks; 24 executed, 72 up-to-date). Release vital lint also passed. The release output is unsigned, as configured by the project.

Reproduction command for Android checks:

```powershell
.\gradlew.bat :app:lintDebug :app:testDebugUnitTest :app:assembleDebug :app:assembleRelease --offline --max-workers=1 --no-parallel --no-configuration-cache
```

The initial offline attempt needed an uncached Espresso dependency. After caching it online, the final run uses one worker to limit host memory pressure. Earlier interrupted build attempts are not counted as successful verification.

After the final WebView release change, `:app:testDebugUnitTest :app:assembleDebug` was rerun successfully (42 tasks; 7 executed). The fresh XML results again report 126 tests with no failures or errors. `adb -s emulator-5554 install -r app/build/outputs/apk/debug/app-debug.apk` returned **Success**.

Artifacts:

- `app/build/outputs/apk/debug/app-debug.apk` — final debug build, installed on the emulator.
- `app/build/outputs/apk/release/app-release-unsigned.apk` — optimized production build, unsigned.
- `app/build/reports/lint-results-debug.html` — fresh lint report.
- `app/build/reports/tests/testDebugUnitTest/index.html` — Android unit-test report.

## Practical limits

The visual review checked every composition against its PNG; typography rendering and real state-dependent values differ from static screenshots. New Phase 8 layouts were also exercised at narrow mobile width. The browser audit is broad functional coverage, not an exhaustive combination of every setting or a substitute for accessibility certification. SMA* intentionally accepts trees rather than arbitrary cyclic graphs. Existing unrelated application work was preserved.
