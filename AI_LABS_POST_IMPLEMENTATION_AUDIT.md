# AI Algorithms Virtual Labs — Post-Implementation Audit

Date: 2026-09-27

Scope: the existing 24 labs, all eight offline asset bundles, shared presentation/lifecycle code, and Android packaging. No labs added. Existing phase work and unrelated workspace changes were preserved.

## Fixed Errors

- **SVG keyboard activation:** SVG elements do not implement `HTMLElement.click()`. Replaced those calls with dispatched mouse events, including shared keyboard handling, Alpha-Beta, Bayesian nodes/edges, and related table handlers. Rechecked Enter/Space in the affected labs; no new console errors.
- **Run/Step races:** manual Step pauses playback before advancing in the original search and Phase 2 controllers. Regression tests cover single timers, double Run, Step during Run, speed changes, reset, completion, page hide, and visibility changes.
- **Stale visual state:** reset restores initial inspection selections in Hill Climbing, MCTS, SARSA, Bayesian inference, MDP, and policy/value iteration. Old animation targets and detached queue departure elements are cleared before replacement renders.
- **Duplicate IDs:** removed repeated, unused `queenArrow` marker definitions from Local Beam miniature boards.
- **Charts:** empty series no longer produce invalid endpoint coordinates; constant series get a nondegenerate vertical range. Logarithmic values below the displayed minimum stay inside the plot. Horizontal ticks, algorithm-specific axis labels, and real iteration coordinates replace misleading sample indices.
- **History restoration:** shared enhancement observers reconnect after a cached page is restored with Back/Forward.
- **Invalid routes:** unknown keys and keys belonging to a different asset bundle show a useful error with a working lab selector.
- **Help and fullscreen:** replaced blocking search help alerts with accessible dialogs. Annealing expands within the lab viewport, with an internal exit button, focus restoration, and Escape support, avoiding dependence on Android's native fullscreen integration.

## UI Improvements

- Shared, restrained card borders, shadows and radii; consistent button hover, active, focus and disabled states.
- Table headers, zebra rows, hover and selected/current emphasis, numeric alignment, sticky headers and bounded scroll containers.
- Custom SVG action icons replace legacy button glyphs where applicable; existing algorithm-specific branding is retained.
- Clearer muted text, visible keyboard focus, label associations and range value announcements.
- Floating tooltips for notation, graph nodes/edges and chart samples. Search tooltips expose relevant costs, status and parent information; Bayesian edges identify conditional dependencies.
- A compact, collapsible guide and a common 24-lab navigation selector on every page.
- Workspace text-selection protection; educational prose remains selectable. Reduced-motion styles disable decorative transitions.

## Interaction Improvements

- Completion disables obsolete Run/Step/Train actions in older shared controllers; Reset reenables them.
- Numeric validation rejects invalid values while retaining the documented empty/unlimited DFS depth setting.
- Enter/Space activates non-native interactive items; dialogs support Escape. Guides are keyboard operable.
- Real chart sample tooltips are limited to approximately 80 hover targets per series while the full curve remains visible.
- Verified policy grid editing, Evaluate/Improve, value extraction, training/pause/reset, Hill restart, Local Beam iteration and pagination, evidence add/remove, probability tabs, Bayesian CPT editing, cycle rejection, drag movement, save and validation.
- Verified annealing surface/contour selection, drag rotation, wheel zoom, fit, expanded view and exit.

## Algorithm Corrections

No new recurrence or search-priority defect was established in this pass. Engine tests and independent comparisons passed. Changes above correct controller behavior and presentation of real algorithm state.

Cross-checks passed:

1. A* with zero heuristic and UCS agree with an independent shortest-path baseline.
2. Alpha-Beta agrees with Minimax for both root players and multiple move orders/depths.
3. SMA* agrees with A* when memory is sufficient and reports insufficient-memory cases.
4. Policy Iteration and Value Iteration converge to equivalent optimal values/policies, allowing ties.
5. Bidirectional search and BFS agree on shortest unweighted path length for every node pair in the tested graph.
6. Q-learning uses the maximum next Q; SARSA uses the actual selected next action. Terminal bootstrap and reward semantics are tested.
7. Bayesian elimination agrees with independent enumeration; HMM recurrences agree with exhaustive enumeration.

## Per-Lab QA

Every row received a corresponding PNG review from `C:\Indian Servers\ML-Algorithms\Mockup Images\AI Algorithms`, live UI inspection, configuration/control checks, direct/menu routing and refresh checks, and an algorithm test. Run/Step/Reset and rapid playback checks apply to the 23 simulators; the construction editor instead received edit/apply/validate/reset checks. Layout fidelity preserves each reference's composition; displayed numbers come from actual execution rather than copying illustrative mockup values.

| Lab | Distinct checks |
|---|---|
| Uniform Cost Search | Priority queue, g costs, node/edge inspection, zoom |
| IDDFS | Depth limits, stack/explored tabs, repeated traversal |
| A* | Heuristic modes, g/h/f visibility, optimal paths |
| SMA* | Memory bounds, forgotten/regenerated nodes, reset |
| Hill Climbing | Candidate inspection, randomization, restart, conflict history |
| Local Beam | Top-k replacement, candidate ranking/pagination, miniature IDs |
| Policy Iteration | Editable map, rewards, evaluation and improvement |
| Value Iteration | Editable map, value sweeps, extracted policy |
| Q Learning | Episode/Train/Pause, environment reset, Q table |
| AI DFS | Depth/traversal controls, unlimited depth via keyboard, backtracking |
| Greedy Best First | Heuristic presets, weighted edges, comparison help |
| Minimax | Root-player radios, tree/board/terminal views, expansion |
| Bayesian Construction | Invalid CPT rejection, cycle rejection, add/delete edge, drag, save |
| Bayesian Inference | Evidence edits, query changes, probability tabs, graph/table views |
| BFS | Queue, graph/order toggles, example, help dialogs |
| Bidirectional | Meeting rule, frontier toggles, reconstructed path |
| Beam Search | Width, generation/heuristic controls, puzzle selection, trend tooltip |
| Simulated Annealing | Schedules/objectives, landscape manipulation, chart axes |
| Genetic Algorithm | Population, crossover/mutation/selection/elitism, fitness history |
| MCTS | Budget/policy/UCB/sort, tree selection, rollout/value history |
| Alpha-Beta | Pruning/order/root-player controls, bounds, SVG keyboard selection |
| MDP Explorer | Transition/action inspection, rewards, terminals, values/arrows |
| SARSA | Training, reward/behavior controls, algorithm radios, selected-state reset |
| HMM | Invalid observations, modes/views, matrix/trellis selection, completed decoding |

Concrete browser results included `P(Alarm=True | Burglary=True, Earthquake=False) = 0.940000`, invalid CPT-row rejection, cycle rejection, and successful HMM decoding after correcting invalid input.

### Responsive and runtime checks

- Final sweep: **120 route/viewport combinations**, using the actual lab selector and waiting for route/load completion.
- Measured CSS widths: **1280, 1367, 1440, 1600, 1920**. The requested 1366 width rounds to 1367 under the host's display scaling.
- No page-level horizontal overflow, duplicate IDs, or invalid SVG coordinates in that sweep.
- Final fresh-tab console check: **zero errors or warnings** in the tested flows. Earlier discovered SVG errors were fixed and specifically retested.
- Back/Forward during playback restores a paused page. Invalid routes recover through the lab selector.

## Build and Test Evidence

Commands used:

```powershell
node --test app/src/test/*.test.js
.\gradlew.bat :app:lintDebug :app:testDebugUnitTest :app:assembleDebug :app:assembleRelease --offline --max-workers=1 --no-parallel --no-configuration-cache
```

- **54 JavaScript test groups passed**, including six new controller/chart/keyboard regressions.
- **49 custom asset JavaScript files passed `node --check`.**
- **126 Kotlin unit tests:** zero failures/errors/skips; Gradle reused valid cached results where inputs were unchanged.
- Kotlin compilation, Android debug lint, release vital lint, debug APK and optimized release APK: **successful**. Final build completed in 1m 17s.
- TypeScript/React/ESLint checks are not applicable: this project uses Kotlin and plain offline JavaScript. No type suppressions or blanket lint disables were added.
- Debug APK installed successfully on `emulator-5554`. Final debug and release APK contents were compared with the audited shared assets and latest fullscreen/help source.

Artifacts: `app/build/outputs/apk/debug/app-debug.apk`, `app/build/outputs/apk/release/app-release-unsigned.apk`, and `app/build/reports/lint-results-debug.html`.

## Remaining Limitations

- Android lint retains **31 warnings**, with zero errors: dependency update notices (5), existing host cleanup/style/resource suggestions (25), and required JavaScript enablement for offline labs (1). They were not suppressed.
- Release APK is unsigned; distribution signing is not configured.
- Responsive UI testing used the Chromium-based in-app browser. This is not a Safari/Firefox, physical-device, or screen-reader certification. The exact 1366 CSS-pixel setting was unavailable because of display scaling.
- Visual QA was comparative inspection, not a claim of pixel-perfect reproduction. Real execution deliberately differs from illustrative mockup snapshots. Algorithm visualizations retain their documented bounds and representative/paginated views for larger state spaces.
- This audit exercises control types and representative generated items; it does not enumerate every possible parameter combination, randomized run, graph node sequence, or long-duration session.
