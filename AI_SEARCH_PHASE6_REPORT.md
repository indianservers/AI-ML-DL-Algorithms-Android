# AI Algorithms Virtual Labs — Phase 6

Implemented only Minimax Search, Construction of Bayesian Network, and Inference from Bayesian Network. Phase 7 has not been started.

## Audit and reuse

- The Phase 3 Alpha-Beta engine already implements correct recursive MAX/MIN search, utility backup, bounds, pruning, and principal variation. It now accepts an optional supplied game tree; its existing generic-tree behavior and route are preserved. Minimax uses this engine, with a presentation adapter for comparison, backup, and root-choice steps.
- Phase 2 MCTS already provides correct Tic-Tac-Toe winner, legal-move, turn, and move functions. Minimax reuses those functions.
- Existing native Bayesian lessons contain scalar Bayes-rule / conjugate-update demonstrations. They do not provide an editable discrete DAG, general CPT model, or variable-elimination factors. The new Construction and Inference pages share one discrete Bayesian model and engine.
- Existing SVG graph patterns, styled table helpers, responsive cards, step/run controller, help dialog, and bundled GSAP were reused. All runtime assets remain offline. No package or CDN dependency was added.
- Earlier uncommitted project changes were preserved.

## Routes

| Topic | Asset route |
| --- | --- |
| Minimax Search | `phase6_labs/index.html?lab=minimax` |
| Construction of Bayesian Network / Bayesian Networks | `phase6_labs/index.html?lab=construction` |
| Inference from Bayesian Network | `phase6_labs/index.html?lab=inference` |

The AI Algorithms catalog now has 18 distinct lab routes. Construction reuses the existing Bayesian Networks topic ID. The separate Bayesian Inference lesson retains its earlier behavior.

## Minimax

- Real Tic-Tac-Toe game states, alternating players, terminal scoring, depth cutoffs, and configurable score perspective.
- Midgame and opening presets; generic utility tree; terminal-only and open-line cutoff evaluation; optional Alpha-Beta pruning.
- Step exposes entry/descent, leaf evaluation, return, comparison, MAX/MIN backup, root action choice, and completion. Run supports pause and speed adjustment.
- Tree, board, and terminal-state views; node inspection; evaluation and ranking tables; zoom, fit, and expanded tree view; progress, current explanation, and best-move result.
- MAX is blue, MIN pink, utility leaves green, and the principal variation blue. Pruned values are marked as bounds. Partial search values are labeled as searching.
- The tree display deliberately shows a labeled subset of siblings and plies. The search evaluates the complete legal tree within the selected depth, including branches outside the display. Node counts and hidden-branch counts are shown. A shallow zero estimate is not reported as a proven draw.

## Bayesian construction

- Add and rename variables, define 1–4 states, drag nodes, move focused nodes with arrow keys, and inspect parent/child relationships.
- Add edges with source selection and a temporary preview; select/delete edges. Duplicate edges, self-loops, and cycles are rejected.
- CPT rows are generated for every parent-state combination. Edits require finite probabilities in [0,1] summing to 1. Invalid rows have inline messages; valid edits apply explicitly.
- Changing states or parent sets clears affected CPTs, making incomplete probabilities visible rather than silently inventing distributions.
- Weather, Alarm, and empty presets; clear/reset; DAG and full-network validation; name/count/status properties; parent/child and CPT summary tables; consistency checks and workflow status.
- Save validates and stores the shared model locally. “Use in Inference Lab” opens that saved network. Storage failures are reported without discarding the current editor state.
- Teaching capacity is capped at eight variables and four states per variable. Larger CPTs scroll inside the editor.

## Bayesian inference

- Exact variable elimination over the same saved network format: construct factors, restrict evidence, multiply relevant factors, sum out hidden variables, multiply the query factor, and normalize.
- Real trace and live factor tables; evidence add/change/remove; selectable query; Alarm, Weather, and saved-network presets; run/pause, step, reset, speed, zoom, and fit.
- Evidence/query colors, observed-state badges, numerical beliefs, compact graph probability tables, posterior bars, conditional marginals, joint evidence probability, and conditional query view.
- Before completion, unobserved graph values are clearly labeled as priors. Completed values are conditioned on the evidence. Impossible evidence produces an undefined-conditional message rather than fabricated probabilities.
- Factor-flow animations are explicitly described as instructional highlights, not literal causal signals.
- Verified examples: `P(Alarm=True | Burglary=True, Earthquake=False) = 0.94`; `P(Burglary=True | JohnCalls=True, MaryCalls=True) ≈ 0.2841718354`; default joint evidence probability `0.000998`.

## Visual and interaction QA

All three specified reference PNGs were opened before implementation and compared with browser screenshots. The implementation follows their header, controls/canvas/analytics columns, node palette, table treatment, floating CPT editor, explanatory cards, and bottom progress/result composition.

The PNGs contain illustrative intermediate states and inconsistent numerical examples. The labs display actual engine values. SVG icons and available fonts differ from the raster artwork; this is not a pixel-identical reproduction.

Desktop testing used a measured 1448-pixel viewport, and mobile testing used approximately 391 pixels. All three pages fit without horizontal page overflow; large graphs and tables scroll internally. Mobile panels stack in controls, canvas, analytics, progress order.

Browser tests covered invalid/valid CPT edits, cycle rejection, multistate variables, node dragging, keyboard movement, edge removal/recreation, local save and inference handoff, Minimax views/depth/pruning/zoom/run, inference factor tables/evidence/query/pause/posterior/probability tabs, and mobile CPT editing. Shared board-height clipping and an SVG highlight animation that displaced nodes were found and fixed. Subsequent editor checks produced no console errors.

## Verification

- **29 JavaScript test groups passed** across Phases 1–6.
- New tests check Minimax against an independent recursive oracle and against Alpha-Beta, terminal scoring, alternating node types, depth cutoffs, root decisions, backups, step stages, and reset.
- Bayesian tests cover graph edits, cycle rejection, state/CPT invalidation, probability validation, factor restriction/multiplication/sum-out, normalization, multistate networks, observed queries, impossible evidence, trace stages, and save-format round trips.
- Variable-elimination results match independent enumeration for every variable in the Alarm and Weather networks under multiple evidence configurations.
- **124 Android unit tests passed**, with zero failures or errors. Route uniqueness and topic-ID reuse are tested.
- Browser regression checks loaded and stepped Alpha-Beta, HMM, Q-learning, SARSA, and BFS without console errors. Earlier engine test groups also pass.
- All new JavaScript files pass syntax checks. This bundle uses JavaScript, not TypeScript.
- `:app:testDebugUnitTest :app:assembleDebug --offline` succeeded.
- The final debug APK installed successfully on the connected Android emulator.
- `git diff --check` passed.

## Main files

- `app/src/main/assets/phase6_labs/`: shared Bayesian engine, Minimax adapter, three pages, shared SVG/UI helpers, HTML loader, and scoped responsive CSS.
- `app/src/main/assets/phase3_labs/alpha-beta-engine.js`: optional supplied-tree support and terminal-root handling.
- `LearnCatalog.kt`, `SearchAlgorithmWebLab.kt`, and `LearnCatalogTest.kt`: catalog entries, stable routes, and routing checks.
- `app/src/test/phase6-engines.test.js`: algorithm and shared-model tests.
