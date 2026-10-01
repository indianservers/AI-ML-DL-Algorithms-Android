# AI Algorithms Virtual Labs — Phase 8

## Delivered

Three offline WebView labs are integrated into the AI Algorithms catalog, bringing the mockup-based collection to 24 entries:

| Lab | Asset route | Reference PNG |
| --- | --- | --- |
| Simplified Memory-Bounded A* | `phase8_labs/index.html?lab=sma` | `04 - Simplified Memory-Bounded A Star Search.png` |
| Hill Climbing | `phase8_labs/index.html?lab=hill` | `05 - Hill Climbing Search.png` |
| Local Beam | `phase8_labs/index.html?lab=local-beam` | `06 - Local Beam Search.png` |

References were opened before implementation. The pages retain the reference-specific compositions: SMA* tree and memory/OPEN/forgotten panels; Hill Climbing board, candidate inspector, landscape and history; Local Beam source/candidate/next board bands, rankings, trend and lineage. Custom SVG icons and queens avoid platform-dependent emoji rendering in the new labs.

## Algorithms and reuse

- SMA* reuses Phase 7 weighted-search validation and adjacency primitives. Its independent engine retains at most M search-node records, including ancestors. Incoming nodes participate in worst-leaf selection before allocation. Expanded parents preserve scalar bounds for forgotten successors, propagate monotonic backups and regenerate children when their bounds become competitive. The implementation explicitly supports finite directed trees with one parent per node; cyclic and multiple-parent inputs are rejected. UI audit records and immutable problem data are outside the resident-search bound and never consulted by the search. This distinction is stated in the UI and engine comments.
- Hill Climbing and Local Beam share immutable one-queen-per-column states, attacking-pair scoring, complete neighborhoods, canonical deduplication and seeded initialization in `nqueens.js`. Hill Climbing selects the best neighbor across all columns; the selected-column inspector does not constrain the search. Equal moves are bounded and avoid revisits. Restart count and best state survive explicit restarts.
- Local Beam generates all k neighborhoods, deduplicates states, deterministically sorts by h and state key, and commits the actual top k. Generate/rank/select/replace are separate transitions. A solved generated state is committed before stopping. Repeated beams and iteration limits prevent endless cycling. Source boards are retained for truthful lineage after replacement.
- Shared Phase 7 presentation primitives, Phase 2 random/chart utilities and bundled GSAP are reused. Engines remain DOM-independent. Run and Step call the same engine transitions.

## Correctness evidence

Eight Phase 8 test groups cover heuristic pair counts, complete immutable neighborhoods, best moves and deltas, local optima, bounded sideways movement, restarts/reset, local-beam ranking/deduplication/termination, SMA* resident bounds at every transition, worst-leaf eviction, backups, regeneration and insufficient-memory failure. Sufficient-memory SMA* agrees with A* on the same test trees.

Browser examples on the reference tree:

- Memory 4, predefined admissible heuristic: A → C → G → L, cost 8; 18 transitions, 2 prunes.
- Memory 4, zero heuristic: same optimal path and cost; 51 transitions, 11 prunes, 3 regenerations.
- Local Beam default first generation: 168 generated moves and 168 unique states, with exactly 3 selected. The default run honestly stops on recurrence at iteration 7 with h=1.

The SMA* reference contains illustrative inconsistencies: the implemented default goal is L, and h(G)=1 respects the actual remaining edge cost. Displayed results come from the engine rather than copying reference numbers.

## Interaction, motion and size

All three direct routes, refresh, Step, Run/Pause, Reset and configuration changes were exercised in the browser. Desktop and approximately 391 CSS-pixel mobile viewports were checked. Controls stack on narrow screens; wide educational canvases scroll inside their cards without page-wide overflow.

Animations are attached to state transitions. The Phase 8 controller owns its tweens, kills them before redraw/reset/unmount, pauses when hidden, and scales durations with playback speed. Reduced-motion settings are respected. Tables and charts derive from snapshots/history.

N is limited to 4–10, beam width to 1–6, and Local Beam runs to 100 iterations. At maximum settings, generation evaluates 540 moves while ranking renders 15 rows per page; representative boards keep the canvas bounded. Hill Climbing permits at most 1000 moves and bounds sideways runs. The SMA* UI uses the 12-node reference tree.

## Final verification

See `AI_SEARCH_FINAL_24_LABS_AUDIT.md` for the combined JavaScript, Android, lint, build, route, architecture and cross-algorithm results. Visual QA was manual against every reference; no pixel-difference claim is made. Computed states, explanatory copy and responsive behavior intentionally differ where a static illustrative value would be misleading.

Final gates: 48 JavaScript groups and 126 Android tests passed; 48 JavaScript files passed syntax checks; lint reported 0 errors and 31 warnings; debug and optimized release builds passed. The final debug APK installed successfully on emulator-5554. The release APK is unsigned. No broad test/lint suppression was added.
