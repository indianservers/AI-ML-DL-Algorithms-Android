# AI Algorithms Virtual Labs — Phase 4

## Delivered

- Hidden Markov Model — Forward & Viterbi Algorithms (reference 24).
- Policy Iteration (reference 07).
- Value Iteration (reference 08).

All three are offline Android WebView labs under `app/src/main/assets/phase4_labs/`. Phase 5 was not started.

## Audit and reuse

The Phase 3 MDP engine already implemented correct synchronous Value Iteration. The new Value Iteration page uses that engine directly. Policy Iteration needed a separate evaluation/improvement state machine.

Both iteration labs use Phase 3's Gridworld, which is also used by MDP Explorer and the SARSA/Q-learning comparison. The shared environment now accepts editable sizes, walls, start, and terminals, while preserving its previous defaults. Expected action returns are shared. There is no second map parser or transition model. The existing native Kotlin Q-learning lab remains intact and its regression test passes.

The old Kotlin HMM contained valid normalized Forward filtering, but called independent posterior maxima a Viterbi path and returned a final posterior maximum as the sequence likelihood. It now accumulates the Forward normalization likelihood and performs log-domain Viterbi with backpointers. The WebView engine exposes the same mathematical recurrences as incremental events for configurable H/T models.

Shared styled tables, controls, charts, Gridworld rendering, and local GSAP utilities are reused. No network dependencies or new libraries were introduced.

## Routes

| Lab | Asset route |
| --- | --- |
| HMM | `phase4_labs/index.html?lab=hmm` |
| Policy Iteration | `phase4_labs/index.html?lab=policy` |
| Value Iteration | `phase4_labs/index.html?lab=value` |

Existing Hidden Markov Models topics resolve to the HMM page. The AI menu aliases the original HMM topic ID. Policy and Value Iteration each have one new topic. Catalog tests verify twelve distinct AI lab routes and preserve the earlier aliases.

## Algorithm and interaction behavior

### HMM

- Validated initial, transition, and emission distributions; two to five hidden states.
- Editable sequences of 1–12 H/T observations, transition/emission presets, and playback speed.
- Forward uses log-sum-exp over every predecessor; Viterbi uses maxima and stored backpointers.
- Step computes one trellis cell, termination, or one backtracking operation. Run uses the same transitions.
- Interactive trellis cells, probability-flow details, matrix row selection, and a complete Forward/Viterbi comparison.
- Graph view selects a source state and shows its outgoing probabilities without overlapping five-state edge labels.
- Forward displays likelihood and final probabilities, with no decoded path. Viterbi reveals its path during backtracking.
- Invalid input pauses playback and disables execution; ordinary typing preserves spaces. Reset restores the last valid sequence.

### Policy Iteration

- Deterministic policy initialized to Up; synchronous iterative policy evaluation followed by greedy improvement and a stability check.
- Step updates one state's value or action, or advances a phase. Evaluation uses a saved previous-sweep array.
- Evaluate Policy stops at improvement; Improve Policy stops at the stability check. Buttons are enabled only in their applicable phases.
- Live active cell, current policy, latest completed improvement snapshot, evaluation sweeps, maximum/average change, policy changes, and stable status.
- Grid size, discount, transition noise, rewards, start, terminals, and add/remove walls all reset and alter the actual environment.

### Value Iteration

- Step and Update Values perform one full synchronous Bellman sweep, explicitly stated in the UI.
- The selected-state panel displays all four expected action returns and their maximum. After a sweep it uses the prior values that produced that backup; inspection previews the next backup.
- Live heatmap, value table, derived-policy grid/table, convergence threshold, and greedy route extraction.
- Routes are computed from the chosen start, with cycle/unreachable outcomes reported. Before convergence, the policy is labeled provisional.
- Manual, empty, and randomized obstacle maps; editable start, goal, pit, discount, and rewards.

Rewards are paid on entry and terminal continuation values are zero. Terminal tiles show entry rewards; value tables show actual V values. These conventions prevent terminal rewards being counted twice.

## Visual and animation review

All three exact reference PNGs were opened before implementation and revisited during browser QA. The pages reproduce their control/visualization/analytics compositions, stacked HMM panels, iteration grids, bottom explanations, state colors, tables, equation panels, and progress indicators. Follow-up passes adjusted panel spacing, action layouts, the heatmap scale, mobile help, and graph readability.

The reference screenshots show partly completed computations and contain illustrative numerical inconsistencies. These pages start from genuine initial state and display calculated results. They do not hardcode the reference's example numbers. Font rendering, simplified icons, and variable-length explanations differ from the raster images; the implementation is not pixel-identical.

At a measured desktop viewport of 1448×1086 and a narrow viewport of approximately 391×844, all three pages fit the available width without page-level horizontal overflow. Large trellises and matrices scroll inside their panels. Seven-by-seven grids were also checked on mobile.

GSAP motion follows engine events: probability edges, active trellis cells, decoded backtracking, individual value updates, changed policy arrows, sweep/phase indicators, numerical interpolation, heatmap transitions, and completion emphasis. Reduced-motion preferences are respected.

## Validation

- **16 JavaScript test groups passed** across Phases 1–4.
- **122 Android unit tests passed**, zero failures/errors/skips.
- HMM Forward/Viterbi checked against exhaustive enumeration, backpointers, invalid distributions, impossible observations, and 2,000-observation numerical stability.
- Policy evaluation/improvement checked for single-state updates, changed actions, stability, and agreement with Value Iteration under deterministic and noisy transitions.
- Value Iteration checked for synchronous backups, convergence, extracted routes, and reset.
- Browser checks covered execution, phase buttons, map edits, terminal removal, model changes, invalid HMM input, sequence typing, comparison, graph selection, keyboard trellis selection, derived-policy view, and help.
- Browser regression checks: MDP Explorer, SARSA, Q-learning comparison, and Alpha-Beta loaded and stepped without console errors.
- JavaScript syntax checks and `git diff --check` passed. This bundle does not use TypeScript.
- `:app:testDebugUnitTest :app:assembleDebug --offline` succeeded.
- Debug APK installed successfully on the connected Android device.

## Implementation map

| File | Purpose |
| --- | --- |
| `phase4_labs/hmm-engine.js` | Stable Forward/Viterbi events and validation |
| `phase4_labs/policy-engine.js` | Evaluation, improvement, stability, route extraction |
| `phase4_labs/hmm.js` | HMM controls, diagrams, matrices, trace and explanations |
| `phase4_labs/iterations.js` | Policy and Value Iteration pages |
| `phase4_labs/common.js` | Playback, probability formatting, matrices and help |
| `phase4_labs/labs.css` | Reference-based and responsive layouts |
| `phase3_labs/gridworld.js` | Shared editable environment and expected returns |
| `phase3_labs/mdp-engine.js` | Reused Value Iteration engine |
| `app/src/test/phase4-engines.test.js` | Algorithm correctness and shared-core regression checks |

Earlier uncommitted project changes were preserved.
