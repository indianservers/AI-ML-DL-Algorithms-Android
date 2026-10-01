# AI Algorithms Virtual Labs — Phase 3

## Delivered

- Alpha-Beta Pruning, based on reference 21.
- Markov Decision Process Explorer, based on reference 22.
- SARSA Learning, based on reference 23.

The three labs use offline Android WebView assets, as in Phases 1 and 2. Their engines are separate from the page renderers. No external requests or new chart libraries are needed.

## Audit and reuse

The repository had no Minimax or Alpha-Beta engine. MDP and SARSA existed as learning topics, while Value Iteration and Policy Iteration were future entries. The existing Kotlin `GridWorld` in `phase5/engine/PracticalAlgorithms.kt` combines a deterministic 4×4 environment and a batch Q-learning loop. It does not expose transitions, obstacles, entry rewards, or single-step events.

Phase 3 therefore adds a shared JavaScript Gridworld environment for its two WebView labs. Value Iteration and SARSA/Q-learning operate independently on that environment. The existing Kotlin Q-learning implementation and its tests remain unchanged. Phase 2's SVG chart helper and the existing local GSAP bundle are reused without modifying them.

## Navigation

Alpha-Beta is a new AI Algorithms search topic. The AI menu also lists MDP Explorer and SARSA Learning using the IDs of the original Markov Decision Process and SARSA topics. Both old names and new display names resolve to the same respective asset route. All nine AI lab routes are explicit and tested; unrelated topics retain their existing screen selection.

Assets live in `app/src/main/assets/phase3_labs/`:

| File | Responsibility |
| --- | --- |
| `gridworld.js` | States, actions, legal movement, rewards, slip outcomes, sampling |
| `alpha-beta-engine.js` | Tree generation, depth-first Minimax events, bounds and pruning |
| `mdp-engine.js` | Synchronous Bellman sweeps, expected action returns, policy extraction |
| `sarsa-engine.js` | ε-greedy behavior, SARSA and Q-learning targets, Q updates and episodes |
| `common.js` | Reused charts, playback, controls, grid rendering and animation helpers |
| `alpha-beta.js`, `mdp.js`, `sarsa.js` | Separate lab layouts and interactions |
| `labs.css`, `index.html` | Responsive presentation and offline dependencies |

## Algorithm behavior

### Alpha-Beta

Step exposes node entry, leaf evaluation, child return, bound update, cutoff check, subtree pruning, backup, and the final choice. MAX updates α; MIN updates β. A cutoff marks all unvisited descendants and counts skipped leaves. Cut branch values are labeled as upper or lower bounds rather than exact values.

Controls support depth 2–4, MAX/MIN roots, pruning on/off, fixed or seeded random utilities, custom leaf utilities, and three move orders. Best First is explicitly labeled as an oracle demonstration because it uses complete Minimax values to illustrate ideal ordering. Nodes and table rows can be selected to inspect bounds. Help, documentation, and settings buttons work.

The default example returns move B, value 5, with 8 of 10 leaves evaluated. Best First evaluates 6 leaves. Full Minimax evaluates all 10 and returns the same root value. The reference image contains inconsistent sample utilities and work-saving counts; the implementation reports calculated results.

### MDP Explorer

The learner can select states and actions, inspect merged stochastic outcomes and expected returns, and change rewards, terminal configurations, γ, or total slip probability. Collisions remain in the same state. Slip probability is split equally between the two perpendicular actions. Outcome paths and labels show their actual probabilities.

Each Step performs one synchronous Bellman sweep using the previous value array. Run repeats those same sweeps until max |ΔV| is below 0.001. Terminal values are zero; their displayed goal/hole rewards are paid on entry. The initial policy is greedy with respect to the initial estimates, not a precomputed solution. State values have table and heatmap views, and policy arrows can be hidden.

### SARSA

The 7×7 environment includes the reference's obstacles, penalty cells, start, goal, a custom SVG robot, trajectory, selectable Q values, and a greedy-policy mini-grid. Logical steps choose an action, sample movement and the actual next action, apply the Q update, and continue or finish the episode.

SARSA bootstraps from the Q value of that actual next action and carries the action forward into the next transition. The comparison mode executes Q-learning with a maximum-next-Q target in the same environment. Switching modes resets to a seeded experiment. Terminal transitions use zero bootstrap; time-limit truncations retain a nonterminal bootstrap value.

Run Episode stops after the current episode. Train advances batches of 20 of the same logical steps per tick. Episodes stop at the goal or 150 moves. The live panels show the sampled transition, old Q, reward, next Q, target, TD error, and new Q. Epsilon decay and its floor feed actual action selection. The reward chart uses completed episode rewards and a 20-episode moving average. Finishing a training budget is not labeled as proof of convergence.

## Visual and interaction review

All three reference PNGs were opened before implementation. Browser review covered the separate page compositions, tree and grid geometry, selected states, analytics, lower panels, and state-driven GSAP motion. A second pass corrected inconsistent grid-cell heights, sidebar spacing, custom leaf application, and transition-path visibility.

The layout retains the references' control/visualization/analytics structure and color families, with responsive stacking for narrow screens. At 1448×1086 and 390×844, all three pages fit the viewport width without page-level horizontal overflow. Dense tree content scrolls inside its own panel on small screens. Page height can exceed the reference image because explanations and live data vary. Browser font and Android WebView rendering can differ from the PNG; this is not a pixel-identical raster reproduction.

GSAP animates evaluated nodes, bound changes, pruned descendants, final principal variation, Gridworld outcome emphasis, changing policy/value estimates, robot movement, transition actions, Q updates, and reward-chart progression. The pages respect the system reduced-motion preference.

## Validation

- Five Phase 3 Node test groups cover Minimax equivalence, cutoff behavior and move ordering; normalized transitions and rewards; Bellman updates and convergence; ε-greedy behavior; the on-policy next action and TD update; terminal bootstrap; Q-learning comparison; episode limits; reset; and epsilon decay.
- Existing Phase 1 and Phase 2 engine tests are included in the final test run.
- Android unit tests include the native Q-learning regression test and a new test for reused topic IDs and nine distinct AI lab routes.
- JavaScript syntax checks and the offline Android debug build pass.
- Browser checks exercised Alpha-Beta completion, help, custom utilities, depth and pruning; MDP sweeps, auto-run, noise, cell selection, environment/reward/terminal presets, heatmap and arrow toggle; and SARSA updates, cell selection, training, comparison, and reset.
- A deterministic MDP browser run converged in nine sweeps. A SARSA browser run completed ten episodes and produced a real reward chart.
- All six Phase 1/2 pages loaded with their expected titles and no browser console errors. Phase 3 browser checks likewise reported no console errors.
- Final totals: 11 JavaScript tests and 121 Android unit tests passed, with zero failures. The final debug APK was installed successfully on the connected emulator.

There are no TypeScript files in these lab bundles; JavaScript syntax, engine behavior, browser execution, and Kotlin compilation are the relevant checks.
