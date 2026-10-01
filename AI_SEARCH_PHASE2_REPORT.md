# AI Algorithms Virtual Labs — Phase 2

## Audit and integration

The app is native Android Compose. Simulated Annealing and Genetic Algorithm already had entries in the Evolutionary Algorithms catalog, but no dedicated engine or interactive page. Monte Carlo Tree Search had no catalog entry or engine. No Three.js or charting package was present. Phase 1 introduced an offline WebView lab bundle with GSAP 3.15; Phase 2 follows that pattern in a separate asset folder so Phase 1 styles and logic remain unchanged.

The existing Simulated Annealing and Genetic Algorithm topic IDs are reused in a new AI Algorithms / Optimization & Evolution section. Their original Evolutionary Algorithms entries remain available. Monte Carlo Tree Search has one new topic under AI Algorithms / Search. The topic list deduplicates reused IDs. The local lab URL uses a `lab` query parameter, so refreshing a lab retains its page.

## Engines and displays

- **Simulated Annealing:** Seeded two-dimensional optimization using Himmelblau, Rastrigin, Ackley, or Rosenbrock. Generate, evaluate, update, cool, and convergence check are separate Step actions. The Metropolis decision uses the current temperature and a recorded random draw. A live canvas renders a computed surface or contour, trajectory, candidate and best solution. Charts and move history come from engine state.
- **Genetic Algorithm:** Seeded binary chromosomes with configurable population, length, selection, crossover, mutation, fitness, and elitism. Step advances one evolutionary stage; Run Generation executes those same steps to replacement. Fitness leaderboard, chromosome stages, fitness trend, and pairwise Hamming diversity are computed from the population.
- **Monte Carlo Tree Search:** Tic-Tac-Toe with selection, one-node expansion, legal rollout, and backpropagation as distinct steps. Q is the mean reward from the root player's view; selection uses Q at root-player turns and −Q at opponent turns, plus `C × sqrt(ln(parent visits) / child visits)`. The tree, action table, rollout boards, result counts, and value trend are generated from actual search nodes. The recommended action is the root child with the most visits.

The UI composition follows reference images `18 - Simulated Annealing.png`, `19 - Genetic Algorithm.png`, and `20 - Monte Carlo Tree Search.png`. GSAP animates state-driven transitions and is skipped under `prefers-reduced-motion`. The annealing surface is computed by a canvas mesh because the project has no existing 3D library; it supports drag rotation, zoom, contour switching, value inspection, fit, and fullscreen.

## Verification

- `node --test app/src/test/phase2-engines.test.js` — 3 tests pass, covering acceptance and cooling, GA cycle and elitism, and MCTS rollout/backpropagation/UCB.
- `./gradlew.bat :app:testDebugUnitTest --offline` — 120 unit tests pass. The catalog's domain count assertion was updated from 14 to 15 to account for the AI Algorithms domain added in Phase 1.
- `./gradlew.bat :app:assembleDebug --offline` — successful. APK installed and launched on `emulator-5554`.
- Browser visual comparison at the mockup desktop width, plus interaction checks for each lab and a console error check. No browser console errors were observed.

The images depict simulations after many steps; each lab opens at an honest initial state and reaches comparable visual density as its engine runs.
