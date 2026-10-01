# AI Algorithms Virtual Labs — Phase 1

## Existing app audit

The project is an Android Jetpack Compose app. Its algorithm library lives in `LearnCatalog` and `LearnModuleScreen`; it has no browser router. A code search found no BFS, Bidirectional Search, Beam Search, shared graph search engine, or GSAP dependency. Existing lesson WebViews provided a compatible integration pattern. Existing studios and their routes were left intact.

## Delivery

Added an **AI Algorithms / Search** section with Breadth First Search, Bidirectional Search, and Beam Search. Opening one of these topics loads a local WebView asset. The pages are self-contained and work offline, including the bundled GSAP 3.15 core. The three pages are selected by the local `lab` query parameter, so direct reload of each asset URL preserves the lab.

The desktop compositions follow reference images `15 - Breadth First Search.png`, `16 - Bidirectional Search.png`, and `17 - Beam Search.png`. At narrow widths the panels stack. The search state is produced by pure engines in `engines.js`; controls, visualization, tables, chart, explanations, and progress read that state after each logical step. Run can be paused, speed adjusted, and Reset reconstructs the engine. GSAP animates queue/frontier updates, candidate arrival, selected nodes, and path chips; animation is skipped for a reduced-motion preference.

The bidirectional mockup repeats the label F for two different graph nodes. The center node is named **M** in the implementation so every node has a unique identity and paths can be reconstructed unambiguously.

## Verification

- `node --test app/src/test/search-engines.test.js` — 3 tests pass. These cover BFS FIFO and shortest path, bidirectional shortest path for every start/goal pair in the preset, and Beam candidate scoring/ranking/width.
- `./gradlew.bat :app:compileDebugKotlin --offline` — successful.
- `./gradlew.bat :app:assembleDebug --offline` — successful. APK installed and launched on `emulator-5554`.
- Browser visual checks at the reference desktop width and a narrow width. Step and goal completion were exercised for all three labs. No browser console errors were reported.

## Technical boundary

GSAP is a JavaScript animation library and cannot run directly in Compose. The native app hosts these three detailed labs in local WebViews; other app screens remain Compose. The reference images depict a wide desktop UI, so a phone presents the same controls and state in a stacked layout.
