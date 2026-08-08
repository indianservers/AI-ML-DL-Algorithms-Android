# Phase 1 Interactive Algorithm Audit

## Existing Learn Module

- Navigation: `LearnModuleScreen.kt` shows a searchable catalog, domain/section accordions, and an in-memory selected topic. Back navigation returns from a lesson to the catalog.
- Catalog/models: `LearnCatalog.kt` defines `LearnDomain`, `LearnSection`, `LearnTopic`, `LearningProfile`, and broad `VisualizationKind` routing for 240+ algorithms.
- Lesson screens: existing lesson stages are `Understand`, `Visualize`, `Explore`, `Train`, `Predict`, `Experiment`, `Compare`, and `Test`.
- Visualization components: `MlLabComponents.kt` provides shared panels, buttons, metric pills, `DatasetGraph`, `LossChart`, and a dark scientific visual language.
- Canvas usage: visualizations are Compose `Canvas` implementations. Existing canvases are illustrative and mostly generic.
- Compose usage: the Learn module is Jetpack Compose only. No XML screen implementation was found for Learn.
- ViewModel architecture: no Learn-specific ViewModel exists; state is local Compose state with shared preferences for completion.
- Repository/data layer: catalog content is static Kotlin data. No external repository is used for Learn lessons.
- Existing datasets: prior Learn visualizations generated synthetic points inside composables; no reusable dataset generator existed.
- Existing ML implementation: `PhaseOneEngines.kt` had basic linear regression, metrics, logistic probability, KNN, Gaussian NB, K-Means, PCA, and small penalty helpers.
- Training engine: only simple linear-regression snapshots existed; no general reusable training-step model or GD mode comparison existed.
- Animation utilities: no reusable animation primitives were present in the Learn module.
- Graph utilities: `DatasetGraph` and `LossChart` are reusable but narrow.
- Theme system: `MlLabComponents.kt` supplies semantic dark lab colors; app theme is in `ui/theme`.
- Dark/light support: the lab currently uses a dark custom palette regardless of Material light/dark state.
- Reusable controls: `SegmentedOption`, `GradientButton`, `MetricPill`, and `SectionTitle` exist; algorithm parameter metadata did not.

## Reusable Components Kept

- `GlassPanel`, `GradientButton`, `SegmentedOption`, `MetricPill`, `SectionTitle`
- Existing catalog navigation and completion storage
- Existing lesson flow for all non-Phase-1 algorithms
- Existing broad `LearnCatalog` taxonomy

## Missing Functionality Before Phase 1

- No dedicated Phase 1 algorithm lab screen
- No reusable interactive dataset canvas with tap, drag, select, delete, query point, residuals, decision boundary, split, and neighbor layers
- No dataset preset generator with deterministic seed/noise/sample controls
- No strongly typed parameter/control architecture
- No reusable training-step model for teaching current iteration state
- No Batch vs SGD vs Mini-Batch comparison
- No coefficient shrinkage chart for Ridge/Lasso/Elastic Net
- No confusion matrix component
- No tree impurity/split computation exposed to the UI
- No tests for Phase 1 numerical correctness across the requested algorithms

## Architectural Changes Required

- Add a Phase 1 topic matcher that intercepts only the requested algorithms.
- Add a reusable pure Kotlin math/state engine for deterministic datasets, regression fits, regularization, classification metrics, KNN, tree split criteria, perceptron update, and GD paths.
- Add a Phase 1 Compose lab screen with standard sections: Learn, Visualize, Train, Experiment, Metrics.
- Keep the original generic Learn screen as fallback for future phases and unrelated algorithms.

## Files Modified

- `app/src/main/java/com/indianservers/ai_ml_dl_algorithms/ml_lab/learn/LearnModuleScreen.kt`

## Files Created

- `app/src/main/java/com/indianservers/ai_ml_dl_algorithms/ml_lab/learn/interactive/PhaseOneInteractiveEngines.kt`
- `app/src/main/java/com/indianservers/ai_ml_dl_algorithms/ml_lab/learn/interactive/PhaseOneAlgorithmLab.kt`
- `app/src/test/java/com/indianservers/ai_ml_dl_algorithms/PhaseOneInteractiveEngineTest.kt`
- `PHASE1_INTERACTIVE_ALGORITHMS_REPORT.md`
- `PHASE1_INTERACTIVE_ALGORITHM_AUDIT.md`

## Implementation Boundary

Phase 1 implements only:

- Simple Linear Regression
- Multiple Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Logistic Regression
- K-Nearest Neighbors
- Perceptron
- Decision Tree Classification
- Decision Tree Regression
- Batch Gradient Descent
- Stochastic Gradient Descent
- Mini-Batch Gradient Descent

Phase 2+ algorithms remain on the existing generic lesson path.
