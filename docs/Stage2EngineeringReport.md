# Phase 2 visualization engineering report

Date: 2026-10-01. Scope: the 60 catalog placements in Unsupervised Learning, Semi-Supervised Learning, and Ensemble Learning. Phase 1 supervised implementations were retained and reused where the same algorithm appears in Ensemble Learning.

## Implementation

The explicit registry routes all 60 Phase 2 placements to native Compose algorithm views. A test checks the catalog placement count and every route. None of these placements enters the legacy general visualization screen.

| Area | Changed files |
|---|---|
| Routing | `AlgorithmVisualizationRegistry.kt`, `LearnModuleScreen.kt`, `StageTwoVisualization.kt`, `StageTwoVisualizationScreen.kt` |
| Shared drawing and model | `StageTwoPrimitives.kt`, `PhaseThreeInteractiveEngines.kt` |
| Clustering | `StageTwoClusteringEngine.kt`, `StageTwoClusteringPanel.kt` |
| Reduction | `StageTwoReductionEngine.kt`, `StageTwoReductionPanel.kt` |
| Association | `StageTwoAssociationEngine.kt`, `StageTwoAssociationPanel.kt` |
| Anomaly | `StageTwoAnomalyEngine.kt`, `StageTwoAnomalyPanel.kt` |
| Semi-supervised | `StageTwoSemiSupervisedEngine.kt`, `StageTwoSemiSupervisedPanel.kt` |
| Ensemble | `StageTwoEnsembleEngine.kt`, `StageTwoEnsemblePanel.kt` |
| Checks and report | `StageTwoVisualizationTest.kt`, `AlgorithmVisualizationAudit.md`, `Stage2EngineeringReport.md`, `stage2_emulator_verify.py`, `stage2_auto_verify.py` |

New code is organized by algorithm family: `StageTwoVisualization.kt`, `StageTwoVisualizationScreen.kt`, `StageTwoPrimitives.kt`, and the clustering, reduction, association, anomaly, semi-supervised, and ensemble `StageTwo*Engine.kt` and `StageTwo*Panel.kt` pairs. `AlgorithmVisualizationRegistry.kt` and `LearnModuleScreen.kt` supply the routing. `PhaseThreeInteractiveEngines.kt` now fits full two-dimensional Gaussian covariance for GMM.

The reusable native primitives include a draggable Cartesian plot, dataset selector, model-derived overlays, sliders for legitimate hyperparameters, and Reset/Previous/Next/Auto step controls. Algorithm panels add dendrograms, reachability views, covariance ellipses, soft-membership rings, feature-loading bars, matrix heatmaps, FP trees, vertical TID sets, anomaly partitions, graph propagation, and ensemble learner cards.

Seven Ensemble Learning placements reuse the corresponding Phase 1 supervised engine and view: Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost. This keeps the mathematical behavior and controls consistent across catalog placements. Bagging, Voting, Soft Voting, Hard Voting, Stacking, and Blending have their own Phase 2 panels.

## Per-placement checklist

`Verified in emulator` means the specific catalog result and Visualization tab were opened on the isolated portrait emulator, a screenshot was captured, scrolling was attempted, and the app remained active. The interaction columns record automated checks where a corresponding control was found. They are evidence of UI response, not a proof that the educational approximation is a production implementation.

| Section | Algorithm | Algorithm-specific view | Status | Checked interactions |
|---|---|---|---|---|
| Clustering | K-Means | centroid assignments and alternating updates | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | K-Means++ | distance-squared seeding and centroid updates | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Mini-Batch K-Means | highlighted batch and incremental centroid trail | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Hierarchical Clustering | linkage merges and dendrogram cut | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Agglomerative Clustering | bottom-up merges and dendrogram | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Divisive Clustering | top-down recursive splits and tree | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step |
| Clustering | DBSCAN | core-border-noise expansion and epsilon circle | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | HDBSCAN | mutual-reachability hierarchy and size-qualified clusters | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | OPTICS | reachability order linked to scatter | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Mean Shift | kernel windows and mode-seeking trails | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Gaussian Mixture Models | EM responsibilities and covariance ellipses | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Spectral Clustering | similarity graph and graph partition | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | BIRCH | streaming CF entries and CF tree | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Affinity Propagation | responsibility/availability messages and exemplars | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Clustering | Fuzzy C-Means | membership proportions and fuzzy centroids | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Dimensionality Reduction | PCA | covariance axes and 1D projection | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset |
| Dimensionality Reduction | Kernel PCA | kernel matrix and nonlinear embedding | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Dimensionality Reduction | Sparse PCA | sparse feature loadings and projection | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Dimensionality Reduction | Incremental PCA | batch-wise evolving principal axis | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step |
| Dimensionality Reduction | Truncated SVD | matrix, singular spectrum and rank-k reconstruction | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, parameter |
| Dimensionality Reduction | Factor Analysis | latent factors, loadings and unique noise | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset |
| Dimensionality Reduction | Independent Component Analysis | source/mixed/recovered signal traces | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, parameter |
| Dimensionality Reduction | t-SNE | neighbour probability affinities and evolving map | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Dimensionality Reduction | UMAP | fuzzy neighbour graph and optimized embedding | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Dimensionality Reduction | Isomap | neighbour geodesics and manifold unfolding | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Dimensionality Reduction | Locally Linear Embedding | local reconstruction weights and unfolding | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Dimensionality Reduction | Multidimensional Scaling | distance matrix and stress-minimized layout | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset |
| Dimensionality Reduction | Autoencoder-based Reduction | encoder bottleneck decoder and reconstruction | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step |
| Association Learning | Apriori | candidate itemsets and support pruning lattice | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Association Learning | FP-Growth | frequency-ordered FP tree and prefix paths | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Association Learning | ECLAT | vertical TID sets and intersections | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Association Learning | Association Rule Mining | rule support confidence lift and item graph | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Anomaly Detection | Isolation Forest | random partitions, path length and ensemble score | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Anomaly Detection | Local Outlier Factor | K-reachability density and LOF ratio | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Anomaly Detection | One-Class SVM | kernel support vectors and enclosing boundary | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Anomaly Detection | Elliptic Envelope | robust covariance ellipse and Mahalanobis score | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Anomaly Detection | Autoencoder Anomaly Detection | reconstruction and error threshold | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Anomaly Detection | Statistical Outlier Detection | distribution tails, z-score and IQR fences | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, parameter |
| Methods | Self-Training | confident pseudo-label admission by iteration | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | Label Propagation | graph probability diffusion with hard seed clamping | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | Label Spreading | soft-clamped graph probability diffusion | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | Co-Training | two feature-view models exchanging labels | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | Pseudo-Labelling | confidence-gated model predictions and retraining | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | Consistency Regularization | perturbation predictions and consistency loss | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | Mean Teacher | student and EMA-teacher predictions over time | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | FixMatch | weak prediction gate and strong augmentation training | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Methods | MixMatch | guess, sharpen and MixUp label pipeline | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Bagging | Bagging | bootstrap duplicates and independent learners | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Bagging | Random Forest | bootstrap and random-feature decision trees | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Bagging | Extra Trees | random-threshold trees and aggregate vote | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Boosting | AdaBoost | weighted mistakes and staged weak learners | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Boosting | Gradient Boosting | sequential pseudo-residual corrections | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Boosting | XGBoost | gradient/Hessian split gains and corrections | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Boosting | LightGBM | binned gradient statistics and leaf-wise growth | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Boosting | CatBoost | ordered category statistics and symmetric trees | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll |
| Combining Models | Voting | independent model class votes | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step |
| Combining Models | Soft Voting | class probability average | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step |
| Combining Models | Hard Voting | discrete class majority | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step |
| Combining Models | Stacking | out-of-fold base predictions and meta-model | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |
| Combining Models | Blending | holdout predictions and meta-model | IMPLEMENTED / VERIFIED IN EMULATOR | screen, scroll, dataset, step, parameter |

## Verification and limitations

The final offline Gradle build and unit-test run was **successful**. The unit suite reports **156 tests in 22 suites, 0 failures**. The tests cover registry exhaustiveness and representative clustering, reduction, association, anomaly, semi-supervised, and ensemble numerical behavior.

The isolated Android API 37 emulator `emulator-5556` (1080 × 2400 portrait) opened **60/60** catalog placements and their Visualization tabs. Across applicable screens, automated checks recorded 51 dataset changes, 39 step interactions, and 44 parameter interactions. The second complete pass is in `outputs/stage2-verification/full-results-second-pass.csv`; corrected long-title and final-APK rechecks are also recorded there, with the best evidence merged into `consolidated-results.csv`. Per-topic top, changed-dataset, step, control, and scrolled screenshots are in the same directory. The common Auto control advanced through at least two distinct K-Means frames and Pause stopped playback. The latest 5,000 emulator logcat lines contained no app fatal exception or app exception entry.

The separate [catalog audit](AlgorithmVisualizationAudit.md) still marks the 60 Phase 2 views **Partial** for full mathematical/product certification. Its current 320-topic routing totals are 40 supervised native, 60 Phase 2 native, 51 other native, 25 offline Web, and 144 legacy general. Cold-start search misses in targeted runs were rerun successfully; they were verification-script timing issues, not app crashes.

The mathematical audit remains conservative. In particular, HDBSCAN uses a mutual-reachability hierarchy with a size-qualified cut but does not implement condensed-tree excess-of-mass stability selection. The Factor Analysis view derives loadings from a principal direction rather than fitting a full probabilistic factor model. The autoencoder panels use a small linear educational model, and the one-class SVM uses a small approximate dual optimizer. t-SNE, UMAP, and ensemble algorithms use intentionally small datasets and educational training loops. These mechanisms are labeled as partial in the separate [catalog audit](AlgorithmVisualizationAudit.md); the emulator status above describes screen verification only.

Phase 3 and Phase 4 still need algorithm-specific work in the remaining catalog domains: Deep Learning, Reinforcement Learning, Natural Language Processing, Computer Vision, Time-Series Algorithms, Probabilistic & Bayesian Learning, Optimization Algorithms, Evolutionary Algorithms, Recommendation Algorithms, Explainable AI, and AI Algorithms.
