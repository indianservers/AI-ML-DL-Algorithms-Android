# Phase 2 Interactive Algorithms Report

## Completion Matrix

| Algorithm | Interactive | Math | Dataset Presets | Parameters | Training Steps | Boundary/Visualization | Metrics | Break-It Case | Tests | Status |
| --------- | ----------- | ---- | --------------- | ---------- | -------------- | ---------------------- | ------- | ------------- | ----- | ------ |
| Gaussian Naive Bayes | Yes | Yes | Yes | Yes | Recompute | Prior x likelihood -> posterior, Gaussian summaries, distribution panel | Yes | Correlated features | Yes | Implemented |
| Multinomial Naive Bayes | Yes | Yes | Text counts | Yes | Recompute | Word count vector, conditional probabilities, posterior scores | Yes | Correlated-feature explanation | Yes | Implemented |
| Bernoulli Naive Bayes | Yes | Yes | Binary text features | Yes | Recompute | Feature presence/absence likelihoods | Yes | Correlated-feature explanation | Yes | Implemented |
| LDA | Yes | Yes | Yes | Yes | Recompute | Projection score, shared-covariance explanation, class summaries | Yes | Different variances | Yes | Implemented |
| QDA | Yes | Yes | Yes | Yes | Recompute | Class-specific covariance summaries and posterior state | Yes | Different covariance shapes | Yes | Implemented |
| Linear SVM | Yes | Yes | Yes | C | Yes | Hyperplane, margins, support vectors, violations | Yes | Circular data | Yes | Implemented |
| Kernel SVM | Yes | Yes | Yes | Kernel, C, gamma, degree | Yes | Kernel trick state, RBF gamma warning, nonlinear presets | Yes | High-gamma/noisy case | Yes | Implemented |
| Soft Margin SVM | Yes | Yes | Yes | C | Yes | Margin violations, hinge-loss penalty | Yes | Outlier/overlap case | Yes | Implemented through SVM topic controls |
| SGD Classifier | Yes | Yes | Yes | Learning rate | Yes | Sample update, hinge loss, boundary movement explanation | Yes | Overlap/noise | Yes | Implemented |
| Bagging Classifier | Yes | Yes | Yes | Trees | Yes | Bootstrap frequencies, OOB samples, votes | Yes | Noisy ensemble | Yes | Implemented |
| Random Forest Classifier | Yes | Yes | Yes | Trees | Yes | Bootstrap + random feature subsets, selected tree, votes | Yes | Noisy/label-noise cases | Yes | Implemented |
| Extra Trees Classifier | Yes | Yes | Yes | Trees | Yes | Randomized thresholds vs optimized-ish forest thresholds | Yes | Noisy ensemble | Yes | Implemented |
| AdaBoost Classifier | Yes | Yes | Yes | Rounds via steps | Yes | Sample weights, stumps, weighted error, learner weight | Yes | Label noise | Yes | Implemented |
| Voting Classifier | Yes | Yes | Yes | Hard/soft mode | Recompute | Hard votes and soft probability averaging | Yes | Disagreement case | Yes | Implemented |
| Stacking Classifier | Yes | Yes | Yes | Recompute | Recompute | Base predictions -> meta-features -> meta-model flow | Yes | Leakage warning | Yes | Implemented |

## Phase 1 Components Reused

- Catalog routing through `LearnModuleScreen`
- Learn completion state and depth labels
- Compose visual language and shared `GlassPanel`, `SegmentedOption`, `MetricPill`, `SectionTitle`, `GradientButton`
- Pure Kotlin deterministic engine approach
- Training-step state model from Phase 1
- Dataset point model `LabPoint`

## New Shared Components

- `PhaseTwoTopicMatcher`
- `PhaseTwoDatasets`
- `PhaseTwoEngines`
- `MultiClassMetrics` and `PerClassMetric`
- `MulticlassCanvas`
- `MulticlassConfusionMatrix`
- `GaussianDistributionVisualizer`
- `TextNaiveBayesPanel`
- `EnsembleVotes`
- `MisclassificationInspector`
- `BootstrapState`, `EnsembleMemberState`, `EnsembleState`, `AdaBoostRound`

## Technical Debt Removed

- Phase 2 avoids adding another generic fake classifier visualization. Model panels now expose algorithm-specific state.
- Multiclass metrics are explicit and separate from binary metrics.
- Ensemble state is strongly typed instead of free-form maps.

## Performance Observations

- Phase 2 avoids dense per-frame decision grids and uses bounded point/model counts.
- Forest visualizations show selected tree plus aggregate summary rather than drawing all trees.
- Tree counts are bounded to 50 computational members with only a few member rows rendered.
- No long-running training jobs are introduced, so rapid slider updates stay lightweight.

## Architecture Extensions

- Centralized Phase 2 matcher for routing.
- Multiclass dataset presets for 2, 3, and 4 classes.
- Shared posterior, covariance, SVM, kernel, bootstrap, ensemble, and boosting states.
- Comparison tab for same-dataset classifier comparison.
- Break-It mode for assumption-violating datasets.

## Remaining Algorithms

Later phases should still implement Gradient Boosting, XGBoost, LightGBM, CatBoost, clustering, dimensionality reduction, anomaly detection, time series, deep learning, reinforcement learning, GANs, diffusion models, and other advanced topics.

## Known Limitations

- The catalog contains one `Support Vector Machine` topic, so Linear SVM, Kernel SVM, and soft-margin controls are exposed inside that SVM lab rather than three separate catalog rows.
- Ensemble learners use educational stumps/bootstrap states for clarity and speed, not production-grade tree induction.
- Adaptive cancellable decision-grid rendering is documented as future foundation work.
