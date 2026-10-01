# Stage 1 supervised visualization engineering report

Date: 2026-10-01. Scope: catalog audit, explicit routing, and the Supervised Learning Regression and Classification placements. Other categories were audited but not changed.

## A. Catalog audit

The compiled `LearnCatalog` contains **320 unique topic IDs**, **327 visible placements**, **15 categories**, and **37 sections**. The full per-topic inventory is in [AlgorithmVisualizationAudit.md](AlgorithmVisualizationAudit.md). The compiled registry currently routes 40 supervised placements to the new native screen, 90 other placements to existing native labs, 25 to existing offline Web labs, and 165 to the old general lab.

Conservative audit status across unique IDs: **0 Correct, 155 Partial, 165 Generic, 0 Incorrect, 0 Missing**. “Partial” means an algorithm-specific route exists but the entire mathematical, interaction, accessibility, and device checklist has not been independently certified. “Generic” identifies the old general implementation; it no longer masquerades as algorithm-specific in the Visualization tab. These counts are not a claim that 155 views are finished.

## B. Architecture

`AlgorithmVisualizationRegistry.kt` explicitly registers every unique topic ID and records implementation, family, controls, dataset type, interaction, animation, graphic, and audit status. A test fails if a supervised placement has no binding. `LearnModuleScreen.kt` uses the registry for Visualization routing while retaining the current Learn, Train & Inference, Quiz, and other-category routes.

The new native Compose drawing infrastructure provides Cartesian plots, draggable samples, fitted functions, residuals, probability fields, decision regions, partition and tree drawings, uncertainty bands, margins, neighbor links, covariance ellipses, coefficient plots, and ensemble cards. Calculations live in the statistical, tree, and boosting engines; learned slopes, coefficients, tree splits, margins, and predictions are outputs of those engines.

## C. Supervised Regression

| Algorithm | Implemented mechanism visible in Visualization |
|---|---|
| Simple Linear Regression | Least-squares line, residuals, equation, R² and MSE update from samples. |
| Multiple Linear Regression | Two-feature fitted plane, samples, heatmap, rotatable projection, and a feature slice at held X2. |
| Polynomial Regression | Degree-controlled nonlinear least-squares curve and residuals. |
| Ridge Regression | L2 fit, coefficient magnitudes and paths as regularization changes. |
| Lasso Regression | L1 fit, coefficient paths, zeroed coefficients and active-feature count. |
| Elastic Net Regression | Combined L1/L2 fit and coefficient paths with strength and ratio controls. |
| Logistic Regression | Binary classification probability field, sigmoid and learned threshold boundary. |
| Bayesian Linear Regression | Posterior predictive mean, credible band and sampled lines. |
| Quantile Regression | Selected conditional quantile with lower and upper reference fits. |
| Robust Regression | Huber fit compared with ordinary least squares on outliers. |
| Support Vector Regression | Kernel fit, ε tube and support-vector emphasis. |
| Decision Tree Regression | Recursive learned splits, piecewise-constant prediction and actual tree nodes. |
| Random Forest Regression | Multiple fitted trees, their individual predictions and aggregate mean. |
| Extra Trees Regression | Randomized threshold trees and their aggregate prediction. |
| Gradient Boosting Regression | Sequential residual-tree corrections and cumulative prediction. |
| AdaBoost Regression | Changing sample weights, weak predictors and weighted aggregate. |
| XGBoost Regression | Gradient/Hessian contributions, regularized split gain and staged trees. |
| LightGBM Regression | Binned features and highest-gain leaf-wise tree growth. |
| CatBoost Regression | Ordered categorical target statistics and symmetric-tree stages. |
| K-Nearest Neighbors Regression | Query point, nearest-sample links and averaged target prediction. |
| Gaussian Process Regression | Kernel posterior mean, confidence band and sampled functions. |

## D. Supervised Classification

| Algorithm | Implemented mechanism visible in Visualization |
|---|---|
| Logistic Regression | Probability field, sigmoid and learned decision boundary; shared engine with its regression-category placement. |
| K-Nearest Neighbors | Query point, K neighbor links, distance/radius and class votes. |
| Gaussian Naive Bayes | Class Gaussian densities, independent-feature factors and posterior result. |
| Multinomial Naive Bayes | Token-count documents, class word likelihoods and log posterior scores. |
| Bernoulli Naive Bayes | Binary presence/absence features, class probabilities and log posterior scores. |
| Decision Tree | Recursive axis-aligned rectangular regions and matching tree nodes. |
| Random Forest | Bootstrap trees, individual votes and majority result. |
| Extra Trees | Randomized split trees and aggregate vote. |
| Support Vector Machine | Linear or kernel boundary, margins and highlighted support vectors. |
| Linear Discriminant Analysis | Pooled-covariance projection and class separation along the learned axis. |
| Quadratic Discriminant Analysis | Class-specific covariance ellipses and quadratic decision boundary. |
| Perceptron | Mistake-driven boundary updates with Previous, Next and Auto controls. |
| SGD Classifier | Single-sample gradient update sequence and loss history. |
| AdaBoost | Weighted samples, weak stumps and staged vote. |
| Gradient Boosting | Pseudo-residual tree corrections and cumulative class probability. |
| XGBoost | Gradient/Hessian bars, regularized tree corrections and class score. |
| LightGBM | Feature histogram and leaf-wise highest-gain growth. |
| CatBoost | Ordered category statistics and symmetric-tree stages. |
| Gaussian Process Classifier | Latent kernel estimate, class probability and uncertainty field. |

## E. Emulator verification

The app was installed on the isolated Android emulator `emulator-5556` at 1080 × 2400 in portrait. `outputs/stage1-verification/full-results-second-pass.csv` records all **40/40** placements opened, their Visualization tab opened, interaction attempted, page scrolled, and no app crash. Numeric dataset presets were changed in **38/38** applicable placements; the two text Naive Bayes placements used token/binary controls. An actual plotted-point drag was checked separately for **37/37** applicable two-dimensional charts in `drag-results.csv`; Multiple Linear Regression was checked through its plane/slice controls and a later targeted sweep. The six latest regression and AdaBoost changes were rechecked on the final APK in `results.csv` with **6/6** opened, preset, interaction, scroll, control and crash checks passing. Earlier targeted sweeps also covered eight boosting and eight tree/step views.

Screenshots for each placement are under `outputs/stage1-verification/` with `-top`, `-dataset`, `-interaction`, `-scroll`, and `-bottom` suffixes. Selected renderings were inspected for readable labels, portrait fit and horizontal overflow. The latest logcat search found no app `FATAL EXCEPTION`. Automated interaction and screenshot differences confirm screen response, while the audit remains Partial because they do not certify every mathematical or accessibility behavior on every device.


### Placement checklist

“Model response” records a visible change after preset, point, token, or control interaction; it is not a full mathematical proof. All rows were checked in portrait with page scrolling. Back navigation was spot checked from Visualization through Learn to Home. `Text` means the Naive Bayes token/binary input rather than a numeric preset.

| Placement | Opened | Interaction | Dataset | Model response | UI |
|---|---|---|---|---|---|
| Regression / Simple Linear Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Multiple Linear Regression | Yes | Plane/slice controls | Preset | Visual response | Portrait/scroll |
| Regression / Polynomial Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Ridge Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Lasso Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Elastic Net Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Logistic Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Bayesian Linear Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Quantile Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Robust Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Support Vector Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Decision Tree Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Random Forest Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Extra Trees Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Gradient Boosting Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / AdaBoost Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / XGBoost Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / LightGBM Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / CatBoost Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / K-Nearest Neighbors Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Regression / Gaussian Process Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Logistic Regression | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / K-Nearest Neighbors | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Gaussian Naive Bayes | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Multinomial Naive Bayes | Yes | Token toggle | Text | Visual response | Portrait/scroll |
| Classification / Bernoulli Naive Bayes | Yes | Token toggle | Text | Visual response | Portrait/scroll |
| Classification / Decision Tree | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Random Forest | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Extra Trees | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Support Vector Machine | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Linear Discriminant Analysis | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Quadratic Discriminant Analysis | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Perceptron | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / SGD Classifier | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / AdaBoost | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Gradient Boosting | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / XGBoost | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / LightGBM | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / CatBoost | Yes | Point drag | Preset | Visual response | Portrait/scroll |
| Classification / Gaussian Process Classifier | Yes | Point drag | Preset | Visual response | Portrait/scroll |

## F. Tests

`.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest --offline` completed successfully on the final source and produced `app/build/outputs/apk/debug/app-debug.apk`. The debug unit suite has **139 tests across 21 suites, 0 failures and 0 errors**. New tests cover registration completeness, CART impurity and XOR splits, regression forests, quantile/Huber fits, Gaussian-process posterior behavior, Bayesian uncertainty, linear and kernel SVM, SVR, and distinct boosting variants. The tests caught a sign error in Gaussian elimination that produced unstable GP predictions; this was fixed before final build.

No instrumentation-test suite was run. Emulator checks above exercised the installed app.

## G. Remaining scope

The later implementation stages cover the other catalog categories: Unsupervised Learning, Deep Learning, Reinforcement Learning, Computer Vision, Natural Language Processing, Time Series, Probabilistic & Bayesian, Optimization, Ensemble Learning, and the remaining catalog sections. Their legacy or existing routes are itemized in the audit. Supervised views should retain a Partial audit label until a deeper independent mathematical and accessibility certification is completed.
