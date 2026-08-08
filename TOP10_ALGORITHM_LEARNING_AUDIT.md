# Top 10 Algorithm Learning Audit

| Algorithm | Theory | Visualization | Interactive | Training | Math | Experiments | Tests | Quality |
| --------- | ------ | ------------- | ----------- | -------- | ---- | ----------- | ----- | ------- |
| Linear Regression | Strong | Working scatter, line, residuals | Points, slope, intercept, dataset controls | Gradient descent path | Equation, MSE, MAE, R2 | Outliers, nonlinear data, learning-rate failure | Phase 1 engine tests | Flagship-ready |
| Logistic Regression | Strong | Working probability field and boundary | Threshold and data controls | Classification metrics update live | Sigmoid, z score, confusion metrics | Overlap, XOR-like, imbalance | Phase 1 engine tests | Strong |
| KNN | Strong | Working query/neighbour lines | Query point, K, distance metric | Lazy learner, no fitting loop needed | Distance and majority vote | K too small/large, metric changes | Phase 1 engine tests | Strong |
| Decision Tree | Strong | Dataset split plus tree summary | Criterion, data controls, split inspection | Best split recomputes live | Gini/entropy and split counts | Underfit/overfit shown through split/depth concepts | Phase 1 engine tests | Good |
| Random Forest | Present | Ensemble/bootstrapping lab exists | Tree/member inspection and votes | Ensemble state recomputes | Bootstrap, feature subset, vote distribution | Noisy ensemble and label noise | Phase 2 tests | Improved route to flagship |
| SVM | Present | Margin, support vector, kernel visual states exist | C/kernel controls in Phase 2 lab | Hinge-loss state | Margin width, hinge loss, support vectors | Linear vs circular/RBF demos | Phase 2 tests | Strong |
| K-Means | Strong | Centroid assignment and history | K, presets, iterations, K-Means++ | Assign/move loop | Inertia objective | Circles/two moons failure | Phase 3 tests | Strong |
| ANN / MLP | Strong | Layered network, neuron and backprop views | Activation/loss/optimizer controls | Forward/backprop and training traces | Weighted sum, activation, gradients | XOR/nonlinear playground | Phase 5 tests | Strong |
| CNN | Strong | 8x8 image, kernels, feature maps, pooling | Shape, kernel, stride/pooling/classifier controls | Tiny local classifier | Convolution patch math and softmax | Noise, filter choice, pooling tradeoffs | Phase 6 tests | Flagship-ready |
| RNN / LSTM | Strong | Sequence timeline, heatmap, gate views | Sequence length, gates, hidden size, clipping | RNN/LSTM/GRU traces | Hidden state, BPTT, gate equations | Vanishing gradient and delayed memory | Phase 7 tests | Flagship-ready |

## Findings

- The Learn module already contains premium work for most of the Top 10, but the entry point buried them in a very large catalog.
- Random Forest from the Supervised Learning classification section did not route to the existing Random Forest interactive lab; that is now fixed.
- The catalog needed a curated Top 10 entry point so students encounter the flagship labs first.
- Other algorithms should remain in the hierarchy, but they should not visually compete with the Top 10 on the first screen.

## Changes Made From Audit

- Added `LearnCatalog.flagshipTopics` with the ten selected algorithms in the requested order.
- Added a Top 10 Flagship Labs panel above the full Learn hierarchy.
- Added concise actual-state promises for each flagship card.
- Routed supervised `Random Forest` into the Phase 2 ensemble lab.
