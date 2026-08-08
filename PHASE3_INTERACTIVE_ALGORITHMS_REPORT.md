# Phase 3 Interactive Algorithms Report

| Algorithm | Core Mechanism Visualized | Interactive Parameters | Step Mode | Dataset Presets | Metrics | Break-It Example | Comparison Mode | Tests | Status |
| --------- | ------------------------- | ---------------------- | --------- | --------------- | ------- | ---------------- | --------------- | ----- | ------ |
| Gradient Boosting Regression | residual -> weak learner -> additive correction | stages, learning rate | Yes | regression synthetic | train/test error | outliers | boosting comparison | Yes | Implemented |
| Gradient Boosting Classification | pseudo-residual concept -> additive score | stages, learning rate | Yes | classification-shaped synthetic | error proxy | label noise/outliers | boosting comparison | Yes | Implemented conceptually |
| XGBoost Concepts | gradient/hessian -> split gain -> regularization | lambda/gamma state | Yes | regression synthetic | gain | excessive regularization/noise | boosting comparison | Yes | Implemented conceptually |
| Histogram Gradient Boosting Concepts | raw values -> bins -> split statistics | bins concept | Yes | generated numeric data | bin counts | coarse bins | boosting comparison | Yes | Implemented conceptually |
| LightGBM Concepts | leaf-wise vs level-wise growth and histogram reuse | depth/leaves concept | Yes | generated numeric data | gain/error concept | deep branch risk | boosting comparison | Yes | Implemented conceptually |
| CatBoost Concepts | categorical encoding -> ordered statistics | ordered encoding concept | Yes | categorical mini-example | leakage explanation | target leakage | boosting comparison | Yes | Implemented conceptually |
| K-Means | centroid -> assignment -> update -> inertia | K, iterations, seed | Yes | blobs/moons/outliers/overlap | inertia, silhouette | two moons | K-Means vs DBSCAN/GMM | Yes | Implemented |
| K-Means++ | distance probability -> spread centers -> K-Means | K, seed | Yes | same clustering presets | inertia, probability state | poor K/nonconvex | random init comparison | Yes | Implemented |
| Mini-Batch K-Means | mini-batch sample -> centroid update | K, batch, iterations | Yes | same clustering presets | inertia | unstable mini-batches | K-Means comparison | Yes | Implemented |
| Hierarchical Agglomerative | pair merges -> dendrogram -> cut | linkage | Yes | small sampled datasets | merge height/count | chaining | K-Means comparison | Yes | Implemented |
| DBSCAN | epsilon neighborhood -> core/border/noise | epsilon, MinPts | Yes | moons/dense-sparse/noise | core/border/noise counts | varying density | K-Means comparison | Yes | Implemented |
| OPTICS | reachability distance ordering | epsilon, MinPts | Concept | density presets | reachability values | varying density | DBSCAN comparison | Yes | Implemented conceptually |
| Mean Shift | bandwidth window -> density mode | bandwidth | Yes | blob/density presets | centers/inertia proxy | bandwidth extremes | K-Means comparison | Yes | Implemented |
| Gaussian Mixture Model | soft probability -> E-step -> M-step | components, iterations | Yes | elongated/overlap | log likelihood | non-Gaussian shapes | K-Means comparison | Yes | Implemented |
| Spectral Clustering | similarity graph -> graph view -> clusters | neighbors | Concept | nonconvex presets | graph edges | graph sensitivity | K-Means/DBSCAN | Yes | Implemented conceptually |
| PCA | variance axis -> projection -> reconstruction error | axis/data | Yes | elongated/high-scale/circles | variance/error | nonlinear circles | PCA vs embeddings | Yes | Implemented |
| Kernel PCA | nonlinear transform concept | kernel concept | Concept | circles | embedding state | parameter sensitivity | PCA comparison | Yes | Implemented conceptually |
| Truncated SVD | matrix -> low-rank approximation | rank | Yes | local matrix | reconstruction error | too-low rank | matrix methods | Yes | Implemented |
| ICA | mixed signals -> recovered source | mixing coefficients concept | Yes | generated signals | signal separation view | dependent sources | matrix methods | Yes | Implemented conceptually |
| t-SNE | random-ish embedding -> local neighborhoods | seed/neighborhood concept | Concept | high-dimensional projection | neighbor overlap | seed sensitivity | UMAP comparison | Yes | Implemented conceptually |
| UMAP | neighborhood graph -> compact embedding | neighbors/min-dist concept | Concept | high-dimensional projection | neighbor overlap | parameter sensitivity | t-SNE comparison | Yes | Implemented conceptually |
| Isomap | graph/geodesic idea -> unfolded embedding | neighbors | Concept | manifold projection | graph links | disconnected graph | PCA comparison | Yes | Implemented conceptually |
| LLE | local reconstruction neighborhoods -> embedding | neighbors | Concept | manifold projection | neighbor overlap | bad neighborhood size | manifold comparison | Yes | Implemented conceptually |

## Reusable Components Created

- `ClusterVisualizationCanvas`
- `DendrogramVisualizer`
- `GraphCanvas3`
- `EmbeddingCanvas3`
- `MatrixRenderer3`
- `NeighborhoodInspector3`
- `RegressionBoostingCanvas`
- `SplitGainPanel3`
- `HistogramPanel3`
- `SignalCanvas3`

## Reusable Engine/State Additions

- `ClusterPoint`, `KMeansState`, `DbscanState`, `DendrogramState`, `GmmState`, `PcaState`, `GraphState`, `EmbeddingState`, `BoostingState`, `SplitGainState`, `SilhouetteState`
- Deterministic clustering, boosting, graph, PCA, SVD, ICA, embedding, and quality metric helpers.

## Performance Constraints

- Educational sample sizes are bounded to 20-500 points.
- Expensive algorithms use deterministic conceptual states rather than production-scale optimizers.
- No dense per-frame grids or heavyweight dependencies were added.

## Deliberately Deferred

- Full industrial XGBoost/LightGBM/CatBoost compatibility.
- Exact t-SNE/UMAP optimizers.
- Robust 3D Swiss-roll renderer.
- Background/cancellable computation pipeline.

## Technical Debt

- Phase 1/2/3 lab UI helpers should be extracted into public shared composables before Phase 4.
- A ViewModel/coroutine state layer is recommended before implementing heavier later phases.
