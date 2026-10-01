package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

// One entry per Stage 2 catalog placement. Renderers share primitives, never a title-only chart.
internal enum class StageTwoVisualization(
    val section: String,
    val title: String,
    val graphic: String,
    val controls: String,
    val dataset: String
) {
    KMeans("Clustering","K-Means","centroid assignments and alternating updates","K; step; auto; seed","unlabelled 2D clusters"),
    KMeansPlus("Clustering","K-Means++","distance-squared seeding and centroid updates","K; seed; compare initialization; step","unlabelled 2D clusters"),
    MiniBatchKMeans("Clustering","Mini-Batch K-Means","highlighted batch and incremental centroid trail","K; batch size; step; auto","unlabelled 2D clusters"),
    Hierarchical("Clustering","Hierarchical Clustering","linkage merges and dendrogram cut","linkage; cut; step","unlabelled 2D clusters"),
    Agglomerative("Clustering","Agglomerative Clustering","bottom-up merges and dendrogram","linkage; cut; step","unlabelled 2D clusters"),
    Divisive("Clustering","Divisive Clustering","top-down recursive splits and tree","split step; auto","unlabelled 2D clusters"),
    Dbscan("Clustering","DBSCAN","core-border-noise expansion and epsilon circle","epsilon; minPts; step","moons, noise, varying-density 2D data"),
    Hdbscan("Clustering","HDBSCAN","mutual-reachability hierarchy and size-qualified clusters","minPts; hierarchy cut","varying-density 2D data"),
    Optics("Clustering","OPTICS","reachability order linked to scatter","epsilon; minPts; step","varying-density 2D data"),
    MeanShift("Clustering","Mean Shift","kernel windows and mode-seeking trails","bandwidth; step; auto","unlabelled 2D clusters"),
    Gmm("Clustering","Gaussian Mixture Models","EM responsibilities and covariance ellipses","components; EM step","overlapping elliptical data"),
    Spectral("Clustering","Spectral Clustering","similarity graph and graph partition","neighbors; graph stage","moons or circles"),
    Birch("Clustering","BIRCH","streaming CF entries and CF tree","threshold; insertion step","streaming 2D clusters"),
    Affinity("Clustering","Affinity Propagation","responsibility/availability messages and exemplars","preference; message step","unlabelled 2D clusters"),
    FuzzyCMeans("Clustering","Fuzzy C-Means","membership proportions and fuzzy centroids","K; fuzziness; step","overlapping 2D clusters"),

    Pca("Dimensionality Reduction","PCA","covariance axes and 1D projection","point drag; dataset","correlated 2D data"),
    KernelPca("Dimensionality Reduction","Kernel PCA","kernel matrix and nonlinear embedding","RBF gamma; point drag","circles or moons"),
    SparsePca("Dimensionality Reduction","Sparse PCA","sparse feature loadings and projection","sparsity strength","multifeature correlated data"),
    IncrementalPca("Dimensionality Reduction","Incremental PCA","batch-wise evolving principal axis","batch step; point drag","streaming correlated data"),
    TruncatedSvd("Dimensionality Reduction","Truncated SVD","matrix, singular spectrum and rank-k reconstruction","retained rank","document-term matrix"),
    FactorAnalysis("Dimensionality Reduction","Factor Analysis","latent factors, loadings and unique noise","point drag; dataset","multifeature correlated data"),
    Ica("Dimensionality Reduction","Independent Component Analysis","source/mixed/recovered signal traces","mixture strength; sample","mixed independent signals"),
    Tsne("Dimensionality Reduction","t-SNE","neighbour probability affinities and evolving map","perplexity; iterations; learning rate","high-dimensional local neighborhoods"),
    Umap("Dimensionality Reduction","UMAP","fuzzy neighbour graph and optimized embedding","neighbors; minDist; step","high-dimensional local graph"),
    Isomap("Dimensionality Reduction","Isomap","neighbour geodesics and manifold unfolding","neighbors; point drag","curved manifold"),
    Lle("Dimensionality Reduction","Locally Linear Embedding","local reconstruction weights and unfolding","neighbors; point drag","curved manifold"),
    Mds("Dimensionality Reduction","Multidimensional Scaling","distance matrix and stress-minimized layout","point drag; dataset","pairwise distances"),
    AutoencoderReduction("Dimensionality Reduction","Autoencoder-based Reduction","encoder bottleneck decoder and reconstruction","training step; point drag","numeric feature vectors"),

    Apriori("Association Learning","Apriori","candidate itemsets and support pruning lattice","min support; min confidence; step","market baskets"),
    FpGrowth("Association Learning","FP-Growth","frequency-ordered FP tree and prefix paths","min support; transaction step","market baskets"),
    Eclat("Association Learning","ECLAT","vertical TID sets and intersections","min support; itemset step","market baskets"),
    AssociationRules("Association Learning","Association Rule Mining","rule support confidence lift and item graph","min support; min confidence; selected rule","market baskets"),

    IsolationForest("Anomaly Detection","Isolation Forest","random partitions, path length and ensemble score","trees; sample; step","global and local outliers"),
    Lof("Anomaly Detection","Local Outlier Factor","K-reachability density and LOF ratio","K; selected sample","dense and sparse local clusters"),
    OneClassSvm("Anomaly Detection","One-Class SVM","kernel support vectors and enclosing boundary","nu; gamma; kernel","normal data and anomalies"),
    EllipticEnvelope("Anomaly Detection","Elliptic Envelope","robust covariance ellipse and Mahalanobis score","threshold; selected sample","elliptical normal cloud"),
    AutoencoderAnomaly("Anomaly Detection","Autoencoder Anomaly Detection","reconstruction and error threshold","threshold; selected sample","normal and anomalous feature vectors"),
    StatisticalOutlier("Anomaly Detection","Statistical Outlier Detection","distribution tails, z-score and IQR fences","method; threshold","univariate data with outliers"),

    SelfTraining("Methods","Self-Training","confident pseudo-label admission by iteration","confidence threshold; step","few labels and many unlabelled samples"),
    LabelPropagation("Methods","Label Propagation","graph probability diffusion with hard seed clamping","neighbors; step","few labels on graph"),
    LabelSpreading("Methods","Label Spreading","soft-clamped graph probability diffusion","neighbors; alpha; step","few labels on graph"),
    CoTraining("Methods","Co-Training","two feature-view models exchanging labels","confidence threshold; step","two-view partially labelled data"),
    PseudoLabelling("Methods","Pseudo-Labelling","confidence-gated model predictions and retraining","confidence threshold; step","few labels and many unlabelled samples"),
    Consistency("Methods","Consistency Regularization","perturbation predictions and consistency loss","augmentation strength; step","unlabelled numeric samples"),
    MeanTeacher("Methods","Mean Teacher","student and EMA-teacher predictions over time","EMA decay; step","augmented unlabelled samples"),
    FixMatch("Methods","FixMatch","weak prediction gate and strong augmentation training","confidence threshold; step","weak/strong augmented samples"),
    MixMatch("Methods","MixMatch","guess, sharpen and MixUp label pipeline","temperature; mix strength; step","labelled and unlabelled batches"),

    Bagging("Bagging","Bagging","bootstrap duplicates and independent learners","learners; bootstrap step","labelled nonlinear data"),
    RandomForest("Bagging","Random Forest","bootstrap and random-feature decision trees","trees; depth; feature subset","labelled nonlinear data"),
    ExtraTrees("Bagging","Extra Trees","random-threshold trees and aggregate vote","trees; depth; feature subset","labelled nonlinear data"),
    AdaBoost("Boosting","AdaBoost","weighted mistakes and staged weak learners","learner stage; learning rate","labelled nonlinear data"),
    GradientBoosting("Boosting","Gradient Boosting","sequential pseudo-residual corrections","learner stage; learning rate","labelled nonlinear data"),
    Xgboost("Boosting","XGBoost","gradient/Hessian split gains and corrections","learner stage; regularization","labelled nonlinear data"),
    Lightgbm("Boosting","LightGBM","binned gradient statistics and leaf-wise growth","learner stage; bins","labelled nonlinear data"),
    Catboost("Boosting","CatBoost","ordered category statistics and symmetric trees","learner stage; category order","categorical labelled data"),
    Voting("Combining Models","Voting","independent model class votes","model subset; query","labelled nonlinear data"),
    SoftVoting("Combining Models","Soft Voting","class probability average","model subset; query","labelled nonlinear data"),
    HardVoting("Combining Models","Hard Voting","discrete class majority","model subset; query","labelled nonlinear data"),
    Stacking("Combining Models","Stacking","out-of-fold base predictions and meta-model","fold; query","labelled nonlinear data"),
    Blending("Combining Models","Blending","holdout predictions and meta-model","holdout fraction; query","labelled nonlinear data");

    val interaction: String get() = when (section) {
        "Association Learning" -> "select transaction or rule; change thresholds"
        "Dimensionality Reduction" -> "change source data or reduction hyperparameters"
        "Methods" -> "advance learner state; change confidence or graph controls"
        else -> "change data and legitimate hyperparameters; advance algorithm state"
    }
    val animation: String get() = when (this) {
        Pca, KernelPca, SparsePca, TruncatedSvd, FactorAnalysis, Ica,
        EllipticEnvelope, StatisticalOutlier, AssociationRules, Voting, SoftVoting, HardVoting -> "recompute after input change"
        else -> "algorithm-state steps"
    }
}
