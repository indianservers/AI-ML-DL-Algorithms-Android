package com.indianservers.ai_ml_dl_algorithms.ml_lab.data

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Algorithm
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.AlgorithmFamily
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.AlgorithmStatus
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LessonSection
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Point2D

object MlLabContent {
    val algorithms = listOf(
        Algorithm("simple-linear-regression", "Simple Linear Regression", AlgorithmFamily.Regression, "Interactive gradient descent flagship", AlgorithmStatus.Interactive, 0xFF19D3FF),
        Algorithm("multiple-linear-regression", "Multiple Linear Regression", AlgorithmFamily.Regression, "Multiple input features", AlgorithmStatus.Interactive, 0xFF3DEB9C),
        Algorithm("polynomial-regression", "Polynomial Regression", AlgorithmFamily.Regression, "Curves with engineered powers", AlgorithmStatus.Interactive, 0xFFFFB84D),
        Algorithm("ridge-regression", "Ridge Regression", AlgorithmFamily.Regression, "L2 regularised regression", AlgorithmStatus.Interactive, 0xFF7C5CFF),
        Algorithm("lasso-regression", "Lasso Regression", AlgorithmFamily.Regression, "L1 regularisation and sparsity", AlgorithmStatus.Interactive, 0xFFFF5AC8),
        Algorithm("elastic-net", "Elastic Net", AlgorithmFamily.Regression, "L1 plus L2 regularisation", AlgorithmStatus.LessonReady, 0xFFB768FF),
        Algorithm("bayesian-regression", "Bayesian Regression", AlgorithmFamily.Regression, "Posterior over parameters", AlgorithmStatus.LessonReady, 0xFF44E88B),
        Algorithm("quantile-regression", "Quantile Regression", AlgorithmFamily.Regression, "Conditional quantiles", AlgorithmStatus.LessonReady, 0xFF25D8C8),
        Algorithm("robust-regression", "Robust Regression", AlgorithmFamily.Regression, "Outlier resistant fitting", AlgorithmStatus.LessonReady, 0xFFFF7A59),
        Algorithm("logistic-regression", "Logistic Regression", AlgorithmFamily.Classification, "Sigmoid, threshold, decision boundary", AlgorithmStatus.Interactive, 0xFF2F7BFF),
        Algorithm("knn", "k-Nearest Neighbours", AlgorithmFamily.Classification, "Distance based local voting", AlgorithmStatus.Interactive, 0xFF25D8C8),
        Algorithm("gaussian-naive-bayes", "Gaussian Naive Bayes", AlgorithmFamily.Classification, "Probabilistic classifier", AlgorithmStatus.Interactive, 0xFFFF7A59),
        Algorithm("decision-tree-classifier", "Decision Tree Classification", AlgorithmFamily.Classification, "Gini, entropy and splits", AlgorithmStatus.Interactive, 0xFFA95CFF),
        Algorithm("k-means", "K-Means Clustering", AlgorithmFamily.Clustering, "Centroid updates and inertia", AlgorithmStatus.Interactive, 0xFFFFA726),
        Algorithm("pca", "Principal Component Analysis", AlgorithmFamily.DimensionalityReduction, "Variance directions and projection", AlgorithmStatus.Interactive, 0xFF44E88B),
        Algorithm("random-forest-regression", "Random Forest Regression", AlgorithmFamily.Regression, "Bagged decision trees", AlgorithmStatus.LessonReady, 0xFF4AA3FF),
        Algorithm("support-vector-regression", "Support Vector Regression", AlgorithmFamily.Regression, "Margin based regression", AlgorithmStatus.LessonReady, 0xFF33D7FF),
        Algorithm("decision-tree-regression", "Decision Tree Regression", AlgorithmFamily.Regression, "Piecewise constant regression", AlgorithmStatus.LessonReady, 0xFFFF8A3D),
        Algorithm("gradient-boosting-regression", "Gradient Boosting Regression", AlgorithmFamily.Regression, "Boosted weak learners", AlgorithmStatus.LessonReady, 0xFFFFB02E),
        Algorithm("adaboost-regression", "AdaBoost Regression", AlgorithmFamily.Regression, "Adaptive reweighting", AlgorithmStatus.LessonReady, 0xFFFF5AC8),
        Algorithm("knn-regression", "k-NN Regression", AlgorithmFamily.Regression, "Neighbour averaged targets", AlgorithmStatus.LessonReady, 0xFF35E58F),
        Algorithm("gaussian-process-regression", "Gaussian Process Regression", AlgorithmFamily.Regression, "Kernel posterior regression", AlgorithmStatus.LessonReady, 0xFF7C5CFF),
        Algorithm("random-forest", "Random Forest", AlgorithmFamily.Classification, "Many trees voting together", AlgorithmStatus.LessonReady, 0xFF31D0AA),
        Algorithm("support-vector-machine", "Support Vector Machine", AlgorithmFamily.Classification, "Maximum margin classification", AlgorithmStatus.LessonReady, 0xFF328BFF),
        Algorithm("perceptron", "Perceptron", AlgorithmFamily.Classification, "Linear classifier and XOR limitation", AlgorithmStatus.Interactive, 0xFFB768FF),
        Algorithm("lda", "Linear Discriminant Analysis", AlgorithmFamily.Classification, "Linear class separation", AlgorithmStatus.LessonReady, 0xFF44E88B),
        Algorithm("qda", "Quadratic Discriminant Analysis", AlgorithmFamily.Classification, "Class-specific covariance", AlgorithmStatus.LessonReady, 0xFF25D8C8),
        Algorithm("sgd-classifier", "SGD Classifier", AlgorithmFamily.Classification, "Online linear learning", AlgorithmStatus.LessonReady, 0xFFFFA726),
        Algorithm("adaboost", "AdaBoost", AlgorithmFamily.Ensemble, "Adaptive boosting", AlgorithmStatus.Future, 0xFFFF5AC8),
        Algorithm("gradient-boosting", "Gradient Boosting", AlgorithmFamily.Ensemble, "Sequential error correction", AlgorithmStatus.Future, 0xFFFFB02E),
        Algorithm("xgboost-concepts", "XGBoost-style Boosting", AlgorithmFamily.Ensemble, "Regularised tree boosting", AlgorithmStatus.Future, 0xFFFF7A59),
        Algorithm("lightgbm-concepts", "LightGBM Concepts", AlgorithmFamily.Ensemble, "Histogram tree boosting", AlgorithmStatus.Future, 0xFF35E58F),
        Algorithm("catboost-concepts", "CatBoost Concepts", AlgorithmFamily.Ensemble, "Categorical boosting", AlgorithmStatus.Future, 0xFFA95CFF),
        Algorithm("bagging", "Bagging", AlgorithmFamily.Ensemble, "Bootstrap aggregation", AlgorithmStatus.Future, 0xFF4AA3FF),
        Algorithm("stacking", "Stacking", AlgorithmFamily.Ensemble, "Meta model ensembles", AlgorithmStatus.Future, 0xFF7C5CFF),
        Algorithm("voting", "Voting", AlgorithmFamily.Ensemble, "Model vote aggregation", AlgorithmStatus.Future, 0xFF25D8C8),
        Algorithm("dbscan", "DBSCAN", AlgorithmFamily.Clustering, "Density based clusters", AlgorithmStatus.Future, 0xFF25D8C8),
        Algorithm("hdbscan", "HDBSCAN Concepts", AlgorithmFamily.Clustering, "Hierarchical density clusters", AlgorithmStatus.Future, 0xFF3DEB9C),
        Algorithm("hierarchical-clustering", "Hierarchical Clustering", AlgorithmFamily.Clustering, "Agglomerative structure", AlgorithmStatus.Future, 0xFF55C1FF),
        Algorithm("mean-shift", "Mean Shift", AlgorithmFamily.Clustering, "Mode seeking clusters", AlgorithmStatus.Future, 0xFFFFA726),
        Algorithm("gaussian-mixture-models", "Gaussian Mixture Models", AlgorithmFamily.Clustering, "Soft probabilistic clusters", AlgorithmStatus.Future, 0xFFFF5AC8),
        Algorithm("spectral-clustering", "Spectral Clustering", AlgorithmFamily.Clustering, "Graph Laplacian clustering", AlgorithmStatus.Future, 0xFF4AA3FF),
        Algorithm("birch", "BIRCH", AlgorithmFamily.Clustering, "Incremental cluster features", AlgorithmStatus.Future, 0xFF44E88B),
        Algorithm("kernel-pca", "Kernel PCA", AlgorithmFamily.DimensionalityReduction, "Nonlinear component analysis", AlgorithmStatus.Future, 0xFFA95CFF),
        Algorithm("svd", "Singular Value Decomposition", AlgorithmFamily.DimensionalityReduction, "Matrix factorisation", AlgorithmStatus.Future, 0xFF2F7BFF),
        Algorithm("ica", "Independent Component Analysis", AlgorithmFamily.DimensionalityReduction, "Independent latent sources", AlgorithmStatus.Future, 0xFF25D8C8),
        Algorithm("factor-analysis", "Factor Analysis", AlgorithmFamily.DimensionalityReduction, "Latent variable model", AlgorithmStatus.Future, 0xFFFFB84D),
        Algorithm("tsne", "t-SNE", AlgorithmFamily.DimensionalityReduction, "Neighbour preserving maps", AlgorithmStatus.Future, 0xFFFF5AC8),
        Algorithm("umap", "UMAP Concepts", AlgorithmFamily.DimensionalityReduction, "Manifold learning placeholder", AlgorithmStatus.Future, 0xFFA95CFF),
        Algorithm("apriori", "Apriori", AlgorithmFamily.Association, "Frequent itemsets", AlgorithmStatus.Future, 0xFFFFB84D),
        Algorithm("fp-growth", "FP-Growth", AlgorithmFamily.Association, "Frequent pattern trees", AlgorithmStatus.Future, 0xFF35E58F),
        Algorithm("eclat", "Eclat", AlgorithmFamily.Association, "Vertical itemset mining", AlgorithmStatus.Future, 0xFF25D8C8),
        Algorithm("isolation-forest", "Isolation Forest", AlgorithmFamily.Anomaly, "Anomaly isolation paths", AlgorithmStatus.Future, 0xFF3DEB9C),
        Algorithm("one-class-svm", "One-Class SVM", AlgorithmFamily.Anomaly, "Novelty boundary", AlgorithmStatus.Future, 0xFF328BFF),
        Algorithm("local-outlier-factor", "Local Outlier Factor", AlgorithmFamily.Anomaly, "Local density deviation", AlgorithmStatus.Future, 0xFFFFA726),
        Algorithm("autoencoder-anomaly", "Autoencoder Detection", AlgorithmFamily.Anomaly, "Reconstruction error", AlgorithmStatus.Future, 0xFFFF5AC8),
        Algorithm("moving-average", "Moving Average", AlgorithmFamily.TimeSeries, "Windowed smoothing", AlgorithmStatus.Future, 0xFF20D9E8),
        Algorithm("exponential-smoothing", "Exponential Smoothing", AlgorithmFamily.TimeSeries, "Weighted recent history", AlgorithmStatus.Future, 0xFF44E88B),
        Algorithm("holt-winters", "Holt-Winters", AlgorithmFamily.TimeSeries, "Trend and seasonality", AlgorithmStatus.Future, 0xFFFFB84D),
        Algorithm("arima", "ARIMA", AlgorithmFamily.TimeSeries, "Autoregressive forecasting", AlgorithmStatus.Future, 0xFF4AA3FF),
        Algorithm("sarima", "SARIMA", AlgorithmFamily.TimeSeries, "Seasonal ARIMA", AlgorithmStatus.Future, 0xFF7C5CFF),
        Algorithm("var", "VAR", AlgorithmFamily.TimeSeries, "Multivariate autoregression", AlgorithmStatus.Future, 0xFFFF7A59),
        Algorithm("prophet-concepts", "Prophet Concepts", AlgorithmFamily.TimeSeries, "Decomposable forecasting", AlgorithmStatus.Future, 0xFF35E58F),
        Algorithm("multi-armed-bandits", "Multi-Armed Bandits", AlgorithmFamily.Reinforcement, "Explore versus exploit", AlgorithmStatus.Future, 0xFFFFB02E),
        Algorithm("value-iteration", "Value Iteration", AlgorithmFamily.Reinforcement, "Dynamic programming values", AlgorithmStatus.Future, 0xFF44E88B),
        Algorithm("policy-iteration", "Policy Iteration", AlgorithmFamily.Reinforcement, "Policy evaluation and improvement", AlgorithmStatus.Future, 0xFF25D8C8),
        Algorithm("sarsa", "SARSA", AlgorithmFamily.Reinforcement, "On-policy TD control", AlgorithmStatus.Future, 0xFFFF5AC8),
        Algorithm("q-learning", "Q-Learning", AlgorithmFamily.Reinforcement, "Value learning from rewards", AlgorithmStatus.Future, 0xFF7C5CFF),
        Algorithm("dqn", "DQN", AlgorithmFamily.Reinforcement, "Deep Q networks", AlgorithmStatus.Future, 0xFF2F7BFF),
        Algorithm("ppo", "PPO", AlgorithmFamily.Reinforcement, "Policy optimisation", AlgorithmStatus.Future, 0xFFA95CFF),
        Algorithm("mlp", "Multilayer Perceptron", AlgorithmFamily.DeepLearning, "Editable dense neural networks", AlgorithmStatus.Interactive, 0xFFB768FF),
        Algorithm("feedforward-nn", "Feedforward Neural Networks", AlgorithmFamily.DeepLearning, "Forward pass, backprop and inspection", AlgorithmStatus.Interactive, 0xFF20D9E8),
        Algorithm("cnn", "Convolutional Neural Networks", AlgorithmFamily.DeepLearning, "Trainable filters and feature maps", AlgorithmStatus.Interactive, 0xFF2F7BFF),
        Algorithm("rnn", "Recurrent Neural Networks", AlgorithmFamily.DeepLearning, "Sequence recurrence and BPTT", AlgorithmStatus.Interactive, 0xFFFFB84D),
        Algorithm("lstm", "Long Short-Term Memory", AlgorithmFamily.DeepLearning, "Inspectable gated memory", AlgorithmStatus.Interactive, 0xFF44E88B),
        Algorithm("gru", "GRU", AlgorithmFamily.DeepLearning, "Reset and update gates", AlgorithmStatus.Interactive, 0xFF35E58F),
        Algorithm("autoencoders", "Autoencoders", AlgorithmFamily.DeepLearning, "Reconstruction and latent space", AlgorithmStatus.Interactive, 0xFFFF7A59),
        Algorithm("vae", "Variational Autoencoders", AlgorithmFamily.DeepLearning, "Reparameterized probabilistic latent space", AlgorithmStatus.Interactive, 0xFFFF5AC8),
        Algorithm("gan", "GAN", AlgorithmFamily.DeepLearning, "Alternating adversarial generation", AlgorithmStatus.Interactive, 0xFFA95CFF),
        Algorithm("attention", "Attention", AlgorithmFamily.DeepLearning, "Inspectable Q, K, V and token weights", AlgorithmStatus.Interactive, 0xFF4AA3FF),
        Algorithm("transformers", "Transformers", AlgorithmFamily.DeepLearning, "Multi-head encoder representations", AlgorithmStatus.Interactive, 0xFFFF5AC8),
        Algorithm("vision-transformers", "Vision Transformers", AlgorithmFamily.DeepLearning, "Patch and CLS attention", AlgorithmStatus.Interactive, 0xFF25D8C8),
        Algorithm("graph-neural-networks", "Graph Neural Networks", AlgorithmFamily.DeepLearning, "Normalized graph message passing", AlgorithmStatus.Interactive, 0xFFFFB02E),
        Algorithm("siamese-networks", "Siamese Networks", AlgorithmFamily.DeepLearning, "Similarity learning", AlgorithmStatus.Future, 0xFF44E88B),
        Algorithm("diffusion", "Diffusion Models", AlgorithmFamily.DeepLearning, "Forward noise and reverse denoising", AlgorithmStatus.Interactive, 0xFFA95CFF)
    )

    val lessonSections = listOf(
        LessonSection(
            "Overview",
            "A model learns a pattern from examples, then uses that pattern for predictions.",
            "The algorithm maps input features to an output using a small set of learned parameters.",
            "We define a hypothesis, fit parameters by minimising an objective, and validate generalisation.",
            "Study the assumptions, optimisation path, conditioning, failure modes, and implementation trade-offs."
        ),
        LessonSection(
            "Mathematics",
            "The model draws a line or boundary that best matches the examples.",
            "For linear regression: y = wx + b. Training reduces average squared error.",
            "Minimise J(w,b) = (1/n) sum (wx_i + b - y_i)^2 with gradients dJ/dw and dJ/db.",
            "Convergence depends on feature scale, learning rate, convexity, regularisation and floating-point stability."
        ),
        LessonSection(
            "Training Process",
            "Try, measure the mistake, nudge the model, repeat.",
            "Each epoch predicts, computes loss, computes gradients, and updates parameters.",
            "Gradient descent updates theta := theta - alpha grad J(theta), producing inspectable snapshots.",
            "Snapshot histories make optimisation auditable and prepare the app for persistent experiment replay."
        ),
        LessonSection(
            "Use And Limits",
            "Use it when the pattern matches the model shape.",
            "Check outliers, missing values, feature scaling and whether the relationship is linear.",
            "Reason about bias, variance, residuals, assumptions, interpretability and complexity.",
            "Inspect numerical stability, leakage, distribution shift, calibration and edge-case behaviour."
        )
    )

    val regressionDatasets = mapOf(
        "Noisy linear" to listOf(
            Point2D(-0.9f, -0.62f), Point2D(-0.7f, -0.42f), Point2D(-0.45f, -0.2f),
            Point2D(-0.2f, -0.08f), Point2D(0.05f, 0.24f), Point2D(0.28f, 0.22f),
            Point2D(0.48f, 0.48f), Point2D(0.72f, 0.62f), Point2D(0.9f, 0.82f)
        ),
        "Perfect linear" to listOf(
            Point2D(-0.9f, -0.75f), Point2D(-0.6f, -0.5f), Point2D(-0.3f, -0.25f),
            Point2D(0f, 0f), Point2D(0.3f, 0.25f), Point2D(0.6f, 0.5f), Point2D(0.9f, 0.75f)
        ),
        "Polynomial" to listOf(
            Point2D(-0.9f, 0.58f), Point2D(-0.65f, 0.28f), Point2D(-0.35f, -0.05f),
            Point2D(0f, -0.18f), Point2D(0.35f, -0.02f), Point2D(0.65f, 0.28f), Point2D(0.9f, 0.6f)
        ),
        "Outliers" to listOf(
            Point2D(-0.9f, -0.72f), Point2D(-0.55f, -0.45f), Point2D(-0.2f, -0.15f),
            Point2D(0.12f, 0.12f), Point2D(0.4f, 0.44f), Point2D(0.65f, -0.72f), Point2D(0.85f, 0.78f)
        )
    )

    val classificationPoints = listOf(
        Point2D(-0.75f, -0.55f, 0), Point2D(-0.55f, -0.22f, 0), Point2D(-0.35f, -0.7f, 0),
        Point2D(0.45f, 0.28f, 1), Point2D(0.62f, 0.65f, 1), Point2D(0.78f, 0.12f, 1),
        Point2D(-0.12f, 0.52f, 1), Point2D(0.08f, -0.38f, 0)
    )
}
