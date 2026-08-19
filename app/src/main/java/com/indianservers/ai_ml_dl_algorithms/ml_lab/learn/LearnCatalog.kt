package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth

data class LearnTopic(
    val id: String,
    val title: String,
    val domain: String,
    val section: String,
    val accent: Long
)

data class LearnSection(val title: String, val topics: List<LearnTopic>)

data class LearnDomain(
    val title: String,
    val description: String,
    val accent: Long,
    val sections: List<LearnSection>
) {
    val topicCount: Int get() = sections.sumOf { it.topics.size }
}

enum class VisualizationKind {
    Regression, Classification, Neighbours, Tree, Clustering, Density, Projection, NeuralNetwork,
    Convolution, Sequence, Attention, Autoencoder, Generative, Graph, Reinforcement,
    TimeSeries, Probability, Optimization, Recommendation, Explanation, Generic
}

data class LearningProfile(
    val definition: String,
    val purpose: String,
    val intuition: String,
    val steps: List<String>,
    val equation: String,
    val equationNote: String,
    val advantages: List<String>,
    val limitations: List<String>,
    val assumptions: List<String>,
    val hyperparameters: List<String>,
    val applications: List<String>,
    val mistakes: List<String>,
    val sampleData: String,
    val kind: VisualizationKind
)

private data class TheoryOverride(
    val definition: String,
    val purpose: String,
    val intuition: String,
    val steps: List<String>,
    val equation: String,
    val equationNote: String,
    val advantages: List<String>,
    val limitations: List<String>,
    val assumptions: List<String>,
    val hyperparameters: List<String>,
    val applications: List<String>,
    val mistakes: List<String>,
    val sampleData: String,
    val kind: VisualizationKind
)

object LearnCatalog {
    private fun slug(value: String) = value.lowercase()
        .replace("+", " plus ").replace(Regex("[^a-z0-9]+"), "-").trim('-')

    private fun domain(
        title: String,
        description: String,
        accent: Long,
        vararg groups: Pair<String, String>
    ): LearnDomain = LearnDomain(title, description, accent, groups.map { (section, names) ->
        LearnSection(section, names.split('|').map { name ->
            LearnTopic("${slug(title)}-${slug(section)}-${slug(name)}", name, title, section, accent)
        })
    })

    val domains = listOf(
        domain("Supervised Learning", "Learn mappings from labelled examples", 0xFF20D9E8,
            "Regression" to "Simple Linear Regression|Multiple Linear Regression|Polynomial Regression|Ridge Regression|Lasso Regression|Elastic Net Regression|Logistic Regression|Bayesian Linear Regression|Quantile Regression|Robust Regression|Support Vector Regression|Decision Tree Regression|Random Forest Regression|Extra Trees Regression|Gradient Boosting Regression|AdaBoost Regression|XGBoost Regression|LightGBM Regression|CatBoost Regression|K-Nearest Neighbors Regression|Gaussian Process Regression",
            "Classification" to "Logistic Regression|K-Nearest Neighbors|Gaussian Naive Bayes|Multinomial Naive Bayes|Bernoulli Naive Bayes|Decision Tree|Random Forest|Extra Trees|Support Vector Machine|Linear Discriminant Analysis|Quadratic Discriminant Analysis|Perceptron|SGD Classifier|AdaBoost|Gradient Boosting|XGBoost|LightGBM|CatBoost|Gaussian Process Classifier"),
        domain("Unsupervised Learning", "Discover structure without target labels", 0xFFA95CFF,
            "Clustering" to "K-Means|K-Means++|Mini-Batch K-Means|Hierarchical Clustering|Agglomerative Clustering|Divisive Clustering|DBSCAN|HDBSCAN|OPTICS|Mean Shift|Gaussian Mixture Models|Spectral Clustering|BIRCH|Affinity Propagation|Fuzzy C-Means",
            "Dimensionality Reduction" to "PCA|Kernel PCA|Sparse PCA|Incremental PCA|Truncated SVD|Factor Analysis|Independent Component Analysis|t-SNE|UMAP|Isomap|Locally Linear Embedding|Multidimensional Scaling|Autoencoder-based Reduction",
            "Association Learning" to "Apriori|FP-Growth|ECLAT|Association Rule Mining",
            "Anomaly Detection" to "Isolation Forest|Local Outlier Factor|One-Class SVM|Elliptic Envelope|Autoencoder Anomaly Detection|Statistical Outlier Detection"),
        domain("Semi-Supervised Learning", "Combine a small labelled set with unlabelled data", 0xFF35E58F,
            "Methods" to "Self-Training|Label Propagation|Label Spreading|Co-Training|Pseudo-Labelling|Consistency Regularization|Mean Teacher|FixMatch|MixMatch"),
        domain("Ensemble Learning", "Combine learners for stronger predictions", 0xFFFFA52E,
            "Bagging" to "Bagging|Random Forest|Extra Trees",
            "Boosting" to "AdaBoost|Gradient Boosting|XGBoost|LightGBM|CatBoost",
            "Combining Models" to "Voting|Soft Voting|Hard Voting|Stacking|Blending"),
        domain("Deep Learning", "Learn layered representations from data", 0xFF9B3DFF,
            "Neural Network Fundamentals" to "Artificial Neuron|Perceptron|Multi-Layer Perceptron|Feedforward Neural Network|Backpropagation|Gradient Descent|Stochastic Gradient Descent|Mini-Batch Gradient Descent|Activation Functions|Loss Functions|Weight Initialization|Batch Normalization|Layer Normalization|Dropout|Regularization",
            "Convolutional Neural Networks" to "CNN|LeNet|AlexNet|VGG|GoogLeNet / Inception|ResNet|DenseNet|MobileNet|EfficientNet|ConvNeXt",
            "Sequence Models" to "Recurrent Neural Network|Bidirectional RNN|LSTM|GRU|Sequence-to-Sequence|Encoder-Decoder|Attention Mechanism",
            "Transformers" to "Transformer|Self-Attention|Multi-Head Attention|Positional Encoding|Encoder Transformer|Decoder Transformer|BERT|GPT-style Decoder Models|T5|Vision Transformer|Swin Transformer",
            "Autoencoders" to "Basic Autoencoder|Sparse Autoencoder|Denoising Autoencoder|Convolutional Autoencoder|Variational Autoencoder",
            "Generative Models" to "GAN|DCGAN|Conditional GAN|CycleGAN|StyleGAN|Wasserstein GAN|Variational Autoencoder|Diffusion Models|Latent Diffusion Models",
            "Graph Neural Networks" to "Graph Neural Network|Graph Convolutional Network|Graph Attention Network|GraphSAGE|Graph Autoencoder"),
        domain("Reinforcement Learning", "Learn actions through reward and interaction", 0xFFFF5AC8,
            "Fundamentals" to "Agent|Environment|State|Action|Reward|Policy|Value Function|Markov Decision Process",
            "Classical RL" to "Multi-Armed Bandit|Dynamic Programming|Monte Carlo Learning|Temporal Difference Learning|SARSA|Q-Learning|Expected SARSA",
            "Deep Reinforcement Learning" to "Deep Q-Network|Double DQN|Dueling DQN|REINFORCE|Policy Gradient|Actor-Critic|A2C|A3C|DDPG|TD3|PPO|SAC",
            "Advanced RL" to "Model-Based Reinforcement Learning|Multi-Agent Reinforcement Learning|Hierarchical Reinforcement Learning|Offline Reinforcement Learning|Imitation Learning|Inverse Reinforcement Learning"),
        domain("Natural Language Processing", "Represent and model human language", 0xFF4AA3FF,
            "Classical NLP" to "Bag of Words|TF-IDF|N-Grams|Naive Bayes Text Classification|Hidden Markov Models|Conditional Random Fields",
            "Word Representation" to "Word2Vec|CBOW|Skip-Gram|GloVe|FastText",
            "Neural NLP" to "RNN|LSTM|GRU|Seq2Seq|Attention|Transformer|BERT|GPT|T5"),
        domain("Computer Vision", "Understand images, objects and spatial structure", 0xFF25D8C8,
            "Vision Tasks and Models" to "Image Classification|CNN|Object Detection|R-CNN|Fast R-CNN|Faster R-CNN|SSD|YOLO|RetinaNet|Semantic Segmentation|U-Net|Mask R-CNN|Instance Segmentation|Pose Estimation|Vision Transformer|Image Embeddings|Image Similarity"),
        domain("Time-Series Algorithms", "Model trend, seasonality and temporal dependence", 0xFFFFB84D,
            "Forecasting" to "Moving Average|Exponential Smoothing|Holt's Method|Holt-Winters|AR|MA|ARMA|ARIMA|SARIMA|VAR|State-Space Models|Kalman Filter|Prophet-style Forecasting|LSTM Forecasting|GRU Forecasting|Temporal CNN|Transformer-based Forecasting"),
        domain("Probabilistic & Bayesian Learning", "Reason explicitly about uncertainty", 0xFF44E88B,
            "Probability Models" to "Bayes Theorem|Bayesian Inference|Maximum Likelihood Estimation|Maximum A Posteriori Estimation|Bayesian Linear Regression|Bayesian Networks|Hidden Markov Models|Gaussian Mixture Models|Gaussian Processes|Markov Chain Monte Carlo|Gibbs Sampling|Metropolis-Hastings|Variational Inference"),
        domain("Optimization Algorithms", "Find parameters that minimize an objective", 0xFFFF7A59,
            "Optimizers" to "Batch Gradient Descent|Stochastic Gradient Descent|Mini-Batch Gradient Descent|Momentum|Nesterov Momentum|AdaGrad|RMSProp|Adam|AdamW|Nadam|Learning Rate Scheduling|Coordinate Descent|Newton's Method|Quasi-Newton / BFGS|L-BFGS"),
        domain("Evolutionary Algorithms", "Search with populations and nature-inspired dynamics", 0xFF3DEB9C,
            "Nature-Inspired Methods" to "Genetic Algorithm|Genetic Programming|Evolution Strategies|Differential Evolution|Particle Swarm Optimization|Ant Colony Optimization|Artificial Bee Colony|Simulated Annealing"),
        domain("Recommendation Algorithms", "Rank relevant items for users", 0xFF2F7BFF,
            "Recommenders" to "Popularity-Based Recommendation|Content-Based Filtering|User-Based Collaborative Filtering|Item-Based Collaborative Filtering|Matrix Factorization|SVD|Alternating Least Squares|Neural Collaborative Filtering|Deep Recommendation Systems"),
        domain("Explainable AI", "Inspect why a model produced an output", 0xFFFF48BE,
            "Explanation Methods" to "Feature Importance|Permutation Importance|Partial Dependence Plot|SHAP|LIME|Saliency Maps|Grad-CAM|Attention Visualization|Counterfactual Explanations")
    )

    val topics: List<LearnTopic> = domains.flatMap { it.sections }.flatMap { it.topics }

    val flagshipTopics: List<LearnTopic> = listOfNotNull(
        topics.firstOrNull { it.title == "Simple Linear Regression" },
        topics.firstOrNull { it.title == "Logistic Regression" && it.section == "Classification" },
        topics.firstOrNull { it.title == "K-Nearest Neighbors" && it.section == "Classification" },
        topics.firstOrNull { it.title == "Decision Tree" && it.section == "Classification" },
        topics.firstOrNull { it.title == "Random Forest" && it.section == "Classification" },
        topics.firstOrNull { it.title == "Support Vector Machine" },
        topics.firstOrNull { it.title == "K-Means" },
        topics.firstOrNull { it.title == "Multi-Layer Perceptron" },
        topics.firstOrNull { it.title == "CNN" },
        topics.firstOrNull { it.title == "LSTM" }
    )

    fun isFlagship(topic: LearnTopic): Boolean = flagshipTopics.any { it.id == topic.id }

    fun profile(topic: LearnTopic, depth: LearningDepth): LearningProfile {
        val name = topic.title
        val lower = name.lowercase()
        val kind = when {
            lower in listOf("dbscan", "hdbscan", "optics") -> VisualizationKind.Density
            name == "Logistic Regression" -> VisualizationKind.Classification
            "knn" in lower || "nearest neighbor" in lower || "nearest neighbour" in lower -> VisualizationKind.Neighbours
            "tree" in lower || "forest" in lower || "boost" in lower -> VisualizationKind.Tree
            topic.section == "Classification" -> VisualizationKind.Classification
            "linear regression" in lower || "regression" in lower -> VisualizationKind.Regression
            topic.section == "Clustering" -> VisualizationKind.Clustering
            topic.section == "Dimensionality Reduction" -> VisualizationKind.Projection
            "cnn" in lower || "convolution" in lower || name in listOf("LeNet", "AlexNet", "VGG", "ResNet", "DenseNet", "MobileNet", "EfficientNet", "ConvNeXt") -> VisualizationKind.Convolution
            "attention" in lower || "transformer" in lower || name in listOf("BERT", "T5", "GPT") -> VisualizationKind.Attention
            "rnn" in lower || "lstm" in lower || "gru" in lower || "sequence" in lower || "markov" in lower -> VisualizationKind.Sequence
            "autoencoder" in lower -> VisualizationKind.Autoencoder
            "gan" in lower || "diffusion" in lower || "generative" in lower -> VisualizationKind.Generative
            "graph" in lower -> VisualizationKind.Graph
            topic.domain == "Deep Learning" -> VisualizationKind.NeuralNetwork
            topic.domain == "Reinforcement Learning" -> VisualizationKind.Reinforcement
            topic.domain == "Time-Series Algorithms" -> VisualizationKind.TimeSeries
            topic.domain == "Probabilistic & Bayesian Learning" -> VisualizationKind.Probability
            topic.domain == "Optimization Algorithms" || "gradient descent" in lower -> VisualizationKind.Optimization
            topic.domain == "Recommendation Algorithms" -> VisualizationKind.Recommendation
            topic.domain == "Explainable AI" -> VisualizationKind.Explanation
            else -> VisualizationKind.Generic
        }
        val equation = equationFor(name, kind)
        val detail = when (depth) {
            LearningDepth.Beginner -> "Focus on the visible input-to-output behavior before the notation."
            LearningDepth.Intro -> "Track each symbol against the sample data and the plotted result."
            LearningDepth.University -> "Check the objective, assumptions, parameter estimation and validation protocol."
            LearningDepth.Advanced -> "Also inspect numerical stability, computational cost, calibration and distribution shift."
        }
        theoryOverride(topic, kind)?.let { override ->
            return LearningProfile(
                definition = override.definition,
                purpose = "${override.purpose} $detail",
                intuition = override.intuition,
                steps = override.steps,
                equation = override.equation,
                equationNote = override.equationNote,
                advantages = override.advantages,
                limitations = override.limitations,
                assumptions = override.assumptions,
                hyperparameters = override.hyperparameters,
                applications = override.applications,
                mistakes = override.mistakes,
                sampleData = override.sampleData,
                kind = override.kind
            )
        }
        return LearningProfile(
            definition = "$name is a ${topic.section.lowercase()} concept in ${topic.domain.lowercase()} that transforms observed data into a useful representation, estimate, or decision.",
            purpose = "Use it to ${purposeFor(kind)}. $detail",
            intuition = intuitionFor(kind),
            steps = stepsFor(kind),
            equation = equation.first,
            equationNote = equation.second,
            advantages = advantagesFor(kind),
            limitations = limitationsFor(kind),
            assumptions = assumptionsFor(kind),
            hyperparameters = hyperparametersFor(name, kind),
            applications = applicationsFor(kind),
            mistakes = listOf("Evaluating on training data only", "Ignoring preprocessing and data leakage", "Tuning parameters before choosing the right metric"),
            sampleData = sampleFor(kind),
            kind = kind
        )
    }

    fun related(topic: LearnTopic): List<LearnTopic> = topics
        .filter { it.id != topic.id && it.section == topic.section }.take(3)

    private fun theoryOverride(topic: LearnTopic, inferredKind: VisualizationKind): TheoryOverride? {
        fun o(
            definition: String,
            purpose: String,
            intuition: String,
            steps: List<String>,
            equation: String,
            equationNote: String,
            advantages: List<String>,
            limitations: List<String>,
            assumptions: List<String>,
            hyperparameters: List<String>,
            applications: List<String>,
            mistakes: List<String>,
            sampleData: String,
            kind: VisualizationKind = inferredKind
        ) = TheoryOverride(definition, purpose, intuition, steps, equation, equationNote, advantages, limitations, assumptions, hyperparameters, applications, mistakes, sampleData, kind)

        return when {
            topic.title == "Simple Linear Regression" -> o(
                "Linear regression models a continuous target as a straight-line relationship between an input feature and an output.",
                "Use it as an interpretable baseline for numeric prediction and for learning residuals, slope, intercept, and least-squares fitting.",
                "A line is moved through the point cloud until vertical prediction errors are as small as possible overall.",
                listOf("Choose numeric feature x and target y", "Fit slope w and intercept b", "Predict y_hat = wx + b", "Measure residuals y - y_hat", "Evaluate MSE, MAE, R2 and residual pattern"),
                "y_hat = wx + b,   minimize sum_i (y_i - y_hat_i)^2",
                "Ordinary least squares chooses the line with minimum squared residual error.",
                listOf("Very interpretable", "Fast and stable", "Good diagnostic baseline"),
                listOf("Only captures linear relationships", "Sensitive to outliers", "Can be misleading when residual assumptions fail"),
                listOf("Relationship is approximately linear", "Residuals are independent with roughly constant variance", "No severe outlier or leakage problem"),
                listOf("fit_intercept", "feature scaling when comparing coefficients", "regularization only in ridge/lasso variants"),
                listOf("Price estimation", "Demand forecasting", "Scientific trend fitting"),
                listOf("Treating correlation as causation", "Ignoring residual plots", "Evaluating only on training data"),
                "A scatter plot of x values with continuous y targets and a best-fit line.",
                VisualizationKind.Regression
            )
            topic.title == "Logistic Regression" -> o(
                "Logistic regression is a linear classifier that maps a weighted feature score through a sigmoid or softmax to estimate class probability.",
                "Use it for interpretable classification, calibrated probability baselines, and decision-threshold experiments.",
                "It draws a linear boundary, then reports how far a point lies on the positive or negative side as a probability.",
                listOf("Scale or encode features", "Compute linear score z = w^T x + b", "Convert score with sigmoid or softmax", "Optimize cross-entropy loss", "Choose a threshold and evaluate precision, recall and calibration"),
                "p(y=1|x) = 1 / (1 + exp(-(w^T x + b)))",
                "The log-odds are linear in the input features; the sigmoid maps them into 0..1.",
                listOf("Interpretable coefficients", "Probabilistic output", "Strong baseline for linearly separable data"),
                listOf("Linear decision boundary unless features are engineered", "Sensitive to class imbalance and outliers", "Needs regularization with many correlated features"),
                listOf("Independent labelled samples", "Reasonable linear separation in feature space", "Training and deployment preprocessing match"),
                listOf("C or lambda regularization", "penalty L1/L2/elastic-net", "class weights", "decision threshold"),
                listOf("Spam detection", "Churn prediction", "Medical risk screening"),
                listOf("Using 0.5 threshold blindly", "Reading coefficients without checking scaling", "Ignoring calibration and imbalance"),
                "Two labelled classes with a linear probability boundary.",
                VisualizationKind.Classification
            )
            topic.title == "Polynomial Regression" -> o(
                "Polynomial regression fits a curved relationship by expanding the input into powers such as x, x^2, and x^3, then applying linear regression.",
                "Use it when a numeric target follows a smooth nonlinear curve but you still want a transparent parametric model.",
                "Instead of bending the model directly, you give linear regression extra curved features to combine.",
                listOf("Choose polynomial degree", "Create powers of each feature", "Fit linear coefficients on expanded features", "Inspect training versus validation error", "Reduce degree or regularize if the curve wiggles too much"),
                "y_hat = b + w1*x + w2*x^2 + ... + wd*x^d",
                "The model is linear in coefficients but nonlinear in the original input.",
                listOf("Captures smooth curves", "Easy extension of linear regression", "Works with regularization"),
                listOf("High degree can overfit", "Feature values can explode without scaling", "Poor extrapolation outside training range"),
                listOf("Curve is reasonably smooth", "Degree is selected using validation data", "Input range at inference is similar to training"),
                listOf("degree", "include_bias", "regularization strength", "feature scaling"),
                listOf("Growth curves", "Calibration curves", "Physics-inspired curve fitting"),
                listOf("Choosing degree from test data", "Trusting extrapolation", "Skipping scaling for high-degree terms"),
                "One numeric input with a curved continuous target.",
                VisualizationKind.Regression
            )
            topic.title == "Ridge Regression" -> o(
                "Ridge regression is linear regression with an L2 penalty that shrinks coefficients toward zero without usually making them exactly zero.",
                "Use it to stabilize linear models when features are correlated or numerous.",
                "The model still fits a line or plane, but large weights become costly, so the fit becomes smoother and less brittle.",
                listOf("Scale numeric features", "Fit linear predictions", "Add lambda * sum(w^2) to the loss", "Tune lambda on validation data", "Inspect coefficients and residuals"),
                "min_w sum_i (y_i - x_i^T w)^2 + lambda * sum_j w_j^2",
                "The L2 penalty reduces variance by discouraging large coefficients.",
                listOf("Handles multicollinearity", "Stable closed-form or optimized fit", "Keeps all features available"),
                listOf("Does not perform feature selection", "Still linear in engineered features", "Requires tuning lambda"),
                listOf("Features are scaled before penalty", "A linear model is useful", "Validation split represents deployment"),
                listOf("alpha or lambda", "fit_intercept", "solver", "feature scaling"),
                listOf("High-dimensional regression", "Forecasting baselines", "Noisy tabular prediction"),
                listOf("Applying penalty to unscaled features", "Assuming small coefficients mean no effect", "Tuning lambda on the test set"),
                "Many numeric features with correlated predictors and a continuous target.",
                VisualizationKind.Regression
            )
            topic.title == "Lasso Regression" -> o(
                "Lasso regression is linear regression with an L1 penalty that can shrink some coefficients exactly to zero.",
                "Use it when you want a sparse, interpretable linear model and built-in feature selection.",
                "Every nonzero coefficient pays a cost, so weak or redundant features can be dropped entirely.",
                listOf("Scale features", "Fit linear predictions", "Add lambda * sum(abs(w)) to the loss", "Tune lambda", "Inspect selected nonzero features"),
                "min_w sum_i (y_i - x_i^T w)^2 + lambda * sum_j |w_j|",
                "The L1 penalty creates sparse solutions by pushing some weights to zero.",
                listOf("Automatic feature selection", "Interpretable sparse models", "Useful with many irrelevant features"),
                listOf("Can be unstable with highly correlated features", "Still linear", "May underfit when lambda is too high"),
                listOf("Features are scaled", "Sparsity is desirable", "Validation data guides lambda"),
                listOf("alpha or lambda", "max iterations", "tolerance", "feature scaling"),
                listOf("Feature selection", "Sparse risk scoring", "High-dimensional tabular models"),
                listOf("Forgetting to scale features", "Treating selected features as causal proof", "Using too much regularization"),
                "A wide tabular dataset where only a subset of predictors should matter.",
                VisualizationKind.Regression
            )
            topic.title == "K-Nearest Neighbors" -> o(
                "K-Nearest Neighbors predicts a class by finding the K closest stored training samples and voting among their labels.",
                "Use it to teach distance-based local prediction and the effect of K, scaling, and distance metrics.",
                "The model does not learn weights; nearby examples decide the answer at prediction time.",
                listOf("Scale features", "Store labelled samples", "Choose K and distance metric", "Find nearest samples for a query", "Vote uniformly or by distance"),
                "class(x) = majority label among N_K(x)",
                "The neighbourhood N_K(x) contains the K training samples closest to the query.",
                listOf("Simple and intuitive", "Flexible nonlinear boundaries", "No training optimization"),
                listOf("Slow prediction on large datasets", "Sensitive to feature scale and irrelevant dimensions", "Can struggle in high dimensions"),
                listOf("Distance is meaningful", "Features are comparably scaled", "Local label structure is useful"),
                listOf("K", "distance metric", "uniform or distance weighting", "feature scaling"),
                listOf("Similarity search classification", "Small tabular datasets", "Recommendation-style neighbour lookup"),
                listOf("Skipping scaling", "Choosing K from the test set", "Using too many irrelevant features"),
                "Two-dimensional labelled points and a draggable query point.",
                VisualizationKind.Neighbours
            )
            topic.title == "Decision Tree" -> o(
                "A decision tree recursively splits feature space into regions whose labels are increasingly pure.",
                "Use it for interpretable nonlinear classification and to teach impurity, split thresholds, and tree paths.",
                "The model asks a sequence of yes/no questions until the sample lands in a leaf with a class distribution.",
                listOf("Evaluate candidate splits", "Choose the split with best impurity reduction", "Repeat recursively", "Stop by depth or leaf limits", "Predict using the reached leaf"),
                "best split = arg max impurity(parent) - weighted impurity(children)",
                "Gini or entropy measures how mixed the class labels are in a node.",
                listOf("Easy to explain", "Handles nonlinear interactions", "Little feature scaling needed"),
                listOf("High variance if deep", "Greedy splits may miss global structure", "Can overfit noisy data"),
                listOf("Training data represents deployment", "Splits are meaningful for feature values", "Pruning or validation controls complexity"),
                listOf("max_depth", "min_samples_leaf", "criterion", "max_features"),
                listOf("Rule extraction", "Risk segmentation", "Tabular classification"),
                listOf("Growing unrestricted trees", "Interpreting unstable splits as permanent rules", "Ignoring class imbalance"),
                "Labelled tabular points with candidate split lines and a highlighted path.",
                VisualizationKind.Tree
            )
            topic.title == "Random Forest" -> o(
                "Random Forest is an ensemble of decision trees trained on bootstrap samples with random feature subsets, then averaged or voted.",
                "Use it as a strong tabular baseline that reduces the variance of a single decision tree.",
                "Many noisy trees make different mistakes; voting averages those mistakes into a more stable prediction.",
                listOf("Draw bootstrap samples", "Train many decorrelated trees", "Randomly subset features at splits", "Aggregate votes or averages", "Estimate generalization with validation or out-of-bag data"),
                "prediction = average_or_vote(tree_1(x), ..., tree_B(x))",
                "Bagging reduces variance; feature subsampling decorrelates individual trees.",
                listOf("Strong performance on tabular data", "Handles nonlinear interactions", "Supports feature importance and OOB estimates"),
                listOf("Less interpretable than one tree", "Large forests can be heavy", "Can be biased with high-cardinality features"),
                listOf("Trees are sufficiently decorrelated", "Enough trees are trained", "Validation metric matches the task"),
                listOf("n_estimators", "max_depth", "max_features", "min_samples_leaf", "bootstrap"),
                listOf("Credit risk", "Fraud detection", "Tabular classification and regression"),
                listOf("Using too few trees", "Trusting impurity importance blindly", "Ignoring leakage in engineered features"),
                "Bootstrap samples feeding many trees whose outputs vote together.",
                VisualizationKind.Tree
            )
            topic.title == "Extra Trees" -> o(
                "Extra Trees is a tree ensemble that adds stronger randomness by choosing split thresholds more randomly than Random Forest.",
                "Use it when you want a fast, high-variance-reducing ensemble with more randomized trees.",
                "It trades carefully optimized individual splits for many very different trees whose average can generalize well.",
                listOf("Sample data, often without bootstrap", "Pick random feature subsets", "Choose random split thresholds", "Train many trees", "Aggregate votes or averages"),
                "prediction = aggregate over extremely randomized trees",
                "Random thresholds reduce correlation between trees and can reduce variance.",
                listOf("Fast tree training", "Strong variance reduction", "Often robust on tabular data"),
                listOf("Can need many trees", "Less interpretable than single trees", "Random splits may underfit weak signals"),
                listOf("Many randomized trees are enough to stabilize", "Features contain usable thresholds", "Validation guides complexity"),
                listOf("n_estimators", "max_features", "max_depth", "min_samples_leaf", "bootstrap"),
                listOf("Tabular baselines", "Noisy structured data", "Fast ensemble experiments"),
                listOf("Expecting every tree to be good alone", "Using impurity importance without checks", "Ignoring calibration"),
                "A set of randomized split trees whose votes are aggregated.",
                VisualizationKind.Tree
            )
            topic.title == "Support Vector Machine" -> o(
                "Support Vector Machine finds a decision boundary with maximum margin, controlled by support vectors near the boundary.",
                "Use it for margin-based classification, especially with scaled features and linear or kernelized boundaries.",
                "Only the hardest boundary-adjacent points define the separating surface; distant points matter less.",
                listOf("Scale features", "Choose linear or kernel representation", "Optimize margin with slack violations", "Identify support vectors", "Tune C and kernel parameters using validation"),
                "min 1/2 ||w||^2 + C * sum xi_i, subject to y_i(w^T x_i + b) >= 1 - xi_i",
                "C balances wide margin against allowing classification violations.",
                listOf("Effective in high dimensions", "Clear margin concept", "Kernel trick handles nonlinear boundaries"),
                listOf("Requires scaling", "Kernel SVM can be slow on large data", "Probabilities need calibration"),
                listOf("Features are scaled", "Margin is meaningful for the task", "Kernel and C are validated"),
                listOf("C", "kernel", "gamma", "degree", "class weights"),
                listOf("Text classification", "Small/medium tabular classification", "Image feature classification"),
                listOf("Skipping scaling", "Using RBF gamma without validation", "Assuming SVM scores are calibrated probabilities"),
                "Two classes with support vectors, margin lines and possible violations.",
                VisualizationKind.Classification
            )
            "Naive Bayes" in topic.title -> o(
                "Naive Bayes is a probabilistic classifier that applies Bayes' theorem while assuming features are conditionally independent given the class.",
                "Use it for fast probabilistic classification and as a strong baseline for text or count-like features.",
                "Each feature contributes evidence to each class, and the class with the largest posterior score wins.",
                listOf("Estimate class priors", "Estimate feature likelihoods per class", "Apply smoothing", "Sum log evidence for a new sample", "Predict the class with highest posterior score"),
                "argmax_c log P(c) + sum_j log P(x_j | c)",
                "The independence assumption makes likelihood estimation simple and fast.",
                listOf("Very fast", "Works well for sparse text", "Handles small datasets surprisingly well"),
                listOf("Independence assumption is often false", "Probability calibration can be poor", "Feature likelihood choice matters"),
                listOf("Feature model matches data type", "Training examples estimate likelihoods well", "Smoothing handles unseen values"),
                listOf("smoothing alpha", "Gaussian vs multinomial vs Bernoulli likelihood", "class priors"),
                listOf("Spam filtering", "Document classification", "Sentiment baselines"),
                listOf("Using Gaussian NB for count text", "Ignoring correlated features", "Forgetting smoothing"),
                "Feature vectors with class priors and per-class feature likelihoods.",
                VisualizationKind.Classification
            )
            topic.title == "Linear Discriminant Analysis" -> o(
                "Linear Discriminant Analysis models each class as a Gaussian with shared covariance and produces a linear decision boundary.",
                "Use it for interpretable classification and dimensional projection that separates class means.",
                "Classes are represented by centers and shared spread; a point is assigned to the class whose Gaussian evidence is strongest.",
                listOf("Estimate class means", "Estimate shared covariance", "Compute discriminant scores", "Classify by largest score", "Validate assumptions and calibration"),
                "delta_k(x) = x^T Sigma^-1 mu_k - 1/2 mu_k^T Sigma^-1 mu_k + log pi_k",
                "Shared covariance makes the boundary linear between class distributions.",
                listOf("Interpretable linear boundary", "Can work with small data", "Provides class probabilities under assumptions"),
                listOf("Shared covariance assumption may fail", "Sensitive to outliers", "Needs enough samples for covariance"),
                listOf("Class-conditional Gaussian structure", "Similar covariance across classes", "Features are not severely collinear"),
                listOf("solver", "shrinkage", "class priors", "number of components"),
                listOf("Classical pattern recognition", "Medical classification", "Low-dimensional projections"),
                listOf("Using it when covariances differ strongly", "Ignoring outliers", "Forgetting covariance regularization"),
                "Labelled Gaussian-like class clouds with linear discriminant boundary.",
                VisualizationKind.Classification
            )
            topic.title == "Gradient Boosting" || topic.title == "Gradient Boosting Regression" -> o(
                "Gradient Boosting builds an additive ensemble of weak learners, usually trees, where each new learner corrects current errors.",
                "Use it for strong tabular prediction and to teach stage-wise residual correction.",
                "The model starts simple, then repeatedly adds small trees that point in the direction of lower loss.",
                listOf("Start with a baseline prediction", "Compute residuals or negative gradients", "Fit a weak learner to those errors", "Add it with a learning rate", "Repeat and validate stage count"),
                "F_m(x) = F_(m-1)(x) + eta * h_m(x)",
                "Each stage adds a small correction to the current model.",
                listOf("High predictive accuracy", "Flexible loss functions", "Captures nonlinear interactions"),
                listOf("Can overfit without regularization", "Sequential training is slower than bagging", "Needs careful tuning"),
                listOf("Weak learners improve the loss", "Validation controls number of stages", "Data leakage is avoided"),
                listOf("n_estimators", "learning_rate", "max_depth", "subsample", "loss"),
                listOf("Tabular regression", "Ranking", "Classification baselines"),
                listOf("Using too many stages", "Ignoring validation curves", "Setting learning rate too high"),
                "A residual table showing before prediction, residual, correction and after prediction.",
                VisualizationKind.Tree
            )
            topic.title == "XGBoost" || topic.title == "XGBoost Regression" -> o(
                "XGBoost is a regularized gradient-boosted tree system that uses first- and second-order loss information plus split penalties.",
                "Use it for high-performance tabular prediction with explicit regularization and missing-value handling.",
                "It chooses tree splits by calculating how much each split improves the objective after accounting for complexity cost.",
                listOf("Compute gradients and Hessians", "Score candidate splits by gain", "Grow regularized trees", "Shrink each tree contribution", "Validate rounds with early stopping"),
                "gain = 1/2 [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)] - gamma",
                "lambda penalizes leaf weights; gamma penalizes adding splits.",
                listOf("Excellent tabular accuracy", "Strong regularization controls", "Handles sparse and missing values"),
                listOf("Many hyperparameters", "Can overfit leakage quickly", "Less transparent than small models"),
                listOf("Validation data represents deployment", "Features are prepared consistently", "Early stopping monitors generalization"),
                listOf("eta", "max_depth", "min_child_weight", "subsample", "lambda", "gamma"),
                listOf("Kaggle-style tabular prediction", "Risk scoring", "Ranking and classification"),
                listOf("Tuning on test data", "Ignoring early stopping", "Using importance without leakage checks"),
                "Candidate split table with gradients, Hessians, gain and regularization.",
                VisualizationKind.Tree
            )
            topic.title == "LightGBM" || topic.title == "LightGBM Regression" -> o(
                "LightGBM is a gradient-boosted tree framework optimized with histogram-based splits and leaf-wise tree growth.",
                "Use it for fast large-scale tabular learning, especially with many rows or features.",
                "It bins feature values into histograms, then grows the leaf that promises the biggest loss reduction.",
                listOf("Bin continuous features", "Compute histogram split gains", "Grow trees leaf-wise", "Apply shrinkage and regularization", "Use validation and early stopping"),
                "F_m(x) = F_(m-1)(x) + eta * tree_m(x)",
                "Histogram splits speed up gain search; leaf-wise growth can be powerful but needs depth/leaf controls.",
                listOf("Very fast on large tabular data", "Handles categorical features in many setups", "High accuracy with tuning"),
                listOf("Leaf-wise growth can overfit", "Sensitive to num_leaves", "Small datasets may need simpler models"),
                listOf("Validation controls complexity", "Bins preserve enough signal", "Categorical handling is configured correctly"),
                listOf("num_leaves", "learning_rate", "max_depth", "min_data_in_leaf", "feature_fraction"),
                listOf("Large tabular datasets", "Ranking", "Real-time model iteration"),
                listOf("Setting num_leaves too high", "Skipping early stopping", "Trusting default categorical handling blindly"),
                "Histogram bins, leaf-wise split choices and validation loss curve.",
                VisualizationKind.Tree
            )
            topic.title == "CatBoost" || topic.title == "CatBoost Regression" -> o(
                "CatBoost is a gradient-boosted tree method designed to handle categorical features with ordered target statistics and symmetric trees.",
                "Use it for tabular data with important categorical variables and reduced target leakage risk.",
                "It encodes categories using ordered information from previous rows, then boosts trees in a leakage-aware way.",
                listOf("Mark categorical features", "Build ordered target statistics", "Train symmetric boosted trees", "Apply learning-rate shrinkage", "Validate with early stopping"),
                "F_m(x) = F_(m-1)(x) + eta * symmetric_tree_m(x)",
                "Ordered boosting reduces target leakage in categorical encodings.",
                listOf("Strong categorical handling", "Good defaults", "Ordered boosting reduces leakage risk"),
                listOf("Can be slower than simpler boosters", "Still needs validation", "Model internals can be complex"),
                listOf("Categorical columns are correctly specified", "Validation split avoids leakage", "Evaluation metric matches task"),
                listOf("iterations", "learning_rate", "depth", "l2_leaf_reg", "cat_features"),
                listOf("Categorical tabular prediction", "CTR/ranking tasks", "Business risk models"),
                listOf("Manual target encoding before split", "Not specifying categorical columns", "Ignoring validation leakage"),
                "Categorical columns, ordered statistics and boosted symmetric trees.",
                VisualizationKind.Tree
            )
            topic.title == "K-Means" -> o(
                "K-Means partitions data into K clusters by alternating between nearest-centroid assignment and centroid recomputation.",
                "Use it to discover compact, roughly spherical groups and to teach inertia and centroid movement.",
                "Centroids move to the middle of their assigned points until assignments stop changing much.",
                listOf("Choose K and initialize centroids", "Assign each point to nearest centroid", "Move each centroid to assigned mean", "Repeat until convergence", "Evaluate inertia and cluster usefulness"),
                "J = sum_i ||x_i - mu_(c_i)||^2",
                "The objective minimizes within-cluster squared distance to centroids.",
                listOf("Simple and fast", "Easy to visualize", "Scales well with mini-batches"),
                listOf("Requires K in advance", "Assumes compact centroid-shaped clusters", "Sensitive to scaling and initialization"),
                listOf("Euclidean distance is meaningful", "Clusters are roughly convex", "Features are scaled"),
                listOf("K", "initialization", "n_init", "max_iter", "distance scaling"),
                listOf("Customer segmentation", "Vector quantization", "Prototype discovery"),
                listOf("Using raw unscaled features", "Choosing K without validation or domain input", "Forcing non-spherical clusters"),
                "Unlabelled 2D points with draggable centroids and assignment colors.",
                VisualizationKind.Clustering
            )
            topic.title == "DBSCAN" -> o(
                "DBSCAN clusters points by density, grouping density-connected regions and marking sparse isolated points as noise.",
                "Use it when clusters may have arbitrary shapes and outlier detection matters.",
                "A cluster grows from dense core points; points that cannot connect to dense regions become noise.",
                listOf("Scale features", "Choose epsilon radius and MinPts", "Identify core points", "Expand density-connected clusters", "Mark non-reachable points as noise"),
                "core point if |N_epsilon(x)| >= MinPts",
                "Density reachability defines clusters without requiring K.",
                listOf("Finds arbitrary-shaped clusters", "Detects noise", "Does not require number of clusters"),
                listOf("Sensitive to epsilon and scale", "Struggles with varying densities", "High dimensions weaken distance"),
                listOf("Density is meaningful at one scale", "Features are scaled", "Distance metric matches the data"),
                listOf("epsilon", "MinPts", "distance metric", "feature scaling"),
                listOf("Spatial clustering", "Outlier detection", "Shape-based grouping"),
                listOf("Using default epsilon", "Skipping scaling", "Expecting good results with varying densities"),
                "A point cloud with core, border and noise points.",
                VisualizationKind.Density
            )
            topic.title == "Gaussian Mixture Models" -> o(
                "Gaussian Mixture Models represent data as a weighted mixture of Gaussian components with soft membership probabilities.",
                "Use it for soft clustering, density estimation, and elliptical cluster shapes.",
                "Each point partly belongs to each Gaussian, and EM alternates between responsibility estimation and parameter updates.",
                listOf("Choose number of components", "Initialize means and covariances", "E-step: compute responsibilities", "M-step: update component parameters", "Evaluate likelihood and model selection criteria"),
                "p(x) = sum_k pi_k * N(x | mu_k, Sigma_k)",
                "Mixture weights, means and covariances define the data density.",
                listOf("Soft cluster membership", "Models elliptical clusters", "Provides density likelihood"),
                listOf("Needs component count", "Can converge to local optima", "Gaussian assumption may be poor"),
                listOf("Data can be approximated by Gaussian components", "Covariance type is suitable", "Initialization is checked"),
                listOf("n_components", "covariance_type", "regularization covariance", "initialization", "max_iter"),
                listOf("Soft segmentation", "Anomaly scoring", "Density modelling"),
                listOf("Ignoring covariance singularities", "Choosing components only by training likelihood", "Interpreting soft clusters as labels without validation"),
                "Unlabelled points with Gaussian ellipses and responsibility shading.",
                VisualizationKind.Clustering
            )
            topic.title == "PCA" -> o(
                "Principal Component Analysis rotates data to new orthogonal axes that capture maximum variance, then keeps the leading components.",
                "Use it for dimensionality reduction, visualization, compression, and noise filtering.",
                "It finds the directions where the data spreads the most and projects points onto those directions.",
                listOf("Center and usually scale features", "Compute covariance or SVD", "Sort principal components by explained variance", "Project data onto top components", "Inspect variance retained and reconstruction error"),
                "Z = XW, where columns of W are top eigenvectors of covariance(X)",
                "Principal components are orthogonal directions of maximum variance.",
                listOf("Fast linear compression", "Removes correlated redundancy", "Useful visualization baseline"),
                listOf("Only linear structure", "Components can be hard to interpret", "Variance may not equal task usefulness"),
                listOf("Linear projection is useful", "Features are centered and scale is handled", "Large-variance directions matter"),
                listOf("n_components", "whitening", "scaling", "SVD solver"),
                listOf("2D visualization", "Noise reduction", "Preprocessing for models"),
                listOf("Running PCA before train/test split", "Forgetting scaling", "Assuming high variance means predictive value"),
                "A high-dimensional table projected onto PC1 and PC2 with explained variance.",
                VisualizationKind.Projection
            )
            topic.title == "Multi-Layer Perceptron" -> o(
                "A Multi-Layer Perceptron is a feedforward neural network made of dense layers, nonlinear activations and trainable weights.",
                "Use it for nonlinear function approximation on tabular or vector features and to teach forward/backpropagation.",
                "Layers repeatedly mix inputs with weights, apply nonlinearities, and learn internal representations.",
                listOf("Normalize inputs", "Run forward pass through dense layers", "Compute task loss", "Backpropagate gradients", "Update weights with an optimizer and validate"),
                "h_l = phi(W_l h_(l-1) + b_l)",
                "Nonlinear activations let stacked linear layers represent nonlinear functions.",
                listOf("Flexible nonlinear modelling", "Works for many vector tasks", "Foundation for deep learning"),
                listOf("Needs tuning and data", "Less interpretable than linear models", "Can overfit or suffer vanishing gradients"),
                listOf("Inputs are scaled", "Architecture capacity matches data", "Validation monitors overfitting"),
                listOf("hidden layers", "neurons", "activation", "learning rate", "batch size", "regularization"),
                listOf("Tabular neural baselines", "Embedding classifiers", "Function approximation"),
                listOf("Skipping normalization", "Using too large a network for tiny data", "Ignoring validation loss"),
                "Vector inputs passing through dense hidden layers to an output.",
                VisualizationKind.NeuralNetwork
            )
            topic.title == "CNN" -> o(
                "A Convolutional Neural Network uses learned local filters, shared across spatial positions, to build feature maps from images or grids.",
                "Use it for image-like data where local patterns and translation structure matter.",
                "The same small detector slides across the image, finding edges or textures wherever they appear.",
                listOf("Arrange input as channels and pixels", "Apply convolution filters", "Apply activation and pooling or striding", "Stack deeper feature maps", "Train end-to-end with backpropagation"),
                "feature_map(i,j,k) = phi(sum_c sum_u sum_v W(u,v,c,k) * X(i+u,j+v,c) + b_k)",
                "Weight sharing makes CNNs parameter-efficient for spatial data.",
                listOf("Excellent for images", "Learns local features", "Parameter sharing improves efficiency"),
                listOf("Needs enough labelled data or transfer learning", "Can be sensitive to distribution shift", "Feature maps are not always easy to interpret"),
                listOf("Local spatial patterns matter", "Input alignment and preprocessing are consistent", "Labels match visible evidence"),
                listOf("kernel size", "stride", "padding", "filters", "pooling", "learning rate"),
                listOf("Image classification", "Medical imaging", "Visual inspection"),
                listOf("Wrong output shape calculation", "Ignoring input normalization", "Using dense layers where convolution is more appropriate"),
                "Small images, convolution kernels, feature maps and pooling windows.",
                VisualizationKind.Convolution
            )
            topic.title == "Recurrent Neural Network" -> o(
                "A Recurrent Neural Network processes a sequence one step at a time while carrying a hidden state forward.",
                "Use it to teach sequence memory and temporal dependence, especially before LSTM and GRU gates.",
                "Each new item updates a running memory that summarizes earlier items.",
                listOf("Encode sequence inputs", "Update hidden state at each time step", "Produce output from state", "Backpropagate through time", "Watch for vanishing or exploding gradients"),
                "h_t = phi(W_x x_t + W_h h_(t-1) + b)",
                "The same recurrent weights are reused at every time step.",
                listOf("Natural sequence model", "Shares parameters across time", "Good conceptual bridge to gated RNNs"),
                listOf("Vanishing/exploding gradients", "Weak long-range memory", "Sequential computation is hard to parallelize"),
                listOf("Order matters", "Hidden state can summarize useful context", "Sequence lengths are handled carefully"),
                listOf("hidden size", "sequence length", "learning rate", "gradient clipping", "activation"),
                listOf("Simple sequence classification", "Time-series baselines", "Character modelling"),
                listOf("Expecting long memory from a plain RNN", "Not clipping gradients", "Leaking future time steps"),
                "A sequence timeline with hidden state vectors passed forward.",
                VisualizationKind.Sequence
            )
            topic.title == "LSTM" -> o(
                "Long Short-Term Memory is a gated recurrent network that uses input, forget and output gates plus a cell state to preserve sequence information.",
                "Use it for sequence tasks where longer-range context matters more than a plain RNN can handle.",
                "The cell state is a controlled memory conveyor; gates decide what to keep, erase and expose.",
                listOf("Encode sequence inputs", "Compute input, forget and output gates", "Update cell state", "Emit hidden state", "Train with backpropagation through time and validation"),
                "c_t = f_t*c_(t-1) + i_t*g_t,   h_t = o_t*tanh(c_t)",
                "Forget, input and output gates regulate memory flow.",
                listOf("Better long-range memory than plain RNN", "Controls forgetting explicitly", "Strong time-series and text baseline"),
                listOf("More parameters than GRU/RNN", "Sequential and slower than attention", "Still can overfit or drift"),
                listOf("Past context is predictive", "Sequences are ordered correctly", "Future leakage is prevented"),
                listOf("hidden size", "layers", "dropout", "learning rate", "sequence length"),
                listOf("Forecasting", "Text sequence modelling", "Sensor streams"),
                listOf("Forget gate settings that erase useful context", "Training on future-leaked windows", "Ignoring gradient clipping"),
                "A sequence with gate values and cell-state memory over time.",
                VisualizationKind.Sequence
            )
            topic.title == "GRU" -> o(
                "Gated Recurrent Unit is a compact gated recurrent network using update and reset gates to manage sequence memory.",
                "Use it as a simpler alternative to LSTM for sequence tasks with fewer parameters.",
                "One gate controls how much old state survives, while another controls how new candidate memory is formed.",
                listOf("Encode sequence inputs", "Compute update and reset gates", "Build candidate hidden state", "Mix old and new state", "Train through time and validate"),
                "h_t = (1 - z_t)*h_(t-1) + z_t*h_tilde_t",
                "The update gate z_t controls memory retention versus replacement.",
                listOf("Fewer parameters than LSTM", "Often trains faster", "Good sequence baseline"),
                listOf("Less explicit memory structure than LSTM", "Still sequential", "Can struggle with very long context"),
                listOf("Ordered context matters", "Gating can capture needed dependencies", "Evaluation avoids future leakage"),
                listOf("hidden size", "layers", "dropout", "learning rate", "sequence length"),
                listOf("Time-series forecasting", "Sequence classification", "Text and sensor streams"),
                listOf("Assuming gates solve all long-context problems", "Not validating sequence window length", "Skipping scaling for time series"),
                "A sequence timeline showing update/reset gates and hidden state.",
                VisualizationKind.Sequence
            )
            topic.title == "Transformer" || topic.title == "Self-Attention" -> o(
                "Transformer self-attention lets each token compute relevance weights over other tokens and mix their value vectors, often inside multi-head attention blocks.",
                "Use it for language, vision and sequence tasks where global context and parallel token processing matter.",
                "Instead of passing memory step by step, every token can directly look at other tokens and decide what matters.",
                listOf("Embed tokens and add positions", "Project embeddings into Q, K and V", "Compute scaled dot-product attention", "Mix values by attention weights", "Stack heads, feed-forward layers, residuals and normalization"),
                "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V",
                "Query-key similarity becomes a normalized weight over value vectors.",
                listOf("Models long-range interactions", "Parallel over tokens", "Multi-head attention learns different relations"),
                listOf("Attention memory grows quadratically with sequence length", "Needs data and compute", "Attention weights are not guaranteed causal explanations"),
                listOf("Positions are encoded", "Context window contains useful evidence", "Masking prevents future leakage when needed"),
                listOf("model dimension", "heads", "context length", "layers", "dropout", "learning rate"),
                listOf("Language models", "Document understanding", "Vision transformers"),
                listOf("Forgetting positional encoding", "Using causal tasks without masks", "Treating attention maps as definitive explanations"),
                "A token sequence with Q/K/V vectors and an attention matrix.",
                VisualizationKind.Attention
            )
            else -> null
        }
    }

    private fun purposeFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Regression -> "estimate a continuous outcome and explain how features move that estimate"
        VisualizationKind.Classification -> "estimate class evidence and draw a decision boundary"
        VisualizationKind.Neighbours -> "predict from the most similar labelled observations"
        VisualizationKind.Tree -> "build readable conditional decisions or an ensemble of them"
        VisualizationKind.Clustering, VisualizationKind.Density -> "find groups and unusual structure without labels"
        VisualizationKind.Projection -> "compress features while preserving useful structure"
        VisualizationKind.NeuralNetwork, VisualizationKind.Convolution, VisualizationKind.Sequence, VisualizationKind.Attention -> "learn layered representations for complex inputs"
        VisualizationKind.Reinforcement -> "choose actions that maximize long-term reward"
        VisualizationKind.TimeSeries -> "forecast future values from ordered observations"
        VisualizationKind.Probability -> "represent uncertainty and update beliefs from evidence"
        VisualizationKind.Optimization -> "move model parameters toward a lower objective value"
        VisualizationKind.Recommendation -> "rank items by expected relevance"
        VisualizationKind.Explanation -> "attribute a prediction to influential inputs or examples"
        else -> "extract a repeatable signal and test it on unseen examples"
    }

    private fun intuitionFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Regression -> "Imagine balancing a line or curve through a cloud of points so the total residual error is small."
        VisualizationKind.Classification -> "Learn a boundary that separates labelled regions, then turn each score into a class decision."
        VisualizationKind.Neighbours -> "Nearby examples vote; changing the neighbourhood changes how local or smooth the answer becomes."
        VisualizationKind.Tree -> "Ask one useful yes/no question at a time until each region is easier to predict."
        VisualizationKind.Clustering -> "Move representatives toward dense groups, then reassign observations and repeat."
        VisualizationKind.Density -> "Dense connected regions become clusters; isolated observations become noise."
        VisualizationKind.Projection -> "Rotate a lower-dimensional viewing axis until it preserves the structure that matters."
        VisualizationKind.Convolution -> "Slide a small reusable detector across the input to reveal local patterns."
        VisualizationKind.Sequence -> "Carry a compact state forward so the current output can use earlier context."
        VisualizationKind.Attention -> "Let each token assign relevance weights to other tokens before mixing their information."
        VisualizationKind.Autoencoder -> "Squeeze the input through a bottleneck, then reconstruct it from the compact code."
        VisualizationKind.Generative -> "Learn the data distribution well enough to create plausible new samples."
        VisualizationKind.Graph -> "Each node updates itself using messages from connected neighbours."
        VisualizationKind.Reinforcement -> "Actions that lead to useful future rewards become more valuable through repeated experience."
        VisualizationKind.TimeSeries -> "Separate recent level, trend and recurring patterns before extending them into the future."
        VisualizationKind.Probability -> "Start with a belief, score the evidence, and normalize the updated possibilities."
        VisualizationKind.Optimization -> "Follow the local slope downhill while controlling step size and momentum."
        VisualizationKind.Recommendation -> "Place users and items in a shared preference space, then rank nearby candidates."
        VisualizationKind.Explanation -> "Perturb or trace the prediction to measure which inputs changed it most."
        else -> "Convert raw observations into a structured signal, fit it, and verify that it generalizes."
    }

    private fun stepsFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Regression -> listOf("Prepare numeric features and targets", "Compute predictions", "Measure residual loss", "Update parameters", "Validate on unseen points")
        VisualizationKind.Classification -> listOf("Prepare features and class labels", "Compute class scores", "Convert scores to probabilities or votes", "Update the boundary", "Validate discrimination and calibration")
        VisualizationKind.Neighbours -> listOf("Scale features", "Store labelled samples", "Choose K and distance metric", "Find nearest samples for a query", "Average or vote from neighbours")
        VisualizationKind.Clustering, VisualizationKind.Density -> listOf("Scale features", "Measure neighbourhood structure", "Assign groups", "Update cluster state", "Check stability and noise")
        VisualizationKind.NeuralNetwork, VisualizationKind.Convolution, VisualizationKind.Sequence, VisualizationKind.Attention -> listOf("Encode the input", "Run the forward pass", "Compute task loss", "Backpropagate gradients", "Update weights and validate")
        VisualizationKind.Reinforcement -> listOf("Observe the state", "Choose an action", "Receive reward and next state", "Update value or policy", "Repeat with controlled exploration")
        else -> listOf("Define inputs and objective", "Apply preprocessing", "Fit or compute the model", "Inspect intermediate output", "Evaluate on held-out data")
    }

    private fun equationFor(name: String, kind: VisualizationKind): Pair<String, String> = when {
        name == "K-Means" || "K-Means" in name -> "J = sum_i ||x_i - mu_(c_i)||^2" to "Alternate assignment to the nearest centroid with recomputing each centroid mean."
        name == "Q-Learning" -> "Q(s,a) <- Q(s,a) + alpha [r + gamma max Q(s',a') - Q(s,a)]" to "The temporal-difference target combines immediate reward with discounted future value."
        "Attention" in name || "Transformer" in name -> "Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V" to "Scaled similarity becomes a normalized mixing weight over value vectors."
        "Bayes" in name -> "P(theta|D) = P(D|theta)P(theta) / P(D)" to "Posterior belief is proportional to likelihood times prior belief."
        name == "K-Nearest Neighbors Regression" -> "y_hat(x) = (1/K) sum_{i in N_K(x)} y_i" to "Find the K closest training samples to the query and average their target values."
        kind == VisualizationKind.Neighbours -> "prediction(x) = vote or average over N_K(x)" to "The neighbourhood is defined by the chosen distance metric and K."
        kind == VisualizationKind.Regression -> "y_hat = f(x; theta),   J(theta) = (1/n) sum L(y_hat_i, y_i)" to "Training chooses parameters that minimize prediction error, optionally with regularization."
        kind == VisualizationKind.Classification -> "p(y=1|x) = 1 / (1 + exp(-(w^T x + b)))" to "A threshold turns a probability or score into a class decision."
        kind == VisualizationKind.Optimization -> "theta_(t+1) = theta_t - alpha * grad J(theta_t)" to "The learning rate controls how far parameters move against the local gradient."
        kind == VisualizationKind.Probability -> "p(z|x) proportional to p(x|z)p(z)" to "Inference combines a generative likelihood with a prior."
        kind == VisualizationKind.NeuralNetwork || kind == VisualizationKind.Convolution -> "h_(l+1) = phi(W_l h_l + b_l)" to "Each layer applies an affine transform followed by a non-linear activation."
        else -> "theta* = arg min_theta J(theta; X, y)" to "The objective formalizes what a good solution means for this method."
    }

    private fun advantagesFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Tree -> listOf("Readable decision logic", "Handles nonlinear interactions", "Little feature scaling required")
        VisualizationKind.Attention -> listOf("Models long-range interactions", "Parallel token processing", "Inspectable attention weights")
        VisualizationKind.Regression -> listOf("Strong, interpretable baseline", "Fast to fit and predict", "Useful diagnostics")
        else -> listOf("Reusable mathematical objective", "Works in an offline experiment", "Supports measurable validation")
    }

    private fun limitationsFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Attention -> listOf("Memory grows quickly with sequence length", "Needs substantial data and compute", "Attention is not always a causal explanation")
        VisualizationKind.Density -> listOf("Sensitive to scale and density variation", "Parameter selection can be difficult", "High dimensions weaken distance measures")
        VisualizationKind.Regression -> listOf("Misses unmodelled nonlinear structure", "Outliers can dominate common losses", "Correlation is not causation")
        else -> listOf("Performance depends on data quality", "Hyperparameters affect behavior", "Distribution shift can invalidate results")
    }

    private fun assumptionsFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Regression -> listOf("Samples represent deployment data", "Residual structure matches the chosen loss", "Features contain a stable signal")
        VisualizationKind.Neighbours, VisualizationKind.Clustering, VisualizationKind.Density -> listOf("Distance is meaningful", "Features are comparably scaled", "Local structure reflects the task")
        VisualizationKind.TimeSeries -> listOf("Temporal order is preserved", "Past patterns carry information", "Evaluation avoids future leakage")
        else -> listOf("Training and evaluation examples are representative", "The objective matches the real goal", "Preprocessing is identical at inference")
    }

    private fun hyperparametersFor(name: String, kind: VisualizationKind) = when {
        "K-Means" in name -> listOf("K: number of clusters", "Initialization: starting centroids", "Max iterations: update budget")
        name == "DBSCAN" -> listOf("epsilon: neighbourhood radius", "MinPts: density threshold", "Distance metric")
        kind == VisualizationKind.Attention -> listOf("Model dimension", "Attention heads", "Context length", "Dropout")
        kind == VisualizationKind.Neighbours -> listOf("K: neighbour count", "Distance metric", "Vote weighting")
        kind == VisualizationKind.Optimization -> listOf("Learning rate", "Momentum or beta values", "Batch size", "Schedule")
        else -> listOf("Capacity or complexity", "Regularization strength", "Learning rate or tolerance", "Training budget")
    }

    private fun applicationsFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Convolution -> listOf("Image classification", "Visual inspection", "Medical imaging")
        VisualizationKind.Sequence, VisualizationKind.Attention -> listOf("Language understanding", "Sequence forecasting", "Document retrieval")
        VisualizationKind.Recommendation -> listOf("Product ranking", "Media discovery", "Personalized feeds")
        VisualizationKind.Reinforcement -> listOf("Control", "Resource allocation", "Simulation-based planning")
        else -> listOf("Forecasting and decision support", "Pattern discovery", "Risk-aware automation")
    }

    private fun sampleFor(kind: VisualizationKind) = when (kind) {
        VisualizationKind.Regression -> "Eight observations with two numeric features and a continuous target."
        VisualizationKind.Neighbours, VisualizationKind.Clustering, VisualizationKind.Density, VisualizationKind.Projection -> "Twelve two-dimensional points arranged in three partially overlapping groups."
        VisualizationKind.Attention, VisualizationKind.Sequence -> "A six-token sentence represented by embeddings and an attention or state sequence."
        VisualizationKind.TimeSeries -> "Thirty ordered values containing level, trend, seasonality and noise."
        VisualizationKind.Reinforcement -> "A 4 x 4 grid with a start, goal, obstacle and step penalty."
        else -> "A small balanced sample split into training, validation and test partitions."
    }
}
