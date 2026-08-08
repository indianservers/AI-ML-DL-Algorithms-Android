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

    fun profile(topic: LearnTopic, depth: LearningDepth): LearningProfile {
        val name = topic.title
        val lower = name.lowercase()
        val kind = when {
            lower in listOf("dbscan", "hdbscan", "optics") -> VisualizationKind.Density
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
