package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.*

internal enum class VisualizationRoute { SupervisedNative, StageTwoNative, StageThreeForecastNative, StageThreeLanguageNative, StageThreeNeuralLanguageNative, StageThreeVisionNative, StageThreeDeepNative, StageFourNative, ExistingNative, ExistingWeb, LegacyGeneral }
internal enum class VisualizationStatus { Correct, Partial, Generic, Incorrect, Missing }

internal enum class SupervisedVisualization {
    SimpleLinear, MultipleLinear, Polynomial, Ridge, Lasso, ElasticNet, Logistic,
    BayesianLinear, Quantile, Robust, Svr, RegressionTree, RegressionForest,
    RegressionExtraTrees, RegressionGradientBoosting, RegressionAdaBoost,
    RegressionXgboost, RegressionLightgbm, RegressionCatboost, RegressionKnn,
    GaussianProcessRegression, ClassificationKnn, GaussianNb, MultinomialNb,
    BernoulliNb, ClassificationTree, ClassificationForest, ClassificationExtraTrees,
    Svm, Lda, Qda, Perceptron, SgdClassifier, ClassificationAdaBoost,
    ClassificationGradientBoosting, ClassificationXgboost, ClassificationLightgbm,
    ClassificationCatboost, GaussianProcessClassifier
}

internal data class VisualizationRegistration(
    val topicId: String,
    val route: VisualizationRoute,
    val implementation: String,
    val family: String,
    val controls: String,
    val dataset: String,
    val interaction: String,
    val animation: String,
    val graphic: String,
    val status: VisualizationStatus,
    val supervised: SupervisedVisualization? = null,
    val stageTwo: StageTwoVisualization? = null,
    val forecast: ForecastVisualization? = null,
    val language: LanguageVisualization? = null,
    val neuralLanguage: NeuralLanguageVisualization? = null,
    val vision: VisionVisualization? = null,
    val deep: DeepVisualization? = null,
    val stageFour: Boolean = false
)

internal object AlgorithmVisualizationRegistry {
    private val canonicalTitleById = LearnCatalog.topics.associate { it.id to it.title }
    private val stageTwoBindings: Map<Pair<String, String>, StageTwoVisualization> =
        StageTwoVisualization.entries.associateBy { it.section to it.title }.also {
            require(it.size == StageTwoVisualization.entries.size)
            require(it.size == 60)
        }
    private val supervisedBindings = mapOf(
        ("Regression" to "Simple Linear Regression") to SupervisedVisualization.SimpleLinear,
        ("Regression" to "Multiple Linear Regression") to SupervisedVisualization.MultipleLinear,
        ("Regression" to "Polynomial Regression") to SupervisedVisualization.Polynomial,
        ("Regression" to "Ridge Regression") to SupervisedVisualization.Ridge,
        ("Regression" to "Lasso Regression") to SupervisedVisualization.Lasso,
        ("Regression" to "Elastic Net Regression") to SupervisedVisualization.ElasticNet,
        ("Regression" to "Logistic Regression") to SupervisedVisualization.Logistic,
        ("Regression" to "Bayesian Linear Regression") to SupervisedVisualization.BayesianLinear,
        ("Regression" to "Quantile Regression") to SupervisedVisualization.Quantile,
        ("Regression" to "Robust Regression") to SupervisedVisualization.Robust,
        ("Regression" to "Support Vector Regression") to SupervisedVisualization.Svr,
        ("Regression" to "Decision Tree Regression") to SupervisedVisualization.RegressionTree,
        ("Regression" to "Random Forest Regression") to SupervisedVisualization.RegressionForest,
        ("Regression" to "Extra Trees Regression") to SupervisedVisualization.RegressionExtraTrees,
        ("Regression" to "Gradient Boosting Regression") to SupervisedVisualization.RegressionGradientBoosting,
        ("Regression" to "AdaBoost Regression") to SupervisedVisualization.RegressionAdaBoost,
        ("Regression" to "XGBoost Regression") to SupervisedVisualization.RegressionXgboost,
        ("Regression" to "LightGBM Regression") to SupervisedVisualization.RegressionLightgbm,
        ("Regression" to "CatBoost Regression") to SupervisedVisualization.RegressionCatboost,
        ("Regression" to "K-Nearest Neighbors Regression") to SupervisedVisualization.RegressionKnn,
        ("Regression" to "Gaussian Process Regression") to SupervisedVisualization.GaussianProcessRegression,
        ("Classification" to "Logistic Regression") to SupervisedVisualization.Logistic,
        ("Classification" to "K-Nearest Neighbors") to SupervisedVisualization.ClassificationKnn,
        ("Classification" to "Gaussian Naive Bayes") to SupervisedVisualization.GaussianNb,
        ("Classification" to "Multinomial Naive Bayes") to SupervisedVisualization.MultinomialNb,
        ("Classification" to "Bernoulli Naive Bayes") to SupervisedVisualization.BernoulliNb,
        ("Classification" to "Decision Tree") to SupervisedVisualization.ClassificationTree,
        ("Classification" to "Random Forest") to SupervisedVisualization.ClassificationForest,
        ("Classification" to "Extra Trees") to SupervisedVisualization.ClassificationExtraTrees,
        ("Classification" to "Support Vector Machine") to SupervisedVisualization.Svm,
        ("Classification" to "Linear Discriminant Analysis") to SupervisedVisualization.Lda,
        ("Classification" to "Quadratic Discriminant Analysis") to SupervisedVisualization.Qda,
        ("Classification" to "Perceptron") to SupervisedVisualization.Perceptron,
        ("Classification" to "SGD Classifier") to SupervisedVisualization.SgdClassifier,
        ("Classification" to "AdaBoost") to SupervisedVisualization.ClassificationAdaBoost,
        ("Classification" to "Gradient Boosting") to SupervisedVisualization.ClassificationGradientBoosting,
        ("Classification" to "XGBoost") to SupervisedVisualization.ClassificationXgboost,
        ("Classification" to "LightGBM") to SupervisedVisualization.ClassificationLightgbm,
        ("Classification" to "CatBoost") to SupervisedVisualization.ClassificationCatboost,
        ("Classification" to "Gaussian Process Classifier") to SupervisedVisualization.GaussianProcessClassifier
    )

    val entries: Map<String, VisualizationRegistration> = LearnCatalog.topics.associate { topic ->
        topic.id to resolve(topic)
    }.also { registrations ->
        require(registrations.size == LearnCatalog.topics.size)
        require(supervisedBindings.size == 40)
        require(LearnCatalog.domains.first { it.title == "Supervised Learning" }
            .sections.flatMap { it.topics }.all { registrations.getValue(it.id).supervised != null })
    }

    fun forTopic(topic: LearnTopic): VisualizationRegistration {
        val registered = entries[topic.id] ?: error("No visualization registration for ${topic.id}")
        return if (topic.title != canonicalTitleById[topic.id]) resolve(topic) else registered
    }

    private fun resolve(topic: LearnTopic): VisualizationRegistration {
        val supervised = if (topic.domain == "Supervised Learning")
            supervisedBindings[topic.section to topic.title] else null
        if (supervised != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.SupervisedNative, "SupervisedVisualizationScreen/${supervised.name}",
            if (topic.section == "Regression" && supervised != SupervisedVisualization.Logistic) "regression" else "classification",
            controlsFor(supervised), datasetFor(supervised), "edit samples and legitimate hyperparameters",
            animationFor(supervised), graphicFor(supervised), VisualizationStatus.Partial, supervised
        )
        val stageTwo = if (topic.domain in setOf(
                "Unsupervised Learning", "Semi-Supervised Learning", "Ensemble Learning"
            )) stageTwoBindings[topic.section to topic.title] else null
        if (stageTwo != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageTwoNative, "StageTwoVisualizationScreen/${stageTwo.name}",
            topic.section, stageTwo.controls, stageTwo.dataset, stageTwo.interaction,
            stageTwo.animation, stageTwo.graphic, VisualizationStatus.Partial, stageTwo = stageTwo
        )
        val forecast = if (topic.domain == "Time-Series Algorithms") ForecastVisualization.byTitle[topic.title] else null
        if (forecast != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageThreeForecastNative, "StageThreeForecastScreen/${forecast.name}",
            "forecasting", "time step; model-specific window or parameter; series seed", "deterministic time series",
            "inspect highlighted inputs, computed components, and forecast", "step through series",
            "signal chart and algorithm-specific state", VisualizationStatus.Partial, forecast = forecast
        )
        val language = if (topic.domain == "Natural Language Processing") LanguageVisualization.byTitle[topic.title] else null
        if (language != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageThreeLanguageNative, "StageThreeLanguageScreen/${language.name}",
            topic.section, "document; selected token or window", "small offline text corpus",
            "inspect token contributions and computed matrix", "step through computation",
            "token flow, matrix and sequence scores", VisualizationStatus.Partial, language = language
        )
        val neuralLanguage = if (topic.domain == "Natural Language Processing" && topic.section == "Neural NLP")
            NeuralLanguageVisualization.byTitle[topic.title] else null
        if (neuralLanguage != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageThreeNeuralLanguageNative,
            "StageThreeNeuralLanguageScreen/${neuralLanguage.name}", topic.section,
            "token or time step", "offline token sequence", "inspect model state or attention weights",
            "step through token positions", "gates or attention matrix", VisualizationStatus.Partial,
            neuralLanguage = neuralLanguage
        )
        val vision = if (topic.domain == "Computer Vision") VisionVisualization.byTitle[topic.title] else null
        if (vision != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageThreeVisionNative, "StageThreeVisionScreen/${vision.name}",
            topic.section, "select feature or object; confidence threshold when relevant", "small offline image example",
            "inspect selected patch, box, mask, or embedding", "step through model pipeline",
            "image overlay and mechanism-specific computation", VisualizationStatus.Partial, vision = vision
        )
        val deep = if (topic.domain == "Deep Learning") DeepVisualization.byPlacement[topic.section to topic.title] else null
        if (deep != null) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageThreeDeepNative, "StageThreeDeepScreen/${deep.name}",
            topic.section, "select component; vary educational parameter", "small deterministic example",
            "inspect distinct architecture computation", "step through model stages",
            "mechanism-specific native diagram", VisualizationStatus.Partial, deep = deep
        )
        if (StageFourVisualization.supports(topic)) return VisualizationRegistration(
            topic.id, VisualizationRoute.StageFourNative,
            "StageFourVisualizationScreen/${StageFourVisualization.family(topic)}:${topic.title}",
            StageFourVisualization.family(topic), "scenario; step; legitimate algorithm parameters",
            "small offline educational dataset", "inspect computed state and step transitions",
            "stepwise mechanism", "topic-specific native chart or diagram",
            VisualizationStatus.Partial, stageFour = true
        )
        val webAsset = searchLabAssetPath(topic.title)
        val nativeImplementation =
            PhaseNineTopicMatcher.kindFor(topic.title, topic.domain)?.let { "PhaseNineAlgorithmLab/${it.name}" }
                ?: PhaseEightTopicMatcher.kindFor(topic.title, topic.domain)?.let { "PhaseEightAlgorithmLab/${it.name}" }
                ?: PhaseSevenTopicMatcher.kindFor(topic.title, topic.domain)?.let { "PhaseSevenAlgorithmLab/${it.name}" }
                ?: PhaseSixTopicMatcher.kindFor(topic.title, topic.domain)?.let { "PhaseSixAlgorithmLab/${it.name}" }
                ?: PhaseFiveTopicMatcher.kindFor(topic.title, topic.domain)?.let { "PhaseFiveAlgorithmLab/${it.name}" }
                ?: PhaseFourTopicMatcher.kindFor(topic.title)?.let { "PhaseFourAlgorithmLab/${it.name}" }
                ?: PhaseThreeTopicMatcher.kindFor(topic.title, topic.section, topic.domain)?.let { "PhaseThreeAlgorithmLab/${it.name}" }
                ?: PhaseTwoTopicMatcher.kindFor(topic.title, topic.section, topic.domain)?.let { "PhaseTwoAlgorithmLab/${it.name}" }
                ?: PhaseOneTopicMatcher.kindFor(topic.title, topic.section)?.let { "PhaseOneAlgorithmLab/${it.name}" }
        val route = when {
            topic.domain in setOf("Deep Learning", "Natural Language Processing", "Computer Vision", "Time-Series Algorithms") && nativeImplementation != null -> VisualizationRoute.ExistingNative
            webAsset != null -> VisualizationRoute.ExistingWeb
            nativeImplementation != null -> VisualizationRoute.ExistingNative
            else -> VisualizationRoute.LegacyGeneral
        }
        return VisualizationRegistration(
            topic.id, route,
            when (route) {
                VisualizationRoute.ExistingWeb -> "OfflineSearchLab/$webAsset"
                VisualizationRoute.ExistingNative -> nativeImplementation ?: error("Missing native implementation")
                VisualizationRoute.LegacyGeneral -> "AlgorithmLabScreen/${LearnCatalog.profile(topic, com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth.Beginner).kind.name}"
                VisualizationRoute.SupervisedNative -> error("handled above")
                VisualizationRoute.StageTwoNative -> error("handled above")
                VisualizationRoute.StageThreeForecastNative -> error("handled above")
                VisualizationRoute.StageThreeLanguageNative -> error("handled above")
                VisualizationRoute.StageThreeNeuralLanguageNative -> error("handled above")
                VisualizationRoute.StageThreeVisionNative -> error("handled above")
                VisualizationRoute.StageThreeDeepNative -> error("handled above")
                VisualizationRoute.StageFourNative -> error("handled above")
            },
            topic.section, "existing controls", "existing dataset", "inspect existing screen",
            "existing behavior", LearnCatalog.profile(topic,
                com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth.Beginner).kind.name,
            if (route == VisualizationRoute.LegacyGeneral) VisualizationStatus.Generic else VisualizationStatus.Partial
        )
    }

    private fun controlsFor(kind: SupervisedVisualization): String = when (kind) {
        SupervisedVisualization.SimpleLinear -> "samples; residual toggle"
        SupervisedVisualization.MultipleLinear -> "two-feature samples; rotation; held feature"
        SupervisedVisualization.Polynomial -> "samples; polynomial degree"
        SupervisedVisualization.Ridge, SupervisedVisualization.Lasso -> "samples; lambda"
        SupervisedVisualization.ElasticNet -> "samples; lambda; L1 ratio"
        SupervisedVisualization.Logistic -> "labelled samples; decision threshold"
        SupervisedVisualization.BayesianLinear -> "samples; observation noise"
        SupervisedVisualization.Quantile -> "samples; quantile"
        SupervisedVisualization.Robust -> "samples; outlier presets"
        SupervisedVisualization.Svr -> "samples; C; epsilon; kernel; gamma"
        SupervisedVisualization.RegressionTree -> "samples; depth; minimum leaf size"
        SupervisedVisualization.RegressionForest -> "samples; trees; depth; minimum leaf size; bootstrap"
        SupervisedVisualization.RegressionExtraTrees -> "samples; trees; depth; minimum leaf size"
        SupervisedVisualization.RegressionGradientBoosting,
        SupervisedVisualization.RegressionAdaBoost,
        SupervisedVisualization.RegressionXgboost,
        SupervisedVisualization.RegressionLightgbm,
        SupervisedVisualization.RegressionCatboost,
        SupervisedVisualization.ClassificationAdaBoost,
        SupervisedVisualization.ClassificationGradientBoosting,
        SupervisedVisualization.ClassificationXgboost,
        SupervisedVisualization.ClassificationLightgbm,
        SupervisedVisualization.ClassificationCatboost -> "samples; learner stage; learning rate"
        SupervisedVisualization.RegressionKnn -> "samples; K; query X"
        SupervisedVisualization.GaussianProcessRegression -> "samples; observation noise; kernel length scale"
        SupervisedVisualization.ClassificationKnn -> "labelled samples; K; distance metric; query point"
        SupervisedVisualization.GaussianNb -> "labelled samples; query point"
        SupervisedVisualization.MultinomialNb -> "document token counts"
        SupervisedVisualization.BernoulliNb -> "binary word presence"
        SupervisedVisualization.ClassificationTree -> "labelled samples; Gini or Entropy; depth; minimum leaf size"
        SupervisedVisualization.ClassificationForest -> "labelled samples; trees; depth; feature subset; bootstrap; minimum leaf size"
        SupervisedVisualization.ClassificationExtraTrees -> "labelled samples; trees; depth; feature subset; minimum leaf size"
        SupervisedVisualization.Svm -> "labelled samples; C; kernel; gamma; polynomial degree"
        SupervisedVisualization.Lda, SupervisedVisualization.Qda -> "labelled samples"
        SupervisedVisualization.Perceptron, SupervisedVisualization.SgdClassifier ->
            "labelled samples; step; previous; next; auto; learning rate"
        SupervisedVisualization.GaussianProcessClassifier -> "labelled samples; query; kernel length scale"
    }

    private fun datasetFor(kind: SupervisedVisualization): String = when (kind) {
        SupervisedVisualization.MultinomialNb -> "word-count documents"
        SupervisedVisualization.BernoulliNb -> "binary features"
        SupervisedVisualization.MultipleLinear -> "two features and numeric target"
        SupervisedVisualization.RegressionCatboost -> "categorical feature and numeric target"
        SupervisedVisualization.ClassificationCatboost -> "categorical feature and class label"
        SupervisedVisualization.Logistic, SupervisedVisualization.ClassificationKnn,
        SupervisedVisualization.GaussianNb, SupervisedVisualization.ClassificationTree,
        SupervisedVisualization.ClassificationForest, SupervisedVisualization.ClassificationExtraTrees,
        SupervisedVisualization.Svm, SupervisedVisualization.Lda, SupervisedVisualization.Qda,
        SupervisedVisualization.Perceptron, SupervisedVisualization.SgdClassifier,
        SupervisedVisualization.ClassificationAdaBoost, SupervisedVisualization.ClassificationGradientBoosting,
        SupervisedVisualization.ClassificationXgboost, SupervisedVisualization.ClassificationLightgbm,
        SupervisedVisualization.GaussianProcessClassifier -> "labelled two-feature samples"
        else -> "numeric features and target"
    }

    private fun animationFor(kind: SupervisedVisualization): String = when (kind) {
        SupervisedVisualization.Perceptron, SupervisedVisualization.SgdClassifier -> "stepwise updates"
        SupervisedVisualization.RegressionTree, SupervisedVisualization.ClassificationTree -> "recursive splits"
        SupervisedVisualization.RegressionGradientBoosting, SupervisedVisualization.RegressionAdaBoost,
        SupervisedVisualization.RegressionXgboost, SupervisedVisualization.RegressionLightgbm,
        SupervisedVisualization.RegressionCatboost, SupervisedVisualization.ClassificationGradientBoosting,
        SupervisedVisualization.ClassificationAdaBoost, SupervisedVisualization.ClassificationXgboost,
        SupervisedVisualization.ClassificationLightgbm, SupervisedVisualization.ClassificationCatboost -> "stagewise learners"
        else -> "model transition after input change"
    }

    private fun graphicFor(kind: SupervisedVisualization): String = when (kind) {
        SupervisedVisualization.SimpleLinear -> "fitted line; residual segments; metrics"
        SupervisedVisualization.MultipleLinear -> "perspective fitted plane; feature heatmap and slice"
        SupervisedVisualization.Polynomial -> "fitted nonlinear curve; residuals"
        SupervisedVisualization.Ridge -> "fit; L2 coefficient shrinkage bars and paths"
        SupervisedVisualization.Lasso -> "fit; sparse coefficient bars and paths"
        SupervisedVisualization.ElasticNet -> "fit; combined L1/L2 coefficient paths"
        SupervisedVisualization.Logistic -> "sigmoid; probability field; threshold boundary"
        SupervisedVisualization.BayesianLinear -> "posterior mean; credible band; sampled lines"
        SupervisedVisualization.Quantile -> "lower, selected, and upper quantile fits"
        SupervisedVisualization.Robust -> "Huber fit compared with least squares"
        SupervisedVisualization.Svr -> "epsilon tube; support vectors; fitted kernel curve"
        SupervisedVisualization.RegressionTree -> "piecewise constant fit; split thresholds; tree nodes"
        SupervisedVisualization.ClassificationTree -> "recursive rectangular regions and tree nodes"
        SupervisedVisualization.RegressionForest, SupervisedVisualization.RegressionExtraTrees,
        SupervisedVisualization.ClassificationForest, SupervisedVisualization.ClassificationExtraTrees -> "multiple trees and aggregate"
        SupervisedVisualization.RegressionGradientBoosting,
        SupervisedVisualization.ClassificationGradientBoosting -> "stagewise residual bars and cumulative tree prediction"
        SupervisedVisualization.RegressionAdaBoost,
        SupervisedVisualization.ClassificationAdaBoost -> "weighted samples; weak trees; aggregate"
        SupervisedVisualization.RegressionXgboost,
        SupervisedVisualization.ClassificationXgboost -> "gradient and Hessian bars; regularized tree corrections"
        SupervisedVisualization.RegressionLightgbm,
        SupervisedVisualization.ClassificationLightgbm -> "feature histogram; leaf-wise tree; aggregate"
        SupervisedVisualization.RegressionCatboost,
        SupervisedVisualization.ClassificationCatboost -> "ordered category statistics; symmetric tree"
        SupervisedVisualization.RegressionKnn -> "query; neighbors; average"
        SupervisedVisualization.GaussianProcessRegression -> "kernel posterior mean; uncertainty band; sampled functions"
        SupervisedVisualization.ClassificationKnn -> "neighbor links; radius; votes"
        SupervisedVisualization.GaussianNb -> "class densities; independent covariance ellipses; posterior factors"
        SupervisedVisualization.MultinomialNb -> "token counts and log posterior scores"
        SupervisedVisualization.BernoulliNb -> "binary feature switches and log posterior scores"
        SupervisedVisualization.Svm -> "support vectors; margin contours; linear and nonlinear kernels"
        SupervisedVisualization.Lda -> "pooled direction and projected observations"
        SupervisedVisualization.Qda -> "class covariance ellipses and quadratic boundary"
        SupervisedVisualization.Perceptron -> "mistake-driven boundary sequence"
        SupervisedVisualization.SgdClassifier -> "single-sample hinge-loss boundary sequence"
        SupervisedVisualization.GaussianProcessClassifier -> "latent GP probability and uncertainty field"
    }
}
