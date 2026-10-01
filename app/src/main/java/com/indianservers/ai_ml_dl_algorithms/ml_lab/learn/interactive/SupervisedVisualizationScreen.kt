package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.SupervisedVisualization
import kotlin.math.cos
import kotlin.math.sin

@Composable
internal fun SupervisedVisualizationScreen(
    topic: LearnTopic,
    kind: SupervisedVisualization,
    sharedPoints: SnapshotStateList<LabPoint>? = null
) {
    val context = LocalContext.current
    val classification = kind in classificationKinds || kind == SupervisedVisualization.Logistic
    val textModel = kind == SupervisedVisualization.MultinomialNb ||
        kind == SupervisedVisualization.BernoulliNb
    val defaultPreset = when (kind) {
        SupervisedVisualization.Polynomial -> DatasetPreset.Polynomial
        SupervisedVisualization.Robust -> DatasetPreset.Outliers
        SupervisedVisualization.ClassificationTree -> DatasetPreset.XorLike
        SupervisedVisualization.RegressionTree -> DatasetPreset.Polynomial
        else -> if (classification) DatasetPreset.TwoClusters else DatasetPreset.LinearNoise
    }
    var preset by remember(topic.id) { mutableStateOf(defaultPreset) }
    val points = remember(topic.id) {
        sharedPoints ?: mutableStateListOf<LabPoint>().apply { addAll(supervisedPreset(kind, defaultPreset)) }
    }
    var message by remember(topic.id) { mutableStateOf("") }
    val loadCsv = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) runCatching {
            val imported = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                reader.lineSequence().mapNotNull { row ->
                    val cells = row.split(',', ';', '\t').map(String::trim)
                    val x = cells.getOrNull(0)?.toDoubleOrNull()
                    val y = cells.getOrNull(1)?.toDoubleOrNull()
                    if (x == null || y == null) null else when {
                        kind == SupervisedVisualization.MultipleLinear ->
                            cells.getOrNull(2)?.toDoubleOrNull()?.let { LabPoint(x, y, target = it) }
                        classification -> LabPoint(x, y, cells.getOrNull(2)?.toIntOrNull() ?: 0)
                        else -> LabPoint(x, y, target = y)
                    }
                }.toList()
            }.orEmpty()
            require(imported.size >= 4) { "Load at least four numeric data rows." }
            points.clear()
            points.addAll(imported)
            message = "Loaded ${imported.size} samples."
        }.onFailure { message = it.message ?: "Could not read this dataset." }
    }
    val presets = when {
        classification -> listOf(DatasetPreset.TwoClusters, DatasetPreset.OverlappingClasses, DatasetPreset.XorLike, DatasetPreset.Circular)
        kind == SupervisedVisualization.Robust -> listOf(DatasetPreset.Outliers, DatasetPreset.LinearNoise, DatasetPreset.Polynomial, DatasetPreset.PerfectLinear)
        else -> listOf(DatasetPreset.LinearNoise, DatasetPreset.Polynomial, DatasetPreset.Outliers, DatasetPreset.PerfectLinear)
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(13.dp)) {
        Text(topic.title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 20.sp)
        Text(guidanceFor(kind), color = LabMuted, fontSize = 13.sp, lineHeight = 18.sp)
        if (!textModel) {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Dataset · ${points.size} samples", color = LabText, fontWeight = FontWeight.SemiBold)
                    presets.chunked(2).forEach { row ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                            row.forEach { option ->
                                SegmentedOption(option.label, option == preset, Modifier.weight(1f)) {
                                    preset = option
                                    points.clear()
                                    points.addAll(supervisedPreset(kind, option))
                                }
                            }
                            if (row.size == 1) Spacer(Modifier.weight(1f))
                        }
                    }
                    SegmentedOption("Load CSV", false, Modifier.fillMaxWidth()) {
                        loadCsv.launch(arrayOf("text/csv", "text/plain", "application/octet-stream"))
                    }
                    if (message.isNotBlank()) Text(message, color = LabCyan, fontSize = 12.sp)
                }
            }
        }
        if (classification) {
            SupervisedClassificationPanel(kind, points)
        } else {
            SupervisedRegressionPanel(kind, points)
        }
        Spacer(Modifier.height(12.dp))
    }
}

private val classificationKinds = setOf(
    SupervisedVisualization.ClassificationKnn, SupervisedVisualization.GaussianNb,
    SupervisedVisualization.MultinomialNb, SupervisedVisualization.BernoulliNb,
    SupervisedVisualization.ClassificationTree, SupervisedVisualization.ClassificationForest,
    SupervisedVisualization.ClassificationExtraTrees, SupervisedVisualization.Svm,
    SupervisedVisualization.Lda, SupervisedVisualization.Qda, SupervisedVisualization.Perceptron,
    SupervisedVisualization.SgdClassifier, SupervisedVisualization.ClassificationAdaBoost,
    SupervisedVisualization.ClassificationGradientBoosting, SupervisedVisualization.ClassificationXgboost,
    SupervisedVisualization.ClassificationLightgbm, SupervisedVisualization.ClassificationCatboost,
    SupervisedVisualization.GaussianProcessClassifier
)

private fun supervisedPreset(kind: SupervisedVisualization, preset: DatasetPreset): List<LabPoint> {
    if (kind == SupervisedVisualization.MultipleLinear) return List(28) { index ->
        val x1 = -0.9 + index * 1.8 / 27
        val x2 = sin(index * 2.4) * 0.72
        val noise = if (preset == DatasetPreset.PerfectLinear) 0.0 else sin(index * 7.2) * 0.045
        val nonlinear = if (preset == DatasetPreset.Polynomial) .42 * x1 * x1 else 0.0
        val outlier = if (preset == DatasetPreset.Outliers && index % 9 == 0) .65 else 0.0
        val target = 0.14 + 0.48 * x1 - 0.32 * x2 + noise + nonlinear + outlier
        LabPoint(x1, x2, target = target)
    }
    val source = PhaseOneDatasets.generate(preset, 28, 0.12, seed = 17)
    if (kind == SupervisedVisualization.RegressionKnn) return source.map { it.copy(target = it.y, label = 0) }
    if (kind == SupervisedVisualization.RegressionCatboost ||
        kind == SupervisedVisualization.ClassificationCatboost) {
        return source.mapIndexed { index, point ->
            val category = index % 3
            val x = listOf(-0.65, 0.0, 0.65)[category]
            if (kind == SupervisedVisualization.RegressionCatboost)
                point.copy(x = x, target = 0.5 * category - 0.45 + sin(index * 3.0) * 0.12)
            else {
                val label = if (category == 2 || (category == 1 && point.y > 0)) 1 else 0
                point.copy(x = x, label = label, target = label.toDouble())
            }
        }
    }
    return source
}

private fun guidanceFor(kind: SupervisedVisualization): String = when (kind) {
    SupervisedVisualization.SimpleLinear -> "Move samples to see least-squares fitting, residuals, and error update."
    SupervisedVisualization.MultipleLinear -> "Two independent features determine a fitted plane; inspect each coefficient and feature slice."
    SupervisedVisualization.Polynomial -> "Change polynomial degree to see underfitting, nonlinear fit, and overfitting."
    SupervisedVisualization.Ridge -> "Increase L2 strength to shrink coefficients while balancing error and complexity."
    SupervisedVisualization.Lasso -> "Increase L1 strength to drive less useful coefficients exactly to zero."
    SupervisedVisualization.ElasticNet -> "Combine L1 sparsity and L2 shrinkage with the L1 ratio."
    SupervisedVisualization.Logistic -> "Move class samples to see learned probabilities and the threshold-derived boundary."
    SupervisedVisualization.BayesianLinear -> "Add observations to see posterior line uncertainty narrow."
    SupervisedVisualization.Quantile -> "Change quantile to fit a different part of the conditional target distribution."
    SupervisedVisualization.Robust -> "Move an outlier and compare Huber fitting against ordinary least squares."
    SupervisedVisualization.Svr -> "The epsilon tube ignores small errors; points on or outside it become support vectors."
    SupervisedVisualization.RegressionTree -> "Recursive splits produce constant predictions in each leaf interval."
    SupervisedVisualization.RegressionForest, SupervisedVisualization.RegressionExtraTrees ->
        "Inspect individual randomized trees and their averaged regression prediction."
    SupervisedVisualization.RegressionKnn -> "Move the query and change K to average nearby target values."
    SupervisedVisualization.GaussianProcessRegression -> "Add observations and watch kernel-based posterior uncertainty contract."
    SupervisedVisualization.ClassificationKnn -> "Move the query or change K to see the selected neighbors and their votes."
    SupervisedVisualization.GaussianNb -> "Inspect class priors and feature likelihoods before the posterior class decision."
    SupervisedVisualization.MultinomialNb -> "Change word counts to see each token's contribution to a document class."
    SupervisedVisualization.BernoulliNb -> "Toggle binary features to see present and absent likelihood contributions."
    SupervisedVisualization.ClassificationTree -> "Move samples to rebuild impurity-driven recursive partitions and tree nodes."
    SupervisedVisualization.ClassificationForest, SupervisedVisualization.ClassificationExtraTrees ->
        "Bootstrap or random thresholds create distinct trees whose predictions vote."
    SupervisedVisualization.Svm -> "Move points to see which become support vectors and define the margin."
    SupervisedVisualization.Lda -> "Project samples along a direction that separates class means relative to within-class scatter."
    SupervisedVisualization.Qda -> "Class-specific covariance shapes curved probability regions."
    SupervisedVisualization.Perceptron -> "Step through mistakes to see the weight vector and boundary move."
    SupervisedVisualization.SgdClassifier -> "Step through single-sample hinge-loss updates and watch the boundary change."
    SupervisedVisualization.GaussianProcessClassifier -> "Compare high-confidence and uncertain predictive probability regions."
    SupervisedVisualization.RegressionAdaBoost, SupervisedVisualization.ClassificationAdaBoost ->
        "Step through reweighted examples and see how later weak learners focus on earlier errors."
    SupervisedVisualization.RegressionGradientBoosting, SupervisedVisualization.ClassificationGradientBoosting ->
        "Watch each shallow tree fit the current loss gradient and update the ensemble prediction."
    SupervisedVisualization.RegressionXgboost, SupervisedVisualization.ClassificationXgboost ->
        "Inspect regularized split gain and how each new tree corrects the current prediction."
    SupervisedVisualization.RegressionLightgbm, SupervisedVisualization.ClassificationLightgbm ->
        "Follow leaf-wise growth and see which high-gain leaf the next split expands."
    SupervisedVisualization.RegressionCatboost, SupervisedVisualization.ClassificationCatboost ->
        "Compare ordered categorical handling and the stagewise ensemble prediction."
}
