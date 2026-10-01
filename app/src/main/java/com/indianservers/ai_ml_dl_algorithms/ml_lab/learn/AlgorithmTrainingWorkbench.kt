package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Point2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.TrainingSnapshot
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.*
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.exp
import kotlin.math.ln

internal data class LogisticModel(val weightX: Double, val weightY: Double, val bias: Double, val loss: Double) {
    fun probability(x: Double, y: Double): Double = 1.0 / (1.0 + exp(-(weightX * x + weightY * y + bias).coerceIn(-30.0, 30.0)))
}

internal fun fitLogistic(points: List<LabPoint>): LogisticModel {
    require(points.any { it.label == 0 } && points.any { it.label == 1 }) { "Both classes are needed for classification." }
    var wx = 0.0; var wy = 0.0; var bias = 0.0
    repeat(500) {
        var gx = 0.0; var gy = 0.0; var gb = 0.0
        points.forEach { p ->
            val probability = 1.0 / (1.0 + exp(-(wx * p.x + wy * p.y + bias).coerceIn(-30.0, 30.0)))
            val error = probability - p.label
            gx += error * p.x; gy += error * p.y; gb += error
        }
        val rate = .4 / points.size
        wx -= rate * gx; wy -= rate * gy; bias -= rate * gb
    }
    val model = LogisticModel(wx, wy, bias, 0.0)
    val loss = points.map { p ->
        val probability = model.probability(p.x, p.y).coerceIn(1e-9, 1.0 - 1e-9)
        -(p.label * ln(probability) + (1 - p.label) * ln(1.0 - probability))
    }.average()
    return model.copy(loss = loss)
}

internal class TrainingWorkspaceState(topic: LearnTopic) {
    private val kind = PhaseOneTopicMatcher.kindFor(topic.title, topic.section)
    private val defaultPreset = when (kind) {
        PhaseOneAlgorithmKind.Knn, PhaseOneAlgorithmKind.LogisticRegression -> DatasetPreset.TwoClusters
        PhaseOneAlgorithmKind.PolynomialRegression -> DatasetPreset.Polynomial
        else -> DatasetPreset.LinearNoise
    }
    val points = mutableStateListOf<LabPoint>().apply {
        addAll(PhaseOneDatasets.generate(defaultPreset, 24))
    }
    var fitted by mutableStateOf<RegressionFit?>(null)
    var fittedKnn by mutableStateOf<List<LabPoint>?>(null)
    var fittedLogistic by mutableStateOf<LogisticModel?>(null)
    fun invalidate() { fitted = null; fittedKnn = null; fittedLogistic = null }
}

internal fun hasLiveTrainer(topic: LearnTopic): Boolean = PhaseOneTopicMatcher.kindFor(topic.title, topic.section) in setOf(
    PhaseOneAlgorithmKind.SimpleLinearRegression,
    PhaseOneAlgorithmKind.PolynomialRegression,
    PhaseOneAlgorithmKind.RidgeRegression,
    PhaseOneAlgorithmKind.LassoRegression,
    PhaseOneAlgorithmKind.ElasticNetRegression,
    PhaseOneAlgorithmKind.LogisticRegression,
    PhaseOneAlgorithmKind.Knn
)

@Composable
internal fun AlgorithmTrainingWorkbench(
    topic: LearnTopic,
    workspace: TrainingWorkspaceState,
    onOpenVisualization: () -> Unit
) {
    val context = LocalContext.current
    val kind = remember(topic) { PhaseOneTopicMatcher.kindFor(topic.title, topic.section) }
    val regression = kind in setOf(
        PhaseOneAlgorithmKind.SimpleLinearRegression,
        PhaseOneAlgorithmKind.PolynomialRegression, PhaseOneAlgorithmKind.RidgeRegression,
        PhaseOneAlgorithmKind.LassoRegression, PhaseOneAlgorithmKind.ElasticNetRegression
    )
    val knn = kind == PhaseOneAlgorithmKind.Knn
    val logistic = kind == PhaseOneAlgorithmKind.LogisticRegression
    if (!hasLiveTrainer(topic)) {
        val concept = topic.domain == "Reinforcement Learning" && topic.section == "Fundamentals"
        LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            item { SectionTitle("Train & Inference", topic.title) }
            item {
                GlassPanel(Modifier.fillMaxWidth()) {
                    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                        Text(if (concept) "Concept simulation" else "Training engine not connected",
                            color = LabText, fontWeight = FontWeight.Bold, fontSize = 17.sp)
                        Text(if (concept)
                            "${topic.title} is a reinforcement-learning concept. Explore its state and interactions in Visualization; there is no standalone model to fit or export here."
                            else "${topic.title} has an interactive visualization, but this app does not currently fit or export a model for it. Explore its computed mechanism and scenario controls in Visualization.",
                            color = LabMuted, fontSize = 13.sp)
                        GradientButton("Open Visualization", Modifier.fillMaxWidth(), onOpenVisualization)
                    }
                }
            }
        }
        return
    }
    val preset = when {
        knn || logistic -> DatasetPreset.TwoClusters
        kind == PhaseOneAlgorithmKind.PolynomialRegression -> DatasetPreset.Polynomial
        else -> DatasetPreset.LinearNoise
    }
    val points = workspace.points
    var xText by remember { mutableStateOf("0.0") }
    var yText by remember { mutableStateOf("0.0") }
    var queryX by remember { mutableStateOf("0.1") }
    var queryY by remember { mutableStateOf("0.1") }
    var pointClass by remember { mutableIntStateOf(0) }
    var message by remember { mutableStateOf("Load a CSV or edit the data, then train.") }
    var fitted by workspace::fitted
    var fittedKnn by workspace::fittedKnn
    var fittedLogistic by workspace::fittedLogistic

    val importCsv = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) runCatching {
            val parsed = context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { reader ->
                reader.lineSequence().mapNotNull { row ->
                    val cells = row.split(',', ';', '\t').map(String::trim)
                    val x = cells.getOrNull(0)?.toDoubleOrNull()
                    val y = cells.getOrNull(1)?.toDoubleOrNull()
                    if (x != null && y != null) LabPoint(x, y, cells.getOrNull(2)?.toIntOrNull() ?: 0, y) else null
                }.toList()
            }.orEmpty()
            require(parsed.size >= 2) { "CSV needs at least two numeric x,y rows." }
            points.clear(); points.addAll(parsed); workspace.invalidate()
            message = "Loaded ${parsed.size} rows."
        }.onFailure { message = it.message ?: "Unable to load CSV." }
    }
    val exportModel = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/json")) { uri: Uri? ->
        if (uri != null) runCatching {
            val model = JSONObject().put("algorithm", topic.title).put("algorithmId", topic.id)
                .put("format", "educational-json-v1")
            fitted?.let { model.put("modelType", "regression").put("weights", JSONArray(it.weights)).put("bias", it.bias) }
            fittedKnn?.let { data ->
                model.put("modelType", "knn").put("k", 5)
                    .put("trainingData", JSONArray(data.map { p -> JSONObject().put("x", p.x).put("y", p.y).put("label", p.label) }))
            }
            fittedLogistic?.let {
                model.put("modelType", "logistic-regression")
                    .put("weights", JSONArray(listOf(it.weightX, it.weightY)))
                    .put("bias", it.bias)
                    .put("trainingLogLoss", it.loss)
            }
            context.contentResolver.openOutputStream(uri)?.bufferedWriter()?.use { it.write(model.toString(2)) }
            message = "Trained model exported as JSON."
        }.onFailure { message = it.message ?: "Unable to export model." }
    }
    val predicted = fitted?.let { fit ->
        val x = queryX.toDoubleOrNull() ?: 0.0
        fit.bias + fit.weights.mapIndexed { i, w -> w * Math.pow(x, i + 1.0) }.sum()
    }
    val predictedClass = fittedKnn?.let { data ->
        PhaseOneEngines.knn(data, LabPoint(queryX.toDoubleOrNull() ?: 0.0, queryY.toDoubleOrNull() ?: 0.0),
            5, DistanceMetric.Euclidean).first
    }
    val predictedProbability = fittedLogistic?.probability(queryX.toDoubleOrNull() ?: 0.0, queryY.toDoubleOrNull() ?: 0.0)

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Train & Inference", "Change data, fit a model, test a query, and export") }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    Text("Dataset • ${points.size} rows", color = LabText, fontWeight = FontWeight.Bold)
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        SegmentedOption("Load CSV", false, Modifier.weight(1f)) { importCsv.launch(arrayOf("text/csv", "text/plain", "application/octet-stream")) }
                        SegmentedOption("Random", false, Modifier.weight(1f)) {
                            points.clear(); points.addAll(PhaseOneDatasets.generate(preset, 24, seed = (1..10000).random()))
                            workspace.invalidate()
                        }
                        SegmentedOption("Clear", false, Modifier.weight(1f)) { points.clear(); workspace.invalidate() }
                    }
                    DatasetGraph(points.map { Point2D(it.x.toFloat(), it.y.toFloat(), it.label) },
                        line = fitted?.takeIf { kind == PhaseOneAlgorithmKind.SimpleLinearRegression }?.let { TrainingSnapshot(0, it.mse.toFloat(), it.weights.first().toFloat(), it.bias.toFloat(), 0f, 0f) })
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        OutlinedTextField(xText, { xText = it }, Modifier.weight(1f), label = { Text("X") }, singleLine = true)
                        OutlinedTextField(yText, { yText = it }, Modifier.weight(1f), label = { Text("Y") }, singleLine = true)
                    }
                    if (knn || logistic) SegmentedOption("New point: Class $pointClass", true, Modifier.fillMaxWidth()) { pointClass = 1 - pointClass }
                    SegmentedOption("Add point", false, Modifier.fillMaxWidth()) {
                        val x = xText.toDoubleOrNull(); val y = yText.toDoubleOrNull()
                        if (x != null && y != null) {
                            points.add(LabPoint(x, y, pointClass, y))
                            workspace.invalidate()
                        } else message = "Enter numeric X and Y values."
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    if (regression || knn || logistic) {
                        GradientButton("Train ${topic.title}", Modifier.fillMaxWidth()) {
                            if (points.size < 2) { message = "Add at least two points."; return@GradientButton }
                            runCatching {
                                fitted = when (kind) {
                                    PhaseOneAlgorithmKind.SimpleLinearRegression -> PhaseOneEngines.fitSimpleLinear(points)
                                    PhaseOneAlgorithmKind.PolynomialRegression -> PhaseOneEngines.fitPolynomial(points, SupervisedFitDefaults.POLYNOMIAL_DEGREE)
                                    PhaseOneAlgorithmKind.RidgeRegression -> PhaseOneEngines.fitRidge(points, SupervisedFitDefaults.REGULARIZATION_ALPHA)
                                    PhaseOneAlgorithmKind.LassoRegression -> PhaseOneEngines.fitLasso(points, SupervisedFitDefaults.REGULARIZATION_ALPHA)
                                    PhaseOneAlgorithmKind.ElasticNetRegression -> PhaseOneEngines.fitElasticNet(points, SupervisedFitDefaults.REGULARIZATION_ALPHA, SupervisedFitDefaults.ELASTIC_L1_RATIO)
                                    else -> null
                                }
                                fittedKnn = if (knn) points.toList() else null
                                fittedLogistic = if (logistic) fitLogistic(points) else null
                                message = "Training complete. Test a query below."
                            }.onFailure { message = it.message ?: "Training failed." }
                        }
                        fitted?.let { fit ->
                            Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                                MetricPill("MSE", "%.3f".format(fit.mse), LabCyan, Modifier.weight(1f))
                                MetricPill("R²", "%.3f".format(fit.r2), LabGreen, Modifier.weight(1f))
                            }
                            Text("Learned weights ${fit.weights.joinToString { "%.3f".format(it) }}; intercept %.3f".format(fit.bias), color = LabMuted, fontSize = 12.sp)
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                            OutlinedTextField(queryX, { queryX = it }, Modifier.weight(1f), label = { Text("Test X") }, singleLine = true)
                            if (knn || logistic) OutlinedTextField(queryY, { queryY = it }, Modifier.weight(1f), label = { Text("Test Y") }, singleLine = true)
                        }
                        predicted?.let { Text("Prediction: %.3f".format(it), color = LabGreen, fontWeight = FontWeight.Bold) }
                        predictedClass?.let { Text("Predicted class: $it", color = LabGreen, fontWeight = FontWeight.Bold) }
                        predictedProbability?.let { Text("Class 1 probability: %.1f%% • Class %d".format(it * 100, if (it >= .5) 1 else 0), color = LabGreen, fontWeight = FontWeight.Bold) }
                        if (fitted != null || fittedKnn != null || fittedLogistic != null) SegmentedOption("Export trained model (.json)", false, Modifier.fillMaxWidth()) {
                            exportModel.launch("${topic.id}-model.json")
                        }
                    }
                    Text(message, color = LabCyan, fontSize = 12.sp)
                }
            }
        }
    }
}
