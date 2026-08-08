package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GradientButton
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBlue
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBorder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPanelSoft
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

private enum class LabSection(val label: String) { Learn("Learn"), Visualize("Visualize"), Train("Train"), Experiment("Experiment"), Metrics("Metrics") }

@Composable
fun PhaseOneAlgorithmLab(
    topic: LearnTopic,
    kind: PhaseOneAlgorithmKind,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(LabSection.Learn) }
    val classification = kind in setOf(
        PhaseOneAlgorithmKind.LogisticRegression,
        PhaseOneAlgorithmKind.Knn,
        PhaseOneAlgorithmKind.Perceptron,
        PhaseOneAlgorithmKind.DecisionTreeClassification
    )
    val defaultPreset = if (classification) DatasetPreset.TwoClusters else if (kind == PhaseOneAlgorithmKind.PolynomialRegression) DatasetPreset.Polynomial else DatasetPreset.LinearNoise
    var preset by remember(topic.id) { mutableStateOf(defaultPreset) }
    var samples by remember(topic.id) { mutableIntStateOf(36) }
    var noise by remember(topic.id) { mutableDoubleStateOf(0.12) }
    var seed by remember(topic.id) { mutableIntStateOf(7) }
    var liveUpdate by remember(topic.id) { mutableStateOf(true) }
    var selectedIndex by remember(topic.id) { mutableIntStateOf(-1) }
    val points = remember(topic.id, seed, preset, samples, noise) {
        mutableStateListOf<LabPoint>().apply { addAll(PhaseOneDatasets.generate(preset, samples, noise, seed)) }
    }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text(kind.family, color = Color(topic.accent), fontSize = 11.sp)
                }
                Column(horizontalAlignment = Alignment.End) {
                    Text(if (completed) "Completed" else depth.title, color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
                    Text(if (liveUpdate) "Live update" else "Manual", color = if (liveUpdate) LabCyan else LabOrange, fontSize = 11.sp)
                }
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                LabSection.entries.forEach { item -> SegmentedOption(item.label, section == item) { section = item } }
            }
        }
        when (section) {
            LabSection.Learn -> LearnSection(kind)
            LabSection.Visualize -> VisualizeSection(kind, points, selectedIndex, onSelected = { selectedIndex = it })
            LabSection.Train -> TrainSection(kind, points)
            LabSection.Experiment -> ExperimentSection(
                kind = kind,
                points = points,
                preset = preset,
                samples = samples,
                noise = noise,
                seed = seed,
                liveUpdate = liveUpdate,
                onPreset = {
                    preset = it
                    selectedIndex = -1
                },
                onSamples = { samples = it },
                onNoise = { noise = it },
                onSeed = {
                    seed += 1
                    selectedIndex = -1
                },
                onLive = { liveUpdate = !liveUpdate },
                onComplete = onComplete
            )
            LabSection.Metrics -> MetricsSection(kind, points)
        }
    }
}

@Composable
private fun LearnSection(kind: PhaseOneAlgorithmKind) {
    val explanation = phaseOneExplanation(kind)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { LearningBlock("What does this algorithm do?", explanation.what) }
        item { EquationBlock(explanation.equation, explanation.symbols) }
        item { LearningBlock("Input -> Algorithm Step -> Visual Change", explanation.flow) }
        item { LearningBlock("Why did this happen?", explanation.why) }
    }
}

@Composable
private fun VisualizeSection(
    kind: PhaseOneAlgorithmKind,
    points: MutableList<LabPoint>,
    selectedIndex: Int,
    onSelected: (Int) -> Unit
) {
    var weight by remember(kind) { mutableDoubleStateOf(0.62) }
    var bias by remember(kind) { mutableDoubleStateOf(-0.08) }
    var degree by remember(kind) { mutableIntStateOf(3) }
    var alpha by remember(kind) { mutableDoubleStateOf(0.25) }
    var l1Ratio by remember(kind) { mutableDoubleStateOf(0.5) }
    var threshold by remember(kind) { mutableDoubleStateOf(0.5) }
    var k by remember(kind) { mutableIntStateOf(5) }
    var metric by remember(kind) { mutableStateOf(DistanceMetric.Euclidean) }
    var criterion by remember(kind) { mutableStateOf(SplitCriterion.Gini) }
    var showResiduals by remember(kind) { mutableStateOf(true) }
    var showOptimal by remember(kind) { mutableStateOf(false) }
    var query by remember(kind) { mutableStateOf(LabPoint(0.16, 0.08, 0)) }

    val regressionFit = when (kind) {
        PhaseOneAlgorithmKind.SimpleLinearRegression -> PhaseOneEngines.fitSimpleLinear(points, weight, bias)
        PhaseOneAlgorithmKind.MultipleLinearRegression -> PhaseOneEngines.fitMultiple(points)
        PhaseOneAlgorithmKind.PolynomialRegression -> PhaseOneEngines.fitPolynomial(points, degree)
        PhaseOneAlgorithmKind.RidgeRegression -> PhaseOneEngines.fitRidge(points, alpha)
        PhaseOneAlgorithmKind.LassoRegression -> PhaseOneEngines.fitLasso(points, alpha)
        PhaseOneAlgorithmKind.ElasticNetRegression -> PhaseOneEngines.fitElasticNet(points, alpha, l1Ratio)
        PhaseOneAlgorithmKind.DecisionTreeRegression -> PhaseOneEngines.fitSimpleLinear(points)
        else -> null
    }
    val logisticMetrics = if (kind == PhaseOneAlgorithmKind.LogisticRegression) {
        PhaseOneEngines.logisticMetrics(points, 3.2, 2.4, -0.05, threshold)
    } else null
    val knnResult = if (kind == PhaseOneAlgorithmKind.Knn) PhaseOneEngines.knn(points, query, k, metric) else null
    val split = when (kind) {
        PhaseOneAlgorithmKind.DecisionTreeClassification -> PhaseOneEngines.bestClassificationSplit(points, criterion)
        PhaseOneAlgorithmKind.DecisionTreeRegression -> PhaseOneEngines.bestRegressionSplit(points)
        else -> null
    }

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Visualize", "Touch data, change parameters, watch the model recompute") }
        item {
            InteractiveDatasetCanvas(
                points = points,
                selectedIndex = selectedIndex,
                showResiduals = showResiduals,
                regressionFit = regressionFit,
                logisticThreshold = if (kind == PhaseOneAlgorithmKind.LogisticRegression) threshold else null,
                knnQuery = if (kind == PhaseOneAlgorithmKind.Knn) query else null,
                knnNeighbours = knnResult?.second?.map { it.first }.orEmpty(),
                treeSplit = split,
                onPointAdded = { points.add(it.copy(label = if (kind == PhaseOneAlgorithmKind.Knn) points.size % 2 else it.label)) },
                onPointChanged = { index, point -> points[index] = point },
                onPointSelected = onSelected,
                onQueryChanged = { query = it }
            )
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    when (kind) {
                        PhaseOneAlgorithmKind.SimpleLinearRegression -> {
                            SliderRow("Slope w", weight, -5.0, 5.0) { weight = it }
                            SliderRow("Intercept b", bias, -1.0, 1.0) { bias = it }
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                SegmentedOption(if (showResiduals) "Residuals on" else "Residuals off", showResiduals, Modifier.weight(1f)) { showResiduals = !showResiduals }
                                SegmentedOption(if (showOptimal) "Best shown" else "Check my line", showOptimal, Modifier.weight(1f)) { showOptimal = !showOptimal }
                            }
                        }
                        PhaseOneAlgorithmKind.PolynomialRegression -> SliderRow("Degree", degree.toDouble(), 1.0, 8.0) { degree = it.toInt().coerceIn(1, 8) }
                        PhaseOneAlgorithmKind.RidgeRegression, PhaseOneAlgorithmKind.LassoRegression -> SliderRow("Lambda alpha", alpha, 0.0, 2.0) { alpha = it }
                        PhaseOneAlgorithmKind.ElasticNetRegression -> {
                            SliderRow("Lambda alpha", alpha, 0.0, 2.0) { alpha = it }
                            SliderRow("L1 ratio", l1Ratio, 0.0, 1.0) { l1Ratio = it }
                        }
                        PhaseOneAlgorithmKind.LogisticRegression -> SliderRow("Decision threshold", threshold, 0.1, 0.9) { threshold = it }
                        PhaseOneAlgorithmKind.Knn -> {
                            SliderRow("K", k.toDouble(), 1.0, 15.0) { k = it.toInt().coerceIn(1, 15) }
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                DistanceMetric.entries.forEach { option ->
                                    SegmentedOption(option.label, metric == option, Modifier.weight(1f)) { metric = option }
                                }
                            }
                        }
                        PhaseOneAlgorithmKind.DecisionTreeClassification -> {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                SplitCriterion.entries.forEach { option ->
                                    SegmentedOption(option.label, criterion == option, Modifier.weight(1f)) { criterion = option }
                                }
                            }
                        }
                        else -> Text("This view is generated from the current dataset and exact model state.", color = LabMuted, fontSize = 12.sp)
                    }
                }
            }
        }
        item { AlgorithmSpecificCard(kind, regressionFit, logisticMetrics, knnResult, split, showOptimal, points, weight, bias) }
    }
}

@Composable
private fun TrainSection(kind: PhaseOneAlgorithmKind, points: List<LabPoint>) {
    var learningRate by remember(kind) { mutableDoubleStateOf(0.08) }
    var iterations by remember(kind) { mutableIntStateOf(20) }
    var batchSize by remember(kind) { mutableIntStateOf(8) }
    val mode = when (kind) {
        PhaseOneAlgorithmKind.StochasticGradientDescent -> TrainingMode.Stochastic
        PhaseOneAlgorithmKind.MiniBatchGradientDescent -> TrainingMode.MiniBatch
        else -> TrainingMode.Batch
    }
    val path = PhaseOneEngines.gradientPath(points, learningRate, iterations, mode, batchSize)
    val current = path.last()
    val diverging = path.size > 4 && path.last().loss > path[path.lastIndex - 3].loss
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Train", "Step through optimization with real gradients") }
        item { LossLandscapeChart(path) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Iteration", iterations.toString(), LabPurple, Modifier.weight(1f))
                MetricPill("Loss", "%.4f".format(current.loss), if (diverging) LabPink else LabCyan, Modifier.weight(1f))
                MetricPill("Mode", mode.name, LabGreen, Modifier.weight(1f))
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderRow("Learning rate", learningRate, 0.0001, 1.0) { learningRate = it }
                    SliderRow("Iterations", iterations.toDouble(), 1.0, 100.0) { iterations = it.toInt().coerceIn(1, 100) }
                    if (mode == TrainingMode.MiniBatch) SliderRow("Batch size", batchSize.toDouble(), 2.0, 24.0) { batchSize = it.toInt().coerceIn(2, 24) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption("Too small", false, Modifier.weight(1f)) { learningRate = 0.002 }
                        SegmentedOption("Good", true, Modifier.weight(1f)) { learningRate = 0.08 }
                        SegmentedOption("Too large", false, Modifier.weight(1f)) { learningRate = 0.75 }
                    }
                }
            }
        }
        item {
            CurrentAlgorithmStepCard(
                TrainingStep(
                    iterations,
                    "Calculate predictions, compute MSE, average gradients, then update w and b.",
                    current.loss,
                    mapOf("Weight" to current.weight, "Bias" to current.bias, "Gradient w" to current.gradW, "Gradient b" to current.gradB),
                    current.sampleIndices,
                    if (diverging) "Learning rate is too high. Loss is increasing." else "The parameter point moves against the gradient toward lower loss."
                )
            )
        }
        item { GradientComparison(points) }
    }
}

@Composable
private fun ExperimentSection(
    kind: PhaseOneAlgorithmKind,
    points: List<LabPoint>,
    preset: DatasetPreset,
    samples: Int,
    noise: Double,
    seed: Int,
    liveUpdate: Boolean,
    onPreset: (DatasetPreset) -> Unit,
    onSamples: (Int) -> Unit,
    onNoise: (Double) -> Unit,
    onSeed: () -> Unit,
    onLive: () -> Unit,
    onComplete: () -> Unit
) {
    val presets = if (kind in setOf(PhaseOneAlgorithmKind.Knn, PhaseOneAlgorithmKind.LogisticRegression, PhaseOneAlgorithmKind.Perceptron, PhaseOneAlgorithmKind.DecisionTreeClassification)) {
        listOf(DatasetPreset.TwoClusters, DatasetPreset.OverlappingClasses, DatasetPreset.XorLike, DatasetPreset.Circular, DatasetPreset.Imbalanced)
    } else {
        listOf(DatasetPreset.LinearNoise, DatasetPreset.PerfectLinear, DatasetPreset.NegativeCorrelation, DatasetPreset.Outliers, DatasetPreset.Polynomial)
    }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Experiment", "Dataset presets, live recompute, and reset behavior") }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Dataset preset", color = LabText, fontWeight = FontWeight.Bold)
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        presets.forEach { item -> SegmentedOption(item.label, preset == item) { onPreset(item) } }
                    }
                    SliderRow("Samples", samples.toDouble(), 10.0, 200.0) { onSamples(it.toInt().coerceIn(10, 200)) }
                    SliderRow("Noise", noise, 0.0, 0.65) { onNoise(it) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption("Randomize", false, Modifier.weight(1f), onSeed)
                        SegmentedOption(if (liveUpdate) "Live update on" else "Live update off", liveUpdate, Modifier.weight(1f), onLive)
                    }
                }
            }
        }
        item { TrainTestSplitControl(points) }
        item { LearningBlock("Step history", "Iteration history is bounded in the training view. Resetting the dataset, parameters, seed, or algorithm clears derived state on recomposition.") }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun MetricsSection(kind: PhaseOneAlgorithmKind, points: List<LabPoint>) {
    val regression = kind !in setOf(
        PhaseOneAlgorithmKind.LogisticRegression,
        PhaseOneAlgorithmKind.Knn,
        PhaseOneAlgorithmKind.Perceptron,
        PhaseOneAlgorithmKind.DecisionTreeClassification
    )
    val fit = if (regression) PhaseOneEngines.fitSimpleLinear(points) else null
    val metrics = if (!regression) PhaseOneEngines.logisticMetrics(points, 3.2, 2.4, -0.05, 0.5) else null
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Metrics", "Only metrics that make sense for this algorithm are shown") }
        if (fit != null) item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("MAE", "%.3f".format(fit.mae), LabCyan, Modifier.weight(1f))
                MetricPill("MSE", "%.3f".format(fit.mse), LabPurple, Modifier.weight(1f))
                MetricPill("RMSE", "%.3f".format(fit.rmse), LabGreen, Modifier.weight(1f))
                MetricPill("R2", "%.2f".format(fit.r2), LabOrange, Modifier.weight(1f))
            }
        }
        if (metrics != null) {
            item {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    MetricPill("Accuracy", "%.2f".format(metrics.accuracy), LabCyan, Modifier.weight(1f))
                    MetricPill("Precision", "%.2f".format(metrics.precision), LabPurple, Modifier.weight(1f))
                    MetricPill("Recall", "%.2f".format(metrics.recall), LabGreen, Modifier.weight(1f))
                    MetricPill("F1", "%.2f".format(metrics.f1), LabOrange, Modifier.weight(1f))
                }
            }
            item { InteractiveConfusionMatrix(metrics) }
        }
        item { LearningBlock("Why these numbers changed", "Metrics are recomputed from the current dataset and visible parameters. Train/test points remain distinguishable so generalization can be compared.") }
    }
}

@Composable
private fun InteractiveDatasetCanvas(
    points: MutableList<LabPoint>,
    selectedIndex: Int,
    showResiduals: Boolean,
    regressionFit: RegressionFit?,
    logisticThreshold: Double?,
    knnQuery: LabPoint?,
    knnNeighbours: List<LabPoint>,
    treeSplit: TreeSplit?,
    onPointAdded: (LabPoint) -> Unit,
    onPointChanged: (Int, LabPoint) -> Unit,
    onPointSelected: (Int) -> Unit,
    onQueryChanged: (LabPoint) -> Unit
) {
    var draggingIndex by remember { mutableIntStateOf(-1) }
    Canvas(
        Modifier
            .fillMaxWidth()
            .height(300.dp)
            .background(Color(0xFF081126), RoundedCornerShape(8.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(8.dp))
            .pointerInput(points.size, knnQuery) {
                fun toData(offset: Offset): LabPoint {
                    val x = (offset.x / size.width * 2.0 - 1.0).coerceIn(-1.0, 1.0)
                    val y = (1.0 - offset.y / size.height * 2.0).coerceIn(-1.0, 1.0)
                    return LabPoint(x, y)
                }
                detectTapGestures(
                    onTap = { tap ->
                        val data = toData(tap)
                        val nearest = points.indices.minByOrNull { index ->
                            val p = points[index]
                            (p.x - data.x).pow(2) + (p.y - data.y).pow(2)
                        }
                        if (nearest != null && sqrt((points[nearest].x - data.x).pow(2) + (points[nearest].y - data.y).pow(2)) < 0.12) {
                            onPointSelected(nearest)
                        } else if (knnQuery != null) {
                            onQueryChanged(data)
                        } else {
                            onPointAdded(data)
                        }
                    },
                    onLongPress = { tap ->
                        val data = toData(tap)
                        val nearest = points.indices.minByOrNull { index ->
                            val p = points[index]
                            (p.x - data.x).pow(2) + (p.y - data.y).pow(2)
                        }
                        if (nearest != null) points.removeAt(nearest)
                    }
                )
            }
            .pointerInput(points.size) {
                fun toOffset(point: LabPoint) = Offset(((point.x + 1.0) / 2.0 * size.width).toFloat(), ((1.0 - (point.y + 1.0) / 2.0) * size.height).toFloat())
                fun toData(offset: Offset): LabPoint {
                    val x = (offset.x / size.width * 2.0 - 1.0).coerceIn(-1.0, 1.0)
                    val y = (1.0 - offset.y / size.height * 2.0).coerceIn(-1.0, 1.0)
                    return LabPoint(x, y)
                }
                detectDragGestures(
                    onDragStart = { start ->
                        draggingIndex = points.indices.minByOrNull { (toOffset(points[it]) - start).getDistanceSquared() } ?: -1
                    },
                    onDragEnd = { draggingIndex = -1 },
                    onDragCancel = { draggingIndex = -1 },
                    onDrag = { change, _ ->
                        val index = draggingIndex
                        if (index in points.indices) {
                            val current = points[index]
                            val moved = toData(change.position)
                            onPointChanged(index, current.copy(x = moved.x, y = moved.y, target = moved.y))
                        }
                    }
                )
            }
            .padding(10.dp)
    ) {
        fun sx(x: Double) = (size.width * (x + 1.0) / 2.0).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.0) / 2.0)).toFloat()
        for (i in 0..4) {
            drawLine(Color.White.copy(alpha = .07f), Offset(size.width * i / 4f, 0f), Offset(size.width * i / 4f, size.height))
            drawLine(Color.White.copy(alpha = .07f), Offset(0f, size.height * i / 4f), Offset(size.width, size.height * i / 4f))
        }
        drawLine(Color.White.copy(alpha = .18f), Offset(0f, sy(0.0)), Offset(size.width, sy(0.0)))
        drawLine(Color.White.copy(alpha = .18f), Offset(sx(0.0), 0f), Offset(sx(0.0), size.height))

        logisticThreshold?.let { threshold ->
            val boundaryBias = -0.05 - kotlin.math.ln(threshold / (1.0 - threshold))
            drawProbabilityField(threshold)
            drawLine(Color.White, Offset(0f, sy((-3.2 * -1.0 - boundaryBias) / 2.4)), Offset(size.width, sy((-3.2 * 1.0 - boundaryBias) / 2.4)), 4f, cap = StrokeCap.Round)
        }
        treeSplit?.let {
            val value = if (it.feature == "x") sx(it.threshold) else sy(it.threshold)
            if (it.feature == "x") drawLine(LabOrange, Offset(value, 0f), Offset(value, size.height), 4f) else drawLine(LabOrange, Offset(0f, value), Offset(size.width, value), 4f)
        }
        regressionFit?.let { fit ->
            val path = Path()
            for (i in 0..80) {
                val x = -1.0 + 2.0 * i / 80.0
                val y = when (fit.weights.size) {
                    1 -> fit.weights[0] * x + fit.bias
                    else -> fit.bias + fit.weights.mapIndexed { index, w -> w * x.pow(index + 1) }.sum()
                }
                if (i == 0) path.moveTo(sx(x), sy(y)) else path.lineTo(sx(x), sy(y))
            }
            drawPath(path, LabGreen, style = Stroke(4f, cap = StrokeCap.Round))
            if (showResiduals) points.forEachIndexed { index, point ->
                val prediction = fit.predictions.getOrNull(index) ?: point.y
                drawLine(LabPink.copy(alpha = .35f), Offset(sx(point.x), sy(point.y)), Offset(sx(point.x), sy(prediction)), 2f)
            }
        }
        knnQuery?.let { query ->
            knnNeighbours.forEach { drawLine(LabGreen.copy(alpha = .45f), Offset(sx(query.x), sy(query.y)), Offset(sx(it.x), sy(it.y)), 3f) }
            drawCircle(Color.White, 13f, Offset(sx(query.x), sy(query.y)))
            drawCircle(LabGreen, 8f, Offset(sx(query.x), sy(query.y)))
        }
        points.forEachIndexed { index, point ->
            val color = if (point.label == 0) LabCyan else LabPink
            val selected = index == selectedIndex || knnNeighbours.any { it.x == point.x && it.y == point.y && it.label == point.label }
            val radius = if (selected) 9f else 6f
            drawCircle(if (point.train) color.copy(alpha = .24f) else Color.White.copy(alpha = .2f), radius + 5f, Offset(sx(point.x), sy(point.y)))
            drawCircle(if (point.train) color else LabOrange, radius, Offset(sx(point.x), sy(point.y)))
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawProbabilityField(threshold: Double) {
    val cells = 16
    val w = size.width / cells
    val h = size.height / cells
    repeat(cells * cells) { index ->
        val col = index % cells
        val row = index / cells
        val x = -1.0 + 2.0 * (col + .5) / cells
        val y = 1.0 - 2.0 * (row + .5) / cells
        val p = PhaseOneEngines.logisticProbability(LabPoint(x, y), 3.2, 2.4, -0.05)
        drawRect(if (p >= threshold) LabCyan.copy(alpha = .08f + p.toFloat() * .16f) else LabPink.copy(alpha = .08f + (1f - p.toFloat()) * .16f), Offset(col * w, row * h), Size(w, h))
    }
}

@Composable
private fun AlgorithmSpecificCard(
    kind: PhaseOneAlgorithmKind,
    fit: RegressionFit?,
    classification: ClassificationMetrics?,
    knn: Pair<Int, List<Pair<LabPoint, Double>>>?,
    split: TreeSplit?,
    showOptimal: Boolean,
    points: List<LabPoint>,
    weight: Double,
    bias: Double
) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            when {
                fit != null -> {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricPill("MSE", "%.4f".format(fit.mse), LabCyan, Modifier.weight(1f))
                        MetricPill("MAE", "%.4f".format(fit.mae), LabPurple, Modifier.weight(1f))
                        MetricPill("R2", "%.3f".format(fit.r2), LabGreen, Modifier.weight(1f))
                    }
                    if (kind == PhaseOneAlgorithmKind.SimpleLinearRegression && showOptimal) {
                        val best = PhaseOneEngines.fitSimpleLinear(points)
                        Text("Your line: w=%.2f, b=%.2f, MSE=%.3f".format(weight, bias, fit.mse), color = LabMuted, fontSize = 12.sp)
                        Text("Optimal least-squares line: w=%.2f, b=%.2f, MSE=%.3f".format(best.weights.first(), best.bias, best.mse), color = LabGreen, fontSize = 12.sp)
                    }
                    if (kind in setOf(PhaseOneAlgorithmKind.RidgeRegression, PhaseOneAlgorithmKind.LassoRegression, PhaseOneAlgorithmKind.ElasticNetRegression)) {
                        CoefficientChart(fit.weights)
                        Text("Penalty component = %.4f. Features effectively removed: %d".format(fit.penalty, fit.weights.count { abs(it) < 0.015 }), color = LabMuted, fontSize = 12.sp)
                    }
                }
                classification != null -> {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricPill("Accuracy", "%.2f".format(classification.accuracy), LabCyan, Modifier.weight(1f))
                        MetricPill("Precision", "%.2f".format(classification.precision), LabPurple, Modifier.weight(1f))
                        MetricPill("Recall", "%.2f".format(classification.recall), LabGreen, Modifier.weight(1f))
                    }
                    InteractiveConfusionMatrix(classification)
                }
                knn != null -> {
                    Text("Prediction: Class ${knn.first}", color = if (knn.first == 0) LabCyan else LabPink, fontWeight = FontWeight.Bold)
                    Text(knn.second.joinToString(limit = 5) { "class ${it.first.label}: d=%.2f".format(it.second) }, color = LabMuted, fontSize = 12.sp)
                }
                split != null -> {
                    Text("${split.feature} < %.2f ?".format(split.threshold), color = LabOrange, fontWeight = FontWeight.Bold)
                    Text("Impurity/error = %.4f, left=${split.leftCount}, right=${split.rightCount}".format(split.impurity), color = LabMuted, fontSize = 12.sp)
                    Text(split.explanation, color = LabMuted, fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun LossLandscapeChart(path: List<GradientPathPoint>) {
    Canvas(Modifier.fillMaxWidth().height(190.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        if (path.size < 2) return@Canvas
        val maxLoss = path.maxOf { it.loss }.coerceAtLeast(0.001)
        val minW = path.minOf { it.weight }.coerceAtMost(-1.0)
        val maxW = path.maxOf { it.weight }.coerceAtLeast(1.0)
        val curve = Path()
        path.forEachIndexed { index, point ->
            val x = size.width * ((point.weight - minW) / (maxW - minW)).toFloat()
            val y = size.height * (1f - (point.loss / maxLoss).toFloat())
            if (index == 0) curve.moveTo(x, y) else curve.lineTo(x, y)
        }
        drawPath(curve, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
        path.zipWithNext().forEach { (a, b) ->
            val ax = size.width * ((a.weight - minW) / (maxW - minW)).toFloat()
            val ay = size.height * (1f - (a.loss / maxLoss).toFloat())
            val bx = size.width * ((b.weight - minW) / (maxW - minW)).toFloat()
            val by = size.height * (1f - (b.loss / maxLoss).toFloat())
            drawLine(LabGreen.copy(alpha = .45f), Offset(ax, ay), Offset(bx, by), 2f)
        }
        val current = path.last()
        drawCircle(LabPurple, 10f, Offset(size.width * ((current.weight - minW) / (maxW - minW)).toFloat(), size.height * (1f - (current.loss / maxLoss).toFloat())))
    }
}

@Composable
private fun GradientComparison(points: List<LabPoint>) {
    val batch = PhaseOneEngines.gradientPath(points, .08, 20, TrainingMode.Batch).last()
    val sgd = PhaseOneEngines.gradientPath(points, .08, 20, TrainingMode.Stochastic).last()
    val mini = PhaseOneEngines.gradientPath(points, .08, 20, TrainingMode.MiniBatch, 8).last()
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Batch vs SGD vs Mini-Batch", color = LabText, fontWeight = FontWeight.Bold)
            Text("Batch uses the entire dataset and moves smoothly. SGD uses one sample and jitters. Mini-batch averages a subset for a balanced path.", color = LabMuted, fontSize = 12.sp)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Batch", "%.3f".format(batch.loss), LabCyan, Modifier.weight(1f))
                MetricPill("SGD", "%.3f".format(sgd.loss), LabPink, Modifier.weight(1f))
                MetricPill("Mini", "%.3f".format(mini.loss), LabGreen, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun CoefficientChart(weights: List<Double>) {
    Canvas(Modifier.fillMaxWidth().height(96.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).padding(8.dp)) {
        val maxAbs = weights.maxOfOrNull { abs(it) }?.coerceAtLeast(.01) ?: .01
        val barWidth = size.width / weights.size.coerceAtLeast(1)
        val mid = size.height / 2f
        drawLine(Color.White.copy(alpha = .2f), Offset(0f, mid), Offset(size.width, mid))
        weights.forEachIndexed { index, value ->
            val height = (abs(value) / maxAbs * size.height * .42).toFloat()
            val top = if (value >= 0) mid - height else mid
            drawRect(if (value >= 0) LabCyan else LabPink, Offset(index * barWidth + 5f, top), Size(barWidth - 10f, height))
        }
    }
}

@Composable
private fun InteractiveConfusionMatrix(metrics: ClassificationMetrics) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Confusion Matrix", color = LabText, fontWeight = FontWeight.Bold)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("TP", metrics.tp.toString(), LabGreen, Modifier.weight(1f))
                MetricPill("FN", metrics.fn.toString(), LabOrange, Modifier.weight(1f))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("FP", metrics.fp.toString(), LabPink, Modifier.weight(1f))
                MetricPill("TN", metrics.tn.toString(), LabCyan, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun CurrentAlgorithmStepCard(step: TrainingStep) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            SectionTitle("Current Algorithm Step", "Iteration ${step.iteration}")
            Text(step.description, color = LabMuted, fontSize = 12.sp)
            step.parameters.forEach { (label, value) -> Text("$label = %.4f".format(value), color = LabText, fontSize = 12.sp) }
            Text(step.explanation, color = if (step.explanation.contains("too high", true)) LabPink else LabGreen, fontSize = 12.sp)
        }
    }
}

@Composable
private fun TrainTestSplitControl(points: List<LabPoint>) {
    val train = points.count { it.train }
    val test = points.size - train
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Train / Test Split", color = LabText, fontWeight = FontWeight.Bold)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                MetricPill("Train", train.toString(), LabCyan, Modifier.weight(1f))
                MetricPill("Test", test.toString(), LabOrange, Modifier.weight(1f))
                MetricPill("Ratio", "%.0f%%".format(100.0 * train / points.size.coerceAtLeast(1)), LabGreen, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun SliderRow(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
        Text("$label: %.3f".format(value), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
        Slider(value = value.toFloat(), onValueChange = { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun EquationBlock(equation: String, symbols: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            SectionTitle("Core Equation", "Values and terms drive the visualization")
            Box(Modifier.fillMaxWidth().background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(12.dp)) {
                Text(equation, color = LabCyan, fontSize = 14.sp, fontWeight = FontWeight.Bold)
            }
            Text(symbols, color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun LearningBlock(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private data class Explanation(val what: String, val equation: String, val symbols: String, val flow: String, val why: String)

private fun phaseOneExplanation(kind: PhaseOneAlgorithmKind): Explanation = when (kind) {
    PhaseOneAlgorithmKind.SimpleLinearRegression -> Explanation(
        "Linear regression finds the straight line that minimizes prediction error between observed and predicted values.",
        "y_hat = wx + b",
        "x = input, y_hat = prediction, w = slope, b = intercept. Changing w rotates the line; changing b translates it.",
        "Tap or drag samples, adjust slope/intercept, inspect residuals, then compare your line with the least-squares optimum.",
        "The line moves because MSE changes. Gradient descent uses the sign and size of the gradients to reduce that error."
    )
    PhaseOneAlgorithmKind.MultipleLinearRegression -> Explanation(
        "Multiple regression predicts a continuous target from more than one feature and is visualized as a plane.",
        "y_hat = b + w1*x1 + w2*x2",
        "w1 and w2 tilt the plane along each feature axis; b shifts the plane up or down.",
        "The lab treats x and y canvas coordinates as two features and the target as height.",
        "A coefficient grows when that feature consistently explains residual direction."
    )
    PhaseOneAlgorithmKind.PolynomialRegression -> Explanation(
        "Polynomial regression fits curved relationships by adding powers of the input.",
        "y_hat = b + w1*x + w2*x^2 + ... + wd*x^d",
        "d controls flexibility. Higher degree can lower training error while increasing test error.",
        "Move degree from 1 to 8 and watch the curve move from underfit to flexible and then overfit.",
        "The curve bends because extra powers give the model more ways to chase local variation."
    )
    PhaseOneAlgorithmKind.RidgeRegression -> regularized("Ridge", "MSE + lambda * sum(w^2)", "squared coefficients", "shrinks coefficients smoothly toward zero")
    PhaseOneAlgorithmKind.LassoRegression -> regularized("Lasso", "MSE + lambda * sum(|w|)", "absolute coefficients", "can drive coefficients exactly to zero")
    PhaseOneAlgorithmKind.ElasticNetRegression -> regularized("Elastic Net", "MSE + lambda * (r*sum(|w|) + (1-r)*sum(w^2))", "mixed L1/L2 coefficients", "interpolates between ridge shrinkage and lasso feature selection")
    PhaseOneAlgorithmKind.LogisticRegression -> Explanation(
        "Logistic regression maps a linear score to a probability for binary classification.",
        "sigma(z) = 1 / (1 + e^-z), z = w1*x1 + w2*x2 + b",
        "The threshold turns probability into a class. Moving it changes TP, FP, TN, and FN.",
        "The probability field, boundary, threshold, and confusion matrix update together.",
        "Points change class when their probability crosses the selected decision threshold."
    )
    PhaseOneAlgorithmKind.Knn -> Explanation(
        "KNN classifies a query by voting among its nearest labelled samples.",
        "class(query) = majority(label of K nearest points)",
        "K controls neighborhood size; Euclidean and Manhattan distances define near differently.",
        "Tap a query point, adjust K, and inspect highlighted neighbors and vote counts.",
        "The prediction flips when a different class wins the local vote."
    )
    PhaseOneAlgorithmKind.Perceptron -> Explanation(
        "The perceptron learns a linear separator by updating only on misclassified samples.",
        "w = w + eta(y - y_hat)x",
        "eta is learning rate, y is the true label, and y_hat is the predicted sign.",
        "Step through samples and watch mistakes move the boundary.",
        "Correct samples cause no update; mistakes push the line toward a boundary that separates them."
    )
    PhaseOneAlgorithmKind.DecisionTreeClassification -> Explanation(
        "A classification tree splits the feature plane into regions with cleaner class labels.",
        "choose split with minimum weighted impurity",
        "Gini and entropy measure class mixing. Lower values are preferred.",
        "Candidate splits update both the dataset regions and the tree summary.",
        "The chosen split is the one that most reduces class impurity in child nodes."
    )
    PhaseOneAlgorithmKind.DecisionTreeRegression -> Explanation(
        "A regression tree splits data into regions and predicts each leaf's mean target.",
        "choose split with minimum within-leaf squared error",
        "Each split makes a piecewise constant prediction function.",
        "The boundary and leaf error are recomputed from the current dataset.",
        "The split is selected because grouping nearby target values reduces squared error."
    )
    PhaseOneAlgorithmKind.BatchGradientDescent, PhaseOneAlgorithmKind.StochasticGradientDescent, PhaseOneAlgorithmKind.MiniBatchGradientDescent -> Explanation(
        "${kind.displayName} moves parameters downhill on a loss landscape.",
        "theta(t+1) = theta(t) - alpha * gradient J(theta)",
        "alpha is learning rate. Batch size controls how many samples estimate each gradient.",
        "Compare smooth batch paths, noisy SGD paths, and balanced mini-batch paths.",
        "Too-small alpha crawls; good alpha converges; too-large alpha overshoots or diverges."
    )
}

private fun regularized(name: String, equation: String, symbols: String, behavior: String) = Explanation(
    "$name regression adds a penalty to ordinary regression so the model prefers simpler coefficients.",
    equation,
    "lambda controls penalty strength over $symbols.",
    "Increase lambda and watch the coefficient chart plus prediction curve change.",
    "The penalty raises the cost of large weights, so optimization $behavior."
)
