package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.SupervisedVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.fitLogistic
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlinx.coroutines.delay

private val classificationTrees = setOf(
    SupervisedVisualization.ClassificationTree, SupervisedVisualization.ClassificationForest,
    SupervisedVisualization.ClassificationExtraTrees
)

@Composable
internal fun SupervisedClassificationPanel(kind: SupervisedVisualization, points: MutableList<LabPoint>) {
    if (kind == SupervisedVisualization.MultinomialNb || kind == SupervisedVisualization.BernoulliNb) {
        TextNaiveBayesVisualization(kind)
        return
    }
    if (points.size < 4 || points.map { it.label }.distinct().size < 2) {
        Text("Add at least four points from two classes.", color = LabOrange)
        return
    }
    var selectedClass by remember(kind) { mutableIntStateOf(0) }
    var addMode by remember(kind) { mutableStateOf(false) }
    var query by remember(kind) { mutableStateOf(LabPoint(.08, .05)) }
    var k by remember(kind) { mutableIntStateOf(5) }
    var metric by remember(kind) { mutableStateOf(DistanceMetric.Euclidean) }
    var threshold by remember(kind) { mutableDoubleStateOf(.5) }
    var criterion by remember(kind) { mutableStateOf(SplitCriterion.Gini) }
    var depth by remember(kind) { mutableIntStateOf(3) }
    var minSamples by remember(kind) { mutableIntStateOf(2) }
    var selectedNode by remember(kind) { mutableStateOf("") }
    var trees by remember(kind) { mutableIntStateOf(5) }
    var featureSubset by remember(kind) { mutableIntStateOf(1) }
    var bootstrap by remember(kind) { mutableStateOf(true) }
    var c by remember(kind) { mutableDoubleStateOf(1.0) }
    var svmKernel by remember(kind) { mutableStateOf(KernelType.Linear) }
    var gamma by remember(kind) { mutableDoubleStateOf(2.0) }
    var polynomialDegree by remember(kind) { mutableIntStateOf(3) }
    var step by remember(kind) { mutableIntStateOf(0) }
    var autoTrain by remember(kind) { mutableStateOf(false) }
    var rate by remember(kind) { mutableDoubleStateOf(.3) }
    var lengthScale by remember(kind) { mutableDoubleStateOf(.38) }
    var stage by remember(kind) { mutableIntStateOf(3) }
    var advancedControls by remember(kind) { mutableStateOf(false) }
    LaunchedEffect(kind, autoTrain) {
        while (autoTrain) {
            delay(650)
            step = (step + 1).coerceAtMost(24)
            if (step == 24) autoTrain = false
        }
    }

    val snapshot = points.toList()
    val logistic = remember(snapshot, kind) {
        if (kind == SupervisedVisualization.Logistic) runCatching { fitLogistic(snapshot) }.getOrNull() else null
    }
    val knn = remember(snapshot, kind, query, k, metric) {
        if (kind == SupervisedVisualization.ClassificationKnn)
            PhaseOneEngines.knn(snapshot, query, k, metric) else null
    }
    val tree = remember(snapshot, kind, depth, minSamples, criterion) {
        if (kind == SupervisedVisualization.ClassificationTree)
            SupervisedTreeEngine.fit(snapshot, false, depth, minSamples, criterion)
        else null
    }
    val forest = remember(snapshot, kind, trees, depth, featureSubset, bootstrap, minSamples) {
        if (kind == SupervisedVisualization.ClassificationForest ||
            kind == SupervisedVisualization.ClassificationExtraTrees)
            SupervisedTreeEngine.forest(snapshot, false, trees, depth,
                extraTrees = kind == SupervisedVisualization.ClassificationExtraTrees,
                featureSubset = featureSubset, minSamples = minSamples,
                bootstrap = bootstrap && kind != SupervisedVisualization.ClassificationExtraTrees)
        else emptyList()
    }
    val svm = remember(snapshot, kind, c, svmKernel, gamma, polynomialDegree) {
        if (kind == SupervisedVisualization.Svm)
            SupervisedStatisticalEngine.kernelSvm(snapshot, c, svmKernel, gamma, polynomialDegree)
        else null
    }
    val boostingVariant = when (kind) {
        SupervisedVisualization.ClassificationGradientBoosting -> BoostingVariant.Gradient
        SupervisedVisualization.ClassificationAdaBoost -> BoostingVariant.Ada
        SupervisedVisualization.ClassificationXgboost -> BoostingVariant.Xgboost
        SupervisedVisualization.ClassificationLightgbm -> BoostingVariant.Lightgbm
        SupervisedVisualization.ClassificationCatboost -> BoostingVariant.Catboost
        else -> null
    }
    val boosting = remember(snapshot, boostingVariant, stage, rate) {
        boostingVariant?.let { SupervisedBoostingEngine.fit(snapshot, false, it, stage, rate) }
    }
    val summaries = remember(snapshot) { PhaseTwoEngines.classSummaries(snapshot) }
    val gaussianNb = if (kind == SupervisedVisualization.GaussianNb)
        PhaseTwoEngines.gaussianNaiveBayes(snapshot, query) else null
    val lda = remember(snapshot, kind) {
        if (kind == SupervisedVisualization.Lda) LdaProjection.fit(snapshot) else null
    }
    val linearStep = remember(snapshot, kind, step, rate) {
        if (kind == SupervisedVisualization.Perceptron || kind == SupervisedVisualization.SgdClassifier)
            classificationStep(snapshot, kind, step, rate) else null
    }
    val gaussianProcess = remember(snapshot, kind, lengthScale) {
        if (kind == SupervisedVisualization.GaussianProcessClassifier)
            SupervisedStatisticalEngine.gaussianProcessClassifier(snapshot, lengthScale)
        else null
    }

    NativeSupervisedPlot(
        points = points, classification = true,
        selectedIndices = (svm?.supportIndices?.toSet().orEmpty() +
            knn?.second.orEmpty().mapNotNull { item -> snapshot.indexOf(item.first).takeIf { it >= 0 } }),
        query = if (kind == SupervisedVisualization.ClassificationKnn || kind == SupervisedVisualization.GaussianNb ||
            kind in classificationTrees || kind == SupervisedVisualization.GaussianProcessClassifier) query else null,
        onQuery = if (addMode) null else ({ query = it }),
        onPointAdded = { points.add(it.copy(label = selectedClass, target = selectedClass.toDouble())) },
        onPointChanged = { index, moved ->
            points[index] = points[index].copy(x = moved.x, y = moved.y)
        },
        onPointRemoved = { points.removeAt(it) },
        onBackgroundTap = if (tree != null && !addMode) ({ point ->
            selectedNode = tree.pathFor(point.x, point.y)
        }) else null
    ) { plot ->
        when {
            logistic != null -> {
                drawClassificationField(plot) { x, y -> logistic.probability(x, y) }
                val logOdds = ln(threshold / (1.0 - threshold))
                drawLinearBoundary(plot, logistic.weightX, logistic.weightY, logistic.bias - logOdds)
            }
            knn != null -> {
                val selected = knn.second
                selected.forEach { item ->
                    drawLine(LabGreen.copy(alpha = .55f), plot.point(query.x, query.y),
                        plot.point(item.first.x, item.first.y), 2.5f)
                }
                val radius = selected.maxOfOrNull { it.second } ?: 0.0
                drawCircle(LabPurple.copy(alpha = .8f),
                    ((plot.right - plot.left) / 2.0 * radius).toFloat(),
                    plot.point(query.x, query.y), style = androidx.compose.ui.graphics.drawscope.Stroke(2f))
            }
            tree != null -> drawTreePartitions(plot, tree, selectedNode)
            forest.isNotEmpty() -> drawClassificationField(plot) { x, y ->
                forest.count { it.predict(x, y) >= .5 } / forest.size.toDouble()
            }
            svm != null -> {
                drawClassificationField(plot) { x, y ->
                    1.0 / (1.0 + exp(-svm.score(x, y).coerceIn(-30.0, 30.0)))
                }
                drawImplicitBoundary(plot, { x, y -> svm.score(x, y) }, 0.0, Color.White)
                drawImplicitBoundary(plot, { x, y -> svm.score(x, y) }, -1.0, LabPurple)
                drawImplicitBoundary(plot, { x, y -> svm.score(x, y) }, 1.0, LabPurple)
            }
            kind == SupervisedVisualization.GaussianNb -> {
                drawClassificationField(plot) { x, y ->
                    PhaseTwoEngines.gaussianNaiveBayes(snapshot, LabPoint(x, y)).second
                        .firstOrNull { it.label == 1 }?.posterior ?: .5
                }
                summaries.forEach { summary ->
                    drawCovarianceEllipse(plot, summary, independent = true)
                }
            }
            kind == SupervisedVisualization.Lda && lda != null -> {
                drawClassificationField(plot) { x, y -> lda.probability(x, y) }
                drawLine(LabGreen, plot.point(-lda.directionX, -lda.directionY),
                    plot.point(lda.directionX, lda.directionY), 3f)
                val norm = sqrt(lda.directionX.pow(2) + lda.directionY.pow(2))
                val ux = lda.directionX / norm
                val uy = lda.directionY / norm
                snapshot.forEach { sample ->
                    val position = (sample.x - lda.midpointX) * ux + (sample.y - lda.midpointY) * uy
                    val projected = plot.point(lda.midpointX + position * ux,
                        lda.midpointY + position * uy)
                    drawLine(LabGreen.copy(alpha = .25f), plot.point(sample.x, sample.y), projected, 1.2f)
                    drawCircle(if (sample.label == 0) LabCyan else LabPink, 3.5f, projected)
                }
                summaries.forEach { drawCircle(LabPurple, 8f, plot.point(it.meanX, it.meanY)) }
            }
            kind == SupervisedVisualization.Qda -> {
                drawClassificationField(plot) { x, y -> qdaProbability(summaries, x, y) }
                summaries.forEach { summary ->
                    drawCovarianceEllipse(plot, summary, independent = false)
                }
            }
            linearStep != null -> drawLinearBoundary(plot, linearStep.first, linearStep.second, linearStep.third)
            boosting != null -> drawClassificationField(plot) { x, y -> boosting.predict(x, y) }
            gaussianProcess != null -> {
                drawClassificationField(plot) { x, y ->
                    gaussianProcess.predict(x, y).first
                }
            }
        }
    }

    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        SegmentedOption("Add Class A", selectedClass == 0, Modifier.weight(1f)) { selectedClass = 0 }
        SegmentedOption("Add Class B", selectedClass == 1, Modifier.weight(1f)) { selectedClass = 1 }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        SegmentedOption("Inspect", !addMode, Modifier.weight(1f)) { addMode = false }
        SegmentedOption("Add sample", addMode, Modifier.weight(1f)) { addMode = true }
    }
    if (logistic != null) {
        NativeSlider("Decision threshold", threshold, .1, .9) { threshold = it }
        SigmoidStrip(logistic.weightX, logistic.bias)
        Text("Learned weights: ${"%.2f".format(logistic.weightX)}, ${"%.2f".format(logistic.weightY)} · log loss ${"%.3f".format(logistic.loss)}",
            color = LabCyan, fontSize = 12.sp)
    }
    if (knn != null) {
        NativeSlider("K neighbors", k.toDouble(), 1.0, 15.0) { k = it.toInt() }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            DistanceMetric.entries.forEach { choice ->
                SegmentedOption(choice.label, choice == metric, Modifier.weight(1f)) { metric = choice }
            }
        }
        Text("Neighbor votes → Class ${knn.first}", color = LabGreen, fontWeight = FontWeight.Bold)
    }
    if (tree != null) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SplitCriterion.entries.forEach { choice ->
                SegmentedOption(choice.label, choice == criterion, Modifier.weight(1f)) { criterion = choice }
            }
        }
        NativeSlider("Maximum depth", depth.toDouble(), 1.0, 5.0) { depth = it.toInt() }
        SegmentedOption(if (advancedControls) "Hide advanced controls" else "Advanced controls",
            advancedControls, Modifier.fillMaxWidth()) { advancedControls = !advancedControls }
        if (advancedControls)
            NativeSlider("Minimum samples per leaf", minSamples.toDouble(), 2.0, 6.0) { minSamples = it.toInt() }
        NativeTreeDiagram(tree, false, selectedNode) { selectedNode = it }
        Text("Tap a tree node or an empty plot region to highlight its partition.",
            color = LabMuted, fontSize = 11.sp)
    }
    if (forest.isNotEmpty()) {
        NativeSlider("Trees", trees.toDouble(), 2.0, 10.0) { trees = it.toInt() }
        NativeSlider("Maximum depth", depth.toDouble(), 1.0, 5.0) { depth = it.toInt() }
        SegmentedOption(if (advancedControls) "Hide advanced controls" else "Advanced controls",
            advancedControls, Modifier.fillMaxWidth()) { advancedControls = !advancedControls }
        if (advancedControls) {
            NativeSlider("Minimum samples per leaf", minSamples.toDouble(), 1.0, 6.0) { minSamples = it.toInt() }
            NativeSlider("Features per split", featureSubset.toDouble(), 1.0, 2.0) { featureSubset = it.toInt() }
        }
        if (kind == SupervisedVisualization.ClassificationForest)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SegmentedOption("Bootstrap on", bootstrap, Modifier.weight(1f)) { bootstrap = true }
                SegmentedOption("Bootstrap off", !bootstrap, Modifier.weight(1f)) { bootstrap = false }
            }
        else Text("Extra Trees uses the full sample and random split thresholds.",
            color = LabMuted, fontSize = 11.sp)
        EnsembleCards(forest, false)
        val votes = forest.map { it.predict(query.x, query.y).toInt() }.groupingBy { it }.eachCount()
        Text("At query: ${votes.entries.joinToString { "Class ${it.key}: ${it.value}" }} · final Class ${SupervisedTreeEngine.forestPrediction(forest, query.x, query.y, false).toInt()}",
            color = LabGreen, fontSize = 12.sp)
    }
    if (svm != null) {
        NativeSlider("C margin penalty", c, .1, 4.0) { c = it }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            KernelType.entries.forEach { choice ->
                SegmentedOption(choice.label, choice == svmKernel, Modifier.weight(1f)) { svmKernel = choice }
            }
        }
        if (svmKernel != KernelType.Linear) {
            SegmentedOption(if (advancedControls) "Hide advanced controls" else "Advanced controls",
                advancedControls, Modifier.fillMaxWidth()) { advancedControls = !advancedControls }
            if (advancedControls) {
                NativeSlider("Kernel gamma", gamma, .2, 5.0) { gamma = it }
                if (svmKernel == KernelType.Polynomial)
                    NativeSlider("Polynomial degree", polynomialDegree.toDouble(), 2.0, 5.0) { polynomialDegree = it.toInt() }
            }
        }
        Text("${svm.supportIndices.size} support vectors · hinge loss ${"%.3f".format(svm.hingeLoss)}",
            color = LabCyan, fontSize = 12.sp)
    }
    if (gaussianNb != null) {
        GaussianPosteriorCards(gaussianNb.second)
    }
    if (lda != null) {
        Text("LDA direction: (${"%.2f".format(lda.directionX)}, ${"%.2f".format(lda.directionY)}) · projected separation ${"%.2f".format(lda.separation)}",
            color = LabCyan, fontSize = 12.sp)
    }
    if (kind == SupervisedVisualization.Qda) {
        Text("Each class has its own covariance ellipse and quadratic density score.",
            color = LabMuted, fontSize = 12.sp)
    }
    if (linearStep != null) {
        NativeSlider("Training step", step.toDouble(), 0.0, 24.0) { step = it.toInt() }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SegmentedOption("Previous", false, Modifier.weight(1f)) {
                autoTrain = false
                step = (step - 1).coerceAtLeast(0)
            }
            SegmentedOption("Next", false, Modifier.weight(1f)) {
                autoTrain = false
                step = (step + 1).coerceAtMost(24)
            }
            SegmentedOption(if (autoTrain) "Pause" else "Auto", autoTrain, Modifier.weight(1f)) {
                if (step == 24) step = 0
                autoTrain = !autoTrain
            }
        }
        NativeSlider("Learning rate", rate, .05, .8) { rate = it }
        Text("Boundary parameters come from ${if (kind == SupervisedVisualization.Perceptron) "mistake-only perceptron" else "single-sample hinge-loss SGD"} updates.",
            color = LabCyan, fontSize = 12.sp)
    }
    if (boosting != null) {
        NativeSlider("Learner stage", stage.toDouble(), 1.0, 8.0) { stage = it.toInt() }
        NativeSlider("Learning rate", rate, .1, .8) { rate = it }
        BoostingCards(boosting)
    }
    if (kind == SupervisedVisualization.GaussianProcessClassifier) {
        NativeSlider("Kernel length scale", lengthScale, .1, .8) { lengthScale = it }
        val estimate = gaussianProcess?.predict(query.x, query.y)
        Text("At query: Class B ${"%.0f".format((estimate?.first ?: .5) * 100)}% · latent uncertainty ${"%.2f".format(estimate?.second ?: 1.0)}",
            color = LabCyan, fontSize = 12.sp)
        Text("Kernel posterior uncertainty grows away from observations; probability follows an approximate logistic link.",
            color = LabMuted, fontSize = 12.sp)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawClassificationField(
    plot: PlotScale, probability: (Double, Double) -> Double
) {
    val cells = 22
    val width = (plot.right - plot.left) / cells
    val height = (plot.bottom - plot.top) / cells
    for (row in 0 until cells) for (column in 0 until cells) {
        val x = -1.0 + (column + .5) * 2.0 / cells
        val y = 1.0 - (row + .5) * 2.0 / cells
        val p = probability(x, y).coerceIn(0.0, 1.0)
        val color = if (p >= .5) LabPink else LabCyan
        drawRect(color.copy(alpha = (.06 + abs(p - .5) * .32).toFloat()),
            Offset(plot.left + column * width, plot.top + row * height), Size(width + .5f, height + .5f))
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLinearBoundary(
    plot: PlotScale, wx: Double, wy: Double, bias: Double, color: Color = Color.White
) {
    if (abs(wy) > 1e-6) {
        drawLine(color, plot.point(-1.0, (-bias + wx) / wy), plot.point(1.0, (-bias - wx) / wy), 3f)
    } else if (abs(wx) > 1e-6) {
        val x = -bias / wx
        drawLine(color, plot.point(x, -1.0), plot.point(x, 1.0), 3f)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawImplicitBoundary(
    plot: PlotScale, score: (Double, Double) -> Double, level: Double, color: Color
) {
    val cells = 38
    val step = 2.0 / cells
    val values = Array(cells + 1) { row ->
        DoubleArray(cells + 1) { column -> score(-1.0 + column * step, -1.0 + row * step) - level }
    }
    for (row in 0 until cells) for (column in 0 until cells) {
        val x = -1.0 + column * step
        val y = -1.0 + row * step
        val corners = arrayOf(
            Triple(x, y, values[row][column]),
            Triple(x + step, y, values[row][column + 1]),
            Triple(x + step, y + step, values[row + 1][column + 1]),
            Triple(x, y + step, values[row + 1][column])
        )
        val crossings = (0..3).mapNotNull { edge ->
            val a = corners[edge]
            val b = corners[(edge + 1) % 4]
            if ((a.third < 0) == (b.third < 0)) null else {
                val ratio = a.third / (a.third - b.third)
                plot.point(a.first + ratio * (b.first - a.first),
                    a.second + ratio * (b.second - a.second))
            }
        }
        if (crossings.size >= 2) drawLine(color, crossings[0], crossings[1], if (level == 0.0) 3f else 1.7f)
        if (crossings.size == 4) drawLine(color, crossings[2], crossings[3], if (level == 0.0) 3f else 1.7f)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawCovarianceEllipse(
    plot: PlotScale, summary: ClassSummary, independent: Boolean
) {
    val covariance = if (independent) 0.0 else summary.covariance
    val trace = summary.varianceX + summary.varianceY
    val delta = sqrt((summary.varianceX - summary.varianceY).pow(2) + 4 * covariance.pow(2))
    val major = sqrt(((trace + delta) / 2).coerceAtLeast(1e-6)) * 1.6
    val minor = sqrt(((trace - delta) / 2).coerceAtLeast(1e-6)) * 1.6
    val angle = .5 * atan2(2 * covariance, summary.varianceX - summary.varianceY)
    val path = Path()
    repeat(65) { index ->
        val t = index * 2 * kotlin.math.PI / 64
        val dx = major * cos(t) * cos(angle) - minor * sin(t) * sin(angle)
        val dy = major * cos(t) * sin(angle) + minor * sin(t) * cos(angle)
        val location = plot.point(summary.meanX + dx, summary.meanY + dy)
        if (index == 0) path.moveTo(location.x, location.y) else path.lineTo(location.x, location.y)
    }
    drawPath(path, if (summary.label == 0) LabCyan else LabPink,
        style = androidx.compose.ui.graphics.drawscope.Stroke(2f))
}

@Composable
private fun SigmoidStrip(weight: Double, bias: Double) {
    Canvas(Modifier.fillMaxWidth().height(78.dp)) {
        val plot = PlotScale(size.width, size.height)
        drawFunction(plot, { x -> 2.0 / (1.0 + exp(-(weight * x + bias))) - 1.0 }, LabGreen, 3f)
    }
    Text("Sigmoid P(Class B | X1) from the learned model", color = LabMuted, fontSize = 11.sp)
}

@Composable
private fun GaussianPosteriorCards(scores: List<PosteriorBreakdown>) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("P(class) × P(X1|class) × P(X2|class)", color = LabText, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            scores.forEach {
                Text("Class ${it.label}: prior ${"%.2f".format(it.prior)} × ${"%.2f".format(it.likelihoodX)} × ${"%.2f".format(it.likelihoodY)} → posterior ${"%.1f".format(it.posterior * 100)}%",
                    color = if (it.label == 0) LabCyan else LabPink, fontSize = 11.sp)
            }
        }
    }
}

private fun classificationStep(
    points: List<LabPoint>, kind: SupervisedVisualization, steps: Int, rate: Double
): Triple<Double, Double, Double> {
    var wx = 0.0
    var wy = 0.0
    var bias = 0.0
    repeat(steps) { index ->
        val update = if (kind == SupervisedVisualization.Perceptron)
            PhaseOneEngines.perceptronStep(points, index, wx, wy, bias, rate)
        else PhaseTwoEngines.sgdClassifierStep(points, index, rate, wx, wy, bias)
        wx = update.parameters.getValue("w1")
        wy = update.parameters.getValue("w2")
        bias = update.parameters.getValue("bias")
    }
    return Triple(wx, wy, bias)
}

internal data class LdaProjection(val directionX: Double, val directionY: Double,
    val midpointX: Double, val midpointY: Double, val separation: Double) {
    fun probability(x: Double, y: Double): Double =
        1.0 / (1.0 + exp(-((x - midpointX) * directionX + (y - midpointY) * directionY).coerceIn(-30.0, 30.0)))
    companion object {
        fun fit(points: List<LabPoint>): LdaProjection {
            val a = points.filter { it.label == 0 }
            val b = points.filter { it.label == 1 }
            require(a.isNotEmpty() && b.isNotEmpty())
            val ax = a.map { it.x }.average(); val ay = a.map { it.y }.average()
            val bx = b.map { it.x }.average(); val by = b.map { it.y }.average()
            val centered = points.map {
                val mx = if (it.label == 0) ax else bx
                val my = if (it.label == 0) ay else by
                (it.x - mx) to (it.y - my)
            }
            val xx = centered.sumOf { it.first * it.first } / points.size + .01
            val xy = centered.sumOf { it.first * it.second } / points.size
            val yy = centered.sumOf { it.second * it.second } / points.size + .01
            val det = (xx * yy - xy * xy).coerceAtLeast(1e-9)
            val dx = bx - ax; val dy = by - ay
            val wx = (yy * dx - xy * dy) / det
            val wy = (xx * dy - xy * dx) / det
            val norm = sqrt(wx * wx + wy * wy).coerceAtLeast(1e-9)
            return LdaProjection(wx / norm * 5, wy / norm * 5,
                (ax + bx) / 2, (ay + by) / 2, sqrt(dx * dx + dy * dy) / sqrt(xx + yy))
        }
    }
}

internal fun qdaProbability(summaries: List<ClassSummary>, x: Double, y: Double): Double {
    val logScores = summaries.map { summary ->
        val determinant = (summary.varianceX * summary.varianceY - summary.covariance.pow(2)).coerceAtLeast(1e-6)
        val dx = x - summary.meanX
        val dy = y - summary.meanY
        val quadratic = (summary.varianceY * dx * dx - 2 * summary.covariance * dx * dy +
            summary.varianceX * dy * dy) / determinant
        summary.label to (ln(summary.prior.coerceAtLeast(1e-9)) - .5 * ln(determinant) - .5 * quadratic)
    }
    val first = logScores.firstOrNull { it.first == 0 }?.second ?: 0.0
    val second = logScores.firstOrNull { it.first == 1 }?.second ?: 0.0
    return 1.0 / (1.0 + exp((first - second).coerceIn(-30.0, 30.0)))
}

@Composable
private fun TextNaiveBayesVisualization(kind: SupervisedVisualization) {
    val vocabulary = PhaseTwoEngines.textVocabulary
    val counts = remember(kind) { mutableStateMapOf<String, Int>().apply {
        vocabulary.forEach { put(it, if (it == "offer" || it == "free") 1 else 0) }
    } }
    val binary = kind == SupervisedVisualization.BernoulliNb
    val state = if (binary) PhaseTwoEngines.bernoulliNaiveBayes(counts.mapValues { it.value > 0 })
        else PhaseTwoEngines.multinomialNaiveBayes(counts)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(if (binary) "Binary feature matrix: present / absent" else "Document token counts",
                color = LabText, fontWeight = FontWeight.Bold)
            vocabulary.chunked(2).forEach { pair ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    pair.forEach { word ->
                        SegmentedOption("$word: ${if (binary) if ((counts[word] ?: 0) > 0) "1" else "0" else counts[word]}",
                            (counts[word] ?: 0) > 0, Modifier.weight(1f)) {
                            counts[word] = if (binary) 1 - (counts[word] ?: 0) else ((counts[word] ?: 0) + 1) % 5
                        }
                    }
                }
            }
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Token log-likelihood contributions", color = LabText, fontWeight = FontWeight.Bold)
            vocabulary.forEach { word ->
                val count = counts[word] ?: 0
                val class0 = if (binary) PhaseTwoEngines.bernoulliFeatureLogLikelihood(word, 0, count > 0)
                    else count * PhaseTwoEngines.multinomialTokenLogLikelihood(word, 0)
                val class1 = if (binary) PhaseTwoEngines.bernoulliFeatureLogLikelihood(word, 1, count > 0)
                    else count * PhaseTwoEngines.multinomialTokenLogLikelihood(word, 1)
                Text("$word: Class 0 %.2f  •  Class 1 %.2f".format(class0, class1),
                    color = LabMuted, fontSize = 12.sp)
            }
            Text("Log posterior (including class prior)", color = LabText, fontWeight = FontWeight.SemiBold)
            state.classScores.toSortedMap().forEach { (label, score) ->
                Text("Class $label: ${"%.3f".format(score)}", color = if (label == 0) LabCyan else LabPink)
            }
            Text("Prediction: Class ${state.prediction}", color = LabGreen, fontWeight = FontWeight.Bold)
        }
    }
}
