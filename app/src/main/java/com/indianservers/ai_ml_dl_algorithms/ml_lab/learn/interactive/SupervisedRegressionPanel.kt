package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.SupervisedVisualization
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

private val regularizedRegression = setOf(
    SupervisedVisualization.Ridge, SupervisedVisualization.Lasso, SupervisedVisualization.ElasticNet
)
private val treeRegression = setOf(
    SupervisedVisualization.RegressionTree, SupervisedVisualization.RegressionForest,
    SupervisedVisualization.RegressionExtraTrees
)

@Composable
internal fun SupervisedRegressionPanel(kind: SupervisedVisualization, points: MutableList<LabPoint>) {
    if (points.size < 4) {
        Text("Add at least four data points to fit this model.", color = LabOrange)
        return
    }
    var alpha by remember(kind) { mutableDoubleStateOf(SupervisedFitDefaults.REGULARIZATION_ALPHA) }
    var ratio by remember(kind) { mutableDoubleStateOf(SupervisedFitDefaults.ELASTIC_L1_RATIO) }
    var degree by remember(kind) { mutableIntStateOf(SupervisedFitDefaults.POLYNOMIAL_DEGREE) }
    var quantile by remember(kind) { mutableDoubleStateOf(.5) }
    var epsilon by remember(kind) { mutableDoubleStateOf(.14) }
    var c by remember(kind) { mutableDoubleStateOf(1.0) }
    var gamma by remember(kind) { mutableDoubleStateOf(2.0) }
    var kernel by remember(kind) { mutableStateOf(KernelType.Rbf) }
    var lengthScale by remember(kind) { mutableDoubleStateOf(.3) }
    var noise by remember(kind) { mutableDoubleStateOf(.12) }
    var trees by remember(kind) { mutableIntStateOf(5) }
    var depth by remember(kind) { mutableIntStateOf(3) }
    var minLeaf by remember(kind) { mutableIntStateOf(2) }
    var bootstrap by remember(kind) { mutableStateOf(true) }
    var stage by remember(kind) { mutableIntStateOf(3) }
    var rate by remember(kind) { mutableDoubleStateOf(.3) }
    var queryX by remember(kind) { mutableDoubleStateOf(.15) }
    var addMode by remember(kind) { mutableStateOf(false) }
    var k by remember(kind) { mutableIntStateOf(5) }
    var showResiduals by remember(kind) { mutableStateOf(true) }
    var selectedNode by remember(kind) { mutableStateOf("") }
    var rotation by remember(kind) { mutableDoubleStateOf(.55) }
    var heldFeature by remember(kind) { mutableDoubleStateOf(0.0) }
    var advancedControls by remember(kind) { mutableStateOf(false) }

    val snapshot = points.toList()
    val fit = remember(snapshot, kind, alpha, ratio, degree) {
        runCatching {
            when (kind) {
                SupervisedVisualization.SimpleLinear -> PhaseOneEngines.fitSimpleLinear(snapshot)
                SupervisedVisualization.MultipleLinear -> PhaseOneEngines.fitMultiple(snapshot)
                SupervisedVisualization.Polynomial -> PhaseOneEngines.fitPolynomial(snapshot, degree)
                SupervisedVisualization.Ridge -> PhaseOneEngines.fitRidge(snapshot, alpha)
                SupervisedVisualization.Lasso -> PhaseOneEngines.fitLasso(snapshot, alpha)
                SupervisedVisualization.ElasticNet -> PhaseOneEngines.fitElasticNet(snapshot, alpha, ratio)
                else -> null
            }
        }.getOrNull()
    }
    val robust = remember(snapshot, kind) {
        if (kind == SupervisedVisualization.Robust) SupervisedStatisticalEngine.robustHuber(snapshot) else null
    }
    val quantiles = remember(snapshot, kind, quantile) {
        if (kind == SupervisedVisualization.Quantile) listOf(.1, quantile, .9).map {
            SupervisedStatisticalEngine.quantile(snapshot, it)
        } else emptyList()
    }
    val posterior = remember(snapshot, kind, lengthScale, noise) {
        val grid = (0..60).map { -1.0 + it / 30.0 }
        when (kind) {
            SupervisedVisualization.BayesianLinear ->
                SupervisedStatisticalEngine.bayesianLinear(snapshot, noise, grid)
            SupervisedVisualization.GaussianProcessRegression ->
                SupervisedStatisticalEngine.gaussianProcess(snapshot, lengthScale, noise, grid)
            else -> null
        }
    }
    val svr = remember(snapshot, kind, c, epsilon, gamma, kernel) {
        if (kind == SupervisedVisualization.Svr)
            SupervisedStatisticalEngine.epsilonSvr(snapshot, c, epsilon, kernel, gamma)
        else null
    }
    val tree = remember(snapshot, kind, depth, minLeaf) {
        if (kind == SupervisedVisualization.RegressionTree)
            SupervisedTreeEngine.fit(snapshot, regression = true, maxDepth = depth, minSamples = minLeaf)
        else null
    }
    val forest = remember(snapshot, kind, trees, depth, minLeaf, bootstrap) {
        if (kind == SupervisedVisualization.RegressionForest ||
            kind == SupervisedVisualization.RegressionExtraTrees)
            SupervisedTreeEngine.forest(snapshot, regression = true, trees = trees,
                maxDepth = depth, extraTrees = kind == SupervisedVisualization.RegressionExtraTrees,
                minSamples = minLeaf, bootstrap = bootstrap && kind != SupervisedVisualization.RegressionExtraTrees)
        else emptyList()
    }
    val boostVariant = when (kind) {
        SupervisedVisualization.RegressionGradientBoosting -> BoostingVariant.Gradient
        SupervisedVisualization.RegressionAdaBoost -> BoostingVariant.Ada
        SupervisedVisualization.RegressionXgboost -> BoostingVariant.Xgboost
        SupervisedVisualization.RegressionLightgbm -> BoostingVariant.Lightgbm
        SupervisedVisualization.RegressionCatboost -> BoostingVariant.Catboost
        else -> null
    }
    val boost = remember(snapshot, boostVariant, stage, rate) {
        boostVariant?.let { SupervisedBoostingEngine.fit(snapshot, true, it, stage, rate) }
    }
    val neighbours = remember(snapshot, kind, queryX, k) {
        if (kind == SupervisedVisualization.RegressionKnn)
            snapshot.withIndex().sortedBy { abs(it.value.x - queryX) }.take(k.coerceIn(1, snapshot.size))
        else emptyList()
    }
    val knnPrediction = neighbours.map { it.value.target }.takeIf { it.isNotEmpty() }?.average()

    if (kind == SupervisedVisualization.MultipleLinear && fit != null) {
        MultipleRegressionPlane(snapshot, fit, rotation)
        NativeSupervisedPlot(points, classification = true, xLabel = "Feature X1", yLabel = "Feature X2",
            onPointChanged = { i, moved -> points[i] = points[i].copy(x = moved.x, y = moved.y) },
            onPointAdded = { p -> points.add(p.copy(target = fit.bias + fit.weights[0] * p.x + fit.weights[1] * p.y)) },
            onPointRemoved = { points.removeAt(it) }) { plot ->
            for (row in 0 until 18) for (column in 0 until 18) {
                val x = -1.0 + column * 2.0 / 18
                val y = -1.0 + row * 2.0 / 18
                val prediction = fit.bias + fit.weights[0] * x + fit.weights[1] * y
                drawRect(if (prediction >= 0) LabCyan.copy(alpha = (.04 + prediction.coerceIn(0.0, 1.0) * .16).toFloat())
                    else LabPurple.copy(alpha = (.04 + abs(prediction).coerceIn(0.0, 1.0) * .16).toFloat()),
                    plot.point(x, y + 2.0 / 18), Size((plot.right - plot.left) / 18, (plot.bottom - plot.top) / 18))
            }
        }
    } else {
        NativeSupervisedPlot(
            points = points, classification = false,
            selectedIndices = svr?.supportIndices?.toSet().orEmpty() + neighbours.map { it.index },
            onPointAdded = { points.add(it.copy(target = it.y)) },
            onPointChanged = { i, moved -> points[i] = moved.copy(label = points[i].label, train = points[i].train) },
            onPointRemoved = { points.removeAt(it) },
            onBackgroundTap = if (kind == SupervisedVisualization.RegressionKnn && !addMode)
                ({ point -> queryX = point.x }) else null
        ) { plot ->
            when {
                fit != null -> {
                    drawFunction(plot, { x -> fit.bias + fit.weights.mapIndexed { i, w -> w * x.pow(i + 1) }.sum() }, LabGreen)
                    if (showResiduals) snapshot.forEach {
                        val prediction = fit.bias + fit.weights.mapIndexed { i, w -> w * it.x.pow(i + 1) }.sum()
                        drawLine(LabPink.copy(alpha = .6f), plot.point(it.x, it.target), plot.point(it.x, prediction), 2f)
                    }
                }
                robust != null -> {
                    val ols = PhaseOneEngines.fitSimpleLinear(snapshot)
                    drawFunction(plot, { x -> ols.bias + ols.weights.first() * x }, LabMuted, 2f)
                    drawFunction(plot, robust::predict, LabGreen, 4f)
                }
                quantiles.isNotEmpty() -> {
                    quantiles.forEachIndexed { index, model ->
                        drawFunction(plot, model::predict, if (index == 1) LabGreen else LabPurple, if (index == 1) 4f else 2f)
                    }
                }
                posterior != null -> {
                    drawUncertainty(plot, posterior.predictions)
                    posterior.sampledFunctions.forEach { values ->
                        val path = Path()
                        posterior.predictions.forEachIndexed { i, item ->
                            val p = plot.point(item.x, values[i])
                            if (i == 0) path.moveTo(p.x, p.y) else path.lineTo(p.x, p.y)
                        }
                        drawPath(path, LabPurple.copy(alpha = .36f), style = Stroke(1.4f))
                    }
                }
                svr != null -> {
                    drawFunction(plot, { x -> svr.predict(x) + epsilon }, LabPurple, 2f)
                    drawFunction(plot, { x -> svr.predict(x) - epsilon }, LabPurple, 2f)
                    drawFunction(plot, svr::predict, LabGreen, 4f)
                }
                tree != null -> {
                    drawStepFunction(plot, { x -> tree.predict(x, 0.0) }, LabGreen, 4f)
                    tree.nodes().filter { !it.isLeaf }.forEach { node ->
                        val x = node.threshold ?: 0.0
                        drawLine(LabPurple.copy(alpha = .75f), plot.point(x, -1.0), plot.point(x, 1.0), 2f)
                    }
                }
                forest.isNotEmpty() -> {
                    forest.forEach { member -> drawStepFunction(plot, { x -> member.predict(x, 0.0) }, LabPurple.copy(alpha = .32f), 1.5f) }
                    drawStepFunction(plot, { x -> SupervisedTreeEngine.forestPrediction(forest, x, 0.0, true) }, LabGreen, 4f)
                }
                boost != null -> {
                    drawFunction(plot, { boost.initial }, LabMuted, 2f)
                    drawStepFunction(plot, { x -> boost.predict(x, 0.0) }, LabGreen, 4f)
                    snapshot.forEachIndexed { i, sample ->
                        val previous = boost.stages.lastOrNull()?.predictions?.getOrNull(i) ?: boost.initial
                        drawLine(LabPink.copy(alpha = .4f), plot.point(sample.x, sample.target), plot.point(sample.x, previous), 2f)
                    }
                }
                knnPrediction != null -> {
                    neighbours.forEach { neighbor ->
                        drawLine(LabPurple.copy(alpha = .7f), plot.point(queryX, knnPrediction),
                            plot.point(neighbor.value.x, neighbor.value.target), 2f)
                    }
                    drawCircle(LabGreen, 9f, plot.point(queryX, knnPrediction))
                    drawLine(LabGreen.copy(alpha = .5f), plot.point(queryX, -1.0), plot.point(queryX, 1.0), 2f)
                }
            }
        }
    }

    if (fit != null) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            MetricPill("MSE", "%.3f".format(fit.mse), LabCyan, Modifier.weight(1f))
            MetricPill("R²", "%.3f".format(fit.r2), LabGreen, Modifier.weight(1f))
        }
        Text("Learned: intercept ${"%.3f".format(fit.bias)} · coefficients ${fit.weights.joinToString { "%.3f".format(it) }}",
            color = LabMuted, fontSize = 12.sp)
    }
    if (kind == SupervisedVisualization.SimpleLinear) {
        SegmentedOption(if (showResiduals) "Hide residuals" else "Show residuals", showResiduals, Modifier.fillMaxWidth()) {
            showResiduals = !showResiduals
        }
    }
    if (kind == SupervisedVisualization.MultipleLinear && fit != null) {
        NativeSlider("Rotate plane", rotation, -.8, .8) { rotation = it }
        NativeSlider("Hold X2 for feature slice", heldFeature, -1.0, 1.0) { heldFeature = it }
        Text("At X2 = ${"%.2f".format(heldFeature)}, predicted Y = ${"%.2f".format(fit.bias + fit.weights[1] * heldFeature)} + ${"%.2f".format(fit.weights[0])}·X1",
            color = LabCyan, fontSize = 12.sp)
        Canvas(Modifier.fillMaxWidth().height(145.dp)
            .background(Color(0xFF081527), RoundedCornerShape(12.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(12.dp))) {
            val plot = PlotScale(size.width, size.height)
            drawLine(LabMuted, Offset(plot.left, plot.bottom), Offset(plot.right, plot.bottom), 2f)
            drawLine(LabMuted, Offset(plot.left, plot.top), Offset(plot.left, plot.bottom), 2f)
            drawFunction(plot, { x -> fit.bias + fit.weights[0] * x + fit.weights[1] * heldFeature },
                LabGreen, 3.5f)
            snapshot.filter { abs(it.y - heldFeature) <= .22 }.forEach {
                drawCircle(LabCyan, 5f, plot.point(it.x, it.target))
            }
        }
        Text("Feature slice: X1 changes while X2 stays fixed; nearby observed samples are cyan.",
            color = LabMuted, fontSize = 11.sp)
    }
    if (kind == SupervisedVisualization.Polynomial) NativeSlider("Degree", degree.toDouble(), 1.0, 6.0) {
        degree = it.toInt().coerceIn(1, 6)
    }
    if (kind in regularizedRegression) {
        NativeSlider("Regularization λ", alpha, 0.0, 1.5) { alpha = it }
        if (kind == SupervisedVisualization.ElasticNet) NativeSlider("L1 ratio", ratio, 0.0, 1.0) { ratio = it }
        fit?.let { CoefficientBars(it.weights) }
        val paths = remember(snapshot, kind, ratio) {
            (0..6).map { index ->
                val strength = index / 4.0
                when (kind) {
                    SupervisedVisualization.Ridge -> PhaseOneEngines.fitRidge(snapshot, strength).weights
                    SupervisedVisualization.Lasso -> PhaseOneEngines.fitLasso(snapshot, strength).weights
                    else -> PhaseOneEngines.fitElasticNet(snapshot, strength, ratio).weights
                }
            }
        }
        CoefficientPathChart(paths, alpha)
        Text("Active coefficients: ${fit?.weights?.count { abs(it) > 1e-5 } ?: 0} · penalty ${"%.3f".format(fit?.penalty ?: 0.0)}",
            color = LabCyan, fontSize = 12.sp)
    }
    if (kind == SupervisedVisualization.Quantile) NativeSlider("Quantile q", quantile, .1, .9) { quantile = it }
    if (kind == SupervisedVisualization.BayesianLinear || kind == SupervisedVisualization.GaussianProcessRegression) {
        NativeSlider("Observation noise", noise, .03, .35) { noise = it }
        if (kind == SupervisedVisualization.GaussianProcessRegression)
            NativeSlider("Kernel length scale", lengthScale, .1, .8) { lengthScale = it }
        Text("Purple band: posterior uncertainty · green: posterior mean · faint curves: posterior samples",
            color = LabMuted, fontSize = 11.sp)
    }
    if (kind == SupervisedVisualization.Svr) {
        NativeSlider("Epsilon tube", epsilon, .03, .4) { epsilon = it }
        NativeSlider("C", c, .1, 3.0) { c = it }
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            KernelType.entries.forEach { choice ->
                SegmentedOption(choice.label, kernel == choice, Modifier.weight(1f)) { kernel = choice }
            }
        }
        SegmentedOption(if (advancedControls) "Hide advanced controls" else "Advanced controls",
            advancedControls, Modifier.fillMaxWidth()) { advancedControls = !advancedControls }
        if (advancedControls && kernel != KernelType.Linear)
            NativeSlider("Gamma", gamma, .2, 5.0) { gamma = it }
        Text("${svr?.supportIndices?.size ?: 0} support vectors on/outside the epsilon tube", color = LabCyan, fontSize = 12.sp)
    }
    if (kind in treeRegression) {
        NativeSlider("Maximum tree depth", depth.toDouble(), 1.0, 5.0) { depth = it.toInt() }
        if (kind != SupervisedVisualization.RegressionTree)
            NativeSlider("Trees", trees.toDouble(), 2.0, 10.0) { trees = it.toInt() }
        SegmentedOption(if (advancedControls) "Hide advanced controls" else "Advanced controls",
            advancedControls, Modifier.fillMaxWidth()) { advancedControls = !advancedControls }
        if (advancedControls)
            NativeSlider("Minimum samples per leaf", minLeaf.toDouble(), 1.0, 6.0) { minLeaf = it.toInt() }
        if (kind == SupervisedVisualization.RegressionForest)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SegmentedOption("Bootstrap on", bootstrap, Modifier.weight(1f)) { bootstrap = true }
                SegmentedOption("Bootstrap off", !bootstrap, Modifier.weight(1f)) { bootstrap = false }
            }
        if (tree != null) NativeTreeDiagram(tree, true, selectedNode) { selectedNode = it }
        if (forest.isNotEmpty()) EnsembleCards(forest, true)
    }
    if (boost != null) {
        NativeSlider("Learner stage", stage.toDouble(), 1.0, 8.0) { stage = it.toInt() }
        NativeSlider("Learning rate", rate, .1, .8) { rate = it }
        BoostingCards(boost)
    }
    if (kind == SupervisedVisualization.RegressionKnn) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            SegmentedOption("Move query", !addMode, Modifier.weight(1f)) { addMode = false }
            SegmentedOption("Add sample", addMode, Modifier.weight(1f)) { addMode = true }
        }
        NativeSlider("K nearest points", k.toDouble(), 1.0, 15.0) { k = it.toInt() }
        Text("Query X ${"%.2f".format(queryX)} · neighbor mean ${"%.3f".format(knnPrediction ?: 0.0)}",
            color = LabGreen, fontWeight = FontWeight.Bold, fontSize = 13.sp)
    }
    if (robust != null) Text("Green Huber fit resists outliers; gray ordinary least squares moves farther.",
        color = LabMuted, fontSize = 12.sp)
}

@Composable
private fun CoefficientPathChart(paths: List<List<Double>>, current: Double) {
    if (paths.isEmpty() || paths.first().isEmpty()) return
    val scale = paths.flatten().maxOf { abs(it) }.coerceAtLeast(.01)
    val colors = listOf(LabCyan, LabPink, LabGreen, LabPurple, LabOrange)
    Canvas(Modifier.fillMaxWidth().height(112.dp)
        .background(Color(0xFF081527), RoundedCornerShape(12.dp))
        .border(1.dp, LabBorder, RoundedCornerShape(12.dp))) {
        val left = 22f
        val right = size.width - 12f
        val centerY = size.height / 2f
        val amplitude = size.height * .42f
        drawLine(LabMuted.copy(alpha = .5f), Offset(left, centerY), Offset(right, centerY), 1f)
        paths.first().indices.forEach { coefficient ->
            val curve = Path()
            paths.forEachIndexed { index, weights ->
                val x = left + (right - left) * index / paths.lastIndex.coerceAtLeast(1)
                val y = centerY - (weights.getOrElse(coefficient) { 0.0 } / scale * amplitude).toFloat()
                if (index == 0) curve.moveTo(x, y) else curve.lineTo(x, y)
            }
            drawPath(curve, colors[coefficient % colors.size], style = Stroke(2.5f))
        }
        val marker = left + (right - left) * (current / 1.5).toFloat()
        drawLine(Color.White.copy(alpha = .8f), Offset(marker, 0f), Offset(marker, size.height), 2f)
    }
    Text("Coefficient paths as λ increases; white line marks the selected strength.",
        color = LabMuted, fontSize = 11.sp)
}

@Composable
internal fun NativeSlider(label: String, value: Double, min: Double, max: Double, onValue: (Double) -> Unit) {
    Column(Modifier.fillMaxWidth()) {
        val shown = if (min % 1.0 == 0.0 && max % 1.0 == 0.0 && value % 1.0 == 0.0)
            value.toInt().toString() else "%.2f".format(value)
        Text("$label · $shown", color = LabText, fontSize = 12.sp)
        Slider(value.toFloat(), { onValue(it.toDouble()) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun CoefficientBars(weights: List<Double>) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text("Learned coefficient magnitudes", color = LabText, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            weights.forEachIndexed { i, weight ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("β${i + 1}", color = LabMuted, fontSize = 11.sp, modifier = Modifier.width(28.dp))
                    Canvas(Modifier.weight(1f).height(15.dp)) {
                        drawRect(LabPanelSoft)
                        drawRect(if (abs(weight) < 1e-5) LabPink else LabCyan,
                            size = Size((size.width * (abs(weight) / 2.0).coerceIn(0.0, 1.0)).toFloat(), size.height))
                    }
                    Text("%.3f".format(weight), color = LabText, fontSize = 11.sp, modifier = Modifier.width(46.dp))
                }
            }
        }
    }
}

@Composable
private fun MultipleRegressionPlane(points: List<LabPoint>, fit: RegressionFit, rotation: Double) {
    Canvas(Modifier.fillMaxWidth().height(240.dp).background(Color(0xFF081527), RoundedCornerShape(14.dp))
        .border(1.dp, LabBorder, RoundedCornerShape(14.dp))) {
        val angle = rotation
        fun project(x1: Double, x2: Double, target: Double): Offset {
            val horizontal = x1 * cos(angle) - x2 * sin(angle)
            val depth = x1 * sin(angle) + x2 * cos(angle)
            return Offset(size.width * (.5f + horizontal.toFloat() * .31f),
                size.height * (.53f - depth.toFloat() * .13f - target.toFloat() * .31f))
        }
        for (step in -4..4) {
            val coordinate = step / 4.0
            fun prediction(x: Double, y: Double) = fit.bias + fit.weights[0] * x + fit.weights[1] * y
            drawLine(LabPurple.copy(alpha = .48f),
                project(-1.0, coordinate, prediction(-1.0, coordinate)),
                project(1.0, coordinate, prediction(1.0, coordinate)), 1.5f)
            drawLine(LabPurple.copy(alpha = .48f),
                project(coordinate, -1.0, prediction(coordinate, -1.0)),
                project(coordinate, 1.0, prediction(coordinate, 1.0)), 1.5f)
        }
        points.forEach {
            val estimated = fit.bias + fit.weights[0] * it.x + fit.weights[1] * it.y
            drawLine(LabPink.copy(alpha = .65f), project(it.x, it.y, it.target),
                project(it.x, it.y, estimated), 2f)
            drawCircle(LabCyan, 5f, project(it.x, it.y, it.target))
        }
    }
    Text("Perspective plane · blue samples · pink residuals · purple fitted plane",
        color = LabMuted, fontSize = 11.sp)
}

@Composable
internal fun EnsembleCards(forest: List<SupervisedTreeNode>, regression: Boolean) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text("${forest.size} distinct trees → ${if (regression) "mean prediction" else "majority vote"}",
                color = LabText, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            Canvas(Modifier.fillMaxWidth().height(96.dp)) {
                val visible = forest.take(4)
                val column = size.width / visible.size
                visible.forEachIndexed { index, tree ->
                    fun render(node: SupervisedTreeNode, x: Float, y: Float, spacing: Float, level: Int) {
                        if (level < 3) {
                            node.left?.let { child ->
                                val childX = x - spacing
                                val childY = y + size.height / 4f
                                drawLine(LabPurple.copy(alpha = .6f), Offset(x, y), Offset(childX, childY), 2f)
                                render(child, childX, childY, spacing / 2f, level + 1)
                            }
                            node.right?.let { child ->
                                val childX = x + spacing
                                val childY = y + size.height / 4f
                                drawLine(LabPurple.copy(alpha = .6f), Offset(x, y), Offset(childX, childY), 2f)
                                render(child, childX, childY, spacing / 2f, level + 1)
                            }
                        }
                        val color = if (node.isLeaf) LabGreen else LabCyan
                        drawCircle(color, if (node.isLeaf) 4f else 5f, Offset(x, y))
                    }
                    render(tree, (index + .5f) * column, 7f, column / 4f, 0)
                }
            }
            forest.take(8).forEachIndexed { i, tree ->
                Text("Tree ${i + 1}: ${tree.nodes().size} nodes · root ${if (tree.feature == 0) "X1" else "X2"} ≤ ${"%.2f".format(tree.threshold ?: 0.0)}",
                    color = if (i % 2 == 0) LabCyan else LabPurple, fontSize = 11.sp)
            }
        }
    }
}

@Composable
internal fun BoostingCards(result: BoostingResult) {
    val current = result.stages.lastOrNull() ?: return
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            val title = when (result.variant) {
                BoostingVariant.Gradient -> "Residual correction"
                BoostingVariant.Ada -> "Adaptive sample weighting"
                BoostingVariant.Xgboost -> "Gradient and Hessian split gain"
                BoostingVariant.Lightgbm -> "Histogram bins and leaf-wise growth"
                BoostingVariant.Catboost -> "Ordered categories and symmetric tree"
            }
            Text(title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            Text("Initial prediction: ${"%.3f".format(result.initial)} · ${result.stages.size} fitted stages",
                color = LabMuted, fontSize = 11.sp)
            when (result.variant) {
                BoostingVariant.Ada -> {
                    Text(if (result.regression) "Final ensemble uses a weighted median."
                        else "Final class uses weighted stump votes.",
                        color = LabCyan, fontSize = 11.sp)
                    DiagnosticBars(current.sampleWeights, LabOrange)
                    Text("Larger bars are samples emphasized in the next learner.",
                        color = LabMuted, fontSize = 11.sp)
                }
                BoostingVariant.Xgboost -> {
                    DiagnosticBars(current.gradients.map { abs(it) }, LabPink)
                    DiagnosticBars(current.hessians, LabCyan)
                    Text("Pink = |gradient| · cyan = Hessian · regularized root gain ${"%.3f".format(current.gain)}",
                        color = LabMuted, fontSize = 11.sp)
                }
                BoostingVariant.Lightgbm -> {
                    FeatureHistogram(result.points, current.tree.threshold)
                    Text("Binned feature values feed splits; the next split expands the leaf with greatest error reduction.",
                        color = LabMuted, fontSize = 11.sp)
                }
                BoostingVariant.Catboost -> {
                    result.categoryEncoding.toSortedMap().forEach { (category, value) ->
                        Text("Category ${"%.2f".format(category)} → ordered target statistic ${"%.2f".format(value)}",
                            color = LabCyan, fontSize = 11.sp)
                    }
                    Text("Each tree level uses one shared split across all nodes.",
                        color = LabMuted, fontSize = 11.sp)
                }
                BoostingVariant.Gradient -> {
                    DiagnosticBars(current.residuals.map { abs(it) }, LabPurple)
                    Text("Bar height is the error presented to the current weak learner.",
                        color = LabMuted, fontSize = 11.sp)
                }
            }
            if (result.variant == BoostingVariant.Lightgbm ||
                result.variant == BoostingVariant.Catboost ||
                result.variant == BoostingVariant.Ada)
                NativeTreeDiagram(current.tree, true, "") {}
            result.stages.forEachIndexed { index, item ->
                Text("${index + 1}. ${item.description} Gain ${"%.3f".format(item.gain)}",
                    color = if (index == result.stages.lastIndex) LabGreen else LabMuted, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun DiagnosticBars(values: List<Double>, color: Color) {
    if (values.isEmpty()) return
    Canvas(Modifier.fillMaxWidth().height(54.dp)) {
        val limit = values.maxOf { abs(it) }.coerceAtLeast(1e-9)
        val width = size.width / values.size
        values.forEachIndexed { index, value ->
            val height = size.height * (abs(value) / limit).toFloat()
            drawRect(color.copy(alpha = .75f),
                Offset(index * width, size.height - height),
                Size((width - 2f).coerceAtLeast(1f), height))
        }
    }
}

@Composable
private fun FeatureHistogram(points: List<LabPoint>, threshold: Double?) {
    if (points.isEmpty()) return
    val bins = IntArray(12)
    points.forEach {
        val index = (((it.x + 1.0) / 2.0) * bins.size).toInt().coerceIn(0, bins.lastIndex)
        bins[index]++
    }
    Canvas(Modifier.fillMaxWidth().height(72.dp)) {
        val maxCount = bins.maxOrNull()?.coerceAtLeast(1) ?: 1
        val width = size.width / bins.size
        bins.forEachIndexed { index, count ->
            val height = (size.height - 8f) * count / maxCount
            drawRect(LabCyan.copy(alpha = .72f), Offset(index * width, size.height - height),
                Size(width - 3f, height))
        }
        threshold?.let {
            val x = ((it + 1.0) / 2.0 * size.width).toFloat()
            drawLine(LabPink, Offset(x, 0f), Offset(x, size.height), 2f)
        }
    }
}
