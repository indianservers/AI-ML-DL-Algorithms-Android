package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GradientButton
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBorder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

private enum class PhaseThreeSection(val label: String) {
    Observe("Observe Data"), Parameters("Choose Parameters"), Run("Run Algorithm"), Inspect("Inspect Result"), Break("Break It"), Compare("Compare")
}

@Composable
fun PhaseThreeAlgorithmLab(
    topic: LearnTopic,
    kind: PhaseThreeAlgorithmKind,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(PhaseThreeSection.Observe) }
    var preset by remember(topic.id) { mutableStateOf(defaultPreset3(kind)) }
    var samples by remember(topic.id) { mutableIntStateOf(120) }
    var groups by remember(topic.id) { mutableIntStateOf(3) }
    var noise by remember(topic.id) { mutableDoubleStateOf(.08) }
    var seed by remember(topic.id) { mutableIntStateOf(33) }
    val points = remember(preset, samples, groups, noise, seed) { PhaseThreeDatasets.clusters(preset, samples, groups, noise, seed) }
    val regression = remember(seed) { PhaseThreeDatasets.regression(60, seed) }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("${kind.category} - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "Phase 3", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                PhaseThreeSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            PhaseThreeSection.Observe -> ObserveSection3(kind, points, regression)
            PhaseThreeSection.Parameters -> ParameterSection3(kind, preset, samples, groups, noise, onPreset = { preset = it }, onSamples = { samples = it }, onGroups = { groups = it }, onNoise = { noise = it }, onSeed = { seed += 1 })
            PhaseThreeSection.Run -> RunSection3(kind, points, regression)
            PhaseThreeSection.Inspect -> InspectSection3(kind, points, regression)
            PhaseThreeSection.Break -> BreakSection3(kind, onBreak = { preset = breakPreset3(kind); seed += 1 }, onComplete = onComplete)
            PhaseThreeSection.Compare -> CompareSection3(kind, points, regression)
        }
    }
}

@Composable
private fun ObserveSection3(kind: PhaseThreeAlgorithmKind, points: List<ClusterPoint>, regression: List<LabPoint>) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { Info3("Core idea", info3(kind)) }
        item {
            if (kind in boostingKinds) RegressionBoostingCanvas(regression, PhaseThreeEngines.gradientBoosting(regression, 4, .15))
            else ClusterVisualizationCanvas(points, null, null, null, null, null)
        }
        item { Equation3(equation3(kind), "Every number shown in the lab is recomputed from the current dataset and parameters.") }
        item { Info3("Unsupervised learning note", "For clustering and dimensionality reduction, the model does not receive true labels. Hidden labels are only available for synthetic comparison when explicitly described.") }
    }
}

@Composable
private fun ParameterSection3(
    kind: PhaseThreeAlgorithmKind,
    preset: ClusterPreset,
    samples: Int,
    groups: Int,
    noise: Double,
    onPreset: (ClusterPreset) -> Unit,
    onSamples: (Int) -> Unit,
    onGroups: (Int) -> Unit,
    onNoise: (Double) -> Unit,
    onSeed: () -> Unit
) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Choose Parameters", "Dataset shape, noise, natural groups, and random seed") }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        presets3(kind).forEach { item -> SegmentedOption(item.label, item == preset) { onPreset(item) } }
                    }
                    Slider3("Samples", samples.toDouble(), 20.0, 500.0) { onSamples(it.toInt().coerceIn(20, 500)) }
                    Slider3("Natural groups", groups.toDouble(), 1.0, 6.0) { onGroups(it.toInt().coerceIn(1, 6)) }
                    Slider3("Noise", noise, 0.0, .45) { onNoise(it) }
                    SegmentedOption("New Seed", false, Modifier.fillMaxWidth(), onSeed)
                }
            }
        }
        item { Info3("Parameter insight", parameterInsight3(kind)) }
    }
}

@Composable
private fun RunSection3(kind: PhaseThreeAlgorithmKind, points: List<ClusterPoint>, regression: List<LabPoint>) {
    var k by remember(kind) { mutableIntStateOf(3) }
    var iterations by remember(kind) { mutableIntStateOf(6) }
    var eps by remember(kind) { mutableDoubleStateOf(.22) }
    var minPts by remember(kind) { mutableIntStateOf(5) }
    var bandwidth by remember(kind) { mutableDoubleStateOf(.28) }
    var learningRate by remember(kind) { mutableDoubleStateOf(.15) }
    var linkage by remember(kind) { mutableStateOf(LinkageMethod.Average) }
    val km = PhaseThreeEngines.kMeans(points, k, iterations, plusPlus = kind == PhaseThreeAlgorithmKind.KMeansPlusPlus, miniBatch = if (kind == PhaseThreeAlgorithmKind.MiniBatchKMeans) 18 else null)
    val db = PhaseThreeEngines.dbscan(points, eps, minPts)
    val pca = PhaseThreeEngines.pca(points)
    val boost = PhaseThreeEngines.gradientBoosting(regression, iterations, learningRate)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Run Algorithm", "Step through the mechanism that creates structure") }
        item {
            when (kind) {
                in boostingKinds -> RegressionBoostingCanvas(regression, boost)
                PhaseThreeAlgorithmKind.Dbscan -> ClusterVisualizationCanvas(points, null, db, null, null, null)
                PhaseThreeAlgorithmKind.Hierarchical -> DendrogramVisualizer(PhaseThreeEngines.hierarchical(points.take(24), linkage))
                PhaseThreeAlgorithmKind.MeanShift -> ClusterVisualizationCanvas(points, PhaseThreeEngines.meanShift(points, bandwidth), null, null, null, null)
                PhaseThreeAlgorithmKind.GaussianMixture -> ClusterVisualizationCanvas(points, null, null, PhaseThreeEngines.gmm(points, k), null, null)
                PhaseThreeAlgorithmKind.Pca -> ClusterVisualizationCanvas(points, null, null, null, pca, null)
                PhaseThreeAlgorithmKind.SpectralClustering, PhaseThreeAlgorithmKind.Isomap -> GraphCanvas3(PhaseThreeEngines.similarityGraph(points.take(80), 5))
                PhaseThreeAlgorithmKind.KernelPca, PhaseThreeAlgorithmKind.Tsne, PhaseThreeAlgorithmKind.Umap, PhaseThreeAlgorithmKind.Lle -> EmbeddingCanvas3(PhaseThreeEngines.embedding(points.take(120), kind, 8))
                else -> ClusterVisualizationCanvas(points, km, null, null, null, null)
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (kind in clusterKinds) {
                        Slider3("K / components", k.toDouble(), 1.0, 10.0) { k = it.toInt().coerceIn(1, 10) }
                        Slider3("Iterations", iterations.toDouble(), 1.0, 20.0) { iterations = it.toInt().coerceIn(1, 20) }
                    }
                    if (kind == PhaseThreeAlgorithmKind.Dbscan || kind == PhaseThreeAlgorithmKind.Optics) {
                        Slider3("epsilon", eps, .04, .7) { eps = it }
                        Slider3("MinPts", minPts.toDouble(), 2.0, 16.0) { minPts = it.toInt().coerceIn(2, 16) }
                    }
                    if (kind == PhaseThreeAlgorithmKind.MeanShift) Slider3("Bandwidth", bandwidth, .05, .8) { bandwidth = it }
                    if (kind in boostingKinds) {
                        Slider3("Stages", iterations.toDouble(), 1.0, 20.0) { iterations = it.toInt().coerceIn(1, 20) }
                        Slider3("Learning rate", learningRate, .02, .8) { learningRate = it }
                    }
                    if (kind == PhaseThreeAlgorithmKind.Hierarchical) {
                        Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            LinkageMethod.entries.forEach { SegmentedOption(it.label, linkage == it) { linkage = it } }
                        }
                    }
                }
            }
        }
        item { StepCard3(kind, km, db, pca, boost) }
    }
}

@Composable
private fun InspectSection3(kind: PhaseThreeAlgorithmKind, points: List<ClusterPoint>, regression: List<LabPoint>) {
    val km = PhaseThreeEngines.kMeans(points, 3, 8, plusPlus = kind == PhaseThreeAlgorithmKind.KMeansPlusPlus)
    val db = PhaseThreeEngines.dbscan(points, .22, 5)
    val pca = PhaseThreeEngines.pca(points)
    val sil = PhaseThreeEngines.silhouette(points, km.assignments)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Inspect Result", "Quality, neighborhoods, matrices, and selected-point state") }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Inertia", "%.2f".format(km.inertia), LabCyan, Modifier.weight(1f))
                MetricPill("Silhouette", sil.overall?.let { "%.2f".format(it) } ?: "n/a", LabGreen, Modifier.weight(1f))
                MetricPill("Noise", db.noise.size.toString(), LabOrange, Modifier.weight(1f))
            }
        }
        item { NeighborhoodInspector3(points) }
        item {
            when (kind) {
                PhaseThreeAlgorithmKind.Pca -> MatrixRenderer3(MatrixState(listOf("x", "y"), listOf("x", "y"), listOf(listOf(pca.variance1, 0.0), listOf(0.0, pca.variance2))))
                PhaseThreeAlgorithmKind.TruncatedSvd -> MatrixRenderer3(PhaseThreeEngines.truncatedSvd(2).first)
                PhaseThreeAlgorithmKind.Ica -> SignalCanvas3(PhaseThreeEngines.ica(.65, .25))
                PhaseThreeAlgorithmKind.XGBoostConcepts -> SplitGainPanel3(PhaseThreeEngines.xgBoostGain(1.0, .1))
                PhaseThreeAlgorithmKind.HistogramGradientBoosting, PhaseThreeAlgorithmKind.LightGbmConcepts -> HistogramPanel3(points)
                PhaseThreeAlgorithmKind.CatBoostConcepts -> CatBoostPanel3()
                else -> Info3("Result explanation", resultExplanation3(kind))
            }
        }
    }
}

@Composable
private fun BreakSection3(kind: PhaseThreeAlgorithmKind, onBreak: () -> Unit, onComplete: () -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Break It", "Load a counterexample and explain what fails") }
        item { Info3("Counterexample", breakText3(kind)) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SegmentedOption("Break It", true, Modifier.weight(1f), onBreak)
                GradientButton("Mark lesson complete", Modifier.weight(1f), onComplete)
            }
        }
        item { Info3("Use this when / avoid this when", chooserText3(kind)) }
    }
}

@Composable
private fun CompareSection3(kind: PhaseThreeAlgorithmKind, points: List<ClusterPoint>, regression: List<LabPoint>) {
    val km = PhaseThreeEngines.kMeans(points, 3, 8)
    val db = PhaseThreeEngines.dbscan(points, .22, 5)
    val gmm = PhaseThreeEngines.gmm(points, 3)
    val pca = PhaseThreeEngines.pca(points)
    val emb = PhaseThreeEngines.embedding(points.take(120), PhaseThreeAlgorithmKind.Umap, 4)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Compare Methods", "Same dataset, different assumptions") }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("K-Means", "%.2f".format(km.inertia), LabCyan, Modifier.weight(1f))
                MetricPill("DBSCAN noise", db.noise.size.toString(), LabPink, Modifier.weight(1f))
                MetricPill("GMM LL", "%.1f".format(gmm.logLikelihood), LabGreen, Modifier.weight(1f))
            }
        }
        item { Info3("Curated comparison", comparisonText3(kind)) }
        item { ClusterVisualizationCanvas(points, km, db, gmm, pca, emb) }
    }
}

@Composable
private fun ClusterVisualizationCanvas(points: List<ClusterPoint>, kMeans: KMeansState?, dbscan: DbscanState?, gmm: GmmState?, pca: PcaState?, embedding: EmbeddingState?) {
    val drawPoints = embedding?.embedding ?: kMeans?.points ?: points
    Canvas(Modifier.fillMaxWidth().height(310.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        fun sx(x: Double) = (size.width * (x + 1.1) / 2.2).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.1) / 2.2)).toFloat()
        val colors = listOf(LabCyan, LabPink, LabOrange, LabGreen, LabPurple, Color.White)
        for (i in 0..4) {
            drawLine(Color.White.copy(alpha = .06f), Offset(size.width * i / 4f, 0f), Offset(size.width * i / 4f, size.height))
            drawLine(Color.White.copy(alpha = .06f), Offset(0f, size.height * i / 4f), Offset(size.width, size.height * i / 4f))
        }
        gmm?.components?.forEachIndexed { index, component ->
            val color = colors[index % colors.size]
            drawOval(color.copy(alpha = .15f), Offset(sx(component.mean.x) - (sqrt(component.varianceX) * 160).toFloat(), sy(component.mean.y) - (sqrt(component.varianceY) * 160).toFloat()), Size((sqrt(component.varianceX) * 320).toFloat(), (sqrt(component.varianceY) * 320).toFloat()), style = Stroke(3f))
            drawCircle(color, 7f, Offset(sx(component.mean.x), sy(component.mean.y)))
        }
        pca?.let {
            val c = Offset(sx(it.mean.x), sy(it.mean.y))
            val d = Offset((it.pc1.x * 160).toFloat(), (-it.pc1.y * 160).toFloat())
            drawLine(LabOrange, c - d, c + d, 5f, cap = StrokeCap.Round)
            val d2 = Offset((it.pc2.x * 90).toFloat(), (-it.pc2.y * 90).toFloat())
            drawLine(LabPurple, c - d2, c + d2, 3f, cap = StrokeCap.Round)
        }
        kMeans?.centers?.forEachIndexed { index, center ->
            drawCircle(Color.White, 11f, Offset(sx(center.x), sy(center.y)))
            drawCircle(colors[index % colors.size], 7f, Offset(sx(center.x), sy(center.y)))
        }
        drawPoints.forEachIndexed { index, point ->
            val cluster = dbscan?.labels?.getOrNull(index) ?: point.cluster.takeIf { it >= 0 } ?: point.hiddenLabel
            val isNoise = dbscan?.noise?.contains(index) == true || point.hiddenLabel == -1
            val isCore = dbscan?.core?.contains(index) == true
            val color = if (isNoise) LabMuted else colors[cluster.mod(colors.size)]
            if (isCore) drawCircle(Color.White.copy(alpha = .35f), 10f, Offset(sx(point.x), sy(point.y)), style = Stroke(2f))
            drawCircle(color, if (isNoise) 4.5f else 6.5f, Offset(sx(point.x), sy(point.y)))
        }
    }
}

@Composable
private fun DendrogramVisualizer(state: DendrogramState) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        val maxHeight = state.merges.maxOfOrNull { it.height }?.coerceAtLeast(.001) ?: 1.0
        state.merges.take(40).forEachIndexed { index, merge ->
            val x = size.width * (index + 1) / (state.merges.take(40).size + 1)
            val y = size.height * (1f - (merge.height / maxHeight).toFloat())
            drawLine(if (merge.height > state.cutHeight) LabPink else LabCyan, Offset(x, size.height), Offset(x, y), 2f)
            drawCircle(LabOrange, 4f, Offset(x, y))
        }
        val cutY = size.height * (1f - (state.cutHeight / maxHeight).toFloat())
        drawLine(Color.White, Offset(0f, cutY), Offset(size.width, cutY), 3f)
    }
}

@Composable
private fun GraphCanvas3(state: GraphState) {
    Canvas(Modifier.fillMaxWidth().height(290.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        fun sx(x: Double) = (size.width * (x + 1.1) / 2.2).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.1) / 2.2)).toFloat()
        state.edges.forEach { (a, b) -> drawLine(LabPurple.copy(alpha = .18f), Offset(sx(state.nodes[a].x), sy(state.nodes[a].y)), Offset(sx(state.nodes[b].x), sy(state.nodes[b].y)), 1.2f) }
        state.nodes.forEachIndexed { index, p -> drawCircle(if (index in state.selectedNeighbors) LabOrange else LabCyan, if (index == 0) 9f else 5f, Offset(sx(p.x), sy(p.y))) }
    }
}

@Composable
private fun EmbeddingCanvas3(state: EmbeddingState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Linked embedding: ${state.neighborOverlap} original neighbors preserved near selected point", color = LabText, fontWeight = FontWeight.Bold)
            ClusterVisualizationCanvas(state.embedding, null, null, null, null, null)
        }
    }
}

@Composable
private fun RegressionBoostingCanvas(points: List<LabPoint>, state: BoostingState) {
    Canvas(Modifier.fillMaxWidth().height(240.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        fun sx(x: Double) = (size.width * (x + 1.1) / 2.2).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.1) / 2.2)).toFloat()
        points.forEach { drawCircle(if (it.train) LabCyan else LabOrange, 5f, Offset(sx(it.x), sy(it.y))) }
        val prediction = state.stages.lastOrNull()?.predictionAfter ?: 0.0
        drawLine(LabGreen, Offset(0f, sy(prediction)), Offset(size.width, sy(prediction)), 4f)
        points.forEach { drawLine(LabPink.copy(alpha = .25f), Offset(sx(it.x), sy(it.y)), Offset(sx(it.x), sy(prediction)), 1.5f) }
    }
}

@Composable
private fun StepCard3(kind: PhaseThreeAlgorithmKind, km: KMeansState, db: DbscanState, pca: PcaState, boost: BoostingState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            SectionTitle("Current Algorithm Step", kind.displayName)
            when {
                kind in boostingKinds -> {
                    val stage = boost.stages.last()
                    Text("F${stage.stage}(x) = F${stage.stage - 1}(x) + eta*h${stage.stage}(x)", color = LabCyan, fontWeight = FontWeight.Bold)
                    Text("Before %.3f, residual %.3f, contribution %.3f, after %.3f".format(stage.predictionBefore, stage.residual, stage.contribution, stage.predictionAfter), color = LabMuted, fontSize = 12.sp)
                }
                kind == PhaseThreeAlgorithmKind.Dbscan -> Text("Average neighbors %.1f, core points ${db.core.size}, border ${db.border.size}, noise ${db.noise.size}".format(db.averageNeighbors), color = LabMuted, fontSize = 12.sp)
                kind == PhaseThreeAlgorithmKind.Pca -> Text("PC1 variance %.3f, PC2 variance %.3f, reconstruction error %.3f".format(pca.variance1, pca.variance2, pca.reconstructionError), color = LabMuted, fontSize = 12.sp)
                else -> Text("${km.stepDescription} Inertia %.3f".format(km.inertia), color = LabMuted, fontSize = 12.sp)
            }
        }
    }
}

@Composable
private fun NeighborhoodInspector3(points: List<ClusterPoint>) {
    val neighbors = points.indices.drop(1).sortedBy { (points[it].x - points[0].x).pow(2) + (points[it].y - points[0].y).pow(2) }.take(6)
    Info3("Neighborhood Inspector", neighbors.joinToString { "#$it d=%.2f".format(sqrt((points[it].x - points[0].x).pow(2) + (points[it].y - points[0].y).pow(2))) })
}

@Composable
private fun MatrixRenderer3(matrix: MatrixState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Matrix Visualization", color = LabText, fontWeight = FontWeight.Bold)
            matrix.values.forEachIndexed { row, values ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    values.forEachIndexed { col, value -> MetricPill("${matrix.rows[row]}-${matrix.columns[col]}", "%.1f".format(value), if (value > 0) LabCyan else LabMuted, Modifier.weight(1f)) }
                }
            }
        }
    }
}

@Composable
private fun SignalCanvas3(signals: Pair<List<Double>, List<Double>>) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun drawSignal(values: List<Double>, color: Color, offset: Float) {
            val path = Path()
            values.forEachIndexed { i, v ->
                val x = size.width * i / (values.size - 1)
                val y = offset + (-v * size.height * .18).toFloat()
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(3f, cap = StrokeCap.Round))
        }
        drawSignal(signals.first, LabCyan, size.height * .35f)
        drawSignal(signals.second, LabOrange, size.height * .7f)
    }
}

@Composable
private fun SplitGainPanel3(state: SplitGainState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Split Gain", color = LabText, fontWeight = FontWeight.Bold)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Split A", "%.2f".format(state.gainA), if (state.selected == "A") LabGreen else LabCyan, Modifier.weight(1f))
                MetricPill("Split B", "%.2f".format(state.gainB), if (state.selected == "B") LabGreen else LabPurple, Modifier.weight(1f))
            }
            Text("lambda %.2f and gamma %.2f regularize whether a split remains worthwhile.".format(state.lambda, state.gamma), color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun HistogramPanel3(points: List<ClusterPoint>) {
    val bins = 8
    val counts = List(bins) { bin -> points.count { (((it.x + 1.0) / 2.0) * bins).toInt().coerceIn(0, bins - 1) == bin } }
    Canvas(Modifier.fillMaxWidth().height(150.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val maxCount = counts.max().coerceAtLeast(1)
        val w = size.width / bins
        counts.forEachIndexed { i, count -> drawRect(LabCyan, Offset(i * w + 3f, size.height * (1f - count / maxCount.toFloat())), Size(w - 6f, size.height * count / maxCount.toFloat())) }
    }
}

@Composable
private fun CatBoostPanel3() {
    Info3("Ordered target statistics", "Naive target encoding can leak the current row's target. Ordered encoding uses previous observations only, then boosting proceeds on leakage-reduced categorical features.")
}

@Composable
private fun Slider3(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: %.2f".format(value), color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun Equation3(equation: String, note: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            SectionTitle("Formula", "Expandable mathematics entry point")
            Box(Modifier.fillMaxWidth().background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(12.dp)) {
                Text(equation, color = LabCyan, fontSize = 14.sp, fontWeight = FontWeight.Bold)
            }
            Text(note, color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun Info3(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private val boostingKinds = setOf(PhaseThreeAlgorithmKind.GradientBoostingRegression, PhaseThreeAlgorithmKind.GradientBoostingClassification, PhaseThreeAlgorithmKind.XGBoostConcepts, PhaseThreeAlgorithmKind.HistogramGradientBoosting, PhaseThreeAlgorithmKind.LightGbmConcepts, PhaseThreeAlgorithmKind.CatBoostConcepts)
private val clusterKinds = setOf(PhaseThreeAlgorithmKind.KMeans, PhaseThreeAlgorithmKind.KMeansPlusPlus, PhaseThreeAlgorithmKind.MiniBatchKMeans, PhaseThreeAlgorithmKind.GaussianMixture, PhaseThreeAlgorithmKind.MeanShift)

private fun defaultPreset3(kind: PhaseThreeAlgorithmKind) = when (kind) {
    PhaseThreeAlgorithmKind.Dbscan, PhaseThreeAlgorithmKind.SpectralClustering, PhaseThreeAlgorithmKind.KernelPca -> ClusterPreset.TwoMoons
    PhaseThreeAlgorithmKind.GaussianMixture -> ClusterPreset.Elongated
    PhaseThreeAlgorithmKind.Pca -> ClusterPreset.Elongated
    PhaseThreeAlgorithmKind.Tsne, PhaseThreeAlgorithmKind.Umap, PhaseThreeAlgorithmKind.Isomap, PhaseThreeAlgorithmKind.Lle -> ClusterPreset.HighDim
    else -> ClusterPreset.Blobs
}

private fun presets3(kind: PhaseThreeAlgorithmKind) = when (kind) {
    PhaseThreeAlgorithmKind.KMeans, PhaseThreeAlgorithmKind.KMeansPlusPlus, PhaseThreeAlgorithmKind.MiniBatchKMeans -> listOf(ClusterPreset.Blobs, ClusterPreset.Elongated, ClusterPreset.Circles, ClusterPreset.TwoMoons, ClusterPreset.Outliers, ClusterPreset.Overlap)
    PhaseThreeAlgorithmKind.Dbscan, PhaseThreeAlgorithmKind.Optics -> listOf(ClusterPreset.TwoMoons, ClusterPreset.DenseSparse, ClusterPreset.Noise, ClusterPreset.Outliers, ClusterPreset.Overlap)
    PhaseThreeAlgorithmKind.Pca, PhaseThreeAlgorithmKind.KernelPca -> listOf(ClusterPreset.Elongated, ClusterPreset.Circles, ClusterPreset.HighDim)
    else -> ClusterPreset.entries.take(8)
}

private fun breakPreset3(kind: PhaseThreeAlgorithmKind) = when (kind) {
    PhaseThreeAlgorithmKind.KMeans, PhaseThreeAlgorithmKind.KMeansPlusPlus, PhaseThreeAlgorithmKind.MiniBatchKMeans -> ClusterPreset.TwoMoons
    PhaseThreeAlgorithmKind.Dbscan -> ClusterPreset.DenseSparse
    PhaseThreeAlgorithmKind.GaussianMixture -> ClusterPreset.TwoMoons
    PhaseThreeAlgorithmKind.Pca -> ClusterPreset.Circles
    PhaseThreeAlgorithmKind.Isomap -> ClusterPreset.Noise
    PhaseThreeAlgorithmKind.Lle -> ClusterPreset.Single
    in boostingKinds -> ClusterPreset.Outliers
    else -> ClusterPreset.Overlap
}

private fun info3(kind: PhaseThreeAlgorithmKind) = when (kind) {
    PhaseThreeAlgorithmKind.KMeans -> "Centroids alternate between assigning points and moving to cluster means."
    PhaseThreeAlgorithmKind.KMeansPlusPlus -> "Initialization spreads centroids using distance-weighted probabilities before ordinary K-Means runs."
    PhaseThreeAlgorithmKind.MiniBatchKMeans -> "Small batches update centroids faster but with noisier movement."
    PhaseThreeAlgorithmKind.Dbscan -> "Density-connected core points form clusters; sparse points become border or noise."
    PhaseThreeAlgorithmKind.Hierarchical -> "Clusters merge one pair at a time, creating a dendrogram that can be cut at different heights."
    PhaseThreeAlgorithmKind.GaussianMixture -> "Components assign soft probabilities instead of hard cluster labels."
    PhaseThreeAlgorithmKind.Pca -> "PCA finds the axis with maximum projected variance, then compresses data onto it."
    in boostingKinds -> "Boosting adds weak learners sequentially so each stage corrects current errors."
    else -> "This lab shows the algorithm's structural transformation from raw data to a lower-dimensional or graph-based view."
}

private fun equation3(kind: PhaseThreeAlgorithmKind) = when (kind) {
    PhaseThreeAlgorithmKind.KMeans, PhaseThreeAlgorithmKind.KMeansPlusPlus, PhaseThreeAlgorithmKind.MiniBatchKMeans -> "argmin sum ||x_i - mu_cluster||^2"
    PhaseThreeAlgorithmKind.Dbscan -> "N_eps(p) = {q | distance(p,q) <= eps}"
    PhaseThreeAlgorithmKind.GaussianMixture -> "p(x) = sum pi_k N(x | mu_k, Sigma_k)"
    PhaseThreeAlgorithmKind.Pca -> "principal axis = eigenvector of covariance with largest eigenvalue"
    PhaseThreeAlgorithmKind.TruncatedSvd -> "A ~= U_k Sigma_k V_k^T"
    PhaseThreeAlgorithmKind.XGBoostConcepts -> "gain = split_score(left) + split_score(right) - split_score(parent) - gamma"
    in boostingKinds -> "F_m(x) = F_(m-1)(x) + eta h_m(x)"
    else -> "structure = transform(data, neighborhood graph, parameters)"
}

private fun parameterInsight3(kind: PhaseThreeAlgorithmKind) = when (kind) {
    PhaseThreeAlgorithmKind.Dbscan -> "epsilon changes neighborhood radius; MinPts changes how much local evidence creates a core point."
    PhaseThreeAlgorithmKind.KMeans -> "K changes how many compact groups the algorithm must force onto the data."
    PhaseThreeAlgorithmKind.Pca -> "The axis angle changes projected variance and reconstruction error."
    in boostingKinds -> "Learning rate controls how strongly each weak learner changes the current prediction."
    else -> "Parameters tune the tradeoff between local detail, global structure, and visual stability."
}

private fun resultExplanation3(kind: PhaseThreeAlgorithmKind) = "The result panel focuses on mechanism-specific state for ${kind.displayName}: assignments, neighborhoods, graph links, projections, or additive corrections."
private fun breakText3(kind: PhaseThreeAlgorithmKind) = "This loads a dataset known to challenge ${kind.displayName}. The goal is to see assumptions fail, then compare a better-suited method or parameter setting."
private fun chooserText3(kind: PhaseThreeAlgorithmKind) = "Use it when the data shape matches its assumptions. Avoid treating the visual output as proof; inspect metrics, failure cases, and parameter sensitivity."
private fun comparisonText3(kind: PhaseThreeAlgorithmKind) = "K-Means assumes compact centroid-shaped clusters; DBSCAN follows density; GMM allows soft elliptical membership; PCA preserves variance, while manifold methods emphasize neighborhoods."
