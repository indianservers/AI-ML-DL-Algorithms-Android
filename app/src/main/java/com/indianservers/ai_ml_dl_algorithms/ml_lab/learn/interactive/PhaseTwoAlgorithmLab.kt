package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.gestures.detectTapGestures
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
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

private enum class PhaseTwoSection(val label: String) { Understand("Understand"), Visualize("Visualize"), Train("Train"), Experiment("Experiment"), Metrics("Metrics"), Compare("Compare") }

@Composable
fun PhaseTwoAlgorithmLab(
    topic: LearnTopic,
    kind: PhaseTwoAlgorithmKind,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(PhaseTwoSection.Understand) }
    var classes by remember(topic.id) { mutableIntStateOf(if (kind in binaryOnlyKinds) 2 else 3) }
    var preset by remember(topic.id) { mutableStateOf(defaultPreset(kind)) }
    var noise by remember(topic.id) { mutableDoubleStateOf(0.12) }
    var samples by remember(topic.id) { mutableIntStateOf(60) }
    var seed by remember(topic.id) { mutableIntStateOf(19) }
    val points = remember(topic.id, preset, noise, samples, seed, classes) {
        mutableStateListOf<LabPoint>().apply { addAll(PhaseTwoDatasets.generate(preset, samples, classes, noise, seed)) }
    }
    var selected by remember(topic.id) { mutableIntStateOf(0) }
    var query by remember(topic.id) { mutableStateOf(LabPoint(0.12, -0.05, 0)) }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("${kind.category} - ${kind.difficulty}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else depth.title, color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                PhaseTwoSection.entries.forEach { item -> SegmentedOption(item.label, section == item) { section = item } }
            }
        }
        when (section) {
            PhaseTwoSection.Understand -> PhaseTwoUnderstand(kind)
            PhaseTwoSection.Visualize -> PhaseTwoVisualize(kind, points, selected, query, onSelect = { selected = it }, onQuery = { query = it })
            PhaseTwoSection.Train -> PhaseTwoTrain(kind, points)
            PhaseTwoSection.Experiment -> PhaseTwoExperiment(kind, preset, classes, samples, noise, seed, onPreset = { preset = it }, onClasses = { classes = it }, onSamples = { samples = it }, onNoise = { noise = it }, onRandomize = { seed += 1 }, onBreak = { preset = breakPreset(kind); seed += 1 }, onComplete = onComplete)
            PhaseTwoSection.Metrics -> PhaseTwoMetrics(kind, points)
            PhaseTwoSection.Compare -> PhaseTwoCompare(kind, points, query)
        }
    }
}

@Composable
private fun PhaseTwoUnderstand(kind: PhaseTwoAlgorithmKind) {
    val info = phaseTwoInfo(kind)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { InfoCard("Intuition", info.intuition) }
        item { EquationCard2(info.equation, info.symbols) }
        item { InfoCard("Assumptions", info.assumptions) }
        item { InfoCard("Prerequisites", info.prerequisites) }
        item { InfoCard("What changed?", info.whatChanged) }
    }
}

@Composable
private fun PhaseTwoVisualize(
    kind: PhaseTwoAlgorithmKind,
    points: MutableList<LabPoint>,
    selected: Int,
    query: LabPoint,
    onSelect: (Int) -> Unit,
    onQuery: (LabPoint) -> Unit
) {
    var c by remember(kind) { mutableDoubleStateOf(1.0) }
    var gamma by remember(kind) { mutableDoubleStateOf(1.0) }
    var degree by remember(kind) { mutableIntStateOf(3) }
    var kernel by remember(kind) { mutableStateOf(KernelType.Rbf) }
    var trees by remember(kind) { mutableIntStateOf(7) }
    var votingMode by remember(kind) { mutableStateOf(VotingMode.Hard) }
    var correlated by remember(kind) { mutableStateOf(false) }
    val sample = points.getOrNull(selected) ?: query
    val summaries = PhaseTwoEngines.classSummaries(points)
    val nb = if (kind == PhaseTwoAlgorithmKind.GaussianNaiveBayes) PhaseTwoEngines.gaussianNaiveBayes(points, sample) else null
    val qda = if (kind == PhaseTwoAlgorithmKind.Qda) PhaseTwoEngines.qdaPredict(points, sample) else null
    val svm = if (kind in svmKinds || kind == PhaseTwoAlgorithmKind.SgdClassifier) PhaseTwoEngines.svmState(points, c) else null
    val kernelState = if (kind == PhaseTwoAlgorithmKind.KernelSvm) PhaseTwoEngines.kernelState(kernel, gamma, degree) else null
    val ensemble = if (kind in ensembleKinds) PhaseTwoEngines.ensemble(points, query, trees, kind, 9) else null
    val boost = if (kind == PhaseTwoAlgorithmKind.AdaBoostClassifier) PhaseTwoEngines.adaBoost(points.filter { it.label <= 1 }, 5) else emptyList()

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Visualize", "Decision regions, internals, selected sample, and model state") }
        item {
            MulticlassCanvas(
                points = points,
                selectedIndex = selected,
                query = if (kind in ensembleKinds || kind == PhaseTwoAlgorithmKind.VotingClassifier || kind == PhaseTwoAlgorithmKind.StackingClassifier) query else null,
                summaries = summaries,
                svm = svm,
                supportVectors = svm?.supportVectorIndices.orEmpty(),
                bootstrap = ensemble?.bootstrapStates?.firstOrNull(),
                sampleWeights = boost.lastOrNull()?.sampleWeights.orEmpty(),
                onSelect = onSelect,
                onQuery = onQuery
            )
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    when (kind) {
                        PhaseTwoAlgorithmKind.KernelSvm -> {
                            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                KernelType.entries.forEach { item -> SegmentedOption(item.label, kernel == item) { kernel = item } }
                            }
                            SliderLine("C", c, .05, 8.0) { c = it }
                            SliderLine("Gamma", gamma, .05, 10.0) { gamma = it }
                            SliderLine("Degree", degree.toDouble(), 2.0, 6.0) { degree = it.toInt().coerceIn(2, 6) }
                        }
                        PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoAlgorithmKind.SoftMarginSvm -> {
                            SliderLine("C soft-margin penalty", c, .05, 8.0) { c = it }
                            Text("Current minimum margin violations: ${svm?.violations?.size ?: 0}", color = LabMuted, fontSize = 12.sp)
                        }
                        PhaseTwoAlgorithmKind.BaggingClassifier, PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoAlgorithmKind.ExtraTreesClassifier -> {
                            SliderLine("Trees", trees.toDouble(), 1.0, 50.0) { trees = it.toInt().coerceIn(1, 50) }
                            Text("Visualizer shows selected tree plus aggregate votes; it does not draw all trees at once.", color = LabMuted, fontSize = 12.sp)
                        }
                        PhaseTwoAlgorithmKind.VotingClassifier -> {
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                                VotingMode.entries.forEach { item -> SegmentedOption(item.label, votingMode == item, Modifier.weight(1f)) { votingMode = item } }
                            }
                        }
                        PhaseTwoAlgorithmKind.GaussianNaiveBayes, PhaseTwoAlgorithmKind.MultinomialNaiveBayes, PhaseTwoAlgorithmKind.BernoulliNaiveBayes -> {
                            SegmentedOption(if (correlated) "Correlated features on" else "Correlated features off", correlated, Modifier.fillMaxWidth()) { correlated = !correlated }
                            Text("Naive Bayes treats feature likelihoods as conditionally independent given the class. Correlation can make that approximation less realistic, but it does not guarantee failure.", color = LabMuted, fontSize = 12.sp)
                        }
                        else -> Text("Parameters update the visible model-specific state from deterministic calculations.", color = LabMuted, fontSize = 12.sp)
                    }
                }
            }
        }
        item { ModelSpecificPanel(kind, sample, nb, qda, svm, kernelState, ensemble, boost, votingMode) }
        if (kind == PhaseTwoAlgorithmKind.GaussianNaiveBayes) item { GaussianDistributionVisualizer(sample.x, summaries.firstOrNull()) }
        if (kind == PhaseTwoAlgorithmKind.MultinomialNaiveBayes) item { TextNaiveBayesPanel(false) }
        if (kind == PhaseTwoAlgorithmKind.BernoulliNaiveBayes) item { TextNaiveBayesPanel(true) }
    }
}

@Composable
private fun PhaseTwoTrain(kind: PhaseTwoAlgorithmKind, points: List<LabPoint>) {
    var iteration by remember(kind) { mutableIntStateOf(0) }
    var rate by remember(kind) { mutableDoubleStateOf(0.08) }
    val step = when (kind) {
        PhaseTwoAlgorithmKind.SgdClassifier -> PhaseTwoEngines.sgdClassifierStep(points.filter { it.label <= 1 }, iteration, rate, -0.4, 0.35, 0.0)
        PhaseTwoAlgorithmKind.AdaBoostClassifier -> {
            val round = PhaseTwoEngines.adaBoost(points.filter { it.label <= 1 }, (iteration + 1).coerceIn(1, 8)).last()
            TrainingStep(round.round, "Fit a decision stump, find weighted mistakes, increase difficult sample weights, then continue.", round.weightedError, mapOf("learner weight" to round.learnerWeight, "weighted error" to round.weightedError), round.misclassified, "Misclassified samples become more prominent for the next weak learner.")
        }
        PhaseTwoAlgorithmKind.BaggingClassifier, PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoAlgorithmKind.ExtraTreesClassifier -> {
            val boot = PhaseTwoEngines.bootstrap(points, 4, iteration, randomThreshold = kind == PhaseTwoAlgorithmKind.ExtraTreesClassifier)
            TrainingStep(iteration + 1, "Draw a bootstrap sample with replacement, fit a small tree, and aggregate with other members.", boot.outOfBag.size.toDouble(), mapOf("out-of-bag samples" to boot.outOfBag.size.toDouble(), "repeated samples" to boot.frequencies.count { it > 1 }.toDouble()), boot.frequencies.indices.filter { boot.frequencies[it] > 1 }, "Repeated and omitted samples are the point: each member sees a different training view.")
        }
        else -> TrainingStep(iteration + 1, "This classifier primarily recomputes from current data and parameters.", 0.0, emptyMap(), emptyList(), "Move data or parameters in Visualize and inspect the changed state.")
    }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Train", "Meaningful steps, not hidden magic") }
        item { PhaseTwoStepCard(step) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderLine("Learning rate", rate, .001, .5) { rate = it }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption("Step", false, Modifier.weight(1f)) { iteration += 1 }
                        SegmentedOption("Reset", false, Modifier.weight(1f)) { iteration = 0 }
                    }
                }
            }
        }
    }
}

@Composable
private fun PhaseTwoExperiment(
    kind: PhaseTwoAlgorithmKind,
    preset: PhaseTwoDatasetPreset,
    classes: Int,
    samples: Int,
    noise: Double,
    seed: Int,
    onPreset: (PhaseTwoDatasetPreset) -> Unit,
    onClasses: (Int) -> Unit,
    onSamples: (Int) -> Unit,
    onNoise: (Double) -> Unit,
    onRandomize: () -> Unit,
    onBreak: () -> Unit,
    onComplete: () -> Unit
) {
    val presets = presetsFor(kind)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Experiment", "Presets, counterexamples, and model complexity") }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Dataset preset", color = LabText, fontWeight = FontWeight.Bold)
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        presets.forEach { item -> SegmentedOption(item.label, item == preset) { onPreset(item) } }
                    }
                    if (kind !in binaryOnlyKinds) SliderLine("Classes", classes.toDouble(), 2.0, 4.0) { onClasses(it.toInt().coerceIn(2, 4)) }
                    SliderLine("Samples", samples.toDouble(), 12.0, 200.0) { onSamples(it.toInt().coerceIn(12, 200)) }
                    SliderLine("Noise / overlap", noise, 0.0, 0.55) { onNoise(it) }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption("Randomize", false, Modifier.weight(1f), onRandomize)
                        SegmentedOption("Break It", true, Modifier.weight(1f), onBreak)
                    }
                }
            }
        }
        item { InfoCard("How can we fix this?", fixSuggestion(kind)) }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun PhaseTwoMetrics(kind: PhaseTwoAlgorithmKind, points: List<LabPoint>) {
    val actual = points.map { it.label }
    val predicted = when (kind) {
        PhaseTwoAlgorithmKind.GaussianNaiveBayes -> points.map { PhaseTwoEngines.gaussianNaiveBayes(points, it).first }
        PhaseTwoAlgorithmKind.Qda -> points.map { PhaseTwoEngines.qdaPredict(points, it).first }
        PhaseTwoAlgorithmKind.Lda -> points.map { PhaseTwoEngines.ldaPredict(points, it).first }
        PhaseTwoAlgorithmKind.BaggingClassifier, PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoAlgorithmKind.ExtraTreesClassifier -> points.map { PhaseTwoEngines.ensemble(points, it, 9, kind, 5).prediction }
        else -> points.map { if (it.x + it.y > 0) 1 else 0 }
    }
    val metrics = PhaseTwoEngines.multiclassMetrics(actual, predicted)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Metrics", "Macro metrics are shown separately from binary metrics") }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Accuracy", "%.2f".format(metrics.accuracy), LabCyan, Modifier.weight(1f))
                MetricPill("Macro P", "%.2f".format(metrics.macroPrecision), LabPurple, Modifier.weight(1f))
                MetricPill("Macro R", "%.2f".format(metrics.macroRecall), LabGreen, Modifier.weight(1f))
                MetricPill("Macro F1", "%.2f".format(metrics.macroF1), LabOrange, Modifier.weight(1f))
            }
        }
        item { MulticlassConfusionMatrix(metrics) }
        item { MisclassificationInspector(points, predicted) }
    }
}

@Composable
private fun PhaseTwoCompare(kind: PhaseTwoAlgorithmKind, points: List<LabPoint>, query: LabPoint) {
    val nb = PhaseTwoEngines.gaussianNaiveBayes(points, query).first
    val lda = PhaseTwoEngines.ldaPredict(points, query).first
    val forest = PhaseTwoEngines.ensemble(points, query, 11, PhaseTwoAlgorithmKind.RandomForestClassifier, 7)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Classifier Comparison", "Same dataset, different assumptions") }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Naive Bayes", "Class $nb", LabCyan, Modifier.weight(1f))
                MetricPill("LDA", "Class $lda", LabPurple, Modifier.weight(1f))
                MetricPill("Forest", "Class ${forest.prediction}", LabGreen, Modifier.weight(1f))
            }
        }
        item { InfoCard("Bias / variance", "One tree can be high variance. Bagging reduces variance by averaging bootstrap models. Random Forest additionally decorrelates trees with random feature subsets; this is useful, but not a universal guarantee.") }
        item { EnsembleVotes(forest) }
        item { InfoCard("Current algorithm focus", "${kind.displayName} is compared here against reusable baseline classifiers so students see assumption differences on the same data.") }
    }
}

@Composable
private fun MulticlassCanvas(
    points: List<LabPoint>,
    selectedIndex: Int,
    query: LabPoint?,
    summaries: List<ClassSummary>,
    svm: SvmState?,
    supportVectors: List<Int>,
    bootstrap: BootstrapState?,
    sampleWeights: List<Double>,
    onSelect: (Int) -> Unit,
    onQuery: (LabPoint) -> Unit
) {
    Canvas(
        Modifier.fillMaxWidth().height(310.dp)
            .background(Color(0xFF081126), RoundedCornerShape(8.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(8.dp))
            .pointerInput(points.size, query) {
                fun data(offset: Offset) = LabPoint((offset.x / size.width * 2.0 - 1.0).coerceIn(-1.0, 1.0), (1.0 - offset.y / size.height * 2.0).coerceIn(-1.0, 1.0))
                detectTapGestures { tap ->
                    val p = data(tap)
                    val nearest = points.indices.minByOrNull { (points[it].x - p.x).pow(2) + (points[it].y - p.y).pow(2) }
                    if (nearest != null && sqrt((points[nearest].x - p.x).pow(2) + (points[nearest].y - p.y).pow(2)) < .14) onSelect(nearest) else onQuery(p)
                }
            }
            .padding(10.dp)
    ) {
        fun sx(x: Double) = (size.width * (x + 1.0) / 2.0).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.0) / 2.0)).toFloat()
        val colors = listOf(LabCyan, LabPink, LabOrange, LabGreen)
        for (i in 0..4) {
            drawLine(Color.White.copy(alpha = .07f), Offset(size.width * i / 4f, 0f), Offset(size.width * i / 4f, size.height))
            drawLine(Color.White.copy(alpha = .07f), Offset(0f, size.height * i / 4f), Offset(size.width, size.height * i / 4f))
        }
        summaries.forEach { summary ->
            val color = colors[summary.label % colors.size]
            drawCircle(color.copy(alpha = .12f), (sqrt(summary.varianceX + summary.varianceY) * 115).toFloat().coerceIn(22f, 90f), Offset(sx(summary.meanX), sy(summary.meanY)))
            drawCircle(Color.White, 5f, Offset(sx(summary.meanX), sy(summary.meanY)))
        }
        bootstrap?.let {
            val value = if (it.splitFeature == "x1") sx(it.threshold) else sy(it.threshold)
            if (it.splitFeature == "x1") drawLine(LabPurple, Offset(value, 0f), Offset(value, size.height), 3f) else drawLine(LabPurple, Offset(0f, value), Offset(size.width, value), 3f)
        }
        svm?.let {
            fun linePoint(x: Double): Double = -(it.weightX * x + it.bias) / it.weightY.coerceAtLeast(1e-6)
            drawLine(Color.White, Offset(sx(-1.0), sy(linePoint(-1.0))), Offset(sx(1.0), sy(linePoint(1.0))), 4f, cap = StrokeCap.Round)
            val marginOffset = 1.0 / sqrt(it.weightX * it.weightX + it.weightY * it.weightY)
            drawLine(LabGreen.copy(alpha = .45f), Offset(sx(-1.0), sy(linePoint(-1.0) + marginOffset)), Offset(sx(1.0), sy(linePoint(1.0) + marginOffset)), 2f)
            drawLine(LabGreen.copy(alpha = .45f), Offset(sx(-1.0), sy(linePoint(-1.0) - marginOffset)), Offset(sx(1.0), sy(linePoint(1.0) - marginOffset)), 2f)
        }
        points.forEachIndexed { index, point ->
            val color = colors[point.label % colors.size]
            val support = index in supportVectors
            val selected = index == selectedIndex
            val weightRadius = sampleWeights.getOrNull(index)?.let { 5f + (it * 90).toFloat().coerceIn(0f, 12f) } ?: 6f
            if (support || selected) drawCircle(Color.White.copy(alpha = .85f), weightRadius + 6f, Offset(sx(point.x), sy(point.y)))
            if (bootstrap?.frequencies?.getOrNull(index) == 0) drawCircle(Color.White.copy(alpha = .18f), weightRadius + 8f, Offset(sx(point.x), sy(point.y)), style = Stroke(2f))
            drawCircle(color, weightRadius, Offset(sx(point.x), sy(point.y)))
        }
        query?.let {
            drawCircle(Color.White, 13f, Offset(sx(it.x), sy(it.y)))
            drawCircle(LabGreen, 8f, Offset(sx(it.x), sy(it.y)))
        }
    }
}

@Composable
private fun ModelSpecificPanel(
    kind: PhaseTwoAlgorithmKind,
    sample: LabPoint,
    nb: Pair<Int, List<PosteriorBreakdown>>?,
    qda: Pair<Int, List<PosteriorBreakdown>>?,
    svm: SvmState?,
    kernel: KernelState?,
    ensemble: EnsembleState?,
    boost: List<AdaBoostRound>,
    votingMode: VotingMode
) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            when {
                nb != null -> {
                    Text("Prior x Likelihood -> Posterior", color = LabText, fontWeight = FontWeight.Bold)
                    nb.second.forEach { item -> Text("Class ${item.label}: P(c)=%.2f, P(x1|c)=%.3f, P(x2|c)=%.3f, posterior=%.2f".format(item.prior, item.likelihoodX, item.likelihoodY, item.posterior), color = classColor(item.label), fontSize = 12.sp) }
                    Text("Prediction: class ${nb.first}", color = classColor(nb.first), fontWeight = FontWeight.Bold)
                }
                qda != null -> {
                    Text("QDA uses class-specific covariance, so boundaries can curve.", color = LabText, fontWeight = FontWeight.Bold)
                    qda.second.forEach { Text("Class ${it.label} posterior %.2f".format(it.posterior), color = classColor(it.label), fontSize = 12.sp) }
                }
                svm != null -> {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricPill("Margin", "%.2f".format(svm.marginWidth), LabCyan, Modifier.weight(1f))
                        MetricPill("Support", svm.supportVectorIndices.size.toString(), LabGreen, Modifier.weight(1f))
                        MetricPill("Hinge", "%.2f".format(svm.hingeLoss), LabPink, Modifier.weight(1f))
                    }
                    Text("Only points on or inside the margin become support-vector candidates in this educational state.", color = LabMuted, fontSize = 12.sp)
                }
                kernel != null -> {
                    Text("Kernel trick: original 2D data is compared through ${kernel.kernel.label} similarity.", color = LabText, fontWeight = FontWeight.Bold)
                    Text("Conceptual transformed separation score %.2f. ${kernel.warning.orEmpty()}".format(kernel.transformedSeparation), color = if (kernel.warning == null) LabMuted else LabPink, fontSize = 12.sp)
                }
                ensemble != null -> {
                    EnsembleVotes(ensemble)
                }
                boost.isNotEmpty() -> {
                    val round = boost.last()
                    Text("AdaBoost round ${round.round}", color = LabText, fontWeight = FontWeight.Bold)
                    Text("Weighted error %.3f -> learner weight %.3f".format(round.weightedError, round.learnerWeight), color = LabMuted, fontSize = 12.sp)
                    Text("Misclassified samples receive larger weights before the next stump.", color = LabMuted, fontSize = 12.sp)
                }
                kind == PhaseTwoAlgorithmKind.VotingClassifier -> {
                    val probs = listOf(mapOf(0 to .7, 1 to .3), mapOf(0 to .45, 1 to .55), mapOf(0 to .62, 1 to .38))
                    val soft = PhaseTwoEngines.softVoting(probs)
                    val hard = PhaseTwoEngines.hardVoting(listOf(0, 1, 0))
                    Text("${votingMode.label}: final class ${if (votingMode == VotingMode.Hard) hard else soft.first}", color = LabText, fontWeight = FontWeight.Bold)
                    Text("Hard voting counts labels. Soft voting averages model probability estimates.", color = LabMuted, fontSize = 12.sp)
                }
                kind == PhaseTwoAlgorithmKind.StackingClassifier -> {
                    val features = PhaseTwoEngines.stackingMetaFeatures(listOf(.82, .61, .74))
                    val probability = PhaseTwoEngines.stackingMetaProbability(features)
                    Text("Base predictions -> meta-features -> meta-model", color = LabText, fontWeight = FontWeight.Bold)
                    Text("Meta-features ${features.joinToString { "%.2f".format(it) }} produce probability %.2f".format(probability), color = LabMuted, fontSize = 12.sp)
                    Text("Leakage warning: a real stack should train the meta-model on out-of-fold predictions.", color = LabOrange, fontSize = 12.sp)
                }
                else -> Text("Selected sample x1=%.2f, x2=%.2f, class=${sample.label}".format(sample.x, sample.y), color = LabMuted, fontSize = 12.sp)
            }
        }
    }
}

@Composable
private fun GaussianDistributionVisualizer(selectedX: Double, summary: ClassSummary?) {
    var mean by remember(summary?.label) { mutableDoubleStateOf(summary?.meanX ?: 0.0) }
    var sigma by remember(summary?.label) { mutableDoubleStateOf(sqrt(summary?.varianceX ?: .12)) }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Gaussian Distribution", "Mean, standard deviation, selected x, and density")
            Canvas(Modifier.fillMaxWidth().height(132.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).padding(8.dp)) {
                val path = Path()
                for (i in 0..100) {
                    val x = -1.5 + 3.0 * i / 100.0
                    val density = exp(-0.5 * ((x - mean) / sigma.coerceAtLeast(.02)).pow(2)) / (sigma.coerceAtLeast(.02) * sqrt(2.0 * Math.PI))
                    val px = size.width * i / 100f
                    val py = size.height * (1f - (density / 2.2).toFloat().coerceIn(0f, 1f))
                    if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
                }
                drawPath(path, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
                val selected = ((selectedX + 1.5) / 3.0).toFloat().coerceIn(0f, 1f) * size.width
                drawLine(LabOrange, Offset(selected, 0f), Offset(selected, size.height), 3f)
            }
            SliderLine("mu", mean, -1.0, 1.0) { mean = it }
            SliderLine("sigma", sigma, 0.05, 1.0) { sigma = it }
        }
    }
}

@Composable
private fun TextNaiveBayesPanel(bernoulli: Boolean) {
    val counts = remember { mutableStateListOf(1, 0, 1, 0, 1, 0) }
    val state = if (bernoulli) {
        PhaseTwoEngines.bernoulliNaiveBayes(PhaseTwoEngines.textVocabulary.mapIndexed { i, w -> w to (counts[i] > 0) }.toMap())
    } else {
        PhaseTwoEngines.multinomialNaiveBayes(PhaseTwoEngines.textVocabulary.mapIndexed { i, w -> w to counts[i] }.toMap())
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(if (bernoulli) "Binary Feature Presence" else "Word Count Vector", color = LabText, fontWeight = FontWeight.Bold)
            PhaseTwoEngines.textVocabulary.forEachIndexed { index, word ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("$word = ${counts[index]}", color = LabMuted, fontSize = 12.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        SegmentedOption("-", false) { counts[index] = (counts[index] - 1).coerceAtLeast(0) }
                        SegmentedOption("+", true) { counts[index] = if (bernoulli) 1 - counts[index].coerceIn(0, 1) else counts[index] + 1 }
                    }
                }
            }
            Text("Prediction: class ${state.prediction}; scores ${state.classScores.mapValues { "%.2f".format(it.value) }}", color = LabGreen, fontSize = 12.sp)
        }
    }
}

@Composable
private fun EnsembleVotes(state: EnsembleState) {
    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
        Text("Tree votes -> aggregate prediction class ${state.prediction}", color = LabText, fontWeight = FontWeight.Bold)
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            state.voteDistribution.entries.take(4).forEach { (label, share) ->
                MetricPill("Class $label", "%.0f%%".format(share * 100), classColor(label), Modifier.weight(1f))
            }
        }
        state.members.take(5).forEach { member -> Text("${member.id} -> class ${member.prediction}, ${member.metadata}", color = LabMuted, fontSize = 11.sp) }
    }
}

@Composable
private fun MulticlassConfusionMatrix(metrics: MultiClassMetrics) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Multiclass Confusion Matrix", color = LabText, fontWeight = FontWeight.Bold)
            metrics.confusion.forEachIndexed { row, values ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    values.forEachIndexed { col, value -> MetricPill("A${metrics.classes[row]} P${metrics.classes[col]}", value.toString(), classColor(metrics.classes[row]), Modifier.weight(1f)) }
                }
            }
            metrics.perClass.forEach { Text("Class ${it.label}: P %.2f, R %.2f, F1 %.2f".format(it.precision, it.recall, it.f1), color = classColor(it.label), fontSize = 12.sp) }
        }
    }
}

@Composable
private fun MisclassificationInspector(points: List<LabPoint>, predicted: List<Int>) {
    val mistakes = points.indices.filter { points[it].label != predicted[it] }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Misclassification Inspector", color = LabText, fontWeight = FontWeight.Bold)
            Text("${mistakes.size} mistakes highlighted by index. Tap a sample in Visualize to inspect model-specific reasons.", color = LabMuted, fontSize = 12.sp)
            Text(mistakes.take(12).joinToString { "#$it true=${points[it].label} pred=${predicted[it]}" }, color = if (mistakes.isEmpty()) LabGreen else LabOrange, fontSize = 11.sp)
        }
    }
}

@Composable
private fun PhaseTwoStepCard(step: TrainingStep) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            SectionTitle("Current Algorithm Step", "Iteration ${step.iteration}")
            Text(step.description, color = LabMuted, fontSize = 12.sp)
            step.parameters.forEach { (name, value) -> Text("$name = %.4f".format(value), color = LabText, fontSize = 12.sp) }
            Text(step.explanation, color = LabGreen, fontSize = 12.sp)
        }
    }
}

@Composable
private fun SliderLine(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: %.2f".format(value), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun EquationCard2(equation: String, symbols: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            SectionTitle("Core Equation", "Linked to the visual state")
            Box(Modifier.fillMaxWidth().background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(12.dp)) {
                Text(equation, color = LabCyan, fontSize = 14.sp, fontWeight = FontWeight.Bold)
            }
            Text(symbols, color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun InfoCard(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private data class PhaseTwoInfo(val intuition: String, val equation: String, val symbols: String, val assumptions: String, val prerequisites: String, val whatChanged: String)

private fun phaseTwoInfo(kind: PhaseTwoAlgorithmKind): PhaseTwoInfo = when (kind) {
    PhaseTwoAlgorithmKind.GaussianNaiveBayes -> PhaseTwoInfo("Estimate each class distribution, multiply prior by feature likelihoods, then normalize into posteriors.", "P(c|x) proportional to P(c) * P(x1|c) * P(x2|c)", "Means and variances define Gaussian likelihoods per class.", "Features are treated as conditionally independent given class.", "Probability, class labels, Gaussian distributions.", "Moving a sample changes likelihood terms and therefore posterior scores.")
    PhaseTwoAlgorithmKind.MultinomialNaiveBayes -> PhaseTwoInfo("Classify count vectors such as tiny word-count examples.", "score(c) = log P(c) + sum count_i log P(word_i|c)", "Counts increase the effect of words seen many times.", "Counts are generated from class-specific categorical frequencies.", "Naive Bayes and logarithms.", "Adding token counts moves the posterior toward classes where those tokens are frequent.")
    PhaseTwoAlgorithmKind.BernoulliNaiveBayes -> PhaseTwoInfo("Classify based on binary feature presence or absence.", "score(c) = log P(c) + sum log P(feature_i present/absent|c)", "Both presence and absence contribute evidence.", "Binary features are conditionally independent approximations.", "Naive Bayes and binary features.", "Toggling a feature changes both positive and negative evidence.")
    PhaseTwoAlgorithmKind.Lda -> PhaseTwoInfo("Find a projection where class means separate relative to within-class spread.", "maximize between-class separation / within-class spread", "Projection axis and shared covariance create linear boundaries.", "Classes share covariance structure.", "Linear classification, covariance.", "More overlap lowers separability; shared covariance keeps the boundary linear.")
    PhaseTwoAlgorithmKind.Qda -> PhaseTwoInfo("Model each class with its own covariance, allowing curved boundaries.", "argmax_c log P(c) + log N(x; mu_c, Sigma_c)", "Each covariance ellipse shapes that class region.", "Each class can have different covariance.", "Gaussian class models.", "Different covariances bend boundaries that LDA would keep linear.")
    PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoAlgorithmKind.SoftMarginSvm -> PhaseTwoInfo("Choose a separating hyperplane with the widest margin.", "min ||w||^2 + C sum hinge_loss", "Support vectors are nearest or violating samples.", "Best suited to margin-separable feature spaces.", "Linear classification and hinge loss.", "Increasing C penalizes violations more strongly and can tighten the margin.")
    PhaseTwoAlgorithmKind.KernelSvm -> PhaseTwoInfo("Use a kernel similarity so nonlinear data can become linearly separable in an implicit feature space.", "K(x,z) = exp(-gamma ||x-z||^2) or (gamma x.z + 1)^d", "Gamma controls locality; degree controls polynomial flexibility.", "Kernel choice must match data geometry.", "Linear SVM, distance, feature maps.", "High gamma makes influence local and can overfit.")
    PhaseTwoAlgorithmKind.SgdClassifier -> PhaseTwoInfo("Train a linear classifier one sample or mini-batch at a time.", "w <- w - alpha gradient loss(w; x_i, y_i)", "Hinge loss updates only margin violations.", "Feature scaling and learning rate matter.", "Gradient descent and linear classification.", "Each selected sample can move the boundary if its margin is too small.")
    PhaseTwoAlgorithmKind.BaggingClassifier -> ensembleInfo("Bagging", "Bootstrap samples create different models whose majority vote reduces variance.")
    PhaseTwoAlgorithmKind.RandomForestClassifier -> ensembleInfo("Random Forest", "Bootstrap samples plus random feature subsets decorrelate trees before voting.")
    PhaseTwoAlgorithmKind.ExtraTreesClassifier -> ensembleInfo("Extra Trees", "Randomized thresholds add more split randomness than Random Forest.")
    PhaseTwoAlgorithmKind.AdaBoostClassifier -> PhaseTwoInfo("Build weak learners sequentially, increasing attention on mistakes.", "alpha_t = 0.5 log((1-error_t)/error_t)", "Sample weights become visible halos around difficult points.", "Weak learners must be better than random on weighted data.", "Decision stumps and classification error.", "Misclassified samples gain weight, steering the next stump.")
    PhaseTwoAlgorithmKind.VotingClassifier -> PhaseTwoInfo("Combine independently trained model predictions.", "hard: majority vote; soft: average probabilities", "Hard uses class labels; soft requires probability estimates.", "Base models should be meaningfully diverse.", "Basic classifiers and probabilities.", "Changing mode changes whether labels or probability estimates are aggregated.")
    PhaseTwoAlgorithmKind.StackingClassifier -> PhaseTwoInfo("Train base models, feed their predictions into a meta-model, then predict.", "meta_features = [p1, p2, p3]", "Base predictions become features for the meta learner.", "Meta-model training must avoid leakage using out-of-fold predictions.", "Voting and validation splits.", "Stacking is not voting: it learns how to combine model outputs.")
}

private fun ensembleInfo(name: String, intuition: String) = PhaseTwoInfo(intuition, "prediction = majority_vote(member_1(x), ..., member_T(x))", "Each member has bootstrap samples, optional feature subsets, and a vote.", "Members should differ enough for aggregation to help.", "Decision trees and train/test validation.", "$name changes visible votes, OOB membership, and aggregate stability.")

private val binaryOnlyKinds = setOf(PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoAlgorithmKind.KernelSvm, PhaseTwoAlgorithmKind.SoftMarginSvm, PhaseTwoAlgorithmKind.SgdClassifier, PhaseTwoAlgorithmKind.AdaBoostClassifier)
private val svmKinds = setOf(PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoAlgorithmKind.KernelSvm, PhaseTwoAlgorithmKind.SoftMarginSvm)
private val ensembleKinds = setOf(PhaseTwoAlgorithmKind.BaggingClassifier, PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoAlgorithmKind.ExtraTreesClassifier)

private fun defaultPreset(kind: PhaseTwoAlgorithmKind) = when (kind) {
    PhaseTwoAlgorithmKind.KernelSvm -> PhaseTwoDatasetPreset.CircularData
    PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoAlgorithmKind.SoftMarginSvm, PhaseTwoAlgorithmKind.SgdClassifier -> PhaseTwoDatasetPreset.PerfectlySeparable
    PhaseTwoAlgorithmKind.AdaBoostClassifier -> PhaseTwoDatasetPreset.LabelNoise
    PhaseTwoAlgorithmKind.BaggingClassifier, PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoAlgorithmKind.ExtraTreesClassifier -> PhaseTwoDatasetPreset.NoisyEnsemble
    else -> PhaseTwoDatasetPreset.WellSeparatedGaussian
}

private fun presetsFor(kind: PhaseTwoAlgorithmKind) = when {
    kind in svmKinds -> listOf(PhaseTwoDatasetPreset.PerfectlySeparable, PhaseTwoDatasetPreset.OneOutlier, PhaseTwoDatasetPreset.OverlappingClasses, PhaseTwoDatasetPreset.CircularData, PhaseTwoDatasetPreset.XorLike)
    kind in ensembleKinds -> listOf(PhaseTwoDatasetPreset.WellSeparatedGaussian, PhaseTwoDatasetPreset.NoisyEnsemble, PhaseTwoDatasetPreset.LabelNoise, PhaseTwoDatasetPreset.XorLike)
    kind == PhaseTwoAlgorithmKind.AdaBoostClassifier -> listOf(PhaseTwoDatasetPreset.PerfectlySeparable, PhaseTwoDatasetPreset.OneOutlier, PhaseTwoDatasetPreset.LabelNoise)
    else -> listOf(PhaseTwoDatasetPreset.WellSeparatedGaussian, PhaseTwoDatasetPreset.DifferentVariances, PhaseTwoDatasetPreset.CorrelatedFeatures, PhaseTwoDatasetPreset.ImbalancedPriors, PhaseTwoDatasetPreset.OverlappingClasses)
}

private fun breakPreset(kind: PhaseTwoAlgorithmKind) = when (kind) {
    PhaseTwoAlgorithmKind.GaussianNaiveBayes, PhaseTwoAlgorithmKind.MultinomialNaiveBayes, PhaseTwoAlgorithmKind.BernoulliNaiveBayes -> PhaseTwoDatasetPreset.CorrelatedFeatures
    PhaseTwoAlgorithmKind.Lda -> PhaseTwoDatasetPreset.DifferentVariances
    PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoAlgorithmKind.SoftMarginSvm -> PhaseTwoDatasetPreset.CircularData
    PhaseTwoAlgorithmKind.KernelSvm -> PhaseTwoDatasetPreset.LabelNoise
    PhaseTwoAlgorithmKind.AdaBoostClassifier -> PhaseTwoDatasetPreset.LabelNoise
    else -> PhaseTwoDatasetPreset.NoisyEnsemble
}

private fun fixSuggestion(kind: PhaseTwoAlgorithmKind) = when (kind) {
    PhaseTwoAlgorithmKind.LinearSvm -> "Try Kernel SVM for nonlinear circles, or reduce overlap with better features."
    PhaseTwoAlgorithmKind.KernelSvm -> "Lower gamma when the boundary becomes too local; use validation data to tune C and gamma."
    PhaseTwoAlgorithmKind.Lda -> "Use QDA when class covariance shapes are genuinely different."
    PhaseTwoAlgorithmKind.GaussianNaiveBayes -> "Inspect correlated features; Naive Bayes can still work, but calibration and assumptions deserve scrutiny."
    PhaseTwoAlgorithmKind.AdaBoostClassifier -> "Label noise can attract too much weight. Reduce estimators or use a model robust to noisy labels."
    else -> "Compare with a model whose assumptions match the counterexample, and validate on held-out data."
}

private fun classColor(label: Int): Color = listOf(LabCyan, LabPink, LabOrange, LabGreen)[label.mod(4)]
