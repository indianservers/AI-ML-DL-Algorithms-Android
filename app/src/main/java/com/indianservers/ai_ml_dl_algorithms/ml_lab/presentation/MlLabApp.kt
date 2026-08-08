package com.indianservers.ai_ml_dl_algorithms.ml_lab.presentation

import android.content.Context
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.algorithms.PhaseOneEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.DatasetGraph
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GradientButton
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.HeroPipeline
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBlue
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGradientBackground
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPanelSoft
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LossChart
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.data.MlLabContent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Algorithm
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.AlgorithmFamily
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.AlgorithmStatus
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Point2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.presentation.DeepLearningScreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.presentation.AiEngineeringStudio
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnModuleScreen
import kotlinx.coroutines.delay

private enum class LabTab(val label: String) {
    Home("Home"),
    Learn("Learn"),
    Deep("Deep"),
    Train("Train"),
    Data("Data"),
    Infer("Studio"),
    Saved("Saved")
}

@Composable
fun MlLabApp() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("ml_lab_phase_one", Context.MODE_PRIVATE) }
    var selectedTab by remember { mutableStateOf(LabTab.Home) }
    var selectedAlgorithm by remember { mutableStateOf(MlLabContent.algorithms.first()) }
    var depth by remember {
        mutableStateOf(LearningDepth.entries.getOrElse(prefs.getInt("learning_depth", 0)) { LearningDepth.Beginner })
    }
    var onboardingDone by remember { mutableStateOf(prefs.getBoolean("onboarding_done", false)) }
    LaunchedEffect(depth) { prefs.edit().putInt("learning_depth", depth.ordinal).apply() }

    LabGradientBackground {
        if (!onboardingDone) {
            OnboardingScreen {
                prefs.edit().putBoolean("onboarding_done", true).apply()
                onboardingDone = true
            }
        } else {
            Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()) {
                Box(Modifier.weight(1f)) {
                    when (selectedTab) {
                        LabTab.Home -> HomeScreen(depth, onDepthChanged = { depth = it }, onOpen = { tab -> selectedTab = tab })
                        LabTab.Learn -> LearnModuleScreen(depth)
                        LabTab.Deep -> DeepLearningScreen()
                        LabTab.Train -> TrainingPlayground(selectedAlgorithm)
                        LabTab.Data -> DatasetLab()
                        LabTab.Infer -> AiEngineeringStudio()
                        LabTab.Saved -> SavedScreen(selectedAlgorithm)
                    }
                }
                BottomNav(selectedTab) { selectedTab = it }
            }
        }
    }
}

@Composable
private fun OnboardingScreen(onStart: () -> Unit) {
    Column(
        Modifier
            .fillMaxSize()
            .statusBarsPadding()
            .navigationBarsPadding()
            .padding(22.dp),
        verticalArrangement = Arrangement.SpaceBetween,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(20.dp))
        HeroPipeline(Modifier.fillMaxWidth())
        Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text("ML & DL Training", color = LabText, fontSize = 30.sp, fontWeight = FontWeight.Bold)
            Text("Master machine learning with live equations, datasets, training snapshots and offline lessons.", color = LabMuted, fontSize = 15.sp)
        }
        GradientButton("Get Started", Modifier.fillMaxWidth(), onStart)
    }
}

@Composable
private fun HomeScreen(depth: LearningDepth, onDepthChanged: (LearningDepth) -> Unit, onOpen: (LabTab) -> Unit) {
    LazyColumn(
        Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Column {
                    Text("ML & Deep Learning", color = LabText, fontSize = 28.sp, fontWeight = FontWeight.Bold)
                    Text("Learn - Train - Inspect - Deploy", color = LabMuted)
                }
                Text("Phase 5", color = LabGreen, fontWeight = FontWeight.Bold)
            }
        }
        item { HeroPipeline(Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("Learning Depth", "Choose the amount of mathematical detail")
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        LearningDepth.entries.forEach {
                            SegmentedOption(it.title, depth == it, Modifier.weight(1f)) { onDepthChanged(it) }
                        }
                    }
                    Text(depth.description, color = LabMuted, fontSize = 12.sp)
                }
            }
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    QuickCard("Learn Algorithms", "Full catalog", LabCyan, Modifier.weight(1f)) { onOpen(LabTab.Learn) }
                    QuickCard("Training Playground", "Live snapshots", LabPurple, Modifier.weight(1f)) { onOpen(LabTab.Train) }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    QuickCard("Dataset Lab", "Edit 2D data", LabGreen, Modifier.weight(1f)) { onOpen(LabTab.Data) }
                    QuickCard("AI Engineering Studio", "Real on-device models", LabOrange, Modifier.weight(1f)) { onOpen(LabTab.Infer) }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    QuickCard("Compare Models", "Metrics ready", LabBlue, Modifier.weight(1f)) { onOpen(LabTab.Train) }
                    QuickCard("Neural Networks", "Build and inspect", LabPink, Modifier.weight(1f)) { onOpen(LabTab.Deep) }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SectionTitle("Complete AI Lifecycle", "Learn - build - understand - deploy - optimize")
                    Text("${LearnCatalog.topics.size} structured lessons, modern architecture labs, real LiteRT/ONNX model import, live media pipelines, tensor inspection, quantization and device benchmarking.", color = LabMuted, fontSize = 13.sp)
                }
            }
        }
    }
}

@Composable
private fun CatalogScreen(selected: Algorithm, depth: LearningDepth, onSelected: (Algorithm) -> Unit) {
    var family by remember { mutableStateOf<AlgorithmFamily?>(null) }
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        SectionTitle("Algorithms", "Classical ML plus Phase 2 neural-network foundations")
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SegmentedOption("All", family == null) { family = null }
                    AlgorithmFamily.entries.take(3).forEach { item ->
                        SegmentedOption(item.title, family == item) { family = item }
                    }
                }
            }
            items(MlLabContent.algorithms.filter { family == null || it.family == family }) { algorithm ->
                AlgorithmRow(algorithm, selected.id == algorithm.id) { onSelected(algorithm) }
            }
            item { AlgorithmDetail(selected, depth) }
        }
    }
}

@Composable
private fun AlgorithmDetail(algorithm: Algorithm, depth: LearningDepth) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SectionTitle(algorithm.title, "${algorithm.family.title} - ${algorithm.subtitle}")
            Text(
                when (algorithm.status) {
                    AlgorithmStatus.Interactive -> if (algorithm.family == AlgorithmFamily.DeepLearning) {
                        "Interactive lab is available in Deep. Build, train and inspect real neural-network parameters offline."
                    } else {
                        "Interactive lab is available in Train or Infer. Theory, visualisation and metrics share the reusable lesson framework."
                    }
                    AlgorithmStatus.LessonReady -> "Lesson structure is ready; full interactive training is scheduled after the Phase 1 flagship set."
                    AlgorithmStatus.Future -> "Catalog placeholder for later phases. Navigation and content architecture already supports this topic."
                },
                color = LabMuted,
                fontSize = 13.sp
            )
            MlLabContent.lessonSections.forEach { section ->
                Text(section.title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 14.sp)
                Text(section.body(depth), color = LabMuted, fontSize = 13.sp)
            }
        }
    }
}

@Composable
private fun TrainingPlayground(algorithm: Algorithm) {
    var datasetName by remember { mutableStateOf("Noisy linear") }
    var learningRate by remember { mutableFloatStateOf(0.08f) }
    var epochs by remember { mutableIntStateOf(70) }
    var autoPlay by remember { mutableStateOf(false) }
    val points = MlLabContent.regressionDatasets.getValue(datasetName)
    val state = remember(points, epochs, learningRate) { PhaseOneEngines.trainLinearRegression(points, epochs, learningRate) }
    var selectedEpoch by remember(state) { mutableIntStateOf(state.snapshots.lastIndex) }
    LaunchedEffect(autoPlay, state) {
        while (autoPlay) {
            selectedEpoch = (selectedEpoch + 1).coerceAtMost(state.snapshots.lastIndex)
            if (selectedEpoch == state.snapshots.lastIndex) autoPlay = false
            delay(120)
        }
    }
    val current = state.snapshots[selectedEpoch]
    val metrics = PhaseOneEngines.regressionMetrics(points, current.weight, current.bias)

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Training Playground", "${algorithm.title} with inspectable snapshots") }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MlLabContent.regressionDatasets.keys.forEach {
                    SegmentedOption(it, datasetName == it) { datasetName = it }
                }
            }
        }
        item {
            DatasetGraph(points, Modifier.fillMaxWidth(), line = current)
        }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                MetricPill("Epoch", "${current.epoch}/$epochs", LabPurple, Modifier.weight(1f))
                MetricPill("Loss", "%.4f".format(current.loss), LabCyan, Modifier.weight(1f))
                MetricPill("Equation", "y=%.2fx%+.2f".format(current.weight, current.bias), LabGreen, Modifier.weight(1f))
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SectionTitle("Hyperparameters", "Training speed, learning rate and epoch control")
                    Text("Learning Rate %.3f".format(learningRate), color = LabMuted, fontSize = 12.sp)
                    Slider(value = learningRate, onValueChange = { learningRate = it }, valueRange = 0.005f..0.18f)
                    Text("Epochs $epochs", color = LabMuted, fontSize = 12.sp)
                    Slider(value = epochs.toFloat(), onValueChange = { epochs = it.toInt().coerceAtLeast(10) }, valueRange = 10f..160f)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption(if (autoPlay) "Pause" else "Play", autoPlay, Modifier.weight(1f)) { autoPlay = !autoPlay }
                        SegmentedOption("Step", false, Modifier.weight(1f)) { selectedEpoch = (selectedEpoch + 1).coerceAtMost(state.snapshots.lastIndex) }
                        SegmentedOption("Reset", false, Modifier.weight(1f)) { selectedEpoch = 0; autoPlay = false }
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("Training History", "Select earlier epochs to inspect model state")
                    LossChart(state.snapshots, selectedEpoch)
                    Slider(value = selectedEpoch.toFloat(), onValueChange = { selectedEpoch = it.toInt() }, valueRange = 0f..state.snapshots.lastIndex.toFloat())
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        metrics.entries.forEach { MetricPill(it.key, "%.3f".format(it.value), LabBlue, Modifier.weight(1f)) }
                    }
                    Text("Gradient dw %.4f, db %.4f".format(current.gradientWeight, current.gradientBias), color = LabMuted, fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun DatasetLab() {
    val points = remember { mutableStateListOf<Point2D>().apply { addAll(MlLabContent.classificationPoints) } }
    var selectedClass by remember { mutableIntStateOf(0) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Dataset Lab", "Regression, classification and clustering datasets") }
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SegmentedOption("Class A", selectedClass == 0) { selectedClass = 0 }
                SegmentedOption("Class B", selectedClass == 1) { selectedClass = 1 }
                SegmentedOption("Clear", false) { points.clear() }
                SegmentedOption("Randomise", false) {
                    points.clear()
                    points.addAll(MlLabContent.classificationPoints.shuffled())
                }
            }
        }
        item {
            DatasetGraph(points, onPointAdded = { points.add(it.copy(label = selectedClass)) })
        }
        item {
            val centers = PhaseOneEngines.kMeans(points, k = 3)
            val pca = PhaseOneEngines.pcaDirection(points)
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("Core Visualisation Engine", "K-Means centroids and PCA direction are computed offline")
                    DatasetGraph(points + centers, pcaDirection = pca)
                }
            }
        }
    }
}

@Composable
private fun InferenceLab() {
    var queryX by remember { mutableFloatStateOf(0.18f) }
    var queryY by remember { mutableFloatStateOf(0.12f) }
    var threshold by remember { mutableFloatStateOf(0.5f) }
    var k by remember { mutableIntStateOf(5) }
    var manhattan by remember { mutableStateOf(false) }
    val query = Point2D(queryX, queryY, 2)
    val logistic = PhaseOneEngines.classifyLogistic(query, threshold)
    val probability = PhaseOneEngines.logisticProbability(query)
    val (knnVote, neighbours) = PhaseOneEngines.knnPredict(MlLabContent.classificationPoints, query, k, manhattan)
    val (bayesVote, bayesProbabilities) = PhaseOneEngines.gaussianNaiveBayes(MlLabContent.classificationPoints, query)

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Model Inference", "Logistic regression, kNN and Naive Bayes run offline") }
        item { DatasetGraph(MlLabContent.classificationPoints + query, selectedNeighbours = neighbours) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SectionTitle("Query Point", "Move the point and watch predictions update")
                    Text("X %.2f".format(queryX), color = LabMuted, fontSize = 12.sp)
                    Slider(queryX, { queryX = it }, valueRange = -1f..1f)
                    Text("Y %.2f".format(queryY), color = LabMuted, fontSize = 12.sp)
                    Slider(queryY, { queryY = it }, valueRange = -1f..1f)
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("Classification Metrics", "Probability, threshold, votes and likelihoods")
                    Text("Threshold %.2f".format(threshold), color = LabMuted, fontSize = 12.sp)
                    Slider(threshold, { threshold = it }, valueRange = 0.1f..0.9f)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        MetricPill("Logistic", "Class $logistic", LabPurple, Modifier.weight(1f))
                        MetricPill("Prob", "%.2f".format(probability), LabCyan, Modifier.weight(1f))
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption("K=$k", true, Modifier.weight(1f)) { k = if (k >= 7) 1 else k + 2 }
                        SegmentedOption(if (manhattan) "Manhattan" else "Euclidean", manhattan, Modifier.weight(1f)) { manhattan = !manhattan }
                        MetricPill("kNN", "Class $knnVote", LabGreen, Modifier.weight(1f))
                    }
                    MetricPill("Naive Bayes", "Class $bayesVote  A %.2f  B %.2f".format(bayesProbabilities[0] ?: 0f, bayesProbabilities[1] ?: 0f), LabOrange, Modifier.fillMaxWidth())
                }
            }
        }
    }
}

@Composable
private fun SavedScreen(selectedAlgorithm: Algorithm) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Bookmarks", "Local saved-learning shell for Phase 1") }
        items(MlLabContent.algorithms.filter { it.status != AlgorithmStatus.Future }.take(9)) {
            AlgorithmRow(it, it.id == selectedAlgorithm.id) {}
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("Notes", "Offline-first content browsing")
                    MlLabContent.lessonSections.forEach { section ->
                        Text(section.title, color = LabText, fontWeight = FontWeight.Bold)
                        Text(section.body(LearningDepth.University), color = LabMuted, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun QuickCard(title: String, subtitle: String, accent: Color, modifier: Modifier = Modifier, onClick: () -> Unit) {
    GlassPanel(modifier) {
        Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Box(Modifier.size(34.dp), contentAlignment = Alignment.Center) {
                Text(title.first().toString(), color = accent, fontWeight = FontWeight.Bold, fontSize = 20.sp)
            }
            Text(title, color = LabText, fontWeight = FontWeight.Bold, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(subtitle, color = LabMuted, fontSize = 12.sp)
            SegmentedOption("Open", false, Modifier.fillMaxWidth(), onClick)
        }
    }
}

@Composable
private fun AlgorithmRow(algorithm: Algorithm, selected: Boolean, onClick: () -> Unit) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(42.dp), contentAlignment = Alignment.Center) {
                Text(algorithm.title.first().toString(), color = Color(algorithm.accent), fontWeight = FontWeight.Bold, fontSize = 20.sp)
            }
            Column(Modifier.weight(1f)) {
                Text(algorithm.title, color = LabText, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("${algorithm.family.title} - ${algorithm.status.name}", color = LabMuted, fontSize = 12.sp)
            }
            SegmentedOption(if (selected) "Selected" else "View", selected, onClick = onClick)
        }
    }
}

@Composable
private fun BottomNav(selected: LabTab, onSelected: (LabTab) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .height(70.dp)
            .padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        LabTab.entries.forEach { tab ->
            val active = selected == tab
            Box(
                Modifier
                    .weight(1f)
                    .height(52.dp)
                    .background(if (active) LabPurple else LabPanelSoft, RoundedCornerShape(8.dp))
                    .border(1.dp, if (active) Color.White.copy(alpha = 0.16f) else Color(0xFF2A365A), RoundedCornerShape(8.dp))
                    .clickable { onSelected(tab) }
                    .padding(horizontal = 2.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(tab.label, color = if (active) Color.White else LabMuted, fontSize = 10.sp, maxLines = 1)
            }
        }
    }
}
