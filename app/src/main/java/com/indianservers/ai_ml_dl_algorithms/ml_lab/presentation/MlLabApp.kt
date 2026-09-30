package com.indianservers.ai_ml_dl_algorithms.ml_lab.presentation

import android.content.Context
import android.app.Activity
import androidx.activity.compose.BackHandler
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
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmIcon
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import kotlinx.coroutines.delay

private enum class LabTab(val label: String) {
    Home("Home"),
    Learn("Learn"),
    Deep("Deep"),
    Train("Train"),
    Data("Data"),
    Infer("Studio"),
    Saved("Saved"),
    Settings("Settings")
}

@Composable
fun MlLabApp() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("ml_lab_phase_one", Context.MODE_PRIVATE) }
    var selectedTab by remember { mutableStateOf(LabTab.Home) }
    var selectedTopic by remember { mutableStateOf<LearnTopic?>(null) }
    var lastTopic by remember {
        mutableStateOf(LearnCatalog.topics.firstOrNull { it.id == prefs.getString("last_topic", null) })
    }
    var showExitDialog by remember { mutableStateOf(false) }
    val history = remember { mutableStateListOf<LabTab>() }
    fun navigate(tab: LabTab) {
        if (tab != selectedTab) history.add(selectedTab)
        selectedTab = tab
    }
    fun back() {
        if (selectedTab == LabTab.Home) showExitDialog = true
        else {
            selectedTab = if (history.isNotEmpty()) history.removeAt(history.lastIndex) else LabTab.Home
            if (selectedTab == LabTab.Home) selectedTopic = null
        }
    }
    var selectedAlgorithm by remember { mutableStateOf(MlLabContent.algorithms.first()) }
    var depth by remember {
        mutableStateOf(LearningDepth.entries.getOrElse(prefs.getInt("learning_depth", 0)) { LearningDepth.Beginner })
    }
    var onboardingDone by remember { mutableStateOf(prefs.getBoolean("onboarding_done", false)) }
    LaunchedEffect(depth) { prefs.edit().putInt("learning_depth", depth.ordinal).apply() }
    BackHandler(enabled = onboardingDone && selectedTab != LabTab.Learn) { back() }

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
                        LabTab.Home -> HomeDashboard(
                            continueTopic = lastTopic,
                            onTopic = {
                                selectedTopic = it
                                lastTopic = it
                                prefs.edit().putString("last_topic", it.id).apply()
                                navigate(LabTab.Learn)
                            },
                            onLibrary = { selectedTopic = null; navigate(LabTab.Learn) },
                            onSettings = { navigate(LabTab.Settings) },
                            onTool = { name ->
                                navigate(when (name) {
                                    "Dataset Lab" -> LabTab.Data
                                    "Training Playground" -> LabTab.Train
                                    "Deep Learning" -> LabTab.Deep
                                    "AI Engineering Studio" -> LabTab.Infer
                                    else -> LabTab.Saved
                                })
                            }
                        )
                        LabTab.Learn -> LearnModuleScreen(depth, selectedTopic, onBackHome = { back() })
                        LabTab.Deep -> DeepLearningScreen()
                        LabTab.Train -> TrainingPlayground(selectedAlgorithm)
                        LabTab.Data -> DatasetLab()
                        LabTab.Infer -> AiEngineeringStudio()
                        LabTab.Saved -> SavedScreen(selectedAlgorithm)
                        LabTab.Settings -> SettingsScreen(depth, onDepthChanged = { depth = it }, onBack = { back() })
                    }
                }
                if (selectedTab != LabTab.Home) HomeOnlyNav { selectedTopic = null; history.clear(); selectedTab = LabTab.Home }
            }
        }
    }
    if (showExitDialog) {
        AlertDialog(
            onDismissRequest = { showExitDialog = false },
            title = { Text("Exit the app?") },
            text = { Text("Would you like to close AI/ML Learning Lab?") },
            confirmButton = { TextButton(onClick = { showExitDialog = false; (context as? Activity)?.finish() }) { Text("Yes") } },
            dismissButton = { TextButton(onClick = { showExitDialog = false }) { Text("No") } }
        )
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
private fun SavedScreen(selectedAlgorithm: Algorithm) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Bookmarks", "Your saved learning notes") }
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
private fun AlgorithmRow(algorithm: Algorithm, selected: Boolean, onClick: () -> Unit) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            AlgorithmIcon(
                title = algorithm.title,
                section = algorithm.family.title,
                domain = algorithm.family.title,
                accent = algorithm.accent,
                size = 42.dp
            )
            Column(Modifier.weight(1f)) {
                Text(algorithm.title, color = LabText, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("${algorithm.family.title} - ${algorithm.status.name}", color = LabMuted, fontSize = 12.sp)
            }
            SegmentedOption(if (selected) "Selected" else "View", selected, onClick = onClick)
        }
    }
}

@Composable
private fun HomeOnlyNav(onHome: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(58.dp).background(Color(0xFF07152F))
            .clickable(onClick = onHome).padding(horizontal = 20.dp),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("⌂", color = LabCyan, fontSize = 28.sp)
        Text("  Home", color = LabText, fontWeight = FontWeight.Bold, fontSize = 15.sp)
    }
}

@Composable
private fun SettingsScreen(depth: LearningDepth, onDepthChanged: (LearningDepth) -> Unit, onBack: () -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item { SegmentedOption("‹  Back", false, onClick = onBack) }
        item { SectionTitle("Settings", "Set your preferred learning experience") }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    SectionTitle("Learning Depth", "Choose the amount of mathematical detail")
                    LearningDepth.entries.forEach { option ->
                        SegmentedOption("${option.title} — ${option.description}", depth == option, Modifier.fillMaxWidth()) {
                            onDepthChanged(option)
                        }
                    }
                }
            }
        }
    }
}
