package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import android.content.Context
import androidx.activity.compose.BackHandler
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
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseOneAlgorithmLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseOneTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourAlgorithmLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFiveDeepLearningLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFiveTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSixCnnLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSixTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenSequenceLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseEightTransformerLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseEightTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseNineGenerativeLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseNineTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeAlgorithmLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoAlgorithmLab
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoTopicMatcher
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin

private enum class LearningStage(val label: String) {
    Understand("Understand"), Visualize("Visualize"), Explore("Explore"), Train("Train"),
    Predict("Predict"), Experiment("Experiment"), Compare("Compare"), Test("Test")
}

@Composable
fun LearnModuleScreen(depth: LearningDepth) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("learn_module", Context.MODE_PRIVATE) }
    var selected by remember { mutableStateOf<LearnTopic?>(null) }
    val completed = remember {
        mutableStateListOf<String>().apply { addAll(prefs.getStringSet("completed", emptySet()).orEmpty()) }
    }

    BackHandler(enabled = selected != null) { selected = null }
    selected?.let { topic ->
        AlgorithmLearningScreen(
            topic = topic,
            depth = depth,
            completed = topic.id in completed,
            onBack = { selected = null },
            onComplete = {
                if (topic.id !in completed) completed.add(topic.id)
                prefs.edit().putStringSet("completed", completed.toSet()).apply()
            }
        )
    } ?: LearnCatalogScreen(completed.toSet(), onOpen = { selected = it })
}

@Composable
private fun LearnCatalogScreen(completed: Set<String>, onOpen: (LearnTopic) -> Unit) {
    var query by remember { mutableStateOf("") }
    val expandedDomains = remember { mutableStateListOf("Supervised Learning") }
    val expandedSections = remember { mutableStateListOf("Supervised Learning/Regression") }
    val matching = remember(query) {
        if (query.isBlank()) emptyList() else LearnCatalog.topics.filter {
            it.title.contains(query, true) || it.section.contains(query, true) || it.domain.contains(query, true)
        }
    }

    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                SectionTitle("Learn", "Top 10 flagship labs plus the full algorithm library")
                Text("${completed.size}/${LearnCatalog.topics.size}", color = LabGreen, fontWeight = FontWeight.Bold)
            }
        }
        item {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                label = { Text("Search algorithms") },
                placeholder = { Text("K-Means, attention, forecasting...") }
            )
        }
        if (query.isNotBlank()) {
            item { Text("${matching.size} results", color = LabMuted, fontSize = 12.sp) }
            items(matching, key = { it.id }) { topic -> TopicRow(topic, topic.id in completed) { onOpen(topic) } }
            if (matching.isEmpty()) item { EmptySearch() }
        } else {
            item { TopTenFlagshipPanel(completed, onOpen) }
            items(LearnCatalog.domains, key = { it.title }) { domain ->
                DomainAccordion(
                    domain = domain,
                    expanded = domain.title in expandedDomains,
                    expandedSections = expandedSections,
                    completed = completed,
                    onToggle = {
                        if (domain.title in expandedDomains) expandedDomains.remove(domain.title) else expandedDomains.add(domain.title)
                    },
                    onToggleSection = { key ->
                        if (key in expandedSections) expandedSections.remove(key) else expandedSections.add(key)
                    },
                    onOpen = onOpen
                )
            }
        }
    }
}

@Composable
private fun TopTenFlagshipPanel(completed: Set<String>, onOpen: (LearnTopic) -> Unit) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Top 10 Flagship Labs", "Premium touch-first lessons for the algorithms students need most")
            LearnCatalog.flagshipTopics.forEachIndexed { index, topic ->
                Row(
                    Modifier.fillMaxWidth().clickable { onOpen(topic) }.background(Color(topic.accent).copy(alpha = .08f), RoundedCornerShape(7.dp)).padding(9.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Box(Modifier.size(30.dp).background(Color(topic.accent).copy(alpha = .18f), RoundedCornerShape(7.dp)), contentAlignment = Alignment.Center) {
                        Text("${index + 1}", color = Color(topic.accent), fontWeight = FontWeight.Bold, fontSize = 12.sp)
                    }
                    Column(Modifier.weight(1f)) {
                        Text(flagshipTitle(topic), color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                        Text(flagshipPromise(topic), color = LabMuted, fontSize = 11.sp, maxLines = 2)
                    }
                    Text(if (topic.id in completed) "Done" else "Open", color = if (topic.id in completed) LabGreen else LabCyan, fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
private fun DomainAccordion(
    domain: LearnDomain,
    expanded: Boolean,
    expandedSections: List<String>,
    completed: Set<String>,
    onToggle: () -> Unit,
    onToggleSection: (String) -> Unit,
    onOpen: (LearnTopic) -> Unit
) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                Modifier.fillMaxWidth().clickable(onClick = onToggle),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    Modifier.size(38.dp).background(Color(domain.accent).copy(alpha = .15f), RoundedCornerShape(7.dp)),
                    contentAlignment = Alignment.Center
                ) { Text(domain.title.take(2), color = Color(domain.accent), fontWeight = FontWeight.Bold) }
                Column(Modifier.weight(1f)) {
                    Text(domain.title, color = LabText, fontWeight = FontWeight.Bold)
                    Text(domain.description, color = LabMuted, fontSize = 11.sp, maxLines = 2)
                }
                Text("${domain.topicCount}  ${if (expanded) "-" else "+"}", color = Color(domain.accent), fontWeight = FontWeight.Bold)
            }
            if (expanded) domain.sections.forEach { section ->
                val key = "${domain.title}/${section.title}"
                val sectionExpanded = key in expandedSections
                Column(
                    Modifier.fillMaxWidth().background(LabPanelSoft.copy(alpha = .62f), RoundedCornerShape(7.dp)).padding(9.dp),
                    verticalArrangement = Arrangement.spacedBy(7.dp)
                ) {
                    Row(
                        Modifier.fillMaxWidth().clickable { onToggleSection(key) },
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(section.title, color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                        Text("${section.topics.size}  ${if (sectionExpanded) "-" else "+"}", color = LabMuted, fontSize = 12.sp)
                    }
                    if (sectionExpanded) section.topics.forEach { topic ->
                        CompactTopicRow(topic, topic.id in completed) { onOpen(topic) }
                    }
                }
            }
        }
    }
}

@Composable
private fun CompactTopicRow(topic: LearnTopic, completed: Boolean, onOpen: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable(onClick = onOpen).padding(vertical = 7.dp, horizontal = 4.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(Modifier.size(7.dp).background(if (completed) LabGreen else Color(topic.accent), RoundedCornerShape(3.dp)))
        Text(topic.title, color = if (completed) LabMuted else LabText, fontSize = 13.sp, modifier = Modifier.weight(1f))
        Text(if (completed) "Done" else ">", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
    }
}

@Composable
private fun TopicRow(topic: LearnTopic, completed: Boolean, onOpen: () -> Unit) {
    GlassPanel(Modifier.fillMaxWidth().clickable(onClick = onOpen)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier.size(40.dp).background(Color(topic.accent).copy(alpha = .16f), RoundedCornerShape(7.dp)),
                contentAlignment = Alignment.Center
            ) { Text(topic.title.first().toString(), color = Color(topic.accent), fontWeight = FontWeight.Bold) }
            Column(Modifier.weight(1f)) {
                Text(topic.title, color = LabText, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("${topic.domain} / ${topic.section}", color = LabMuted, fontSize = 11.sp, maxLines = 1)
            }
            Text(if (completed) "Done" else ">", color = if (completed) LabGreen else LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun EmptySearch() {
    GlassPanel(Modifier.fillMaxWidth()) {
        Text("No matching algorithm. Try a domain, task, or shorter model name.", color = LabMuted, fontSize = 13.sp)
    }
}

private fun flagshipTitle(topic: LearnTopic): String = when (topic.title) {
    "Simple Linear Regression" -> "Linear Regression"
    "Multi-Layer Perceptron" -> "ANN / MLP"
    "LSTM" -> "RNN / LSTM"
    else -> topic.title
}

private fun flagshipPromise(topic: LearnTopic): String = when (topic.title) {
    "Simple Linear Regression" -> "Drag points, fit y = wx + b, inspect residuals, MSE, R2, and gradient descent."
    "Logistic Regression" -> "Move the threshold and inspect sigmoid probability, confusion matrix, precision, recall, and F1."
    "K-Nearest Neighbors" -> "Place a query, change K, switch distance metrics, and watch the nearest-neighbor vote."
    "Decision Tree" -> "Explore impurity, candidate splits, tree path, and underfit/overfit behavior."
    "Random Forest" -> "Inspect bootstraps, individual tree votes, ensemble stability, and noisy-data behavior."
    "Support Vector Machine" -> "See margins, support vectors, C, violations, and linear-vs-RBF behavior."
    "K-Means" -> "Animate assign/move steps, drag centroids, compare K-Means++, and inspect inertia."
    "Multi-Layer Perceptron" -> "Tap neurons and weights, inspect forward/backprop numbers, and train XOR."
    "CNN" -> "Move kernels over 8x8 images, inspect feature maps, pooling, and tiny shape classification."
    "LSTM" -> "Step through sequence memory, gradient flow, gates, and RNN vs LSTM delayed prediction."
    else -> "Open the flagship interactive lab."
}

@Composable
private fun AlgorithmLearningScreen(
    topic: LearnTopic,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    PhaseNineTopicMatcher.kindFor(topic.title, topic.domain)?.let { concept ->
        PhaseNineGenerativeLab(
            topic = topic,
            concept = concept,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseEightTopicMatcher.kindFor(topic.title, topic.domain)?.let { concept ->
        PhaseEightTransformerLab(
            topic = topic,
            concept = concept,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseSevenTopicMatcher.kindFor(topic.title, topic.domain)?.let { concept ->
        PhaseSevenSequenceLab(
            topic = topic,
            concept = concept,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseSixTopicMatcher.kindFor(topic.title, topic.domain)?.let { concept ->
        PhaseSixCnnLab(
            topic = topic,
            concept = concept,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseFiveTopicMatcher.kindFor(topic.title, topic.domain)?.let { concept ->
        PhaseFiveDeepLearningLab(
            topic = topic,
            concept = concept,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseFourTopicMatcher.kindFor(topic.title)?.let { kind ->
        PhaseFourAlgorithmLab(
            topic = topic,
            kind = kind,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseThreeTopicMatcher.kindFor(topic.title, topic.section, topic.domain)?.let { kind ->
        PhaseThreeAlgorithmLab(
            topic = topic,
            kind = kind,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseTwoTopicMatcher.kindFor(topic.title, topic.section, topic.domain)?.let { kind ->
        PhaseTwoAlgorithmLab(
            topic = topic,
            kind = kind,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    PhaseOneTopicMatcher.kindFor(topic.title, topic.section)?.let { kind ->
        PhaseOneAlgorithmLab(
            topic = topic,
            kind = kind,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onComplete = onComplete
        )
        return
    }
    val profile = remember(topic, depth) { LearnCatalog.profile(topic, depth) }
    var stage by remember(topic) { mutableStateOf(LearningStage.Understand) }

    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("${topic.domain} / ${topic.section}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else depth.title, color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                LearningStage.entries.forEach { item ->
                    SegmentedOption(item.label, stage == item) { stage = item }
                }
            }
        }
        when (stage) {
            LearningStage.Understand -> UnderstandStage(profile)
            LearningStage.Visualize -> InteractiveStage(topic, profile, "Interactive visualization", false)
            LearningStage.Explore -> ExploreStage(topic, profile)
            LearningStage.Train -> TrainStage(profile)
            LearningStage.Predict -> PredictStage(topic, profile)
            LearningStage.Experiment -> InteractiveStage(topic, profile, "Mini-lab", true)
            LearningStage.Compare -> CompareStage(topic, profile)
            LearningStage.Test -> QuizStage(topic, completed, onComplete)
        }
    }
}

@Composable
private fun UnderstandStage(profile: LearningProfile) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { LearningCard("What it is", profile.definition) }
        item { LearningCard("Why it is used", profile.purpose) }
        item { LearningCard("Intuition", profile.intuition) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    SectionTitle("How it works", "Step-by-step")
                    profile.steps.forEachIndexed { index, step -> Text("${index + 1}. $step", color = LabMuted, fontSize = 13.sp) }
                }
            }
        }
        item { EquationCard(profile) }
        item { TripleListCard("Advantages", profile.advantages, "Limitations", profile.limitations, "Assumptions", profile.assumptions) }
        item { TripleListCard("Hyperparameters", profile.hyperparameters, "Applications", profile.applications, "Common mistakes", profile.mistakes) }
    }
}

@Composable
private fun InteractiveStage(topic: LearnTopic, profile: LearningProfile, title: String, experiment: Boolean) {
    var primary by remember(topic) { mutableFloatStateOf(if (experiment) .72f else .42f) }
    var secondary by remember(topic) { mutableFloatStateOf(if (experiment) .28f else .58f) }
    var run by remember(topic) { mutableIntStateOf(0) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle(title, visualSubtitle(profile.kind)) }
        item { AlgorithmCanvas(profile.kind, primary, secondary, run, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text(parameterLabel(profile.kind, true, primary), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(primary, { primary = it })
                    Text(parameterLabel(profile.kind, false, secondary), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(secondary, { secondary = it })
                    GradientButton(if (experiment) "Run sample experiment" else "Advance one iteration", Modifier.fillMaxWidth()) { run++ }
                }
            }
        }
        item { LearningCard("Example dataset", profile.sampleData) }
        item { LearningCard("Observation", observationFor(profile.kind, primary, secondary, run)) }
    }
}

@Composable
private fun ExploreStage(topic: LearnTopic, profile: LearningProfile) {
    var parameter by remember(topic) { mutableFloatStateOf(.5f) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Explore", "Connect the equation to model behavior") }
        item { EquationCard(profile) }
        item { AlgorithmCanvas(profile.kind, parameter, 1f - parameter, 1, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(parameterLabel(profile.kind, true, parameter), color = LabText, fontWeight = FontWeight.Bold)
                    Slider(parameter, { parameter = it })
                    Text("Move the control and identify which term in the equation changes the visible output.", color = LabMuted, fontSize = 12.sp)
                }
            }
        }
    }
}

@Composable
private fun TrainStage(profile: LearningProfile) {
    var steps by remember { mutableIntStateOf(0) }
    var rate by remember { mutableFloatStateOf(.35f) }
    val loss = exp(-steps * (.025f + rate * .025f)).coerceAtLeast(.012f)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Train", "Inspect optimization rather than hiding it") }
        item { TrainingCurve(steps, rate) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Steps", steps.toString(), LabPurple, Modifier.weight(1f))
                MetricPill("Loss", "%.4f".format(loss), LabCyan, Modifier.weight(1f))
                MetricPill("Score", "%.1f%%".format((1f - loss) * 100), LabGreen, Modifier.weight(1f))
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("Learning rate %.3f".format(.005f + rate * .195f), color = LabText, fontWeight = FontWeight.Bold)
                    Slider(rate, { rate = it })
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SegmentedOption("Step", false, Modifier.weight(1f)) { steps = (steps + 1).coerceAtMost(100) }
                        SegmentedOption("Train 20", true, Modifier.weight(1f)) { steps = (steps + 20).coerceAtMost(100) }
                        SegmentedOption("Reset", false, Modifier.weight(1f)) { steps = 0 }
                    }
                }
            }
        }
        item { LearningCard("Training process", profile.steps.joinToString(" -> ")) }
    }
}

@Composable
private fun PredictStage(topic: LearnTopic, profile: LearningProfile) {
    var input by remember(topic) { mutableFloatStateOf(.5f) }
    val score = (1f / (1f + exp(-((input - .42f) * 7f)))).coerceIn(0f, 1f)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Predict", "Run inference on a held-out sample") }
        item { AlgorithmCanvas(profile.kind, input, score, 2, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("Sample input %.2f".format(input), color = LabText, fontWeight = FontWeight.Bold)
                    Slider(input, { input = it })
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricPill("Output", "%.3f".format(score), LabCyan, Modifier.weight(1f))
                        MetricPill("Decision", if (score >= .5f) "Positive" else "Negative", if (score >= .5f) LabGreen else LabPink, Modifier.weight(1f))
                    }
                }
            }
        }
        item { LearningCard("Inference check", "Use exactly the preprocessing fitted on training data. A plausible score is not evidence of calibration or robustness.") }
    }
}

@Composable
private fun CompareStage(topic: LearnTopic, profile: LearningProfile) {
    val related = remember(topic) { LearnCatalog.related(topic) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Compare", "Choose a method from evidence, not familiarity") }
        items(related) { candidate ->
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                    Text("${topic.title} vs ${candidate.title}", color = LabText, fontWeight = FontWeight.Bold)
                    Text("Compare validation metric, inference cost, interpretability and sensitivity to data scale on the same split.", color = LabMuted, fontSize = 12.sp)
                }
            }
        }
        item { TripleListCard("When to use", profile.advantages, "When not to use", profile.limitations, "Check first", profile.assumptions) }
    }
}

@Composable
private fun QuizStage(topic: LearnTopic, completed: Boolean, onComplete: () -> Unit) {
    var answer by remember(topic) { mutableIntStateOf(-1) }
    val options = listOf(
        "Match preprocessing, objective and evaluation to the deployment task",
        "Choose the largest model and evaluate on its training set",
        "Tune on the test set until the score improves"
    )
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Knowledge check", "Finish the learning loop") }
        item { LearningCard("Question", "Which workflow gives the most trustworthy evidence for ${topic.title}?") }
        items(options.indices.toList()) { index ->
            SegmentedOption(options[index], answer == index, Modifier.fillMaxWidth()) { answer = index }
        }
        if (answer >= 0) item {
            LearningCard(
                if (answer == 0) "Correct" else "Review this",
                if (answer == 0) "A held-out evaluation with deployment-equivalent preprocessing tests generalization." else "Training or tuning on the test set leaks information and produces an optimistic result."
            )
        }
        if (answer == 0 && !completed) item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
        if (completed) item { LearningCard("Lesson completed", "Progress is stored locally and remains available offline.") }
    }
}

@Composable
private fun LearningCard(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

@Composable
private fun EquationCard(profile: LearningProfile) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            SectionTitle("Mathematical explanation", "Important equation")
            Box(
                Modifier.fillMaxWidth().background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(12.dp)
            ) { Text(profile.equation, color = LabCyan, fontSize = 13.sp, fontWeight = FontWeight.SemiBold) }
            Text(profile.equationNote, color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun TripleListCard(first: String, firstItems: List<String>, second: String, secondItems: List<String>, third: String, thirdItems: List<String>) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            ListBlock(first, firstItems, LabGreen)
            ListBlock(second, secondItems, LabPink)
            ListBlock(third, thirdItems, LabOrange)
        }
    }
}

@Composable
private fun ListBlock(title: String, values: List<String>, accent: Color) {
    Text(title, color = accent, fontWeight = FontWeight.Bold, fontSize = 13.sp)
    values.forEach { Text("- $it", color = LabMuted, fontSize = 12.sp) }
}

@Composable
private fun AlgorithmCanvas(kind: VisualizationKind, primary: Float, secondary: Float, iteration: Int, modifier: Modifier) {
    Canvas(
        modifier.height(250.dp).background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(10.dp)
    ) {
        val points = List(12) { index ->
            Offset(
                size.width * (.08f + .84f * ((index * 37 + iteration * 7) % 101) / 100f),
                size.height * (.10f + .80f * ((index * 61 + 19) % 101) / 100f)
            )
        }
        fun grid() {
            repeat(5) { i ->
                val x = size.width * i / 4f
                val y = size.height * i / 4f
                drawLine(Color.White.copy(alpha = .06f), Offset(x, 0f), Offset(x, size.height))
                drawLine(Color.White.copy(alpha = .06f), Offset(0f, y), Offset(size.width, y))
            }
        }
        grid()
        when (kind) {
            VisualizationKind.Regression, VisualizationKind.Optimization -> {
                points.forEach { drawCircle(LabCyan, 7f, it) }
                val y1 = size.height * (.82f - secondary * .3f)
                val y2 = size.height * (.18f + (1f - primary) * .26f)
                drawLine(LabGreen, Offset(0f, y1), Offset(size.width, y2), 6f, cap = StrokeCap.Round)
                points.forEach { point ->
                    val predicted = y1 + (y2 - y1) * point.x / size.width
                    drawLine(LabPink.copy(alpha = .3f), point, Offset(point.x, predicted), 2f)
                }
            }
            VisualizationKind.Classification -> {
                points.forEachIndexed { index, point ->
                    val label = point.y < size.height * (.72f - point.x / size.width * .45f)
                    drawCircle(if (label) LabCyan else LabPink, 8f, point)
                }
                val boundaryY1 = size.height * (.78f - secondary * .2f)
                val boundaryY2 = size.height * (.22f + (1f - primary) * .2f)
                val margin = 24f + secondary * 34f
                drawLine(LabGreen.copy(alpha = .35f), Offset(0f, boundaryY1 - margin), Offset(size.width, boundaryY2 - margin), 2f)
                drawLine(Color.White, Offset(0f, boundaryY1), Offset(size.width, boundaryY2), 5f, cap = StrokeCap.Round)
                drawLine(LabGreen.copy(alpha = .35f), Offset(0f, boundaryY1 + margin), Offset(size.width, boundaryY2 + margin), 2f)
            }
            VisualizationKind.Clustering, VisualizationKind.Density -> {
                val centers = listOf(Offset(size.width * .25f, size.height * .35f), Offset(size.width * .7f, size.height * .62f), Offset(size.width * .63f, size.height * .22f))
                points.forEachIndexed { index, point -> drawCircle(listOf(LabCyan, LabPink, LabOrange)[index % 3], 8f, point) }
                centers.take(1 + (primary * 3).toInt().coerceAtMost(2)).forEach { center ->
                    drawCircle(LabGreen.copy(alpha = .12f), 35f + secondary * 70f, center)
                    drawCircle(Color.White, 10f, center); drawCircle(LabGreen, 6f, center)
                }
            }
            VisualizationKind.Neighbours -> {
                points.forEachIndexed { index, point -> drawCircle(if (index % 2 == 0) LabCyan else LabPink, 8f, point) }
                val query = Offset(size.width * primary, size.height * secondary)
                points.sortedBy { (it - query).getDistance() }.take(1 + (primary * 6).toInt()).forEach { drawLine(LabGreen.copy(alpha = .55f), query, it, 3f) }
                drawCircle(Color.White, 13f, query); drawCircle(LabGreen, 8f, query)
            }
            VisualizationKind.Tree -> {
                val root = Offset(size.width / 2f, 24f)
                val layer1 = listOf(Offset(size.width * .28f, size.height * .43f), Offset(size.width * .72f, size.height * .43f))
                val leaves = listOf(.12f, .38f, .62f, .88f).map { Offset(size.width * it, size.height * .84f) }
                layer1.forEach { drawLine(LabPurple, root, it, 5f) }
                leaves.forEachIndexed { index, leaf -> drawLine(LabCyan, layer1[index / 2], leaf, 4f) }
                (listOf(root) + layer1 + leaves).forEachIndexed { index, node -> drawCircle(if (index < 3) LabPurple else if (index % 2 == 0) LabGreen else LabPink, 13f, node) }
            }
            VisualizationKind.Projection -> {
                points.forEach { drawCircle(LabCyan, 7f, it) }
                val angle = primary * 3.14f
                val direction = Offset(cos(angle), sin(angle))
                val center = Offset(size.width / 2f, size.height / 2f)
                drawLine(LabOrange, center - direction * size.width, center + direction * size.width, 5f)
                points.forEach { point ->
                    val delta = point - center
                    val projected = center + direction * (delta.x * direction.x + delta.y * direction.y)
                    drawLine(LabMuted.copy(alpha = .35f), point, projected, 2f)
                    drawCircle(LabGreen, 5f, projected)
                }
            }
            VisualizationKind.Attention -> {
                val cells = 6
                val w = size.width / cells; val h = size.height / cells
                repeat(cells * cells) { index ->
                    val row = index / cells; val column = index % cells
                    val value = ((sin((row + 1) * (column + 2f) + primary * 4f) + 1f) / 2f) * (.3f + secondary * .7f)
                    drawRect(LabPurple.copy(alpha = .08f + value * .85f), Offset(column * w, row * h), Size(w - 3f, h - 3f))
                }
            }
            VisualizationKind.Convolution -> {
                val cells = 8; val w = size.width * .52f / cells; val h = size.height / cells
                repeat(cells * cells) { index ->
                    val value = (sin(index * .8f + iteration) + 1f) / 2f
                    drawRect(Color(value, primary, 1f - value, 1f), Offset(index % cells * w, index / cells * h), Size(w - 2f, h - 2f))
                }
                val x = size.width * .57f
                repeat(3) { layer ->
                    val left = x + layer * size.width * .13f
                    drawRect(listOf(LabCyan, LabPurple, LabGreen)[layer].copy(alpha = .35f), Offset(left, 35f + layer * 22f), Size(size.width * .1f, size.height * .55f - layer * 20f))
                }
            }
            VisualizationKind.Reinforcement -> {
                val cells = 4; val w = size.width / cells; val h = size.height / cells
                repeat(cells * cells) { index -> drawRect(LabBorder, Offset(index % cells * w, index / cells * h), Size(w - 2f, h - 2f), style = Stroke(2f)) }
                val path = listOf(0, 1, 5, 9, 10, 11, 15).take((2 + iteration).coerceAtMost(7))
                path.zipWithNext().forEach { (a, b) ->
                    val start = Offset((a % 4 + .5f) * w, (a / 4 + .5f) * h); val end = Offset((b % 4 + .5f) * w, (b / 4 + .5f) * h)
                    drawLine(LabCyan, start, end, 7f, cap = StrokeCap.Round)
                }
                drawCircle(LabGreen, 14f, Offset(3.5f * w, 3.5f * h))
            }
            VisualizationKind.TimeSeries, VisualizationKind.Sequence -> {
                val path = Path()
                repeat(30) { index ->
                    val x = size.width * index / 29f
                    val y = size.height * (.5f - .25f * sin(index * (.35f + primary * .2f)) - index / 120f * secondary)
                    if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
                }
                drawPath(path, LabCyan, style = Stroke(5f, cap = StrokeCap.Round))
            }
            VisualizationKind.NeuralNetwork, VisualizationKind.Autoencoder, VisualizationKind.Generative, VisualizationKind.Graph -> {
                val layers = if (kind == VisualizationKind.Autoencoder) listOf(5, 3, 1, 3, 5) else listOf(4, 6, 3)
                val nodeLayers = layers.mapIndexed { layer, count -> List(count) { node -> Offset(size.width * (layer + 1) / (layers.size + 1f), size.height * (node + 1) / (count + 1f)) } }
                nodeLayers.zipWithNext().forEach { (left, right) -> left.forEach { a -> right.forEach { b -> drawLine(LabPurple.copy(alpha = .16f + secondary * .18f), a, b, 2f) } } }
                nodeLayers.flatten().forEachIndexed { index, node -> drawCircle(if (index % 2 == 0) LabCyan else LabPink, 7f + primary * 4f, node) }
            }
            VisualizationKind.Probability, VisualizationKind.Recommendation, VisualizationKind.Explanation, VisualizationKind.Generic -> {
                points.forEachIndexed { index, point ->
                    val importance = ((sin(index + primary * 5f) + 1f) / 2f)
                    drawLine(LabPurple.copy(alpha = .25f), Offset(size.width / 2f, size.height / 2f), point, 1f + importance * 5f)
                    drawCircle(if (index % 2 == 0) LabCyan else LabOrange, 5f + importance * 7f, point)
                }
            }
        }
    }
}

@Composable
private fun TrainingCurve(steps: Int, rate: Float) {
    Canvas(Modifier.fillMaxWidth().height(190.dp).background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(10.dp)) {
        val path = Path()
        repeat(101) { index ->
            val x = size.width * index / 100f
            val loss = exp(-index * (.025f + rate * .025f))
            val y = size.height * (1f - loss)
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, LabCyan, style = Stroke(5f, cap = StrokeCap.Round))
        val x = size.width * steps / 100f
        val y = size.height * (1f - exp(-steps * (.025f + rate * .025f)))
        drawCircle(LabPurple, 11f, Offset(x, y)); drawCircle(Color.White, 4f, Offset(x, y))
    }
}

private fun visualSubtitle(kind: VisualizationKind) = when (kind) {
    VisualizationKind.Regression -> "Parameter changes update the fitted line and residuals"
    VisualizationKind.Classification -> "Threshold and margin reshape the decision boundary"
    VisualizationKind.Clustering -> "Cluster representatives respond to K and iteration"
    VisualizationKind.Density -> "Radius and density reveal clusters and noise"
    VisualizationKind.Attention -> "Token-to-token weights change by head and scale"
    VisualizationKind.Convolution -> "Kernel responses flow into feature maps"
    VisualizationKind.Reinforcement -> "The policy improves its route through the grid"
    else -> "Controls update this algorithm's structural view"
}

private fun parameterLabel(kind: VisualizationKind, primary: Boolean, value: Float): String = when (kind) {
    VisualizationKind.Clustering -> if (primary) "Clusters ${1 + (value * 3).toInt()}" else "Centroid movement %.2f".format(value)
    VisualizationKind.Density -> if (primary) "MinPts ${2 + (value * 7).toInt()}" else "Epsilon %.2f".format(.1f + value * .9f)
    VisualizationKind.Neighbours -> if (primary) "K ${1 + (value * 8).toInt()}" else "Query Y %.2f".format(value)
    VisualizationKind.Attention -> if (primary) "Head ${1 + (value * 7).toInt()}" else "Attention scale %.2f".format(value)
    VisualizationKind.Optimization, VisualizationKind.Regression -> if (primary) "Learning rate %.3f".format(.001f + value * .199f) else "Regularization %.2f".format(value)
    VisualizationKind.Classification -> if (primary) "Decision threshold %.2f".format(value) else "Margin %.2f".format(value)
    else -> if (primary) "Model parameter %.2f".format(value) else "Data variation %.2f".format(value)
}

private fun observationFor(kind: VisualizationKind, primary: Float, secondary: Float, run: Int): String = when (kind) {
    VisualizationKind.Density -> "A larger epsilon joins more neighbours; MinPts controls how much local evidence is required. Run $run."
    VisualizationKind.Regression -> "The learning rate changes convergence speed while regularization pulls parameters toward simpler values."
    VisualizationKind.Attention -> "Different heads can focus on different token relationships; diffuse weights mix more context."
    else -> "The visible state was recomputed from parameter %.2f, variation %.2f, and iteration %d.".format(primary, secondary, run)
}
