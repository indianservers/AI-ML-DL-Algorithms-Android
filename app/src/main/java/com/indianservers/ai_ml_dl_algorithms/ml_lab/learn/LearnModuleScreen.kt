package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import android.content.Context
import android.webkit.WebView
import android.webkit.WebViewClient
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
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
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
import androidx.compose.ui.viewinterop.AndroidView
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.AlgorithmLessonSeedFactory
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonPageRecord
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonProgressRecord
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonRepository
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.McqOptionRecord
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.McqQuestionRecord
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.QuizAttemptRecord
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
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class LearningStage(val label: String) {
    Understand("Understand"), Visualize("Visualize"), Explore("Explore"), Train("Train"),
    Predict("Predict"), Experiment("Experiment"), Compare("Compare"), Test("Test")
}

private enum class DatasetPresetUi(val label: String) {
    Clean("Clean"), Noisy("Noisy"), Overlap("Overlap"), Outliers("Outliers")
}

private enum class AlgorithmLearningMode(val label: String) {
    Lesson("Lesson"), Lab("Lab")
}

@Composable
fun LearnModuleScreen(depth: LearningDepth) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("learn_module", Context.MODE_PRIVATE) }
    val lessonRepository = remember(context) { LessonRepository.get(context) }
    var selected by remember { mutableStateOf<LearnTopic?>(null) }
    val completed = remember {
        mutableStateListOf<String>().apply { addAll(prefs.getStringSet("completed", emptySet()).orEmpty()) }
    }
    LaunchedEffect(lessonRepository, depth) {
        withContext(Dispatchers.IO) {
            lessonRepository.seedLessonsIfNeeded(depth)
        }
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
    var mode by remember(topic) { mutableStateOf(AlgorithmLearningMode.Lesson) }

    when (mode) {
        AlgorithmLearningMode.Lesson -> HtmlLessonReaderScreen(
            topic = topic,
            depth = depth,
            completed = completed,
            onBack = onBack,
            onOpenLab = { mode = AlgorithmLearningMode.Lab },
            onComplete = onComplete
        )
        AlgorithmLearningMode.Lab -> AlgorithmLabScreen(
            topic = topic,
            depth = depth,
            completed = completed,
            onBack = { mode = AlgorithmLearningMode.Lesson },
            onComplete = onComplete
        )
    }
}

@Composable
private fun AlgorithmLabScreen(
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
            LearningStage.Visualize -> if (topic.isKnnRegression()) KnnRegressionStage(topic, "Visualize KNN Regression") else InteractiveStage(topic, profile, "Interactive visualization", false)
            LearningStage.Explore -> if (topic.isKnnRegression()) KnnRegressionStage(topic, "Explore KNN Regression") else ExploreStage(topic, profile)
            LearningStage.Train -> if (topic.isKnnRegression()) KnnLazyTrainingStage(profile) else TrainStage(profile)
            LearningStage.Predict -> if (topic.isKnnRegression()) KnnRegressionStage(topic, "Predict with KNN Regression") else PredictStage(topic, profile)
            LearningStage.Experiment -> InteractiveStage(topic, profile, "Mini-lab", true)
            LearningStage.Compare -> CompareStage(topic, profile)
            LearningStage.Test -> QuizStage(topic, completed, onComplete)
        }
    }
}

@Composable
private fun HtmlLessonReaderScreen(
    topic: LearnTopic,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onOpenLab: () -> Unit,
    onComplete: () -> Unit
) {
    val context = LocalContext.current
    val repository = remember(context) { LessonRepository.get(context) }
    val scope = rememberCoroutineScope()
    var pages by remember(topic) { mutableStateOf<List<LessonPageRecord>>(emptyList()) }
    var questions by remember(topic) { mutableStateOf<List<Pair<McqQuestionRecord, List<McqOptionRecord>>>>(emptyList()) }
    var pageIndex by remember(topic) { mutableIntStateOf(0) }
    var showHtml by remember(topic) { mutableStateOf(false) }
    var showQuiz by remember(topic) { mutableStateOf(false) }
    var bestScoreText by remember(topic) { mutableStateOf("No score yet") }
    var submittedAttempt by remember(topic) { mutableStateOf<QuizAttemptRecord?>(null) }
    val selectedOptionIds = remember(topic) { mutableStateMapOf<Long, Long>() }
    val fallbackPages = remember(topic, depth) { AlgorithmLessonSeedFactory.build(topic, depth).pages }

    LaunchedEffect(topic, depth) {
        val loaded = withContext(Dispatchers.IO) {
            repository.seedLessonsIfNeeded(depth)
            val storedPages = repository.pagesFor(topic.id)
            val storedQuestions = repository.questionsFor(topic.id)
            val progress = repository.progressFor(topic.id)
            Triple(storedPages, storedQuestions, progress)
        }
        pages = loaded.first.ifEmpty { fallbackPages }
        questions = loaded.second
        pageIndex = ((loaded.third?.lastPageNumber ?: 1) - 1).coerceIn(0, 4)
        bestScoreText = loaded.third?.bestScoreLabel() ?: "No score yet"
    }

    val activePages = pages.ifEmpty { fallbackPages }
    val page = activePages.getOrNull(pageIndex) ?: return

    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 10.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("${topic.domain} / ${topic.section}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "HTML Lesson", color = if (completed) LabGreen else LabCyan, fontSize = 11.sp)
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        SectionTitle("5-Page Lesson", "Storytelling, realtime examples, applications and expert tips")
                        Text("${pageIndex + 1}/${activePages.size}", color = LabGreen, fontWeight = FontWeight.Bold)
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        activePages.forEachIndexed { index, lessonPage ->
                            Box(
                                Modifier
                                    .weight(1f)
                                    .height(8.dp)
                                    .background(
                                        if (index <= pageIndex) Color(topic.accent) else LabPanelSoft,
                                        RoundedCornerShape(4.dp)
                                    )
                                    .clickable {
                                        pageIndex = index
                                        repository.updateProgress(topic.id, lessonPage.pageNumber)
                                    }
                            )
                        }
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SegmentedOption("Lesson", !showQuiz, Modifier.weight(1f)) { showQuiz = false }
                        SegmentedOption("Quiz", showQuiz, Modifier.weight(1f)) { showQuiz = true }
                        SegmentedOption("Open Lab", false, Modifier.weight(1f), onOpenLab)
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricPill("Questions", questions.size.toString(), LabCyan, Modifier.weight(1f))
                        MetricPill("Best Score", bestScoreText, LabGreen, Modifier.weight(1f))
                    }
                }
            }
        }
        if (showQuiz) {
            item {
                LessonQuizScreen(
                    topic = topic,
                    questions = questions,
                    selectedOptionIds = selectedOptionIds,
                    submittedAttempt = submittedAttempt,
                    onSelect = { questionId, optionId ->
                        if (submittedAttempt == null) selectedOptionIds[questionId] = optionId
                    },
                    onSubmit = {
                        scope.launch {
                            val attempt = withContext(Dispatchers.IO) {
                                repository.recordQuizAttempt(topic.id, selectedOptionIds.toMap())
                            }
                            submittedAttempt = attempt
                            bestScoreText = "${attempt.score}/${attempt.totalQuestions}"
                        }
                    },
                    onRetry = {
                        selectedOptionIds.clear()
                        submittedAttempt = null
                    }
                )
            }
        } else {
            item { HtmlLessonDocument(page, topic) }
            item { HtmlLessonStudyNotes(page, topic) }
            item {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SegmentedOption("Previous", false, Modifier.weight(1f)) {
                        pageIndex = (pageIndex - 1).coerceAtLeast(0)
                        repository.updateProgress(topic.id, activePages[pageIndex].pageNumber)
                    }
                    GradientButton(
                        if (pageIndex == activePages.lastIndex) "Complete Lesson" else "Next Page",
                        Modifier.weight(1f)
                    ) {
                        if (pageIndex == activePages.lastIndex) {
                            repository.updateProgress(topic.id, activePages.last().pageNumber, completed = true)
                            onComplete()
                            showQuiz = true
                        } else {
                            pageIndex += 1
                            repository.updateProgress(topic.id, activePages[pageIndex].pageNumber)
                        }
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        SectionTitle("HTML Source", "Stored in SQLite for styled rendering and future export")
                        SegmentedOption(if (showHtml) "Hide" else "Show", showHtml) { showHtml = !showHtml }
                    }
                    if (showHtml) {
                        Text(page.htmlContent.compactHtml(), color = LabMuted, fontSize = 10.sp, lineHeight = 14.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun LessonQuizScreen(
    topic: LearnTopic,
    questions: List<Pair<McqQuestionRecord, List<McqOptionRecord>>>,
    selectedOptionIds: Map<Long, Long>,
    submittedAttempt: QuizAttemptRecord?,
    onSelect: (Long, Long) -> Unit,
    onSubmit: () -> Unit,
    onRetry: () -> Unit
) {
    if (questions.isEmpty()) {
        LearningCard("Quiz is preparing", "The local SQLite lesson database is loading questions for ${topic.title}.")
        return
    }

    val allAnswered = questions.all { (question, _) -> selectedOptionIds.containsKey(question.id) }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                SectionTitle("MCQ Score Test", "Answer all questions, submit, then review explanations")
                Text(submittedAttempt?.scoreLabel() ?: "${selectedOptionIds.size}/${questions.size}", color = LabGreen, fontWeight = FontWeight.Bold)
            }
            submittedAttempt?.let { attempt ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    MetricPill("Score", "${attempt.score}/${attempt.totalQuestions}", LabGreen, Modifier.weight(1f))
                    MetricPill("Percent", "%.0f%%".format(attempt.percentage), LabCyan, Modifier.weight(1f))
                    MetricPill("Result", if (attempt.percentage >= 80.0) "Mastered" else "Practice", LabOrange, Modifier.weight(1f))
                }
            }

            questions.forEach { (question, options) ->
                QuizQuestionCard(
                    question = question,
                    options = options,
                    selectedOptionId = selectedOptionIds[question.id],
                    submitted = submittedAttempt != null,
                    onSelect = { onSelect(question.id, it) }
                )
            }

            if (submittedAttempt == null) {
                if (!allAnswered) {
                    Text("Answer ${questions.size - selectedOptionIds.size} more question(s) to unlock scoring.", color = LabMuted, fontSize = 12.sp)
                }
                GradientButton("Submit MCQ Test", Modifier.fillMaxWidth()) {
                    if (allAnswered) onSubmit()
                }
            } else {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SegmentedOption("Retry", false, Modifier.weight(1f), onRetry)
                    SegmentedOption("Review Done", true, Modifier.weight(1f)) {}
                }
            }
        }
    }
}

@Composable
private fun QuizQuestionCard(
    question: McqQuestionRecord,
    options: List<McqOptionRecord>,
    selectedOptionId: Long?,
    submitted: Boolean,
    onSelect: (Long) -> Unit
) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(LabPanelSoft.copy(alpha = .64f), RoundedCornerShape(8.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(8.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(9.dp)
    ) {
        Text("Q${question.questionNumber}. ${question.question}", color = LabText, fontWeight = FontWeight.Bold, fontSize = 14.sp, lineHeight = 19.sp)
        options.forEach { option ->
            val selected = selectedOptionId == option.id
            val accent = when {
                submitted && option.isCorrect -> LabGreen
                submitted && selected && !option.isCorrect -> LabPink
                selected -> LabCyan
                else -> LabBorder
            }
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(accent.copy(alpha = if (selected || option.isCorrect && submitted) .16f else .08f), RoundedCornerShape(8.dp))
                    .border(1.dp, accent.copy(alpha = .45f), RoundedCornerShape(8.dp))
                    .clickable(enabled = !submitted) { onSelect(option.id) }
                    .padding(10.dp),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    Modifier.size(24.dp).background(accent.copy(alpha = .20f), RoundedCornerShape(7.dp)),
                    contentAlignment = Alignment.Center
                ) {
                    Text(option.optionNumber.toOptionLetter(), color = accent, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                }
                Text(option.optionText, color = if (selected || option.isCorrect && submitted) LabText else LabMuted, fontSize = 13.sp, lineHeight = 18.sp, modifier = Modifier.weight(1f))
            }
        }
        if (submitted) {
            Text("Explanation: ${question.explanation}", color = LabMuted, fontSize = 12.sp, lineHeight = 17.sp)
        }
    }
}

@Composable
private fun HtmlLessonDocument(page: LessonPageRecord, topic: LearnTopic) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(720.dp)
            .background(Color(0xFF071126), RoundedCornerShape(8.dp))
            .border(1.dp, Color(topic.accent).copy(alpha = .35f), RoundedCornerShape(8.dp))
    ) {
        AndroidView(
            modifier = Modifier.fillMaxSize(),
            factory = { context ->
                WebView(context).apply {
                    webViewClient = WebViewClient()
                    setBackgroundColor(android.graphics.Color.TRANSPARENT)
                    isVerticalScrollBarEnabled = false
                    settings.javaScriptEnabled = false
                    settings.domStorageEnabled = false
                    settings.loadsImagesAutomatically = false
                    settings.builtInZoomControls = false
                    settings.displayZoomControls = false
                    settings.textZoom = 100
                }
            },
            update = { webView ->
                webView.loadDataWithBaseURL(
                    null,
                    page.asHtmlDocument(topic),
                    "text/html",
                    "UTF-8",
                    null
                )
            }
        )
    }
}

@Composable
private fun HtmlLessonStudyNotes(page: LessonPageRecord, topic: LearnTopic) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SectionTitle("Study Notes", "Structured fields saved beside the HTML")
            LessonHtmlSection("story", "Story", page.story, LabCyan)
            LessonHtmlSection("simple", "Simple Explanation", page.explanation, LabGreen)
            LessonHtmlSection("teacher-tip", "ML Expert Teacher Tip", page.teacherTip, LabPink)
        }
    }
}

@Composable
private fun LessonHtmlSection(tag: String, title: String, body: String, accent: Color) {
    Column(
        Modifier
            .fillMaxWidth()
            .background(accent.copy(alpha = .09f), RoundedCornerShape(8.dp))
            .border(1.dp, accent.copy(alpha = .24f), RoundedCornerShape(8.dp))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Text(tag, color = accent, fontSize = 10.sp, fontWeight = FontWeight.Bold)
        Text(title, color = LabText, fontSize = 15.sp, fontWeight = FontWeight.Bold)
        Text(body, color = LabMuted, fontSize = 13.sp, lineHeight = 19.sp)
    }
}

private fun String.compactHtml(): String = lines()
    .map { it.trim() }
    .filter { it.isNotBlank() }
    .take(24)
    .joinToString("\n")

private fun QuizAttemptRecord.scoreLabel(): String = "${score}/${totalQuestions}"

private fun LessonProgressRecord.bestScoreLabel(): String =
    if (bestScoreTotal <= 0) "No score yet" else "$bestScore/$bestScoreTotal"

private fun Int.toOptionLetter(): String = when (this) {
    1 -> "A"
    2 -> "B"
    3 -> "C"
    4 -> "D"
    else -> toString()
}

private fun LessonPageRecord.asHtmlDocument(topic: LearnTopic): String {
    val accent = topic.accent.toCssColor()
    return """
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              :root {
                color-scheme: dark;
                --bg: #071126;
                --panel: #111a31;
                --soft: #17213b;
                --text: #f5f7ff;
                --muted: #a9b3c8;
                --border: #2a365a;
                --accent: $accent;
                --green: #35e58f;
                --orange: #ffa52e;
                --pink: #ff48be;
                --cyan: #20d9e8;
              }
              * { box-sizing: border-box; }
              html, body {
                margin: 0;
                min-height: 100%;
                background: radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 20%, transparent), transparent 34%),
                            linear-gradient(180deg, #071126 0%, #081126 100%);
                color: var(--text);
                font-family: sans-serif;
              }
              body { padding: 16px; }
              article.lesson-page {
                border: 1px solid color-mix(in srgb, var(--accent) 38%, var(--border));
                border-radius: 12px;
                background: rgba(17, 26, 49, .86);
                padding: 18px;
              }
              header {
                border-bottom: 1px solid rgba(255,255,255,.08);
                padding-bottom: 14px;
                margin-bottom: 14px;
              }
              .eyebrow {
                color: var(--accent);
                font-size: 12px;
                font-weight: 800;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin: 0 0 8px;
              }
              h1 {
                margin: 0;
                font-size: 28px;
                line-height: 1.12;
              }
              h2 {
                margin: 0 0 8px;
                font-size: 16px;
                color: var(--text);
              }
              p {
                margin: 0;
                color: var(--muted);
                font-size: 15px;
                line-height: 1.55;
              }
              section, aside.teacher-tip {
                border-radius: 10px;
                margin-top: 12px;
                padding: 14px;
                border: 1px solid rgba(255,255,255,.08);
              }
              section.story { background: rgba(32, 217, 232, .10); }
              section.simple { background: rgba(53, 229, 143, .10); }
              section.realtime { background: rgba(255, 165, 46, .10); }
              section.applications { background: rgba(155, 61, 255, .11); }
              aside.teacher-tip {
                background: rgba(255, 72, 190, .11);
                color: var(--muted);
                font-size: 15px;
                line-height: 1.55;
              }
              aside.teacher-tip strong {
                display: block;
                color: var(--pink);
                margin-bottom: 6px;
              }
            </style>
          </head>
          <body>
            ${htmlContent}
          </body>
        </html>
    """.trimIndent()
}

private fun Long.toCssColor(): String = "#%06X".format(this and 0xFFFFFF)

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
    var preset by remember(topic) { mutableStateOf(DatasetPresetUi.Clean) }
    var samples by remember(topic) { mutableFloatStateOf(.45f) }
    var noise by remember(topic) { mutableFloatStateOf(.18f) }
    var speed by remember(topic) { mutableFloatStateOf(.42f) }
    var compare by remember(topic) { mutableStateOf(false) }
    var trainSplit by remember(topic) { mutableStateOf(true) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle(algorithmVisualTitle(topic, title), visualSubtitle(profile.kind)) }
        item { VisualLegend(profile.kind) }
        item {
            AlgorithmCanvas(
                kind = profile.kind,
                primary = primary,
                secondary = secondary,
                iteration = run,
                modifier = Modifier.fillMaxWidth(),
                noise = noise,
                samples = samples,
                compare = compare,
                preset = preset
            )
        }
        if (compare) item { LearningCard("Before / after", compareExplanation(profile.kind, primary, secondary)) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text(parameterLabel(profile.kind, true, primary), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(primary, { primary = it })
                    Text(parameterLabel(profile.kind, false, secondary), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(secondary, { secondary = it })
                    Text("Samples ${12 + (samples * 188).toInt()}", color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(samples, { samples = it })
                    Text("Noise / overlap %.2f".format(noise), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(noise, { noise = it })
                    Text("Animation speed %.2fx".format(.25f + speed * 2.75f), color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Slider(speed, { speed = it })
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        DatasetPresetUi.entries.forEach { option ->
                            SegmentedOption(option.label, preset == option) { preset = option; run++ }
                        }
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SegmentedOption("Train/test", trainSplit, Modifier.weight(1f)) { trainSplit = !trainSplit }
                        SegmentedOption("Compare", compare, Modifier.weight(1f)) { compare = !compare }
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        GradientButton(if (experiment) "Run sample experiment" else "Step animation", Modifier.weight(1f)) { run++ }
                        SegmentedOption("Reset", false, Modifier.weight(1f)) {
                            primary = if (experiment) .72f else .42f
                            secondary = if (experiment) .28f else .58f
                            samples = .45f
                            noise = .18f
                            speed = .42f
                            preset = DatasetPresetUi.Clean
                            compare = false
                            trainSplit = true
                            run = 0
                        }
                    }
                }
            }
        }
        item { MetricReadout(profile.kind, primary, secondary, samples, noise, trainSplit) }
        item { LearningCard("Equation link", equationFocus(profile.kind, primary)) }
        item { LearningCard("Example dataset", "${profile.sampleData} Preset: ${preset.label}, sample count: ${12 + (samples * 188).toInt()}.") }
        item { LearningCard("Observation", observationFor(profile.kind, primary, secondary, run)) }
        algorithmWarning(profile.kind, primary, secondary, noise)?.let { item { LearningCard("Setting warning", it) } }
        item { LearningCard("Share result", "Current lab state is ready for screenshot sharing from the device controls.") }
    }
}

@Composable
private fun KnnRegressionStage(topic: LearnTopic, title: String) {
    var kValue by remember(topic) { mutableFloatStateOf(3f) }
    var queryX by remember(topic) { mutableFloatStateOf(5.6f) }
    var weighted by remember(topic) { mutableStateOf(false) }
    val k = kValue.toInt().coerceIn(1, 15)
    val prediction = remember(k, queryX, weighted) { knnRegressionPrediction(queryX.toDouble(), k, weighted) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle(title, "Choose a query x, find the K nearest samples, then average their y values") }
        item { KnnRegressionCanvas(queryX, k, weighted, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("K - Number of Neighbors: $k", color = LabText, fontWeight = FontWeight.Bold)
                    Slider(kValue, { kValue = it.coerceIn(1f, 15f) }, valueRange = 1f..15f, steps = 13)
                    Text("Query x %.2f".format(queryX), color = LabText, fontWeight = FontWeight.Bold)
                    Slider(queryX, { queryX = it.coerceIn(0.6f, 10.2f) }, valueRange = .6f..10.2f)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SegmentedOption("Uniform", !weighted, Modifier.weight(1f)) { weighted = false }
                        SegmentedOption("Distance weighted", weighted, Modifier.weight(1f)) { weighted = true }
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    SectionTitle("How was this prediction made?", "Nearest neighbours at query x = %.2f".format(queryX))
                    prediction.neighbors.forEach { point ->
                        Text("x %.2f -> y %.2f, distance %.2f".format(point.x, point.y, abs(point.x - queryX)), color = LabMuted, fontSize = 12.sp)
                    }
                    Text(prediction.formula, color = LabGreen, fontWeight = FontWeight.SemiBold, fontSize = 13.sp)
                    Text("Predicted value y_hat = %.2f".format(prediction.value), color = LabGreen, fontWeight = FontWeight.Bold)
                }
            }
        }
        item { LearningCard("KNN regression", "KNN does not fit a global line or train by gradient descent. It stores the samples and predicts from nearby target values at inference time.") }
    }
}

@Composable
private fun KnnLazyTrainingStage(profile: LearningProfile) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Train", "KNN stores examples instead of optimizing weights") }
        item { KnnRegressionCanvas(5.6f, 3, false, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    MetricPill("Fit step", "Store scaled samples", LabPurple, Modifier.fillMaxWidth())
                    MetricPill("Learned weights", "None", LabCyan, Modifier.fillMaxWidth())
                    MetricPill("Prediction rule", "Average nearest y values", LabGreen, Modifier.fillMaxWidth())
                }
            }
        }
        item { LearningCard("Training process", profile.steps.joinToString(" -> ")) }
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
    var showDetails by remember(topic) { mutableStateOf(true) }
    val score = (1f / (1f + exp(-((input - .42f) * 7f)))).coerceIn(0f, 1f)
    val uncertainty = 1f - abs(score - .5f) * 2f
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Predict", "Run inference on a held-out sample") }
        item { VisualLegend(profile.kind) }
        item { AlgorithmCanvas(profile.kind, input, score, 2, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("Sample input %.2f".format(input), color = LabText, fontWeight = FontWeight.Bold)
                    Slider(input, { input = it })
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        MetricPill("Output", "%.3f".format(score), LabCyan, Modifier.weight(1f))
                        MetricPill("Decision", if (score >= .5f) "Positive" else "Negative", if (score >= .5f) LabGreen else LabPink, Modifier.weight(1f))
                        MetricPill("Uncertainty", "%.0f%%".format(uncertainty * 100f), LabOrange, Modifier.weight(1f))
                    }
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        SegmentedOption("Details", showDetails, Modifier.weight(1f)) { showDetails = !showDetails }
                        SegmentedOption("Reset", false, Modifier.weight(1f)) { input = .5f }
                    }
                }
            }
        }
        if (showDetails) item { LearningCard("Prediction breakdown", predictionBreakdown(profile.kind, score, uncertainty)) }
        if (uncertainty > .72f) item { LearningCard("Setting warning", "This sample sits near a boundary or ambiguous region. Inspect calibration and nearby training examples.") }
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
private fun VisualLegend(kind: VisualizationKind) {
    val items = legendItems(kind)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Legend", color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            items.chunked(2).forEach { row ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    row.forEach { item ->
                        Row(
                            Modifier.weight(1f),
                            horizontalArrangement = Arrangement.spacedBy(7.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Box(Modifier.size(10.dp).background(item.second, RoundedCornerShape(5.dp)))
                            Text(item.first, color = LabMuted, fontSize = 11.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        }
                    }
                    if (row.size == 1) Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}

@Composable
private fun MetricReadout(
    kind: VisualizationKind,
    primary: Float,
    secondary: Float,
    samples: Float,
    noise: Float,
    trainSplit: Boolean
) {
    val metrics = metricsFor(kind, primary, secondary, samples, noise, trainSplit)
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        metrics.forEach { metric ->
            MetricPill(metric.label, metric.value, metric.color, Modifier.weight(1f))
        }
    }
}

private data class ReadoutMetric(val label: String, val value: String, val color: Color)

private fun legendItems(kind: VisualizationKind): List<Pair<String, Color>> = when (kind) {
    VisualizationKind.Regression -> listOf("Training points" to LabCyan, "Prediction curve" to LabGreen, "Residuals" to LabPink, "Test split" to LabOrange)
    VisualizationKind.Classification -> listOf("Class A" to LabCyan, "Class B" to LabPink, "Boundary" to LabText, "Margin/confidence" to LabGreen)
    VisualizationKind.Neighbours -> listOf("Samples" to LabCyan, "Query" to LabGreen, "Nearest links" to LabPurple, "Vote/average" to LabOrange)
    VisualizationKind.Tree -> listOf("Root/split" to LabPurple, "Branches" to LabCyan, "Leaves" to LabGreen, "Impurity" to LabOrange)
    VisualizationKind.Clustering, VisualizationKind.Density -> listOf("Cluster samples" to LabCyan, "Other groups" to LabPink, "Centroid/radius" to LabGreen, "Noise/outliers" to LabOrange)
    VisualizationKind.Projection -> listOf("Original samples" to LabCyan, "Projection axis" to LabOrange, "Projected points" to LabGreen, "Reconstruction" to LabMuted)
    VisualizationKind.Attention -> listOf("Low weight" to LabPanelSoft, "High weight" to LabPurple, "Query token" to LabCyan, "Context mix" to LabGreen)
    VisualizationKind.Convolution -> listOf("Input pixels" to LabCyan, "Kernel window" to LabPurple, "Feature maps" to LabGreen, "Activation" to LabOrange)
    VisualizationKind.Sequence, VisualizationKind.TimeSeries -> listOf("Observed values" to LabCyan, "State/trend" to LabGreen, "Forecast" to LabOrange, "Uncertainty" to LabPurple)
    VisualizationKind.Reinforcement -> listOf("State grid" to LabBorder, "Policy path" to LabCyan, "Goal/reward" to LabGreen, "Penalty" to LabPink)
    else -> listOf("Inputs" to LabCyan, "Model structure" to LabPurple, "Output signal" to LabGreen, "Variation" to LabOrange)
}

private fun metricsFor(kind: VisualizationKind, primary: Float, secondary: Float, samples: Float, noise: Float, trainSplit: Boolean): List<ReadoutMetric> {
    val sampleCount = 12 + (samples * 188).toInt()
    val stability = (1f - noise * .72f).coerceIn(.05f, .99f)
    val split = if (trainSplit) "80/20" else "All"
    return when (kind) {
        VisualizationKind.Regression -> listOf(
            ReadoutMetric("MSE", "%.3f".format((1f - stability) + secondary * .12f), LabCyan),
            ReadoutMetric("R2", "%.2f".format((stability - primary * .08f).coerceIn(0f, .99f)), LabGreen),
            ReadoutMetric("Split", split, LabOrange)
        )
        VisualizationKind.Classification -> listOf(
            ReadoutMetric("Accuracy", "%.1f%%".format((stability * 100f).coerceIn(40f, 99f)), LabGreen),
            ReadoutMetric("F1", "%.2f".format((stability - abs(primary - .5f) * .2f).coerceIn(0f, .99f)), LabCyan),
            ReadoutMetric("Samples", sampleCount.toString(), LabPurple)
        )
        VisualizationKind.Clustering, VisualizationKind.Density -> listOf(
            ReadoutMetric("Groups", "${1 + (primary * 5).toInt()}", LabPurple),
            ReadoutMetric("Noise", "%.2f".format(noise), LabOrange),
            ReadoutMetric("Stability", "%.0f%%".format(stability * 100f), LabGreen)
        )
        VisualizationKind.Projection -> listOf(
            ReadoutMetric("Variance", "%.0f%%".format((52f + primary * 43f).coerceIn(0f, 99f)), LabGreen),
            ReadoutMetric("Error", "%.2f".format((1f - primary) * (.3f + noise)), LabPink),
            ReadoutMetric("Samples", sampleCount.toString(), LabCyan)
        )
        else -> listOf(
            ReadoutMetric("Signal", "%.0f%%".format(stability * 100f), LabGreen),
            ReadoutMetric("Control", "%.2f".format(primary), LabCyan),
            ReadoutMetric("Samples", sampleCount.toString(), LabPurple)
        )
    }
}

private data class KnnSample(val x: Double, val y: Double)
private data class KnnPrediction(val value: Double, val neighbors: List<KnnSample>, val formula: String)

private val knnRegressionSamples = listOf(
    KnnSample(.65, .1), KnnSample(1.05, 1.8), KnnSample(1.22, .8), KnnSample(1.62, 2.6),
    KnnSample(2.05, 3.3), KnnSample(2.42, 3.1), KnnSample(2.75, 4.2), KnnSample(3.1, 4.8),
    KnnSample(3.48, 2.6), KnnSample(3.82, 6.3), KnnSample(4.15, 5.4), KnnSample(5.1, 4.8),
    KnnSample(5.8, 5.6), KnnSample(6.2, 6.1), KnnSample(6.85, 7.4), KnnSample(7.35, 8.1),
    KnnSample(7.68, 6.3), KnnSample(8.05, 8.5), KnnSample(8.5, 6.4), KnnSample(9.05, 5.1),
    KnnSample(9.55, 5.5), KnnSample(10.15, 3.2)
)

private fun LearnTopic.isKnnRegression(): Boolean =
    title == "K-Nearest Neighbors Regression"

private fun knnRegressionPrediction(queryX: Double, k: Int, weighted: Boolean): KnnPrediction {
    val neighbors = knnRegressionSamples.sortedBy { abs(it.x - queryX) }.take(k.coerceIn(1, knnRegressionSamples.size))
    val value = if (weighted) {
        val weights = neighbors.map { 1.0 / abs(it.x - queryX).coerceAtLeast(.05) }
        neighbors.zip(weights).sumOf { (point, weight) -> point.y * weight } / weights.sum()
    } else {
        neighbors.sumOf { it.y } / neighbors.size
    }
    val formula = if (weighted) {
        "Prediction = distance-weighted average of the nearest y values"
    } else {
        "Prediction = (${neighbors.joinToString(" + ") { "%.2f".format(it.y) }}) / ${neighbors.size}"
    }
    return KnnPrediction(value, neighbors, formula)
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
private fun KnnRegressionCanvas(queryX: Float, k: Int, weighted: Boolean, modifier: Modifier) {
    val prediction = remember(queryX, k, weighted) { knnRegressionPrediction(queryX.toDouble(), k, weighted) }
    Canvas(
        modifier.height(340.dp).background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(14.dp)
    ) {
        fun xToCanvas(x: Double) = (size.width * ((x / 10.5).coerceIn(0.0, 1.0))).toFloat()
        fun yToCanvas(y: Double) = (size.height * (1.0 - ((y + 2.0) / 12.0).coerceIn(0.0, 1.0))).toFloat()
        fun point(sample: KnnSample) = Offset(xToCanvas(sample.x), yToCanvas(sample.y))

        repeat(6) { i ->
            val x = size.width * i / 5f
            val y = size.height * i / 5f
            drawLine(Color.White.copy(alpha = .06f), Offset(x, 0f), Offset(x, size.height))
            drawLine(Color.White.copy(alpha = .06f), Offset(0f, y), Offset(size.width, y))
        }

        val curve = Path()
        repeat(96) { index ->
            val x = 10.5 * index / 95.0
            val y = knnRegressionPrediction(x, k, weighted).value
            val canvasX = xToCanvas(x)
            val canvasY = yToCanvas(y)
            if (index == 0) curve.moveTo(canvasX, canvasY) else curve.lineTo(canvasX, canvasY)
        }
        drawPath(curve, LabGreen, style = Stroke(5f, cap = StrokeCap.Round))

        val queryPoint = Offset(xToCanvas(queryX.toDouble()), yToCanvas(prediction.value))
        drawLine(Color.White.copy(alpha = .4f), Offset(queryPoint.x, size.height), queryPoint, 2f)
        drawLine(LabGreen.copy(alpha = .45f), Offset(0f, queryPoint.y), queryPoint, 2f)

        knnRegressionSamples.forEach { sample ->
            drawCircle(LabCyan, 7f, point(sample))
        }
        prediction.neighbors.forEach { sample ->
            val neighbor = point(sample)
            drawLine(LabPurple.copy(alpha = .72f), queryPoint, neighbor, 3f)
            drawCircle(LabPurple, 11f, neighbor)
            drawCircle(LabCyan, 5f, neighbor)
        }
        drawCircle(Color.White, 14f, queryPoint)
        drawCircle(LabPurple, 10f, queryPoint)
        drawCircle(LabGreen, 7f, Offset(queryPoint.x, queryPoint.y))
    }
}

@Composable
private fun AlgorithmCanvas(
    kind: VisualizationKind,
    primary: Float,
    secondary: Float,
    iteration: Int,
    modifier: Modifier,
    noise: Float = .18f,
    samples: Float = .45f,
    compare: Boolean = false,
    preset: DatasetPresetUi = DatasetPresetUi.Clean
) {
    Canvas(
        modifier.height(250.dp).background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(10.dp)
    ) {
        val count = (12 + samples * 40).toInt().coerceIn(12, 52)
        val presetShift = when (preset) {
            DatasetPresetUi.Clean -> 0
            DatasetPresetUi.Noisy -> 17
            DatasetPresetUi.Overlap -> 31
            DatasetPresetUi.Outliers -> 47
        }
        val points = List(count) { index ->
            val outlier = preset == DatasetPresetUi.Outliers && index % 13 == 0
            val overlap = if (preset == DatasetPresetUi.Overlap) .12f else 0f
            val jitterX = (((index * 19 + iteration * 11 + presetShift) % 21) - 10) / 10f * noise * .12f
            val jitterY = (((index * 23 + presetShift) % 21) - 10) / 10f * noise * .14f
            Offset(
                size.width * (.08f + .84f * ((index * 37 + iteration * 7 + presetShift) % 101) / 100f + jitterX).coerceIn(.03f, .97f),
                size.height * (.10f + .80f * ((index * 61 + 19 + presetShift) % 101) / 100f + jitterY + overlap + if (outlier) .25f else 0f).coerceIn(.03f, .97f)
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
                if (compare) drawLine(LabMuted.copy(alpha = .55f), Offset(0f, size.height * .74f), Offset(size.width, size.height * .28f), 3f, cap = StrokeCap.Round)
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
                if (compare) drawLine(LabMuted.copy(alpha = .45f), Offset(0f, size.height * .65f), Offset(size.width, size.height * .35f), 3f)
                drawLine(LabGreen.copy(alpha = .35f), Offset(0f, boundaryY1 - margin), Offset(size.width, boundaryY2 - margin), 2f)
                drawLine(Color.White, Offset(0f, boundaryY1), Offset(size.width, boundaryY2), 5f, cap = StrokeCap.Round)
                drawLine(LabGreen.copy(alpha = .35f), Offset(0f, boundaryY1 + margin), Offset(size.width, boundaryY2 + margin), 2f)
            }
            VisualizationKind.Clustering, VisualizationKind.Density -> {
                val centers = listOf(Offset(size.width * .25f, size.height * .35f), Offset(size.width * .7f, size.height * .62f), Offset(size.width * .63f, size.height * .22f))
                points.forEachIndexed { index, point -> drawCircle(listOf(LabCyan, LabPink, LabOrange)[index % 3], 8f, point) }
                centers.take(1 + (primary * 3).toInt().coerceAtMost(2)).forEach { center ->
                    drawCircle(LabGreen.copy(alpha = .12f), 35f + secondary * 70f + noise * 45f, center)
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
    VisualizationKind.Neighbours -> "Query points, K, and distance define the local prediction"
    VisualizationKind.Tree -> "Splits, paths, leaves, and impurity explain the decision"
    VisualizationKind.Clustering -> "Cluster representatives respond to K and iteration"
    VisualizationKind.Density -> "Radius and density reveal clusters and noise"
    VisualizationKind.Projection -> "Projection axes preserve structure while reducing dimensions"
    VisualizationKind.Attention -> "Token-to-token weights change by head and scale"
    VisualizationKind.Convolution -> "Kernel responses flow into feature maps"
    VisualizationKind.Sequence -> "Hidden state carries context through time"
    VisualizationKind.Reinforcement -> "The policy improves its route through the grid"
    VisualizationKind.TimeSeries -> "Trend, seasonality, and recent history shape the forecast"
    VisualizationKind.Probability -> "Priors, likelihoods, and evidence update uncertainty"
    VisualizationKind.Recommendation -> "User-item signals produce ranked candidates"
    VisualizationKind.Explanation -> "Feature or sample influence is highlighted around the prediction"
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
    VisualizationKind.Classification -> "Moving the threshold or margin changes false positives and false negatives before it changes accuracy."
    VisualizationKind.Neighbours -> "A smaller K follows local detail; a larger K smooths the prediction. Run $run."
    VisualizationKind.Tree -> "Deeper split paths reduce impurity locally but can overfit noisy pockets."
    VisualizationKind.Clustering -> "Changing K or density settings can merge natural groups or split one group into fragments."
    VisualizationKind.Projection -> "A good projection keeps neighbours close while discarding low-value variation."
    VisualizationKind.Attention -> "Different heads can focus on different token relationships; diffuse weights mix more context."
    VisualizationKind.Convolution -> "Kernel size and stride change which local image features survive into the next map."
    VisualizationKind.Sequence -> "The state trace shows how earlier context can influence a later output."
    VisualizationKind.Reinforcement -> "Exploration tries uncertain actions; exploitation follows the currently best route."
    else -> "The visible state was recomputed from parameter %.2f, variation %.2f, and iteration %d.".format(primary, secondary, run)
}

private fun algorithmVisualTitle(topic: LearnTopic, fallback: String): String = when {
    topic.title.contains("Regression", true) -> "Visualize ${topic.title}"
    topic.section == "Classification" -> "Visualize ${topic.title} Classification"
    topic.section == "Clustering" -> "Visualize ${topic.title} Clustering"
    topic.section == "Dimensionality Reduction" -> "Visualize ${topic.title} Projection"
    else -> fallback
}

private fun equationFocus(kind: VisualizationKind, primary: Float): String = when (kind) {
    VisualizationKind.Regression -> "The active control changes prediction f(x); residuals show y - y_hat for each sample."
    VisualizationKind.Classification -> "The active control changes the score threshold or margin used to turn evidence into a class."
    VisualizationKind.Neighbours -> "The active control changes N_K(x), the set of neighbours used for voting or averaging."
    VisualizationKind.Tree -> "The active control changes split depth or impurity pressure in the recursive partition."
    VisualizationKind.Clustering -> "The active control changes assignments by moving K, radius, or density thresholds."
    VisualizationKind.Density -> "The active control changes which points count as density-reachable neighbours."
    VisualizationKind.Projection -> "The active control rotates or compresses the projection axis and changes reconstruction error."
    VisualizationKind.Attention -> "The active control changes the selected head or the scale before softmax."
    VisualizationKind.Convolution -> "The active control changes the local receptive field that creates the feature map."
    VisualizationKind.Sequence, VisualizationKind.TimeSeries -> "The active control changes how much previous context contributes to the next state."
    VisualizationKind.Reinforcement -> "The active control changes the exploration/exploitation balance in the policy."
    else -> "The active control changes the visible model state connected to the displayed equation."
}

private fun compareExplanation(kind: VisualizationKind, primary: Float, secondary: Float): String = when (kind) {
    VisualizationKind.Regression -> "Muted line is the baseline; green line is the current fit. Compare residual length and slope."
    VisualizationKind.Classification -> "Muted boundary is the baseline; white boundary is current. Compare changed class regions."
    VisualizationKind.Neighbours -> "Compare mode keeps the old neighbourhood idea visible while K/query controls reshape the local evidence."
    VisualizationKind.Clustering, VisualizationKind.Density -> "Compare group compactness and noise handling as radius or group count changes."
    VisualizationKind.Projection -> "Compare projected positions against original point positions to judge information loss."
    else -> "Compare the baseline structure with current controls %.2f and %.2f.".format(primary, secondary)
}

private fun algorithmWarning(kind: VisualizationKind, primary: Float, secondary: Float, noise: Float): String? = when {
    noise > .7f -> "High noise can make the visual metric unstable. Use a held-out split before trusting the result."
    kind == VisualizationKind.Regression && secondary > .85f -> "Strong regularization or pressure toward simplicity can underfit curved data."
    kind == VisualizationKind.Classification && (primary < .15f || primary > .85f) -> "Extreme thresholds usually trade recall for precision or precision for recall."
    kind == VisualizationKind.Neighbours && primary < .15f -> "Very small K can overfit individual samples."
    kind == VisualizationKind.Clustering && primary > .8f -> "Too many clusters can split one natural group into artificial pieces."
    kind == VisualizationKind.Attention && secondary > .85f -> "Very sharp attention can hide useful broader context."
    else -> null
}

private fun predictionBreakdown(kind: VisualizationKind, score: Float, uncertainty: Float): String = when (kind) {
    VisualizationKind.Regression -> "The output is a continuous estimate. Inspect residuals on similar held-out samples before trusting the magnitude."
    VisualizationKind.Classification -> "Score %.3f is converted into a class by the threshold; uncertainty is highest near 0.50.".format(score)
    VisualizationKind.Neighbours -> "The prediction should be checked against nearby stored samples; distance scale can change the selected neighbours."
    VisualizationKind.Tree -> "Follow the highlighted split path from root to leaf, then compare the leaf distribution."
    VisualizationKind.Clustering, VisualizationKind.Density -> "The output is structural, so inspect assignment confidence and whether the point is near a boundary."
    VisualizationKind.Projection -> "The prediction view emphasizes whether the projected point keeps its local neighbourhood."
    VisualizationKind.Attention -> "The output depends on which tokens received high attention weight for the selected query."
    VisualizationKind.Convolution -> "The output depends on local feature maps; inspect the strongest activation before the decision."
    VisualizationKind.Sequence, VisualizationKind.TimeSeries -> "The output depends on recent context and hidden state; uncertainty %.0f%% marks ambiguous continuation.".format(uncertainty * 100f)
    else -> "The output combines the current input with the visible model state. Check preprocessing and validation evidence."
}
