package com.indianservers.ai_ml_dl_algorithms.ml_lab.presentation

import androidx.compose.foundation.Image
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.R
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBlue
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmIcon
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnDomain
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonRepository
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmIconKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.algorithmIconKind
import kotlin.math.sin

private val homeShape = RoundedCornerShape(17.dp)

@Composable
internal fun HomeDashboard(
    continueTopic: LearnTopic?,
    onTopic: (LearnTopic) -> Unit,
    onLibrary: () -> Unit,
    onSettings: () -> Unit,
    onTool: (String) -> Unit
) {
    var query by remember { mutableStateOf("") }
    var expanded by remember { mutableStateOf("Supervised Learning") }
    val domains = remember { LearnCatalog.domains }
    val results = remember(query) {
        if (query.isBlank()) emptyList() else LearnCatalog.topics.filter {
            it.title.contains(query, true) || it.section.contains(query, true) || it.domain.contains(query, true)
        }
    }
    LazyColumn(
        Modifier.fillMaxSize().padding(horizontal = 15.dp),
        verticalArrangement = Arrangement.spacedBy(11.dp)
    ) {
        item { HomeHeader(onSettings) }
        item { HomeHero() }
        item {
            OutlinedTextField(
                value = query,
                onValueChange = { query = it },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                leadingIcon = { Text("⌕", color = LabCyan, fontSize = 26.sp) },
                label = { Text("Search algorithms, topics, or concepts") }
            )
        }
        if (query.isBlank() && continueTopic != null) item { ContinueLearningCard(continueTopic, onTopic) }
        if (query.isNotBlank()) {
            item { Text("${results.size} results", color = LabCyan, fontSize = 13.sp) }
            items(results, key = { it.id }) { topic -> HomeTopicCard(topic, Modifier.fillMaxWidth(), onTopic) }
        } else {
            item {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                    Text("Algorithm Categories", color = LabText, fontSize = 21.sp, fontWeight = FontWeight.Bold)
                    Text("${domains.size} categories", color = LabCyan, fontSize = 12.sp)
                }
            }
            items(domains, key = { it.title }) { domain ->
                HomeDomain(domain, expanded == domain.title, { expanded = if (expanded == domain.title) "" else domain.title }, onTopic, onLibrary)
            }
            item { Text("More to explore", color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 10.dp)) }
            item {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    listOf(
                        "All algorithms" to "Browse all ${LearnCatalog.topics.size} lessons",
                        "Dataset Lab" to "Create and edit learning data",
                        "Training Playground" to "Watch training change with each epoch",
                        "Deep Learning" to "Build and inspect neural networks",
                        "AI Engineering Studio" to "Import and run on-device models",
                        "Saved" to "Return to learning notes"
                    ).forEach { (title, subtitle) ->
                        Row(
                            Modifier.fillMaxWidth().clip(homeShape).background(LabPanel)
                                .border(1.dp, LabBlue.copy(alpha = .55f), homeShape)
                                .clickable { if (title == "All algorithms") onLibrary() else onTool(title) }
                                .padding(15.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(title, color = LabText, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                            Text(subtitle, color = LabMuted, fontSize = 11.sp, modifier = Modifier.weight(1.3f))
                            Text("→", color = LabCyan, fontSize = 19.sp)
                        }
                    }
                }
            }
            item { Box(Modifier.height(12.dp)) }
        }
    }
}

@Composable
private fun ContinueLearningCard(topic: LearnTopic, onTopic: (LearnTopic) -> Unit) {
    val context = LocalContext.current
    val progress = remember(topic) { LessonRepository.get(context).progressFor(topic.id) }
    val page = progress?.lastPageNumber ?: 1
    Row(
        Modifier.fillMaxWidth().clip(homeShape)
            .background(Brush.horizontalGradient(listOf(Color(0xFF121D4D), Color(0xFF03264B))))
            .border(1.dp, LabBlue, homeShape).clickable { onTopic(topic) }.padding(12.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        AlgorithmIcon(topic, size = 48.dp)
        Column(Modifier.weight(1f)) {
            Text("Continue Learning", color = LabCyan, fontSize = 11.sp)
            Text(topic.title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp, maxLines = 1)
            Text("Lesson page $page of 5", color = LabMuted, fontSize = 11.sp)
        }
        Text("Continue  →", color = LabText, fontWeight = FontWeight.Bold, fontSize = 12.sp)
    }
}

@Composable
private fun HomeHeader(onSettings: () -> Unit) {
    Row(Modifier.fillMaxWidth().padding(top = 14.dp, bottom = 3.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(
            Modifier.size(49.dp).clip(RoundedCornerShape(14.dp))
                .background(Brush.linearGradient(listOf(LabBlue, LabPurple)))
                .border(1.dp, LabCyan.copy(alpha = .8f), RoundedCornerShape(14.dp)),
            contentAlignment = Alignment.Center
        ) { BrainMark(Modifier.size(33.dp)) }
        Column(Modifier.weight(1f).padding(start = 11.dp)) {
            Text("AI/ML Learning Lab", color = LabText, fontSize = 19.sp, fontWeight = FontWeight.Bold)
            Text("Learn  •  Visualize  •  Build  •  Grow", color = LabMuted, fontSize = 11.sp)
        }
        Box(
            Modifier.size(38.dp).clip(RoundedCornerShape(12.dp)).background(LabPanel)
                .border(1.dp, LabPurple, RoundedCornerShape(12.dp)).clickable(onClick = onSettings),
            contentAlignment = Alignment.Center
        ) { Text("⚙", color = LabText, fontSize = 23.sp) }
    }
}

@Composable
private fun BrainMark(modifier: Modifier = Modifier) {
    Canvas(modifier) {
        val w = size.width
        val h = size.height
        val glow = Color(0xFF36E5FF)
        val left = Path().apply {
            moveTo(.49f * w, .17f * h)
            cubicTo(.28f * w, .04f * h, .14f * w, .24f * h, .19f * w, .38f * h)
            cubicTo(.03f * w, .5f * h, .16f * w, .69f * h, .27f * w, .7f * h)
            cubicTo(.21f * w, .9f * h, .39f * w, .97f * h, .49f * w, .84f * h)
        }
        val right = Path().apply {
            moveTo(.51f * w, .17f * h)
            cubicTo(.72f * w, .04f * h, .86f * w, .24f * h, .81f * w, .38f * h)
            cubicTo(.97f * w, .5f * h, .84f * w, .69f * h, .73f * w, .7f * h)
            cubicTo(.79f * w, .9f * h, .61f * w, .97f * h, .51f * w, .84f * h)
        }
        drawPath(left, Color.White, style = Stroke(3f))
        drawPath(right, Color.White, style = Stroke(3f))
        drawLine(glow, Offset(.5f * w, .12f * h), Offset(.5f * w, .88f * h), 2f)
        listOf(
            .27f to .34f, .35f to .5f, .28f to .7f,
            .73f to .34f, .65f to .5f, .72f to .7f
        ).forEach { (x, y) ->
            drawLine(glow, Offset(.5f * w, y * h), Offset(x * w, y * h), 1.5f)
            drawCircle(glow, 2.5f, Offset(x * w, y * h))
        }
    }
}

@Composable
private fun HomeHero() {
    Box(Modifier.fillMaxWidth().height(164.dp).clip(homeShape).border(1.dp, LabCyan.copy(alpha = .8f), homeShape)) {
        Image(
            painter = painterResource(R.drawable.learning_hero),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize()
        )
        Box(Modifier.fillMaxSize().background(Brush.horizontalGradient(listOf(Color(0xE5000A23), Color(0x86000A23), Color.Transparent))))
        Column(Modifier.align(Alignment.CenterStart).padding(start = 17.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Explore ${LearnCatalog.topics.size} Algorithms", color = LabText, fontWeight = FontWeight.ExtraBold, fontSize = 22.sp)
            Text("Learn, visualize, train, and build with AI", color = Color(0xFFDDE7FF), fontSize = 12.sp)
            Text("A smarter you, for a more intelligent tomorrow.", color = LabCyan, fontSize = 10.sp)
        }
    }
}

@Composable
private fun HomeDomain(domain: LearnDomain, expanded: Boolean, onToggle: () -> Unit, onTopic: (LearnTopic) -> Unit, onLibrary: () -> Unit) {
    val accent = Color(domain.accent)
    Column(
        Modifier.fillMaxWidth().clip(homeShape)
            .background(Brush.verticalGradient(listOf(accent.copy(alpha = if (expanded) .23f else .15f), Color(0xFF07152F))))
            .border(1.dp, accent.copy(alpha = .82f), homeShape).padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(Modifier.fillMaxWidth().clickable(onClick = onToggle), verticalAlignment = Alignment.CenterVertically) {
            AlgorithmIcon(domain.sections.first().topics.first(), size = 39.dp)
            Column(Modifier.weight(1f).padding(start = 10.dp)) {
                Text(domain.title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 15.sp)
                if (expanded) Text(domain.description, color = LabMuted, fontSize = 11.sp, maxLines = 1)
            }
            Text("${domain.topicCount}  ${if (expanded) "⌃" else "⌄"}", color = if (expanded) accent else LabMuted, fontSize = 12.sp)
        }
        if (expanded) {
            featuredTopics(domain).chunked(2).forEach { pair ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    pair.forEach { topic -> HomeTopicCard(topic, Modifier.weight(1f), onTopic) }
                    if (pair.size == 1) Box(Modifier.weight(1f))
                }
            }
            Row(Modifier.fillMaxWidth().clickable(onClick = onLibrary).padding(7.dp), horizontalArrangement = Arrangement.Center) {
                Text("View all ${domain.topicCount} topics  →", color = accent, fontWeight = FontWeight.Bold, fontSize = 12.sp)
            }
        }
    }
}

private fun featuredTopics(domain: LearnDomain): List<LearnTopic> {
    val all = domain.sections.flatMap { it.topics }
    val titles = when (domain.title) {
        "Supervised Learning" -> listOf("Simple Linear Regression", "Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Random Forest", "Support Vector Machine")
        "Unsupervised Learning" -> listOf("K-Means", "Hierarchical Clustering", "DBSCAN", "PCA", "Gaussian Mixture Models", "t-SNE")
        "Deep Learning" -> listOf("Multi-Layer Perceptron", "CNN", "Recurrent Neural Network", "LSTM", "Transformer", "Basic Autoencoder")
        else -> emptyList()
    }
    return if (titles.isEmpty()) all.take(6) else titles.mapNotNull { name ->
        all.firstOrNull { it.title == name && (name != "Logistic Regression" || it.section == "Classification") }
    }
}

@Composable
private fun HomeTopicCard(topic: LearnTopic, modifier: Modifier, onTopic: (LearnTopic) -> Unit) {
    val accent = Color(topic.accent)
    Column(
        modifier.height(145.dp).clip(RoundedCornerShape(13.dp))
            .background(Color(0xD7081A38)).border(1.dp, accent.copy(alpha = .5f), RoundedCornerShape(13.dp))
            .clickable { onTopic(topic) }.padding(10.dp),
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.Top) {
            Column(Modifier.weight(1f)) {
            Text(topic.title, color = LabText, fontWeight = FontWeight.SemiBold, fontSize = 12.sp, maxLines = 2, overflow = TextOverflow.Ellipsis)
            Text(topic.section, color = LabMuted, fontSize = 10.sp, maxLines = 1)
            }
            Text("↗", color = LabCyan, fontSize = 18.sp)
        }
        AlgorithmPreview(topic, Modifier.fillMaxWidth().height(65.dp))
    }
}

@Composable
private fun AlgorithmPreview(topic: LearnTopic, modifier: Modifier = Modifier) {
    val kind = algorithmIconKind(topic)
    val accent = Color(topic.accent)
    Canvas(modifier) {
        val w = size.width
        val h = size.height
        val cyan = Color(0xFF16D8FF)
        val pink = Color(0xFFFF48BE)
        fun dot(x: Float, y: Float, color: Color, radius: Float = 3.4f) {
            drawCircle(color.copy(alpha = .2f), radius * 2.1f, Offset(x * w, y * h))
            drawCircle(color, radius, Offset(x * w, y * h))
        }
        repeat(4) { i ->
            drawLine(Color.White.copy(alpha = .06f), Offset(0f, h * (i + 1) / 5f), Offset(w, h * (i + 1) / 5f))
        }
        if (topic.title == "Logistic Regression") {
            val curve = Path().apply {
                moveTo(.08f * w, .84f * h)
                cubicTo(.34f * w, .82f * h, .39f * w, .78f * h, .48f * w, .5f * h)
                cubicTo(.58f * w, .19f * h, .7f * w, .15f * h, .92f * w, .14f * h)
            }
            drawPath(curve, cyan, style = Stroke(3f))
            repeat(12) { i ->
                val x = .1f + i * .072f
                val y = (1f / (1f + kotlin.math.exp(((x - .5f) * 13f)))).coerceIn(.1f, .9f)
                dot(x, y, if (i % 3 == 0) pink else cyan, 2.7f)
            }
        } else when (kind) {
            AlgorithmIconKind.Regression, AlgorithmIconKind.Optimization, AlgorithmIconKind.TimeSeries -> {
                drawLine(cyan, Offset(.08f * w, .84f * h), Offset(.92f * w, .12f * h), 3f)
                repeat(13) { i ->
                    val x = .1f + i * .064f
                    val y = .77f - i * .052f + sin(i * 2.1).toFloat() * .12f
                    dot(x, y, if (i % 3 == 0) pink else cyan)
                }
            }
            AlgorithmIconKind.Tree, AlgorithmIconKind.Forest, AlgorithmIconKind.Boosting -> {
                fun branch(x1: Float, y1: Float, x2: Float, y2: Float) {
                    drawLine(accent, Offset(x1 * w, y1 * h), Offset(x2 * w, y2 * h), 2.5f)
                    dot(x2, y2, if (x2 < .5f) cyan else pink, 3f)
                }
                dot(.5f, .1f, accent, 5f)
                branch(.5f, .1f, .28f, .42f); branch(.5f, .1f, .72f, .42f)
                branch(.28f, .42f, .15f, .83f); branch(.28f, .42f, .4f, .83f)
                branch(.72f, .42f, .6f, .83f); branch(.72f, .42f, .85f, .83f)
            }
            AlgorithmIconKind.Neighbours, AlgorithmIconKind.Clustering, AlgorithmIconKind.Density, AlgorithmIconKind.Probability -> {
                repeat(18) { i ->
                    val group = i % 3
                    val cx = listOf(.23f, .52f, .78f)[group]
                    val cy = listOf(.65f, .27f, .64f)[group]
                    val dx = sin(i * 13.0).toFloat() * .095f
                    val dy = sin(i * 9.0).toFloat() * .14f
                    dot(cx + dx, cy + dy, listOf(cyan, accent, pink)[group], 2.7f)
                }
                drawCircle(Color.White, 7f, Offset(.47f * w, .55f * h), style = Stroke(2f))
            }
            AlgorithmIconKind.NeuralNetwork, AlgorithmIconKind.Attention, AlgorithmIconKind.Graph, AlgorithmIconKind.Autoencoder -> {
                val xs = listOf(.15f, .48f, .83f)
                val ys = listOf(listOf(.25f, .5f, .75f), listOf(.2f, .4f, .6f, .8f), listOf(.33f, .67f))
                repeat(2) { layer ->
                    ys[layer].forEach { a -> ys[layer + 1].forEach { b ->
                        drawLine(accent.copy(alpha = .5f), Offset(xs[layer] * w, a * h), Offset(xs[layer + 1] * w, b * h), 1.3f)
                    } }
                }
                ys.forEachIndexed { layer, group -> group.forEach { dot(xs[layer], it, if (layer == 1) accent else cyan, 3.4f) } }
            }
            AlgorithmIconKind.Sequence, AlgorithmIconKind.Language -> {
                repeat(5) { i ->
                    val x = (.12f + i * .19f) * w
                    drawCircle(accent.copy(alpha = .22f), 15f, Offset(x, .5f * h))
                    drawCircle(accent, 6f, Offset(x, .5f * h))
                    if (i < 4) drawLine(cyan, Offset(x + 9f, .5f * h), Offset(x + .19f * w - 9f, .5f * h), 2f)
                }
            }
            else -> {
                val path = Path().apply {
                    moveTo(.07f * w, .86f * h)
                    cubicTo(.38f * w, .8f * h, .4f * w, .2f * h, .92f * w, .12f * h)
                }
                drawPath(path, accent, style = Stroke(3f))
                repeat(10) { i -> dot(.1f + i * .08f, .78f - sin(i * .58).toFloat() * .45f, if (i % 2 == 0) cyan else pink, 2.8f) }
            }
        }
    }
}
