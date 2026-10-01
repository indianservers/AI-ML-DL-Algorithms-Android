package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import kotlinx.coroutines.delay
import kotlin.math.abs
import kotlin.math.max

internal object StageFourVisualization {
    val domains = setOf(
        "Reinforcement Learning", "Probabilistic & Bayesian Learning", "Optimization Algorithms",
        "Evolutionary Algorithms", "Recommendation Algorithms", "Explainable AI", "AI Algorithms"
    )

    fun supports(topic: LearnTopic): Boolean = topic.domain in domains ||
        topic.title == "Hidden Markov Model – Forward & Viterbi Algorithms"

    fun family(topic: LearnTopic): String = when (topic.domain) {
        "Natural Language Processing" -> "hidden-state inference"
        "Reinforcement Learning" -> "reinforcement learning"
        "Probabilistic & Bayesian Learning" -> "probability and inference"
        "Optimization Algorithms" -> "optimization"
        "Evolutionary Algorithms" -> "evolutionary search"
        "Recommendation Algorithms" -> "recommendation"
        "Explainable AI" -> "explanation"
        else -> when (topic.section) {
            "Search" -> "AI search"
            "Probabilistic Graphs" -> "Bayesian graph"
            "Dynamic Programming" -> "dynamic programming"
            else -> "AI algorithm"
        }
    }
}

@Composable
internal fun StageFourVisualizationScreen(topic: LearnTopic) {
    if (topic.title == "Hidden Markov Model – Forward & Viterbi Algorithms") {
        StageFourProbabilityScreen(topic)
        return
    }
    when (topic.domain) {
        "Reinforcement Learning" -> StageFourRlScreen(topic)
        "Probabilistic & Bayesian Learning" -> StageFourProbabilityScreen(topic)
        "Optimization Algorithms" -> StageFourOptimizationScreen(topic)
        "Evolutionary Algorithms" -> StageFourEvolutionScreen(topic)
        "Recommendation Algorithms" -> StageFourRecommendationScreen(topic)
        "Explainable AI" -> StageFourExplanationScreen(topic)
        else -> when (topic.section) {
            "Search" -> StageFourSearchScreen(topic)
            "Probabilistic Graphs" -> StageFourProbabilityScreen(topic)
            "Dynamic Programming" -> StageFourRlScreen(topic)
            else -> StageFourSearchScreen(topic)
        }
    }
}

@Composable
internal fun FourPanel(title: String, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    GlassPanel(modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(title, color = LabText, fontSize = 16.sp, fontWeight = FontWeight.Bold)
            content()
        }
    }
}

@Composable
internal fun FourText(text: String, color: Color = LabMuted) {
    Text(text, color = color, fontSize = 13.sp, lineHeight = 19.sp)
}

@Composable
internal fun FourMetric(label: String, value: String, color: Color = LabCyan) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = LabMuted, fontSize = 12.sp)
        Text(value, color = color, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
internal fun FourSteps(step: Int, maxStep: Int, onStep: (Int) -> Unit) {
    var playing by remember { mutableStateOf(false) }
    LaunchedEffect(playing, step, maxStep) {
        if (playing) {
            delay(850)
            if (step >= maxStep) playing = false else onStep(step + 1)
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        FourMetric("Step", "$step / $maxStep")
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
            SegmentedOption("Reset", false, Modifier.weight(1f)) { playing = false; onStep(0) }
            SegmentedOption("Previous", false, Modifier.weight(1f)) { playing = false; onStep((step - 1).coerceAtLeast(0)) }
            SegmentedOption("Next", false, Modifier.weight(1f)) { playing = false; onStep((step + 1).coerceAtMost(maxStep)) }
            SegmentedOption(if (playing) "Pause" else "Auto", playing, Modifier.weight(1f)) { playing = !playing }
        }
    }
}

@Composable
internal fun FourBars(values: List<Pair<String, Double>>, selected: Int = -1, color: Color = LabCyan) {
    val data = values.take(10)
    if (data.isEmpty()) return
    val scale = max(1e-6, data.maxOf { abs(it.second) })
    Row(Modifier.fillMaxWidth().height(155.dp), horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.Bottom) {
        data.forEachIndexed { i, (label, value) ->
            Column(Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Bottom) {
                Text("%.2f".format(value), color = if (i == selected) LabOrange else LabText, fontSize = 10.sp, maxLines = 1)
                Box(Modifier.fillMaxWidth().height((10 + abs(value) / scale * 88).dp).background(
                    if (i == selected) LabOrange else if (value < 0) LabPink else color,
                    RoundedCornerShape(topStart = 4.dp, topEnd = 4.dp)
                ))
                Text(label, color = LabMuted, fontSize = 9.sp, maxLines = 1)
            }
        }
    }
}

@Composable
internal fun FourLine(values: List<Double>, secondary: List<Double> = emptyList(), color: Color = LabCyan,
                      markers: List<Pair<Double, Double>> = emptyList(), connectMarkers: Boolean = false) {
    val all = values + secondary
    if (values.isEmpty()) return
    val lo = all.minOrNull() ?: 0.0
    val hi = all.maxOrNull() ?: 1.0
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF071C36), RoundedCornerShape(8.dp)).padding(10.dp)) {
        val yFor: (Double) -> Float = { v -> size.height - 12f - ((v - lo) / (hi - lo).coerceAtLeast(1e-6) * (size.height - 24f)).toFloat() }
        repeat(4) { n ->
            val y = size.height * (n + 1) / 5f
            drawLine(LabMuted.copy(alpha = .18f), Offset(0f, y), Offset(size.width, y), 1f)
        }
        fun pathFor(series: List<Double>): Path = Path().apply {
            series.forEachIndexed { i, v ->
                val x = i.toFloat() / (series.size - 1).coerceAtLeast(1) * size.width
                if (i == 0) moveTo(x, yFor(v)) else lineTo(x, yFor(v))
            }
        }
        drawPath(pathFor(values), color, style = Stroke(3f))
        if (secondary.isNotEmpty()) drawPath(pathFor(secondary), LabOrange, style = Stroke(2.5f))
        fun marker(point: Pair<Double, Double>) = Offset(
            (point.first.coerceIn(0.0, 1.0) * size.width).toFloat(), yFor(point.second))
        if (connectMarkers) markers.zipWithNext().forEach { (a, b) ->
            drawLine(LabOrange, marker(a), marker(b), 3f)
        }
        markers.forEachIndexed { index, point ->
            drawCircle(if (index == markers.lastIndex) LabOrange else LabGreen,
                if (index == markers.lastIndex) 7f else 5f, marker(point))
        }
    }
}

@Composable
internal fun FourWaterfall(base: Double, contributions: List<Double>, labels: List<String>) {
    val cumulative = contributions.runningFold(base) { value, effect -> value + effect }
    val low = cumulative.minOrNull() ?: base
    val high = cumulative.maxOrNull() ?: base
    val margin = (high - low).coerceAtLeast(.2) * .12
    val paint = remember {
        android.graphics.Paint().apply {
            isAntiAlias = true
            textSize = 25f
            color = android.graphics.Color.WHITE
        }
    }
    Canvas(Modifier.fillMaxWidth().height(225.dp).background(Color(0xFF071C36), RoundedCornerShape(8.dp)).padding(10.dp)) {
        val left = size.width * .30f
        val width = size.width * .66f
        fun x(value: Double) = left + ((value - low + margin) / (high - low + 2 * margin) * width).toFloat()
        val rowHeight = size.height / (contributions.size + 2)
        for (row in 0..contributions.size + 1) {
            val value = if (row == 0) base else cumulative[(row - 1).coerceAtMost(contributions.size)]
            val y = rowHeight * (row + .5f)
            val label = when (row) {
                0 -> "Baseline"
                contributions.size + 1 -> "Prediction"
                else -> labels.getOrElse(row - 1) { "Feature $row" }
            }
            drawContext.canvas.nativeCanvas.drawText(label, 4f, y + 8f, paint)
            if (row > 0 && row <= contributions.size) {
                val previous = cumulative[row - 1]
                val effect = contributions[row - 1]
                drawRect(if (effect >= 0) LabGreen else LabPink,
                    topLeft = Offset(minOf(x(previous), x(value)), y - 12f),
                    size = Size(kotlin.math.abs(x(value) - x(previous)).coerceAtLeast(3f), 24f))
                drawLine(LabMuted.copy(alpha = .4f), Offset(x(previous), y - rowHeight + 12f),
                    Offset(x(previous), y - 12f), 1.5f)
            }
            drawCircle(if (row == contributions.size + 1) LabOrange else LabCyan, 5f, Offset(x(value), y))
        }
    }
}

@Composable
internal fun FourFlow(stages: List<String>, active: Int = 0) {
    Column(Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(5.dp)) {
        stages.chunked(2).forEachIndexed { rowIndex, row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp), verticalAlignment = Alignment.CenterVertically) {
                row.forEachIndexed { columnIndex, stage ->
                    val index = rowIndex * 2 + columnIndex
                    if (columnIndex > 0) Text("→", color = LabMuted)
                    Box(Modifier.weight(1f).background(
                        if (index == active) LabPurple.copy(alpha = .32f) else Color(0xFF0C2445),
                        RoundedCornerShape(7.dp)
                    ).padding(9.dp)) {
                        Text(stage, color = if (index == active) LabText else LabMuted, fontSize = 11.sp, lineHeight = 14.sp)
                    }
                }
                if(row.size==1)Box(Modifier.weight(1f))
            }
        }
    }
}

@Composable
internal fun FourGrid(values: List<List<Double>>, labels: Boolean = true, normalize: Boolean = true) {
    val all = values.flatten()
    val min = all.minOrNull() ?: 0.0
    val max = all.maxOrNull() ?: 1.0
    Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
        values.forEach { row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(3.dp)) {
                row.forEach { value ->
                    val t = (if (normalize) (value - min) / (max - min).coerceAtLeast(1e-9)
                        else value.coerceIn(0.0, 1.0)).toFloat()
                    Box(Modifier.weight(1f).height(40.dp).background(
                        LabCyan.copy(alpha = .12f + .64f * t), RoundedCornerShape(4.dp)
                    ), contentAlignment = Alignment.Center) {
                        if (labels) Text("%.2f".format(value), color = LabText, fontSize = 10.sp)
                    }
                }
            }
        }
    }
}

@Composable
internal fun FourLandscape(points: List<Pair<Double, Double>>, selected: Int = points.lastIndex,
                           connect: Boolean = true) {
    Canvas(Modifier.fillMaxWidth().height(225.dp).background(Color(0xFF071C36), RoundedCornerShape(8.dp)).padding(8.dp)) {
        val center = Offset(size.width * .70f, size.height * .58f)
        repeat(5) { i ->
            val fraction = (i + 1) / 6f
            drawOval(LabPurple.copy(alpha = .22f + i * .07f),
                topLeft = Offset(center.x - size.width * fraction * .32f, center.y - size.height * fraction * .42f),
                size = Size(size.width * fraction * .64f, size.height * fraction * .84f),
                style = Stroke(1.5f))
        }
        fun map(point: Pair<Double, Double>) = Offset(
            ((point.first + 2.5) / 5.0 * size.width).toFloat().coerceIn(0f,size.width),
            ((2.5 - point.second) / 5.0 * size.height).toFloat().coerceIn(0f,size.height))
        if (connect) points.take((selected + 1).coerceAtLeast(0)).zipWithNext().forEach { (a,b) ->
            drawLine(LabCyan,map(a),map(b),3f)
        }
        points.take((selected + 1).coerceAtLeast(0)).forEachIndexed { i,p ->
            drawCircle(if(i==selected) LabOrange else LabCyan,if(i==selected)7f else 4f,map(p))
        }
        drawCircle(LabGreen,6f,center)
    }
}

@Composable
internal fun FourSearchGraph(current: String?, visited: Set<String>, frontier: Set<String>,
                             scores: Map<String, Double> = emptyMap(), pruned: Set<String> = emptySet()) {
    val paint = remember {
        android.graphics.Paint().apply {
            isAntiAlias = true
            textAlign = android.graphics.Paint.Align.CENTER
            textSize = 31f
            color = android.graphics.Color.WHITE
        }
    }
    val positions = mapOf("S" to (.08f to .5f), "A" to (.30f to .2f), "B" to (.30f to .8f),
        "C" to (.56f to .12f), "D" to (.57f to .52f), "G" to (.88f to .5f))
    val edges = listOf("S" to "A", "S" to "B", "A" to "C", "A" to "D", "B" to "D",
        "B" to "G", "C" to "G", "D" to "G")
    Canvas(Modifier.fillMaxWidth().height(205.dp).background(Color(0xFF071C36), RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun p(label: String): Offset = positions.getValue(label).let { Offset(it.first * size.width,it.second * size.height) }
        edges.forEach { (a,b) -> drawLine(LabMuted.copy(alpha=.35f),p(a),p(b),2f) }
        positions.keys.forEach { label ->
            val color = when { label in pruned -> LabPink.copy(alpha=.4f);label==current -> LabOrange
                label in visited -> LabGreen;label in frontier -> LabCyan;else -> LabPurple }
            drawCircle(color,22f,p(label))
            drawContext.canvas.nativeCanvas.drawText(label,p(label).x,p(label).y+10f,paint)
            scores[label]?.let { score ->
                paint.textSize=21f;paint.color=android.graphics.Color.LTGRAY
                drawContext.canvas.nativeCanvas.drawText("%.1f".format(score),p(label).x,p(label).y-30f,paint)
                paint.textSize=31f;paint.color=android.graphics.Color.WHITE
            }
        }
    }
}

@Composable
internal fun FourGameTree(leafValues: List<Int>, highlighted: Int, pruned: Set<Int>) {
    val paint = remember {
        android.graphics.Paint().apply {
            isAntiAlias = true
            textAlign = android.graphics.Paint.Align.CENTER
            textSize = 27f
            color = android.graphics.Color.WHITE
        }
    }
    val xs = listOf(.125f,.375f,.625f,.875f)
    Canvas(Modifier.fillMaxWidth().height(205.dp).background(Color(0xFF071C36), RoundedCornerShape(8.dp)).padding(8.dp)) {
        val root=Offset(size.width*.5f,size.height*.14f)
        val children=listOf(Offset(size.width*.25f,size.height*.48f),Offset(size.width*.75f,size.height*.48f))
        val leaves=xs.map{Offset(size.width*it,size.height*.85f)}
        children.forEach{drawLine(LabMuted,root,it,2f)}
        leaves.forEachIndexed{i,p->drawLine(if(i in pruned)LabPink.copy(alpha=.3f)else LabMuted,children[i/2],p,2f)}
        drawCircle(LabPurple,22f,root);drawContext.canvas.nativeCanvas.drawText("MAX",root.x,root.y+9f,paint)
        children.forEach{drawCircle(LabCyan,22f,it);drawContext.canvas.nativeCanvas.drawText("MIN",it.x,it.y+9f,paint)}
        leaves.forEachIndexed{i,p->
            drawCircle(if(i in pruned)LabPink.copy(alpha=.35f)else if(i==highlighted)LabOrange else LabGreen,22f,p)
            drawContext.canvas.nativeCanvas.drawText(if(i in pruned)"×" else leafValues[i].toString(),p.x,p.y+9f,paint)
        }
    }
}
