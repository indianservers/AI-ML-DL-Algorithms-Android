package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import kotlin.math.abs
import kotlin.math.sqrt

internal object StageThreeFundamentalsEngine {
    val activations = listOf(listOf(1.0, 2.0, -1.0, 3.0), listOf(2.0, 4.0, 1.0, 4.0),
        listOf(-1.0, 3.0, 2.0, 5.0), listOf(4.0, 5.0, 1.0, 6.0))
    fun batchNorm(data: List<List<Double>>): List<List<Double>> = data.map { row -> row.indices.map { c ->
        val values = data.map { it[c] }; val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        (row[c] - mean) / sqrt(variance + 1e-5)
    } }
    fun layerNorm(data: List<List<Double>>): List<List<Double>> = data.map { row ->
        val mean = row.average(); val variance = row.map { (it - mean) * (it - mean) }.average()
        row.map { (it - mean) / sqrt(variance + 1e-5) }
    }
    fun mask(rate: Double, seed: Int, count: Int = 12): List<Boolean> = List(count) { i ->
        (((i * 37 + seed * 19 + 11) % 101) / 101.0) >= rate
    }
    fun penalty(weights: List<Double>, lambda: Double, l1: Boolean): Double =
        lambda * weights.sumOf { if (l1) abs(it) else it * it }
    fun initialization(kind: Int, fanIn: Int = 8): List<Double> = List(32) { i ->
        val centered = (((i * 37 + 13) % 101) / 50.0 - 1.0)
        when (kind) {
            0 -> 0.0
            1 -> centered
            2 -> centered * sqrt(6.0 / (fanIn + fanIn))
            else -> centered * sqrt(2.0 / fanIn)
        }
    }
}

@Composable
internal fun StageThreeFundamentalsScreen(topic: LearnTopic, kind: DeepVisualization) {
    var parameter by remember(topic.id) { mutableIntStateOf(4) }
    var mode by remember(topic.id) { mutableIntStateOf(if (kind == DeepVisualization.WeightInit) 2 else 0) }
    var seed by remember(topic.id) { mutableIntStateOf(0) }
    val raw = StageThreeFundamentalsEngine.activations
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, when (kind) {
            DeepVisualization.BatchNorm -> "Compare each feature before and after normalization across samples."
            DeepVisualization.LayerNorm -> "Compare each sample before and after normalization across features."
            DeepVisualization.WeightInit -> "Switch initializers and compare weight spread and activation scale."
            DeepVisualization.Dropout -> "Change the dropout rate and see which training activations survive."
            DeepVisualization.Regularization -> "Compare L1 and L2 penalties on the same learned weights."
            else -> "Inspect the computation."
        }) }
        when (kind) {
            DeepVisualization.BatchNorm, DeepVisualization.LayerNorm -> {
                val batch = kind == DeepVisualization.BatchNorm
                val normalized = if (batch) StageThreeFundamentalsEngine.batchNorm(raw) else StageThreeFundamentalsEngine.layerNorm(raw)
                val selectedValues = if (batch) raw.map { it[mode % 4] } else raw[mode % 4]
                val mean = selectedValues.average()
                val variance = selectedValues.map { (it-mean)*(it-mean) }.average()
                val normalizedValues = if (batch) normalized.map { it[mode % 4] } else normalized[mode % 4]
                item { GlassPanel(Modifier.fillMaxWidth()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(if (batch) "Columns normalized across the batch ↓" else "Rows normalized across each sample →", color = LabText, fontWeight = FontWeight.Bold)
                        Text("Rows = samples   Columns = features", color = LabMuted, fontSize = 11.sp)
                        MatrixHeatmap(raw, mode % 4, batch)
                        Text("Raw activations → mean and variance → normalized values → γx̂ + β", color = LabCyan, fontSize = 12.sp)
                        MatrixHeatmap(normalized, mode % 4, batch)
                        Text("Raw: ${selectedValues.joinToString { "%.2f".format(it) }}", color = LabMuted, fontSize = 12.sp)
                        Text("μ = %.3f; variance = %.3f; γ = 1; β = 0".format(mean, variance), color = LabCyan, fontSize = 12.sp)
                        Text("Normalized: ${normalizedValues.joinToString { "%.2f".format(it) }}", color = LabMuted, fontSize = 12.sp)
                        SegmentedOption("Select ${if (batch) "feature" else "sample"}", false, Modifier.fillMaxWidth()) { mode++ }
                    }
                } }
            }
            DeepVisualization.WeightInit -> item { GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Weight distribution and activation scale", color = LabText, fontWeight = FontWeight.Bold)
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        listOf("Zero", "Random", "Xavier", "He").forEachIndexed { i, label ->
                            SegmentedOption(label, mode == i, Modifier.weight(1f)) { mode = i }
                        }
                    }
                    val values = StageThreeFundamentalsEngine.initialization(mode)
                    ValueBars(values)
                    Text("mean = %.3f; variance = %.3f".format(values.average(), values.map { (it-values.average())*(it-values.average()) }.average()), color = LabCyan, fontSize = 12.sp)
                    Text(when (mode) {
                        0 -> "Zero weights make neurons in a layer learn identically."
                        1 -> "Unscaled random values can expand or shrink layer activations."
                        2 -> "Xavier scales variance with input and output width."
                        else -> "He scales variance for ReLU networks."
                    }, color = LabMuted, fontSize = 12.sp)
                }
            } }
            DeepVisualization.Dropout -> item { GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    val rate = parameter / 10.0
                    val mask = StageThreeFundamentalsEngine.mask(rate, seed)
                    Text("Training mask", color = LabText, fontWeight = FontWeight.Bold)
                    Row(horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        mask.forEach { active -> Text(if (active) "●" else "○", color = if (active) LabCyan else LabMuted, fontSize = 20.sp) }
                    }
                    Text("${mask.count { it }} of ${mask.size} units active; inverted scaling = %.2f".format(1.0/(1-rate)), color = LabCyan, fontSize = 12.sp)
                    Text("At inference all units are active; inverted training scaling keeps expected activation stable.", color = LabMuted, fontSize = 12.sp)
                    Text("Dropout rate %.1f".format(rate), color = LabText, fontSize = 12.sp)
                    Slider(parameter.toFloat(), { parameter = it.toInt().coerceIn(1, 8) }, valueRange = 1f..8f)
                    SegmentedOption("New mask", false, Modifier.fillMaxWidth()) { seed++ }
                }
            } }
            DeepVisualization.Regularization -> item { GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    val weights = listOf(1.8, -.9, .45, .05)
                    val lambda = parameter / 10.0
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        listOf("None", "L1", "L2").forEachIndexed { i, label -> SegmentedOption(label, mode == i, Modifier.weight(1f)) { mode = i } }
                    }
                    ValueBars(weights)
                    val cost = if (mode == 0) 0.0 else StageThreeFundamentalsEngine.penalty(weights, lambda, mode == 1)
                    Text("Data loss 0.30 + penalty %.3f = objective %.3f".format(cost, .3+cost), color = LabCyan, fontSize = 12.sp)
                    Text(if (mode == 1) "L1 can drive smaller weights exactly to zero."
                    else if (mode == 2) "L2 shrinks weights smoothly without promoting exact zeros."
                    else "Without a penalty, all fitted weight magnitude comes from data loss.", color = LabMuted, fontSize = 12.sp)
                    Text("Regularization λ = %.1f".format(lambda), color = LabText, fontSize = 12.sp)
                    Slider(parameter.toFloat(), { parameter = it.toInt().coerceIn(1, 9) }, valueRange = 1f..9f)
                }
            } }
            else -> error("Not a fundamentals topic")
        }
    }
}

@Composable private fun MatrixHeatmap(matrix: List<List<Double>>, selected: Int, byColumn: Boolean) {
    Canvas(Modifier.fillMaxWidth().height(138.dp).background(Color(0xFF071B37))) {
        val rows = matrix.size; val cols = matrix.first().size
        val cw = size.width / cols; val ch = size.height / rows
        matrix.forEachIndexed { r, row -> row.forEachIndexed { c, value ->
            val color = if ((if (byColumn) c else r) == selected) LabCyan else LabOrange
            drawRect(color.copy(alpha = (.18 + .5 * (abs(value)/6.0).coerceIn(0.0,1.0)).toFloat()),
                Offset(c*cw+2,r*ch+2), Size(cw-4,ch-4))
        } }
    }
}

@Composable private fun ValueBars(values: List<Double>) {
    Canvas(Modifier.fillMaxWidth().height(135.dp).background(Color(0xFF071B37))) {
        val middle = size.height/2
        val extent = values.maxOf { abs(it) }.coerceAtLeast(.1)
        values.forEachIndexed { i, value ->
            val x = (i+.5f)*size.width/values.size
            val y = middle-(value/extent).toFloat()*middle*.8f
            drawLine(if (value>=0) LabCyan else LabOrange, Offset(x,middle),Offset(x,y),
                (size.width/values.size*.65f).coerceAtLeast(2f))
        }
        drawLine(Color(0xFF345276),Offset(0f,middle),Offset(size.width,middle),1f)
    }
}
