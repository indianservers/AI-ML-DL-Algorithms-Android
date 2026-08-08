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
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
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

private enum class TransformerSection(val label: String) { Tokens("Tokens"), Qkv("Q/K/V"), Matrix("Attention"), Heads("Heads"), Position("Position"), Block("Block"), Predict("Predict") }

@Composable
fun PhaseEightTransformerLab(
    topic: LearnTopic,
    concept: PhaseEightConcept,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(defaultTransformerSection(concept)) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("Attention & Transformer - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "Phase 8", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                TransformerSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            TransformerSection.Tokens -> TokensSection()
            TransformerSection.Qkv -> QkvSection()
            TransformerSection.Matrix -> AttentionSection()
            TransformerSection.Heads -> HeadsSection()
            TransformerSection.Position -> PositionSection()
            TransformerSection.Block -> BlockSection()
            TransformerSection.Predict -> PredictionSection(onComplete)
        }
    }
}

@Composable
private fun TokensSection() {
    var selected by remember { mutableIntStateOf(1) }
    val state = PhaseEightEngines.attention()
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Token Sequence", "Tap a token and inspect its representation") }
        item { TokenSequenceVisualizer(state.tokens, selected) { selected = it } }
        item { VectorCard("Embedding for ${state.tokens[selected]}", state.embeddings[selected]) }
        item { InfoT("Attention intuition", "Query asks what information this token is looking for. Key describes what each token contains. Value is the information that gets mixed into the output.") }
    }
}

@Composable
private fun QkvSection() {
    var selected by remember { mutableIntStateOf(1) }
    val state = PhaseEightEngines.attention()
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Query, Key, Value", "All values are derived from embedding projection matrices") }
        item { TokenSequenceVisualizer(state.tokens, selected) { selected = it } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                VectorCard("Embedding", state.embeddings[selected], Modifier.weight(1f))
                VectorCard("Query", state.queries[selected], Modifier.weight(1f))
            }
        }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                VectorCard("Key", state.keys[selected], Modifier.weight(1f))
                VectorCard("Value", state.values[selected], Modifier.weight(1f))
            }
        }
        item { InfoT("Matrix math", "Q = XWQ, K = XWK, V = XWV. The same sequence creates all three projections for self-attention.") }
    }
}

@Composable
private fun AttentionSection() {
    var selected by remember { mutableIntStateOf(1) }
    var causal by remember { mutableStateOf(false) }
    val state = PhaseEightEngines.attention(causal = causal)
    val row = state.cells[selected]
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Scaled Dot-Product Attention", "dot product -> scale -> softmax -> weighted Values") }
        item { AttentionMatrixVisualizer(state, selected) }
        item { AttentionLinks(state, selected) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderT("Selected query token", selected.toDouble(), 0.0, (state.tokens.lastIndex).toDouble()) { selected = it.toInt() }
                    SegmentedOption(if (causal) "Causal mask on" else "Causal mask off", causal, Modifier.fillMaxWidth()) { causal = !causal }
                }
            }
        }
        item { WeightBars(state, selected) }
        item { InfoT("Weighted value output", row.joinToString("\n") { "${state.tokens[selected]} -> ${state.tokens[it.key]} weight %.2f, dot %.2f, scaled %s".format(it.weight, it.dot, if (it.masked) "masked" else "%.2f".format(it.scaled)) }) }
    }
}

@Composable
private fun HeadsSection() {
    var heads by remember { mutableIntStateOf(2) }
    var selectedHead by remember { mutableIntStateOf(0) }
    val state = PhaseEightEngines.multiHead(heads = heads)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Multi-Head Attention", "Different heads produce different attention matrices") }
        item {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                (0 until heads).forEach { SegmentedOption("Head ${it + 1}", selectedHead == it) { selectedHead = it } }
            }
        }
        item { AttentionMatrixVisualizer(state.heads[selectedHead], 1) }
        item { SliderT("Heads", heads.toDouble(), 1.0, 4.0) { heads = it.toInt().coerceIn(1, 4); selectedHead = selectedHead.coerceAtMost(heads - 1) } }
        item { InfoT("Shapes", "Model dim 4, heads $heads, concatenated vector length ${state.concatenated.first().size}. Heads do not necessarily map to clean human concepts.") }
    }
}

@Composable
private fun PositionSection() {
    var position by remember { mutableIntStateOf(2) }
    val embedding = PhaseEightEngines.embeddings(listOf("cat"), 4).first()
    val pos = PhaseEightEngines.positional(position, 4)
    val combined = embedding.indices.map { embedding[it] + pos[it] }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Positional Encoding", "Self-attention needs order information added to token vectors") }
        item { PositionHeatmap(position) }
        item { SliderT("Position", position.toDouble(), 0.0, 10.0) { position = it.toInt() } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                VectorCard("Token", embedding, Modifier.weight(1f))
                VectorCard("Position", pos, Modifier.weight(1f))
            }
        }
        item { VectorCard("Combined", combined, Modifier.fillMaxWidth()) }
    }
}

@Composable
private fun BlockSection() {
    val block = PhaseEightEngines.encoderBlock()
    val params = PhaseEightEngines.parameterCount(8, 4, 2, 8)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Transformer Encoder Block", "Attention, residuals, norm, feed-forward, residuals, norm") }
        item { TransformerBlockDiagram() }
        item { VectorCard("Selected token after attention", block.attention.output[1]) }
        item { VectorCard("After residual + LayerNorm", block.residualNorm[1]) }
        item { VectorCard("After feed-forward block", block.output[1]) }
        item { InfoT("Parameter count", params.entries.joinToString { "${it.key}: ${it.value}" }) }
    }
}

@Composable
private fun PredictionSection(onComplete: () -> Unit) {
    var causal by remember { mutableStateOf(true) }
    val pred = PhaseEightEngines.tokenPrediction()
    val attn = PhaseEightEngines.attention(pred.sequence, causal = causal)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Tiny Offline Token Prediction", "Synthetic vocabulary, local softmax, inspectable attention") }
        item { TokenSequenceVisualizer(pred.sequence + listOf("?"), pred.sequence.lastIndex) {} }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                pred.probabilities.forEachIndexed { i, p -> MetricPill(PhaseEightEngines.vocab[i], "%.0f%%".format(p * 100), if (PhaseEightEngines.vocab[i] == pred.predicted) LabGreen else LabCyan, Modifier.weight(1f)) }
            }
        }
        item { AttentionMatrixVisualizer(attn, pred.sequence.lastIndex) }
        item { InfoT("Why this token?", "Expected ${pred.expected}, predicted ${pred.predicted}, loss %.3f. Attention patterns contribute to the representation, but attention weight is not guaranteed causal explanation.".format(pred.loss)) }
        item { SegmentedOption(if (causal) "Causal mask on" else "Causal mask off", causal, Modifier.fillMaxWidth()) { causal = !causal } }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun TokenSequenceVisualizer(tokens: List<String>, selected: Int, onSelect: (Int) -> Unit) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
        tokens.forEachIndexed { i, token -> SegmentedOption(token, i == selected) { onSelect(i) } }
    }
}

@Composable
private fun AttentionMatrixVisualizer(state: AttentionState, selectedRow: Int) {
    Canvas(Modifier.fillMaxWidth().height(240.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val n = state.tokens.size
        val cellW = size.width / n
        val cellH = size.height / n
        state.cells.forEachIndexed { r, row ->
            row.forEachIndexed { c, cell ->
                val alpha = if (cell.masked) .08f else cell.weight.toFloat().coerceIn(.05f, .95f)
                drawRect((if (r == selectedRow) LabOrange else LabCyan).copy(alpha = alpha), Offset(c * cellW, r * cellH), androidx.compose.ui.geometry.Size(cellW - 2f, cellH - 2f))
                if (cell.masked) drawLine(LabPink, Offset(c * cellW, r * cellH), Offset((c + 1) * cellW, (r + 1) * cellH), 2f)
            }
        }
    }
}

@Composable
private fun AttentionLinks(state: AttentionState, selected: Int) {
    Canvas(Modifier.fillMaxWidth().height(120.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val n = state.tokens.size
        val y = size.height * .55f
        val xs = List(n) { size.width * (it + .5f) / n }
        xs.forEachIndexed { i, x -> drawCircle(if (i == selected) LabOrange else LabPurple, 10f, Offset(x, y)) }
        state.cells[selected].forEachIndexed { i, cell ->
            if (!cell.masked) drawLine(LabCyan.copy(alpha = .25f + cell.weight.toFloat() * .7f), Offset(xs[selected], y - 12f), Offset(xs[i], y - 36f), (1f + cell.weight.toFloat() * 8f), cap = StrokeCap.Round)
        }
    }
}

@Composable
private fun WeightBars(state: AttentionState, selected: Int) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text("Attention Weight Bars", color = LabText, fontWeight = FontWeight.Bold)
            state.cells[selected].forEachIndexed { i, cell ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text("${state.tokens[selected]} -> ${state.tokens[i]}", color = LabMuted, fontSize = 11.sp, modifier = Modifier.weight(1f))
                    Box(Modifier.weight(2f).height(14.dp).background(LabCyan.copy(alpha = cell.weight.toFloat().coerceIn(.05f, .95f)), RoundedCornerShape(4.dp)))
                    Text("%.0f%%".format(cell.weight * 100), color = LabText, fontSize = 11.sp)
                }
            }
        }
    }
}

@Composable
private fun VectorCard(title: String, vector: List<Double>, modifier: Modifier = Modifier.fillMaxWidth()) {
    GlassPanel(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            vector.forEach { v -> Box(Modifier.fillMaxWidth().height(12.dp).background((if (v >= 0) LabCyan else LabPink).copy(alpha = kotlin.math.abs(v).toFloat().coerceIn(.08f, .9f)), RoundedCornerShape(4.dp))) }
            Text(vector.joinToString { "%.2f".format(it) }, color = LabMuted, fontSize = 11.sp)
        }
    }
}

@Composable
private fun PositionHeatmap(position: Int) {
    val rows = (0..10).map { PhaseEightEngines.positional(it, 8) }
    Canvas(Modifier.fillMaxWidth().height(170.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val h = size.height / rows.size
        val w = size.width / rows.first().size
        rows.forEachIndexed { r, row ->
            row.forEachIndexed { c, v ->
                drawRect((if (v >= 0) LabGreen else LabPink).copy(alpha = kotlin.math.abs(v).toFloat().coerceIn(.08f, .9f)), Offset(c * w, r * h), androidx.compose.ui.geometry.Size(w - 2f, h - 2f))
                if (r == position) drawRect(LabOrange, Offset(c * w, r * h), androidx.compose.ui.geometry.Size(w - 2f, h - 2f), style = Stroke(2f))
            }
        }
    }
}

@Composable
private fun TransformerBlockDiagram() {
    InfoT("Encoder flow", "Input -> Multi-Head Self-Attention -> Add & Norm -> Feed Forward -> Add & Norm -> Output. Residual connections add the original representation back before normalization.")
}

@Composable
private fun SliderT(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: %.0f".format(value), color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun InfoT(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private fun defaultTransformerSection(concept: PhaseEightConcept) = when (concept) {
    PhaseEightConcept.Attention -> TransformerSection.Tokens
    PhaseEightConcept.Qkv -> TransformerSection.Qkv
    PhaseEightConcept.ScaledAttention, PhaseEightConcept.SelfAttention, PhaseEightConcept.AttentionMatrix -> TransformerSection.Matrix
    PhaseEightConcept.MultiHead -> TransformerSection.Heads
    PhaseEightConcept.PositionalEncoding -> TransformerSection.Position
    PhaseEightConcept.EncoderBlock -> TransformerSection.Block
    PhaseEightConcept.DecoderCausal, PhaseEightConcept.TokenPrediction -> TransformerSection.Predict
}
