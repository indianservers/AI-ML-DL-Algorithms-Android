package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseEightEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenEngines

internal enum class NeuralLanguageVisualization(val title: String) {
    Rnn("RNN"), Lstm("LSTM"), Gru("GRU"), Seq2Seq("Seq2Seq"), Attention("Attention"),
    Transformer("Transformer"), Bert("BERT"), Gpt("GPT"), T5("T5");
    companion object { val byTitle = entries.associateBy { it.title } }
}

@Composable
internal fun StageThreeNeuralLanguageScreen(topic: LearnTopic, kind: NeuralLanguageVisualization) {
    var selected by remember(topic.id) { mutableIntStateOf(if (kind == NeuralLanguageVisualization.Bert) 3 else 2) }
    val tokens = when (kind) {
        NeuralLanguageVisualization.Bert -> listOf("[CLS]", "The", "cat", "[MASK]", "on", "the", "mat", "[SEP]")
        NeuralLanguageVisualization.T5 -> listOf("summarize:", "The", "cat", "sat", "on", "the", "mat")
        else -> listOf("The", "cat", "sat", "on", "the", "mat")
    }
    val index = selected.coerceIn(tokens.indices)
    val input = tokens.mapIndexed { i, token -> (token.length + i % 3) / 8.0 }
    val causal = kind == NeuralLanguageVisualization.Gpt
    val attention = remember(kind, tokens) { PhaseEightEngines.attention(tokens, causal = causal) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, when (kind) {
            NeuralLanguageVisualization.Rnn -> "Pass one hidden state through the token sequence"
            NeuralLanguageVisualization.Lstm -> "Inspect how gates preserve or replace cell memory"
            NeuralLanguageVisualization.Gru -> "Inspect update and reset gates in a compact recurrence"
            NeuralLanguageVisualization.Seq2Seq -> "Follow source encoding into a teacher-forced decoder"
            NeuralLanguageVisualization.Attention -> "Compare decoder alignment weights over source tokens"
            NeuralLanguageVisualization.Transformer -> "Trace positions through attention, residuals, and feed-forward layers"
            NeuralLanguageVisualization.Bert -> "Use both directions to inspect a masked token"
            NeuralLanguageVisualization.Gpt -> "Mask future tokens in next-token prediction"
            NeuralLanguageVisualization.T5 -> "Follow a task prefix through encoder and decoder"
        }) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Token sequence", color = LabText, fontWeight = FontWeight.Bold)
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        tokens.forEachIndexed { i, token -> SegmentedOption(token, i == index) { selected = i } }
                    }
                    Text("Selected: ${tokens[index]}", color = LabMuted, fontSize = 12.sp)
                }
            }
        }
        when (kind) {
            NeuralLanguageVisualization.Rnn, NeuralLanguageVisualization.Lstm, NeuralLanguageVisualization.Gru -> item {
                val rnn = PhaseSevenEngines.rnnForward(input)
                val lstm = PhaseSevenEngines.lstmForward(input)
                val gru = PhaseSevenEngines.gruForward(input)
                GlassPanel(Modifier.fillMaxWidth()) {
                    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                        val explanation = when (kind) {
                            NeuralLanguageVisualization.Rnn -> "One hidden state flows left to right through the words."
                            NeuralLanguageVisualization.Lstm -> "The cell state carries memory through forget, input, and output gates."
                            else -> "Update and reset gates mix the prior hidden state. There is no separate cell state."
                        }
                        Text(explanation, color = LabMuted, fontSize = 12.sp)
                        when (kind) {
                            NeuralLanguageVisualization.Rnn -> {
                                val step = rnn.steps[index]
                                ScoreLine("Previous h", step.previousHidden)
                                ScoreLine("Input × weight", step.inputContribution)
                                ScoreLine("Memory × weight", step.memoryContribution)
                                ScoreLine("New h = tanh(sum)", step.hidden)
                            }
                            NeuralLanguageVisualization.Lstm -> {
                                val step = lstm[index]
                                ScoreLine("Forget gate", step.forget)
                                ScoreLine("Input gate", step.inputGate)
                                ScoreLine("Candidate", step.candidate)
                                ScoreLine("Cell memory", step.cell)
                                ScoreLine("Output gate", step.outputGate)
                                ScoreLine("Hidden state", step.hidden)
                            }
                            else -> {
                                val step = gru[index]
                                ScoreLine("Reset gate", step.resetGate)
                                ScoreLine("Update gate", step.updateGate)
                                ScoreLine("Candidate", step.candidate)
                                ScoreLine("Hidden state", step.hidden)
                            }
                        }
                        StateStrip(when (kind) {
                            NeuralLanguageVisualization.Rnn -> rnn.steps.map { it.hidden }
                            NeuralLanguageVisualization.Lstm -> lstm.map { it.cell }
                            else -> gru.map { it.hidden }
                        }, index)
                    }
                }
            }
            NeuralLanguageVisualization.Seq2Seq -> item {
                val encoded = PhaseSevenEngines.rnnForward(input)
                val target = listOf("Le", "chat", "est", "assis")
                val decoder = PhaseSevenEngines.teacherForcedDecoder(
                    encoded.steps.last().hidden,
                    listOf(0.0) + target.dropLast(1).map { it.length / 6.0 }
                )
                val targetStep = index.coerceAtMost(target.lastIndex)
                GlassPanel(Modifier.fillMaxWidth()) {
                    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                        Text("Encoder → context → teacher-forced decoder", color = LabText, fontWeight = FontWeight.Bold)
                        Text("Encoder hidden states", color = LabMuted, fontSize = 12.sp)
                        StateStrip(encoded.steps.map { it.hidden }, index)
                        ScoreLine("Encoder context", encoded.steps.last().hidden)
                        Text("Decoder hidden states", color = LabMuted, fontSize = 12.sp)
                        StateStrip(decoder, targetStep)
                        ScoreLine("Decoder state at step ${targetStep + 1}", decoder[targetStep])
                        Text("Example target: ${target.take(targetStep + 1).joinToString(" ")}", color = LabCyan)
                        Text("Each decoder state uses the prior target token and encoder context. The French words are supplied teaching data, not predictions from a trained translator.", color = LabMuted, fontSize = 12.sp)
                    }
                }
            }
            else -> item {
                GlassPanel(Modifier.fillMaxWidth()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(when (kind) {
                            NeuralLanguageVisualization.Bert -> "Bidirectional masked-token model"
                            NeuralLanguageVisualization.Gpt -> "Causal next-token model"
                            NeuralLanguageVisualization.T5 -> "Task prefix → encoder → decoder"
                            NeuralLanguageVisualization.Transformer -> "Positions → attention → residual norm → feed-forward"
                            else -> "Decoder query aligns to encoder tokens"
                        }, color = LabText, fontWeight = FontWeight.Bold)
                        if (kind == NeuralLanguageVisualization.Bert) {
                            Text("The cat [MASK] on the mat", color = LabCyan, fontSize = 12.sp)
                            Text("[MASK] can attend to tokens on both sides; [CLS] pools sequence information.", color = LabMuted, fontSize = 12.sp)
                            val candidates=listOf("sat","slept","ran")
                            val vectors=PhaseEightEngines.embeddings(candidates)
                            val query=attention.output[3]
                            val scores=vectors.map { vector -> query.zip(vector).sumOf { it.first*it.second } }
                            val probabilities=StageThreeActivationEngine.softmax(scores)
                            candidates.forEachIndexed { i,candidate -> ScoreLine("P($candidate | context)",probabilities[i]) }
                            Text("Illustrative probabilities from the tiny untrained attention projection.",color=LabMuted,fontSize=11.sp)
                        }
                        if (kind == NeuralLanguageVisualization.Gpt) {
                            Text("Future positions are blocked. Generate one token from the visible prefix.", color = LabMuted, fontSize = 12.sp)
                            Text("Prefix: ${tokens.take(index + 1).joinToString(" ")} → next token", color = LabCyan, fontSize = 12.sp)
                            val candidates=listOf("on","near","under")
                            val vectors=PhaseEightEngines.embeddings(candidates)
                            val query=attention.output[index]
                            val probabilities=StageThreeActivationEngine.softmax(vectors.map { vector -> query.zip(vector).sumOf { it.first*it.second } })
                            candidates.forEachIndexed { i,candidate -> ScoreLine("P(next = $candidate)",probabilities[i]) }
                            Text("Illustrative probabilities from the tiny untrained causal projection.",color=LabMuted,fontSize=11.sp)
                        }
                        if (kind == NeuralLanguageVisualization.T5) {
                            Text("summarize: is a task prefix. Encoder reads input; causal decoder writes summary tokens.", color = LabMuted, fontSize = 12.sp)
                            Text("Illustrative target summary: A cat sat.", color = LabCyan, fontSize = 12.sp)
                        }
                        if (kind == NeuralLanguageVisualization.Transformer) {
                            val block = PhaseEightEngines.encoderBlock(tokens)
                            ScoreLine("Positioned input", block.positioned[index].average())
                            ScoreLine("Attention output", block.attention.output[index].average())
                            ScoreLine("Feed-forward output", block.output[index].average())
                        }
                        AttentionGrid(attention.cells.map { row -> row.map { it.weight } }, index, causal)
                        if(kind==NeuralLanguageVisualization.T5) {
                            val decoder=PhaseEightEngines.attention(listOf("A","cat","sat"),causal=true)
                            Text("Decoder causal self-attention shown below. Cross-attention to the source is part of T5, but is not computed in this panel.",color=LabOrange,fontSize=12.sp)
                            AttentionGrid(decoder.cells.map { row -> row.map { it.weight } },2,true)
                        }
                        Text("Attention for ${tokens[index]}: ${attention.cells[index].map { "%.2f".format(it.weight) }.joinToString(" · ")}",
                            color = LabMuted, fontSize = 11.sp)
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column {
                    Text("Inspect time or query position: ${index + 1}", color = LabText, fontSize = 12.sp)
                    Slider(index.toFloat(), { selected = it.toInt().coerceIn(tokens.indices) }, valueRange = 0f..tokens.lastIndex.toFloat())
                }
            }
        }
    }
}

@Composable private fun ScoreLine(label: String, score: Double) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = LabMuted, fontSize = 12.sp)
        Text("%.3f".format(score), color = LabCyan, fontSize = 12.sp)
    }
}

@Composable private fun StateStrip(values: List<Double>, selected: Int) {
    Canvas(Modifier.fillMaxWidth().height(75.dp).background(Color(0xFF071B37))) {
        val mid = size.height / 2f
        drawLine(Color(0xFF294666), Offset(0f, mid), Offset(size.width, mid), 1.dp.toPx())
        values.forEachIndexed { i, v ->
            val x = (i + .5f) * size.width / values.size
            val y = mid - v.toFloat() * mid * .8f
            val color = if (i == selected) LabCyan else Color(0xFF8466DA)
            drawLine(color, Offset(x, mid), Offset(x, y), 9.dp.toPx())
            drawCircle(color, 4.dp.toPx(), Offset(x, y))
        }
    }
}

@Composable private fun AttentionGrid(weights: List<List<Double>>, selected: Int, causal: Boolean) {
    Canvas(Modifier.fillMaxWidth().height(175.dp).background(Color(0xFF071B37))) {
        val n = weights.size
        val width = size.width / n
        val height = size.height / n
        weights.forEachIndexed { row, cells -> cells.forEachIndexed { col, value ->
            val color = if (causal && col > row) Color(0xFF172036) else
                Color(0.1f + value.toFloat() * .25f, .24f + value.toFloat() * .55f, .5f + value.toFloat() * .5f, 1f)
            drawRect(color, Offset(col * width + 1, row * height + 1), androidx.compose.ui.geometry.Size(width - 2, height - 2))
            if (row == selected) drawRect(Color(0x553BEAFF), Offset(col * width + 1, row * height + 1),
                androidx.compose.ui.geometry.Size(width - 2, height - 2))
        } }
    }
}
