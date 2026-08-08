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
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
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
import kotlin.math.abs

private enum class SeqSection(val label: String) { Timeline("Timeline"), Rnn("RNN Cell"), Gradients("BPTT"), Lstm("LSTM"), Gru("GRU"), Train("Compare") }

@Composable
fun PhaseSevenSequenceLab(
    topic: LearnTopic,
    concept: PhaseSevenConcept,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(defaultSeqSection(concept)) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("Sequence Learning - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "Phase 7", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                SeqSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            SeqSection.Timeline -> TimelineSection()
            SeqSection.Rnn -> RnnCellSection()
            SeqSection.Gradients -> GradientSection()
            SeqSection.Lstm -> LstmSection()
            SeqSection.Gru -> GruSection()
            SeqSection.Train -> CompareSection(onComplete)
        }
    }
}

@Composable
private fun TimelineSection() {
    var preset by remember { mutableStateOf(SequencePreset.Increasing) }
    var length by remember { mutableIntStateOf(6) }
    var timestep by remember { mutableIntStateOf(0) }
    val inputs = PhaseSevenEngines.sequence(preset, length)
    val state = PhaseSevenEngines.rnnForward(inputs, target = inputs.last())
    val selected = state.steps[timestep.coerceIn(state.steps.indices)]
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Sequence Timeline", "Input plus previous memory becomes the new hidden state") }
        item { SequenceTimelineVisualizer(state, timestep) }
        item {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                SequencePreset.entries.forEach { SegmentedOption(it.label, preset == it) { preset = it; timestep = 0 } }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderSeq("Sequence length", length.toDouble(), 3.0, 30.0) { length = it.toInt().coerceIn(3, 30); timestep = timestep.coerceAtMost(length - 1) }
                    SliderSeq("Active timestep", timestep.toDouble(), 0.0, (length - 1).toDouble()) { timestep = it.toInt() }
                }
            }
        }
        item { RnnEquationCard(selected) }
        item { HiddenHeatmap(state.steps.map { listOf(it.hidden, it.previousHidden, it.output) }, listOf("h", "prev h", "out")) }
    }
}

@Composable
private fun RnnCellSection() {
    var x by remember { mutableDoubleStateOf(.8) }
    var h by remember { mutableDoubleStateOf(.3) }
    val step = PhaseSevenEngines.rnnStep(x, h)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("RNN Cell", "The same cell parameters are reused at every timestep") }
        item { RnnCellCanvas(step) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column {
                    SliderSeq("Current input x_t", x, -1.0, 1.0) { x = it }
                    SliderSeq("Previous hidden h_t-1", h, -1.0, 1.0) { h = it }
                }
            }
        }
        item { RnnEquationCard(step) }
        item { InfoSeq("Weight sharing", "Tap Wh conceptually: the same recurrent weight affects t1, t2, t3, and every later step.") }
    }
}

@Composable
private fun GradientSection() {
    var wh by remember { mutableDoubleStateOf(.4) }
    var clip by remember { mutableStateOf(false) }
    val inputs = PhaseSevenEngines.sequence(SequencePreset.Delayed, 12)
    val bptt = PhaseSevenEngines.bptt(inputs, target = 1.0, wh = wh, clip = if (clip) .25 else 10.0)
    val check = PhaseSevenEngines.finiteDifferenceRecurrentGradient()
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Backpropagation Through Time", "Loss sends gradients backward through unrolled hidden states") }
        item { GradientTimeline(bptt.gradients) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderSeq("Recurrent weight Wh", wh, .1, 1.4) { wh = it }
                    SegmentedOption(if (clip) "Gradient clipping on" else "Gradient clipping off", clip, Modifier.fillMaxWidth()) { clip = !clip }
                }
            }
        }
        item {
            InfoSeq(
                "Gradient state",
                "Accumulated recurrent gradient %.4f, after clipping %.4f. %s".format(bptt.recurrentGradient, bptt.clippedGradient, bptt.warning ?: "Gradient flow is stable in this setting.")
            )
        }
        item { InfoSeq("Gradient check", "BPTT %.6f vs finite difference %.6f".format(check.first, check.second)) }
    }
}

@Composable
private fun LstmSection() {
    var forget by remember { mutableDoubleStateOf(.92) }
    var inputGate by remember { mutableDoubleStateOf(.25) }
    var outputGate by remember { mutableDoubleStateOf(.67) }
    val step = PhaseSevenEngines.lstmStep(.6, .2, .8, forget, inputGate, outputGate)
    val sequence = PhaseSevenEngines.lstmForward(PhaseSevenEngines.sequence(SequencePreset.Delayed, 8))
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("LSTM Gates", "Gates modify a persistent cell-state memory highway") }
        item { LstmGateCanvas(step) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column {
                    SliderSeq("Forget gate", forget, 0.0, 1.0) { forget = it }
                    SliderSeq("Input gate", inputGate, 0.0, 1.0) { inputGate = it }
                    SliderSeq("Output gate", outputGate, 0.0, 1.0) { outputGate = it }
                }
            }
        }
        item { InfoSeq("Memory Inspector", "C_prev %.3f -> retained %.3f, candidate %.3f x input %.3f, C_new %.3f, h %.3f".format(step.previousCell, step.forget * step.previousCell, step.candidate, step.inputGate, step.cell, step.hidden)) }
        item { HiddenHeatmap(sequence.map { listOf(it.cell, it.hidden, it.forget, it.inputGate, it.outputGate) }, listOf("C", "h", "f", "i", "o")) }
    }
}

@Composable
private fun GruSection() {
    var reset by remember { mutableDoubleStateOf(.45) }
    var update by remember { mutableDoubleStateOf(.8) }
    val step = PhaseSevenEngines.gruStep(.6, .7, reset, update)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("GRU", "A simpler gated recurrent unit with update and reset gates") }
        item { GruCanvas(step) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column {
                    SliderSeq("Reset gate", reset, 0.0, 1.0) { reset = it }
                    SliderSeq("Update gate", update, 0.0, 1.0) { update = it }
                }
            }
        }
        item { InfoSeq("GRU hidden update", "candidate %.3f, previous h %.3f, update %.3f -> new h %.3f".format(step.candidate, step.previousHidden, step.updateGate, step.hidden)) }
    }
}

@Composable
private fun CompareSection(onComplete: () -> Unit) {
    var hidden by remember { mutableIntStateOf(4) }
    val rnn = PhaseSevenEngines.train(RecurrentModelKind.Rnn, SequencePreset.Delayed, 30, hidden)
    val lstm = PhaseSevenEngines.train(RecurrentModelKind.Lstm, SequencePreset.Delayed, 30, hidden)
    val gru = PhaseSevenEngines.train(RecurrentModelKind.Gru, SequencePreset.Delayed, 30, hidden)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("RNN vs LSTM vs GRU", "Same deterministic sequence, different memory mechanisms") }
        item { LossCompareChart(listOf(rnn, lstm, gru)) }
        item { SliderSeq("Hidden size", hidden.toDouble(), 1.0, 16.0) { hidden = it.toInt().coerceIn(1, 16) } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("RNN", "${rnn.parameterCount} params", LabCyan, Modifier.weight(1f))
                MetricPill("LSTM", "${lstm.parameterCount} params", LabGreen, Modifier.weight(1f))
                MetricPill("GRU", "${gru.parameterCount} params", LabOrange, Modifier.weight(1f))
            }
        }
        item { InfoSeq("Symbol prediction", "A B C A B ? uses one-hot A=[1,0,0], B=[0,1,0], C=[0,0,1]. Output probabilities come from softmax over recurrent output scores.") }
        item { InfoSeq("Break it", "Long sequence + simple RNN can shrink early gradients. Forget gate near zero erases LSTM memory; near one retains it strongly. High recurrent weight can cause exploding gradients.") }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun SequenceTimelineVisualizer(state: SequenceState, active: Int) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val n = state.steps.size
        val gap = size.width / n.coerceAtLeast(1)
        state.steps.forEachIndexed { i, step ->
            val x = gap * (i + .5f)
            val inputY = size.height * .22f
            val hiddenY = size.height * .62f
            drawCircle(if (i == active) LabOrange else LabCyan, 12f, Offset(x, inputY))
            drawLine(LabMuted, Offset(x, inputY + 14f), Offset(x, hiddenY - 14f), 2f)
            if (i > 0) drawLine(LabGreen, Offset(gap * (i - .5f), hiddenY), Offset(x - 12f, hiddenY), 4f, cap = StrokeCap.Round)
            drawCircle(if (i == active) Color.White else LabPurple, 15f, Offset(x, hiddenY))
            drawCircle(if (step.hidden >= 0) LabGreen else LabPink, 7f, Offset(x, hiddenY))
        }
    }
}

@Composable
private fun RnnCellCanvas(step: RnnStepState) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val leftTop = Offset(size.width * .18f, size.height * .28f)
        val leftBottom = Offset(size.width * .18f, size.height * .72f)
        val center = Offset(size.width * .55f, size.height * .5f)
        val out = Offset(size.width * .86f, size.height * .5f)
        drawCircle(LabCyan, 13f, leftTop)
        drawCircle(LabPurple, 13f, leftBottom)
        drawLine(LabCyan, leftTop, center, (2f + abs(step.inputContribution).toFloat() * 4f).coerceIn(2f, 8f))
        drawLine(LabPurple, leftBottom, center, (2f + abs(step.memoryContribution).toFloat() * 4f).coerceIn(2f, 8f))
        drawCircle(LabOrange, 24f, center)
        drawLine(LabGreen, center, out, 5f)
        drawCircle(LabGreen, 13f, out)
    }
}

@Composable
private fun LstmGateCanvas(step: LstmStepState) {
    Canvas(Modifier.fillMaxWidth().height(230.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val y = size.height * .35f
        drawLine(LabGreen, Offset(30f, y), Offset(size.width - 30f, y), (3f + step.forget.toFloat() * 6f), cap = StrokeCap.Round)
        listOf(step.forget to LabPink, step.inputGate to LabCyan, step.outputGate to LabOrange).forEachIndexed { i, (v, color) ->
            val x = size.width * (.25f + i * .25f)
            drawCircle(color.copy(alpha = .25f), 26f, Offset(x, size.height * .68f))
            drawCircle(color, (8f + v.toFloat() * 14f), Offset(x, size.height * .68f))
        }
    }
}

@Composable
private fun GruCanvas(step: GruStepState) {
    Canvas(Modifier.fillMaxWidth().height(210.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val old = Offset(size.width * .18f, size.height * .5f)
        val cand = Offset(size.width * .56f, size.height * .32f)
        val out = Offset(size.width * .82f, size.height * .5f)
        drawCircle(LabPurple, 15f, old)
        drawLine(LabCyan, old, cand, (2f + step.resetGate.toFloat() * 5f))
        drawCircle(LabOrange, 18f, cand)
        drawLine(LabGreen, old, out, (2f + step.updateGate.toFloat() * 6f))
        drawLine(LabOrange, cand, out, (2f + (1f - step.updateGate.toFloat()) * 6f))
        drawCircle(LabGreen, 15f, out)
    }
}

@Composable
private fun RnnEquationCard(step: RnnStepState) {
    InfoSeq("Live RNN equation", "h_t = tanh(Wx*x_t + Wh*h_t-1 + b)\ninput %.3f + memory %.3f + bias %.3f = z %.3f\nh_t = tanh(z) = %.3f, output %.3f".format(step.inputContribution, step.memoryContribution, step.bias, step.z, step.hidden, step.output))
}

@Composable
private fun HiddenHeatmap(rows: List<List<Double>>, labels: List<String>) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Memory Trace", color = LabText, fontWeight = FontWeight.Bold)
            labels.indices.forEach { row ->
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(labels[row], color = LabMuted, fontSize = 11.sp, modifier = Modifier.weight(.5f))
                    rows.forEach { values ->
                        val v = values.getOrElse(row) { 0.0 }
                        Box(Modifier.weight(1f).height(20.dp).background((if (v >= 0) LabCyan else LabPink).copy(alpha = abs(v).toFloat().coerceIn(.08f, .85f)), RoundedCornerShape(4.dp)))
                    }
                }
            }
        }
    }
}

@Composable
private fun GradientTimeline(values: List<Double>) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Gradient Flow Timeline", color = LabText, fontWeight = FontWeight.Bold)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                val maxValue = values.max().coerceAtLeast(1e-9)
                values.forEach { value ->
                    Box(Modifier.weight(1f).height(34.dp).background(LabOrange.copy(alpha = (value / maxValue).toFloat().coerceIn(.08f, .95f)), RoundedCornerShape(5.dp)))
                }
            }
            Text(values.joinToString { "%.4f".format(it) }, color = LabMuted, fontSize = 11.sp)
        }
    }
}

@Composable
private fun LossCompareChart(states: List<SequenceTrainingState>) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val colors = listOf(LabCyan, LabGreen, LabOrange)
        states.forEachIndexed { s, state ->
            val maxLoss = state.losses.max().coerceAtLeast(.001)
            val path = Path()
            state.losses.forEachIndexed { i, loss ->
                val x = size.width * i / state.losses.lastIndex
                val y = size.height * (loss / maxLoss).toFloat().coerceIn(0f, 1f)
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, colors[s], style = Stroke(4f, cap = StrokeCap.Round))
        }
    }
}

@Composable
private fun SliderSeq(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: %.2f".format(value), color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun InfoSeq(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private fun defaultSeqSection(concept: PhaseSevenConcept) = when (concept) {
    PhaseSevenConcept.Rnn, PhaseSevenConcept.HiddenState -> SeqSection.Timeline
    PhaseSevenConcept.Bptt, PhaseSevenConcept.Gradients -> SeqSection.Gradients
    PhaseSevenConcept.Lstm, PhaseSevenConcept.LstmGates -> SeqSection.Lstm
    PhaseSevenConcept.Gru -> SeqSection.Gru
    PhaseSevenConcept.SequencePrediction -> SeqSection.Train
}
