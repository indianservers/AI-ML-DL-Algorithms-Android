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
import androidx.compose.ui.geometry.Size
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
import kotlin.math.ln
import kotlin.math.pow

private enum class DlSection(val label: String) { Neuron("Neuron"), Activations("Activations"), Mlp("MLP"), Forward("Forward"), Backprop("Backprop"), Train("Train"), Regularize("Regularize") }

@Composable
fun PhaseFiveDeepLearningLab(
    topic: LearnTopic,
    concept: PhaseFiveConcept,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(defaultSection(concept)) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("Deep Learning Foundations - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "Phase 5", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                DlSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            DlSection.Neuron -> SingleNeuronSection()
            DlSection.Activations -> ActivationSection()
            DlSection.Mlp -> MlpSection()
            DlSection.Forward -> ForwardSection()
            DlSection.Backprop -> BackpropSection()
            DlSection.Train -> TrainingSection()
            DlSection.Regularize -> RegularizationSection(onComplete)
        }
    }
}

@Composable
private fun SingleNeuronSection() {
    var x1 by remember { mutableDoubleStateOf(.8) }
    var x2 by remember { mutableDoubleStateOf(-.4) }
    var w1 by remember { mutableDoubleStateOf(.7) }
    var w2 by remember { mutableDoubleStateOf(.3) }
    var bias by remember { mutableDoubleStateOf(.2) }
    var activation by remember { mutableStateOf(DlActivation.Relu) }
    val calc = PhaseFiveEngines.neuron(listOf(x1, x2), listOf(w1, w2), bias, activation)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Single Neuron Lab", "Input -> weighted sum -> bias -> activation -> output") }
        item { NeuralNetworkPlayground(PhaseFiveEngines.forward(listOf(x1, x2), listOf(listOf(listOf(w1, w2))), listOf(listOf(bias)), activation, target = 1.0), selectedLayer = 0, selectedNeuron = 0) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    SliderDl("x1", x1, -1.0, 1.0) { x1 = it }
                    SliderDl("x2", x2, -1.0, 1.0) { x2 = it }
                    SliderDl("w1", w1, -2.0, 2.0) { w1 = it }
                    SliderDl("w2", w2, -2.0, 2.0) { w2 = it }
                    SliderDl("bias", bias, -2.0, 2.0) { bias = it }
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        DlActivation.entries.forEach { SegmentedOption(it.label, activation == it) { activation = it } }
                    }
                }
            }
        }
        item { NeuronCalculationCard(calc) }
    }
}

@Composable
private fun ActivationSection() {
    var z by remember { mutableDoubleStateOf(-1.2) }
    var activation by remember { mutableStateOf(DlActivation.Sigmoid) }
    var gradient by remember { mutableStateOf(false) }
    val softmax = PhaseFiveEngines.softmax(listOf(2.1, 1.2, z))
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Activation Functions", "How z becomes a neuron output") }
        item { ActivationGraph(activation, z, gradient) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderDl("z", z, -6.0, 6.0) { z = it }
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        DlActivation.entries.forEach { SegmentedOption(it.label, activation == it) { activation = it } }
                    }
                    SegmentedOption(if (gradient) "Gradient shown" else "Show Gradient", gradient, Modifier.fillMaxWidth()) { gradient = !gradient }
                }
            }
        }
        item {
            InfoDl("Softmax", "Cat %.0f%%, Dog %.0f%%, Bird %.0f%%. Probabilities sum to %.3f.".format(softmax[0] * 100, softmax[1] * 100, softmax[2] * 100, softmax.sum()))
        }
    }
}

@Composable
private fun MlpSection() {
    var hiddenLayers by remember { mutableIntStateOf(2) }
    var neurons by remember { mutableIntStateOf(3) }
    val sizes = listOf(2) + List(hiddenLayers) { neurons } + listOf(1)
    val (weights, biases) = PhaseFiveEngines.initialize(sizes, 8)
    val state = PhaseFiveEngines.forward(listOf(.5, .8), weights, biases, DlActivation.Relu, 1.0)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Neural Network Playground", "Every connection is a real weight; every neuron is a real calculation") }
        item { NeuralNetworkPlayground(state, selectedLayer = 0, selectedNeuron = 1) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderDl("Hidden layers", hiddenLayers.toDouble(), 1.0, 3.0) { hiddenLayers = it.toInt().coerceIn(1, 3) }
                    SliderDl("Neurons per layer", neurons.toDouble(), 1.0, 8.0) { neurons = it.toInt().coerceIn(1, 8) }
                }
            }
        }
        item { InfoDl("Why hidden layers?", "A single perceptron draws one linear boundary. Hidden neurons build intermediate features that the output neuron can combine into nonlinear decisions such as XOR.") }
        item { XorCanvas(hidden = true) }
    }
}

@Composable
private fun ForwardSection() {
    val (weights, biases) = PhaseFiveEngines.initialize(listOf(2, 3, 1), 4)
    val state = PhaseFiveEngines.forward(listOf(.5, .8), weights, biases, target = 1.0)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Forward Propagation", "Layer values are computed in sequence") }
        item { NeuralNetworkPlayground(state, selectedLayer = 1, selectedNeuron = 0) }
        item { MatrixMathCard(state) }
        item {
            InfoDl(
                "Forward Step",
                "Input ${state.inputs.map { "%.2f".format(it) }} -> hidden ${state.layers.first().activations.map { "%.2f".format(it) }} -> output %.3f -> loss %.3f".format(state.prediction, state.loss)
            )
        }
    }
}

@Composable
private fun BackpropSection() {
    val (weights, biases) = PhaseFiveEngines.initialize(listOf(2, 2, 1), 4)
    val state = PhaseFiveEngines.forward(listOf(.7, -.2), weights, biases, target = 1.0)
    val back = PhaseFiveEngines.backprop(state)
    val check = PhaseFiveEngines.finiteDifferenceGradient()
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Backpropagation", "Loss -> gradients -> weight updates") }
        item { NeuralNetworkPlayground(state, selectedLayer = 0, selectedNeuron = 0, gradients = back.weightGradients) }
        item { InfoDl("What just happened?", back.explanation) }
        item { InfoDl("Chain rule path", "dL/dw = dL/dy x dy/dh x dh/dw. Output gradient %.4f, first hidden gradient %.4f.".format(back.outputGradient, back.hiddenGradients.first())) }
        item { InfoDl("Finite difference check", "Backprop %.6f vs numerical %.6f".format(check.first, check.second)) }
    }
}

@Composable
private fun TrainingSection() {
    var lr by remember { mutableDoubleStateOf(.08) }
    val sgd = PhaseFiveEngines.trainingTrace(DlOptimizerKind.Sgd, 45, lr)
    val adam = PhaseFiveEngines.trainingTrace(DlOptimizerKind.Adam, 45, lr)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Training", "SGD vs Adam on the same simplified loss slice") }
        item { LossTraceChart(sgd.losses, adam.losses) }
        item { SliderDl("Learning rate", lr, .001, .4) { lr = it } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("SGD loss", "%.3f".format(sgd.losses.last()), LabCyan, Modifier.weight(1f))
                MetricPill("Adam loss", "%.3f".format(adam.losses.last()), LabGreen, Modifier.weight(1f))
                MetricPill("Epoch", "45", LabOrange, Modifier.weight(1f))
            }
        }
        item { InfoDl("Adam internals", "For one representative weight, Adam tracks a first moment, second moment, and an effective update. It is adaptive, not magically always better.") }
    }
}

@Composable
private fun RegularizationSection(onComplete: () -> Unit) {
    var dropout by remember { mutableDoubleStateOf(.25) }
    var l2 by remember { mutableDoubleStateOf(.02) }
    val (weights, biases) = PhaseFiveEngines.initialize(listOf(2, 5, 1), 11)
    val mask = PhaseFiveEngines.dropoutMask(5, dropout, 6)
    val state = PhaseFiveEngines.forward(listOf(.4, -.7), weights, biases, dropoutMask = listOf(mask), target = 0.0)
    val penalty = PhaseFiveEngines.l2Penalty(weights, l2)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Dropout / L2", "Regularization changes paths and weight pressure") }
        item { NeuralNetworkPlayground(state, selectedLayer = 0, selectedNeuron = 0) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderDl("Dropout rate", dropout, 0.0, .7) { dropout = it }
                    SliderDl("L2 lambda", l2, 0.0, .2) { l2 = it }
                    Text("Active hidden neurons: ${mask.count { it }}/${mask.size}", color = LabMuted, fontSize = 12.sp)
                    Text("Data loss %.3f + L2 penalty %.3f = total %.3f".format(state.loss, penalty, state.loss + penalty), color = LabMuted, fontSize = 12.sp)
                }
            }
        }
        item { InfoDl("Break the network", "Too much dropout removes too many paths. A deep sigmoid network can shrink gradients. Zero weights make neurons symmetric. High learning rate can bounce around the loss.") }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun NeuralNetworkPlayground(state: NetworkState, selectedLayer: Int, selectedNeuron: Int, gradients: List<List<List<Double>>> = emptyList()) {
    Canvas(Modifier.fillMaxWidth().height(280.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        val layerValues = listOf(state.inputs) + state.layers.map { it.activations }
        val xs = layerValues.indices.map { size.width * (it + 1) / (layerValues.size + 1) }
        fun nodeY(layer: Int, index: Int): Float = size.height * (index + 1) / (layerValues[layer].size + 1)
        state.layers.forEachIndexed { layer, dense ->
            dense.weights.forEachIndexed { neuron, ws ->
                ws.forEachIndexed { input, weight ->
                    val grad = gradients.getOrNull(layer)?.getOrNull(neuron)?.getOrNull(input) ?: 0.0
                    val color = if (weight >= 0) LabCyan else LabPink
                    val stroke = (1.2f + abs(weight).toFloat() * 3f + abs(grad).toFloat() * 8f).coerceIn(1f, 8f)
                    drawLine(color.copy(alpha = .25f + abs(weight).toFloat().coerceIn(0f, 1f) * .45f), Offset(xs[layer], nodeY(layer, input)), Offset(xs[layer + 1], nodeY(layer + 1, neuron)), stroke, cap = StrokeCap.Round)
                }
            }
        }
        layerValues.forEachIndexed { layer, values ->
            values.forEachIndexed { i, value ->
                val dropped = layer > 0 && state.layers.getOrNull(layer - 1)?.dropoutMask?.getOrNull(i) == false
                val selected = layer - 1 == selectedLayer && i == selectedNeuron
                drawCircle(if (selected) Color.White else if (dropped) LabMuted else LabPurple, if (selected) 15f else 11f, Offset(xs[layer], nodeY(layer, i)))
                drawCircle(if (value >= 0) LabGreen else LabOrange, 6f, Offset(xs[layer], nodeY(layer, i)))
            }
        }
    }
}

@Composable
private fun NeuronCalculationCard(calc: NeuronCalculation) {
    InfoDl("Selected neuron calculation", "z = (${calc.inputs[0].format()} x ${calc.weights[0].format()}) + (${calc.inputs[1].format()} x ${calc.weights[1].format()}) + ${calc.bias.format()} = ${calc.z.format()}\n${calc.activation.label}(${calc.z.format()}) = ${calc.output.format()}")
}

@Composable
private fun ActivationGraph(activation: DlActivation, z: Double, gradient: Boolean) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun sx(x: Double) = (size.width * (x + 6.0) / 12.0).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.1) / 2.4)).toFloat()
        val path = Path()
        for (i in 0..160) {
            val x = -6.0 + 12.0 * i / 160.0
            val y = if (gradient) PhaseFiveEngines.derivative(x, activation) else PhaseFiveEngines.activate(x, activation)
            if (i == 0) path.moveTo(sx(x), sy(y)) else path.lineTo(sx(x), sy(y))
        }
        drawPath(path, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
        val y = if (gradient) PhaseFiveEngines.derivative(z, activation) else PhaseFiveEngines.activate(z, activation)
        drawCircle(LabOrange, 9f, Offset(sx(z), sy(y)))
    }
}

@Composable
private fun MatrixMathCard(state: NetworkState) {
    val layer = state.layers.first()
    InfoDl("Matrix view", "z = W x + b\nFirst hidden row: [${layer.weights.first().joinToString { it.format() }}] x [${state.inputs.joinToString { it.format() }}] + ${layer.biases.first().format()} = ${layer.z.first().format()}")
}

@Composable
private fun XorCanvas(hidden: Boolean) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun sx(x: Double) = (size.width * (x + 1.2) / 2.4).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.2) / 2.4)).toFloat()
        for (ix in 0..28) for (iy in 0..28) {
            val x = -1.1 + 2.2 * ix / 28.0
            val y = -1.1 + 2.2 * iy / 28.0
            val p = PhaseFiveEngines.xorPrediction(x, y, hidden)
            drawRect(if (p > .5) LabCyan.copy(alpha = .12f) else LabPink.copy(alpha = .12f), Offset(sx(x), sy(y)), Size(size.width / 28f + 1f, size.height / 28f + 1f))
        }
        PhaseFiveEngines.xorData().forEach { p -> drawCircle(if (p.label == 1) LabCyan else LabPink, 9f, Offset(sx(p.x), sy(p.y))) }
    }
}

@Composable
private fun LossTraceChart(sgd: List<Double>, adam: List<Double>) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun drawLoss(values: List<Double>, color: Color) {
            val maxLoss = values.max().coerceAtLeast(.001)
            val path = Path()
            values.forEachIndexed { i, loss ->
                val x = size.width * i / values.lastIndex
                val y = size.height * (loss / maxLoss).toFloat().coerceIn(0f, 1f)
                if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, color, style = Stroke(4f, cap = StrokeCap.Round))
        }
        drawLoss(sgd, LabCyan)
        drawLoss(adam, LabGreen)
    }
}

@Composable
private fun SliderDl(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: ${value.format()}", color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun InfoDl(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private fun defaultSection(concept: PhaseFiveConcept) = when (concept) {
    PhaseFiveConcept.ArtificialNeuron -> DlSection.Neuron
    PhaseFiveConcept.ActivationFunctions -> DlSection.Activations
    PhaseFiveConcept.Mlp -> DlSection.Mlp
    PhaseFiveConcept.ForwardPropagation -> DlSection.Forward
    PhaseFiveConcept.Backpropagation -> DlSection.Backprop
    PhaseFiveConcept.LossFunctions -> DlSection.Train
    PhaseFiveConcept.Optimizers -> DlSection.Train
    PhaseFiveConcept.Regularization -> DlSection.Regularize
}

private fun Double.format(): String = "%.3f".format(this)
