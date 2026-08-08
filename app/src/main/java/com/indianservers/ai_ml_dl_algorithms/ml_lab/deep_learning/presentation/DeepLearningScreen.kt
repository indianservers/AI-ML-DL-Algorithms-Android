package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.presentation

import android.content.Context
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.data.DeepLearningContent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Activation
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.EpochSnapshot
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Initializer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.LossFunction
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NetworkTrace
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NeuralNetwork
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NeuralSample
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.OptimizerType
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.presentation.ArchitectureLabs
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.presentation.ModernArchitectureLabs
import kotlin.math.ln

private enum class DeepPage(val title: String) {
    Modern("Modern"), Architectures("CNN/RNN"), Playground("MLP"), Forward("Forward"), Backprop("Backprop"), Activations("Activations"),
    Losses("Losses"), Optimizers("Optimizers"), Health("Health")
}

@Composable
fun DeepLearningScreen() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("ml_lab_phase_two", Context.MODE_PRIVATE) }
    var page by remember { mutableStateOf(DeepPage.Modern) }
    var sizes by remember { mutableStateOf(listOf(2, 4, 4, 1)) }
    var activation by remember { mutableStateOf(Activation.Tanh) }
    var initializer by remember { mutableStateOf(Initializer.Xavier) }
    var optimizer by remember { mutableStateOf(OptimizerType.Adam) }
    var learningRate by remember { mutableFloatStateOf(0.03f) }
    var batchSize by remember { mutableIntStateOf(4) }
    var clipGradients by remember { mutableStateOf(true) }
    var networkKey by remember { mutableIntStateOf(0) }
    var network by remember(networkKey) {
        mutableStateOf(NeuralNetwork(sizes, activation, Activation.Sigmoid, LossFunction.BinaryCrossEntropy, initializer, 42))
    }
    var snapshots by remember(networkKey) { mutableStateOf(emptyList<EpochSnapshot>()) }
    var selectedSample by remember { mutableIntStateOf(1) }
    var inspectedLayer by remember { mutableIntStateOf(0) }
    var inspectedNeuron by remember { mutableIntStateOf(0) }
    val sample = DeepLearningContent.xor[selectedSample]
    val trace = network.trace(sample)

    fun rebuild(newSizes: List<Int> = sizes) {
        sizes = newSizes.map { it.coerceIn(1, 32) }
        networkKey++
    }

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Column {
                    Text("Deep Learning", color = LabText, fontSize = 27.sp, fontWeight = FontWeight.Bold)
                    Text("Build - Train - Inspect - Understand", color = LabMuted, fontSize = 13.sp)
                }
                Text("Phase 4", color = LabPink, fontWeight = FontWeight.Bold)
            }
        }
        item {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                DeepPage.entries.forEach { item -> SegmentedOption(item.title, page == item) { page = item } }
            }
        }
        when (page) {
            DeepPage.Modern -> item { ModernArchitectureLabs() }
            DeepPage.Architectures -> item { ArchitectureLabs() }
            DeepPage.Playground -> playgroundItems(
                sizes, activation, initializer, optimizer, learningRate, batchSize, clipGradients,
                network, snapshots,
                onPreset = { preset -> activation = preset.activation; rebuild(preset.sizes) },
                onSizes = { rebuild(it) },
                onActivation = { activation = it; rebuild() },
                onInitializer = { initializer = it; rebuild() },
                onOptimizer = { optimizer = it },
                onLearningRate = { learningRate = it },
                onBatch = { batchSize = it },
                onClip = { clipGradients = it },
                onStep = { network.trainBatch(DeepLearningContent.xor, learningRate, optimizer, if (clipGradients) 5f else null); snapshots = snapshots + snapshot(network, snapshots.size + 1) },
                onTrain = { snapshots = network.train(DeepLearningContent.xor, 600, learningRate, optimizer, batchSize, if (clipGradients) 5f else null, 20) },
                onReset = { rebuild() },
                onSave = {
                    prefs.edit().putString("architecture", sizes.joinToString(",")).putFloat("accuracy", network.evaluate(DeepLearningContent.xor).second).apply()
                }
            )
            DeepPage.Forward -> forwardItems(trace, selectedSample, inspectedLayer, inspectedNeuron,
                onSample = { selectedSample = it },
                onLayer = { inspectedLayer = it.coerceIn(0, network.layers.lastIndex); inspectedNeuron = 0 },
                onNeuron = { inspectedNeuron = it })
            DeepPage.Backprop -> backpropItems(network, sample, trace, learningRate, optimizer)
            DeepPage.Activations -> activationItems()
            DeepPage.Losses -> lossItems()
            DeepPage.Optimizers -> optimizerItems()
            DeepPage.Health -> healthItems(network, snapshots, prefs.getString("architecture", null), prefs.getFloat("accuracy", -1f))
        }
        if (page != DeepPage.Modern && page != DeepPage.Architectures) {
            item {
                GlassPanel(Modifier.fillMaxWidth()) {
                    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                        SectionTitle("Coming Later", "Phase 4 modern architectures")
                        Text("Attention - Transformers - Vision Transformers - GANs - Diffusion - Graph Neural Networks", color = LabMuted, fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.playgroundItems(
    sizes: List<Int>, activation: Activation, initializer: Initializer, optimizer: OptimizerType,
    learningRate: Float, batchSize: Int, clip: Boolean, network: NeuralNetwork, snapshots: List<EpochSnapshot>,
    onPreset: (com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.data.NetworkPreset) -> Unit,
    onSizes: (List<Int>) -> Unit, onActivation: (Activation) -> Unit, onInitializer: (Initializer) -> Unit,
    onOptimizer: (OptimizerType) -> Unit, onLearningRate: (Float) -> Unit, onBatch: (Int) -> Unit,
    onClip: (Boolean) -> Unit, onStep: () -> Unit, onTrain: () -> Unit, onReset: () -> Unit, onSave: () -> Unit
) {
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Neural Network Playground", "A real dense network running fully on-device")
                NetworkDiagram(network)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    MetricPill("Architecture", sizes.joinToString(" -> "), LabPurple, Modifier.weight(1.4f))
                    MetricPill("Parameters", network.parameterCount().toString(), LabCyan, Modifier.weight(1f))
                    MetricPill("FLOPs", network.flopEstimate().toString(), LabGreen, Modifier.weight(1f))
                }
            }
        }
    }
    item {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Presets", "Editable, phone-safe architectures")
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                DeepLearningContent.presets.forEach { preset -> SegmentedOption(preset.name, sizes == preset.sizes) { onPreset(preset) } }
            }
        }
    }
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Network Builder", "Up to five hidden layers and 32 neurons per layer")
                sizes.drop(1).dropLast(1).forEachIndexed { index, neurons ->
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Hidden ${index + 1}", color = LabText, modifier = Modifier.weight(1f))
                        SegmentedOption("-", false) { onSizes(sizes.toMutableList().also { it[index + 1] = (neurons - 1).coerceAtLeast(1) }) }
                        Text(neurons.toString(), color = LabCyan, fontWeight = FontWeight.Bold)
                        SegmentedOption("+", false) { onSizes(sizes.toMutableList().also { it[index + 1] = (neurons + 1).coerceAtMost(32) }) }
                        SegmentedOption("Remove", false) { onSizes(sizes.toMutableList().also { it.removeAt(index + 1) }) }
                    }
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    SegmentedOption("Add Layer", false, Modifier.weight(1f)) { if (sizes.size < 7) onSizes(sizes.toMutableList().also { it.add(it.lastIndex, 4) }) }
                    SegmentedOption("Reset 2-4-4-1", false, Modifier.weight(1f)) { onSizes(listOf(2, 4, 4, 1)) }
                }
                Text("Hidden activation", color = LabMuted, fontSize = 12.sp)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    listOf(Activation.Sigmoid, Activation.Tanh, Activation.ReLU, Activation.LeakyReLU).forEach { SegmentedOption(it.label, activation == it) { onActivation(it) } }
                }
                Text("Initialization", color = LabMuted, fontSize = 12.sp)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    Initializer.entries.forEach { SegmentedOption(it.label, initializer == it) { onInitializer(it) } }
                }
            }
        }
    }
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SectionTitle("Live Training", "XOR - binary cross entropy - deterministic seed 42")
                DecisionBoundary(network, DeepLearningContent.xor)
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    OptimizerType.entries.forEach { SegmentedOption(it.label, optimizer == it) { onOptimizer(it) } }
                }
                Text("Learning rate %.4f".format(learningRate), color = LabMuted, fontSize = 12.sp)
                Slider(learningRate, onLearningRate, valueRange = 0.001f..0.2f)
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Batch size $batchSize", color = LabMuted)
                    Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                        listOf(1, 2, 4).forEach { SegmentedOption(it.toString(), batchSize == it) { onBatch(it) } }
                    }
                }
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
                    Text("Clip gradient norm at 5.0", color = LabMuted)
                    Switch(clip, onClip)
                }
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    SegmentedOption("Single batch", false, Modifier.weight(1f), onStep)
                    GradientButton("Train 600", Modifier.weight(1f), onTrain)
                    SegmentedOption("Reset", false, Modifier.weight(1f), onReset)
                }
                SegmentedOption("Save Model", false, Modifier.fillMaxWidth(), onSave)
            }
        }
    }
    item {
        val metrics = network.evaluate(DeepLearningContent.xor)
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SectionTitle("Training Dashboard", "The boundary and metrics come from the current weights")
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    MetricPill("Loss", "%.4f".format(metrics.first), LabPink, Modifier.weight(1f))
                    MetricPill("Accuracy", "%.0f%%".format(metrics.second * 100f), LabGreen, Modifier.weight(1f))
                    MetricPill("Grad norm", "%.4f".format(network.gradientNorm()), LabOrange, Modifier.weight(1f))
                }
                EpochChart(snapshots)
                DeepLearningContent.xor.forEach { sample ->
                    Text("[${sample.input.joinToString { "%.0f".format(it) }}] -> %.3f  target %.0f".format(network.predict(sample.input)[0], sample.target[0]), color = LabMuted, fontSize = 12.sp)
                }
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.forwardItems(
    trace: NetworkTrace, sampleIndex: Int, layerIndex: Int, neuronIndex: Int,
    onSample: (Int) -> Unit, onLayer: (Int) -> Unit, onNeuron: (Int) -> Unit
) {
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Forward Propagation Explorer", "Tap through one sample, layer and neuron at a time")
                Row(horizontalArrangement = Arrangement.spacedBy(7.dp)) { (0..3).forEach { SegmentedOption("Sample ${it + 1}", sampleIndex == it) { onSample(it) } } }
                Text("Input [${trace.layers.first().input.joinToString { "%.2f".format(it) }}]", color = LabCyan)
                trace.layers.forEachIndexed { index, layer ->
                    SegmentedOption("Layer ${index + 1}: ${layer.input.size} -> ${layer.output.size}", layerIndex == index, Modifier.fillMaxWidth()) { onLayer(index) }
                }
                val layer = trace.layers[layerIndex.coerceIn(trace.layers.indices)]
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    layer.output.indices.forEach { SegmentedOption("N${it + 1}", neuronIndex == it) { onNeuron(it) } }
                }
                val neuron = neuronIndex.coerceIn(layer.output.indices)
                Text("z = sum(w*x) + b = %.6f".format(layer.weighted[neuron]), color = LabText, fontWeight = FontWeight.Bold)
                Text("activation(z) = %.6f".format(layer.output[neuron]), color = LabGreen)
                Text("Prediction %.6f - target %.1f - loss %.6f".format(trace.prediction[0], trace.target[0], trace.loss), color = LabMuted)
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.backpropItems(network: NeuralNetwork, sample: NeuralSample, trace: NetworkTrace, learningRate: Float, optimizer: OptimizerType) {
    item {
        var stage by remember { mutableIntStateOf(0) }
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Backpropagation Explorer", "Exact values from the current 2 -> 4 -> 4 -> 1 network")
                val stages = listOf("Forward pass", "Calculate loss", "Output gradient", "Activation gradient", "Weight gradient", "Bias gradient", "Propagate backward", "Update parameters")
                stages.forEachIndexed { index, title ->
                    Text("${index + 1}. $title", color = if (index == stage) LabPink else LabMuted, fontWeight = if (index == stage) FontWeight.Bold else FontWeight.Normal)
                }
                Text("dL/dw = dL/da * da/dz * dz/dw", color = LabCyan, fontWeight = FontWeight.Bold)
                Text("Prediction %.5f, target %.1f, BCE loss %.5f".format(trace.prediction[0], sample.target[0], trace.loss), color = LabText)
                val output = network.layers.last()
                Text("Output weight %.5f - gradient %.6f - update %.6f".format(output.weights.values.first(), output.weights.gradients.first(), -learningRate * output.weights.gradients.first()), color = LabMuted)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    SegmentedOption("Back", false, Modifier.weight(1f)) { stage = (stage - 1).coerceAtLeast(0) }
                    SegmentedOption("Step", true, Modifier.weight(1f)) { if (stage == 2) { network.zeroGrad(); network.forward(sample.input); network.backward(sample.target) }; if (stage == 7) network.trainBatch(listOf(sample), learningRate, optimizer); stage = (stage + 1).coerceAtMost(7) }
                    SegmentedOption("Reset", false, Modifier.weight(1f)) { stage = 0 }
                }
            }
        }
    }
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SectionTitle("Chain Rule", "Local derivatives compose into an end-to-end gradient")
                Text("x -> multiply by w -> sum + bias -> activation -> prediction -> loss", color = LabText)
                Text("Each node receives an upstream gradient, multiplies it by its local derivative, and passes the result backward.", color = LabMuted, fontSize = 13.sp)
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.activationItems() {
    item {
        var activation by remember { mutableStateOf(Activation.ReLU) }
        var x by remember { mutableFloatStateOf(-2.4f) }
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Activation Function Lab", "Compare output and derivative across the full input range")
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    Activation.entries.forEach { SegmentedOption(it.label, activation == it) { activation = it } }
                }
                ActivationChart(activation, x)
                Text("Input x = %.2f".format(x), color = LabMuted)
                Slider(x, { x = it }, valueRange = -10f..10f)
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    MetricPill("Output", "%.5f".format(activation.apply(x)), LabPurple, Modifier.weight(1f))
                    MetricPill("Derivative", "%.5f".format(activation.derivative(x)), LabCyan, Modifier.weight(1f))
                }
                Text(activation.formula, color = LabText, fontWeight = FontWeight.Bold)
                Text(if (activation == Activation.Sigmoid) "Smooth probability output, but deep saturated units can make gradients vanish." else if (activation == Activation.ReLU) "Fast and sparse, but negative units can become permanently inactive." else "Inspect its range and derivative before choosing it for hidden or output layers.", color = LabMuted, fontSize = 13.sp)
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.lossItems() {
    item {
        var loss by remember { mutableStateOf(LossFunction.BinaryCrossEntropy) }
        var prediction by remember { mutableFloatStateOf(0.72f) }
        val target = floatArrayOf(1f)
        val value = loss.value(floatArrayOf(prediction), target)
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Loss Function Lab", "Turn prediction quality into an optimization signal")
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    LossFunction.entries.forEach { SegmentedOption(it.label, loss == it) { loss = it } }
                }
                LossCurve(loss, prediction)
                Text("Actual value 1 - prediction %.3f".format(prediction), color = LabMuted)
                Slider(prediction, { prediction = it }, valueRange = 0.001f..0.999f)
                MetricPill("${loss.label} loss", "%.6f".format(value), LabPink, Modifier.fillMaxWidth())
                if (loss == LossFunction.BinaryCrossEntropy) Text("L = -[1 log(%.3f) + 0 log(1 - %.3f)] = %.6f".format(prediction, prediction, -ln(prediction)), color = LabText, fontSize = 13.sp)
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.optimizerItems() {
    item {
        val curves = remember {
            listOf(OptimizerType.SGD, OptimizerType.Momentum, OptimizerType.RMSProp, OptimizerType.Adam).associateWith { type ->
                NeuralNetwork(listOf(2, 4, 1), Activation.Tanh, Activation.Sigmoid, LossFunction.BinaryCrossEntropy, Initializer.Xavier, 7)
                    .train(DeepLearningContent.xor, 300, if (type == OptimizerType.SGD) 0.08f else 0.03f, type, 4, 5f, 10)
            }
        }
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                SectionTitle("Optimizer Lab", "Same network, seed and dataset - different update rules")
                OptimizerComparison(curves)
                curves.forEach { (type, values) ->
                    val last = values.last()
                    Text("${type.label}: loss %.4f - accuracy %.0f%%".format(last.loss, last.accuracy * 100f), color = when(type) { OptimizerType.SGD -> LabCyan; OptimizerType.Momentum -> LabPink; OptimizerType.RMSProp -> LabOrange; else -> LabGreen })
                }
                Text("Momentum carries an exponential moving average of gradients. RMSProp scales by recent squared gradients. Adam combines both with bias correction.", color = LabMuted, fontSize = 13.sp)
            }
        }
    }
}

private fun androidx.compose.foundation.lazy.LazyListScope.healthItems(network: NeuralNetwork, snapshots: List<EpochSnapshot>, savedArchitecture: String?, savedAccuracy: Float) {
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SectionTitle("Gradient Health", "Vanishing, exploding, saturation and dead-unit diagnostics")
                network.gradientHealth().forEachIndexed { index, health ->
                    Text("Layer ${index + 1} - norm %.5f - mean %.5f - range %.5f..%.5f".format(health.norm, health.mean, health.min, health.max), color = LabText, fontSize = 13.sp)
                    Text("Dead ReLUs ${health.deadReluCount} - saturated units ${health.saturatedCount}", color = if (health.deadReluCount + health.saturatedCount > 0) LabOrange else LabGreen, fontSize = 12.sp)
                }
                Text("Weight norm %.4f - estimated memory %.1f KB".format(network.weightNorm(), network.parameterCount() * 8f / 1024f), color = LabMuted)
                Text(when {
                    network.gradientNorm() > 25f -> "Warning: gradients are exploding. Lower the learning rate or enable clipping."
                    network.gradientNorm() in 0f..0.00001f && snapshots.isNotEmpty() -> "Early-layer gradients are near zero. Compare sigmoid with ReLU or Xavier with He initialization."
                    else -> "No critical numerical warning in the current snapshot."
                }, color = LabPink)
            }
        }
    }
    item {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                SectionTitle("Saved Neural Model", "Local offline experiment state")
                Text(if (savedArchitecture == null) "No saved model yet. Use Save Model in the playground." else "Architecture ${savedArchitecture.replace(',', '-')} - XOR accuracy %.0f%%".format(savedAccuracy * 100f), color = LabMuted)
                SectionTitle("Capacity and Generalization", "Overfitting lab foundation")
                Text("A larger network can fit noisy training samples while validation loss rises. Compare architecture size, L1/L2 penalties, dropout, and a held-out split before trusting training loss alone.", color = LabMuted, fontSize = 13.sp)
            }
        }
    }
}

@Composable
private fun NetworkDiagram(network: NeuralNetwork) {
    Canvas(Modifier.fillMaxWidth().height(210.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        val sizes = network.sizes
        fun center(layer: Int, neuron: Int): Offset {
            val x = 18f + (size.width - 36f) * layer / sizes.lastIndex.coerceAtLeast(1)
            val visible = sizes[layer].coerceAtMost(8)
            val y = (neuron + 1f) * size.height / (visible + 1f)
            return Offset(x, y)
        }
        for (layer in 0 until sizes.lastIndex) {
            val from = sizes[layer].coerceAtMost(8)
            val to = sizes[layer + 1].coerceAtMost(8)
            for (i in 0 until from) for (j in 0 until to) {
                val weight = network.layers[layer].weights.values[j * sizes[layer] + i]
                drawLine(if (weight >= 0) LabCyan.copy(alpha = 0.35f) else LabPink.copy(alpha = 0.35f), center(layer, i), center(layer + 1, j), 1f + kotlin.math.abs(weight) * 2.2f)
            }
        }
        sizes.forEachIndexed { layer, count ->
            repeat(count.coerceAtMost(8)) { neuron ->
                val accent = if (layer == 0) LabCyan else if (layer == sizes.lastIndex) LabGreen else LabPurple
                drawCircle(accent.copy(alpha = 0.25f), 11f, center(layer, neuron))
                drawCircle(accent, 6f, center(layer, neuron))
            }
            if (count > 8) drawCircle(LabMuted, 3f, Offset(center(layer, 7).x, size.height - 5f))
        }
    }
}

@Composable
private fun DecisionBoundary(network: NeuralNetwork, samples: List<NeuralSample>) {
    Canvas(Modifier.fillMaxWidth().height(230.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val grid = 22
        val cellW = size.width / grid
        val cellH = size.height / grid
        for (gx in 0 until grid) for (gy in 0 until grid) {
            val x = gx / (grid - 1f)
            val y = 1f - gy / (grid - 1f)
            val probability = network.predict(floatArrayOf(x, y))[0]
            drawRect(if (probability >= 0.5f) LabPink.copy(alpha = 0.08f + probability * 0.25f) else LabCyan.copy(alpha = 0.08f + (1f - probability) * 0.25f), Offset(gx * cellW, gy * cellH), androidx.compose.ui.geometry.Size(cellW + 1f, cellH + 1f))
        }
        samples.forEach { sample ->
            val point = Offset(sample.input[0] * size.width, (1f - sample.input[1]) * size.height)
            val color = if (sample.target[0] > 0.5f) LabPink else LabCyan
            drawCircle(Color.White.copy(alpha = 0.8f), 11f, point)
            drawCircle(color, 7f, point)
        }
    }
}

@Composable
private fun EpochChart(snapshots: List<EpochSnapshot>) {
    Canvas(Modifier.fillMaxWidth().height(120.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        if (snapshots.size < 2) return@Canvas
        val maxLoss = snapshots.maxOf { it.loss }.coerceAtLeast(0.01f)
        val path = Path()
        snapshots.forEachIndexed { index, item ->
            val x = index * size.width / snapshots.lastIndex
            val y = size.height * (1f - item.loss / maxLoss)
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
    }
}

@Composable
private fun ActivationChart(activation: Activation, selectedX: Float) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val axisY = size.height / 2f
        drawLine(Color.White.copy(alpha = 0.15f), Offset(0f, axisY), Offset(size.width, axisY), 1f)
        val path = Path()
        repeat(121) { index ->
            val x = -10f + 20f * index / 120f
            val output = activation.apply(x).coerceIn(-5f, 5f)
            val px = size.width * index / 120f
            val py = axisY - output * size.height / 10f
            if (index == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        drawPath(path, LabPurple, style = Stroke(4f, cap = StrokeCap.Round))
        val px = size.width * (selectedX + 10f) / 20f
        val py = axisY - activation.apply(selectedX).coerceIn(-5f, 5f) * size.height / 10f
        drawCircle(LabCyan, 7f, Offset(px, py))
    }
}

@Composable
private fun LossCurve(loss: LossFunction, selectedPrediction: Float) {
    Canvas(Modifier.fillMaxWidth().height(150.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val path = Path()
        repeat(100) { index ->
            val p = 0.001f + 0.998f * index / 99f
            val value = loss.value(floatArrayOf(p), floatArrayOf(1f)).coerceAtMost(7f)
            val x = size.width * index / 99f
            val y = size.height * (1f - value / 7f)
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, LabPink, style = Stroke(4f, cap = StrokeCap.Round))
        val value = loss.value(floatArrayOf(selectedPrediction), floatArrayOf(1f)).coerceAtMost(7f)
        drawCircle(LabCyan, 7f, Offset(size.width * selectedPrediction, size.height * (1f - value / 7f)))
    }
}

@Composable
private fun OptimizerComparison(curves: Map<OptimizerType, List<EpochSnapshot>>) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val colors = listOf(LabCyan, LabPink, LabOrange, LabGreen)
        curves.entries.forEachIndexed { curveIndex, (_, values) ->
            val peak = values.maxOf { it.loss }.coerceAtLeast(0.01f)
            val path = Path()
            values.forEachIndexed { index, item ->
                val x = index * size.width / values.lastIndex
                val y = size.height * (1f - item.loss / peak)
                if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
            }
            drawPath(path, colors[curveIndex], style = Stroke(3f, cap = StrokeCap.Round))
        }
    }
}

private fun snapshot(network: NeuralNetwork, epoch: Int): EpochSnapshot {
    val metrics = network.evaluate(DeepLearningContent.xor)
    return EpochSnapshot(epoch, metrics.first, metrics.second, network.gradientNorm(), network.weightNorm())
}
