package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.presentation

import android.content.Context
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
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
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.data.PhaseThreeContent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.CnnEpoch
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.Conv2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.DenseAutoencoder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.EmbeddingLayer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.GruCell
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.ImageSample
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.LstmCell
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.Pool2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.PoolingType
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.TensorImage
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.TinyCnnClassifier
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.VanillaRnn
import kotlin.math.abs
import kotlin.system.measureNanoTime

private enum class ArchitectureLab(val title: String) { Vision("CNN Vision"), Sequence("Sequences"), Autoencoder("Autoencoder"), Embeddings("Embeddings"), Inspector("Inspector") }
private enum class SequenceKind { RNN, LSTM, GRU }

@Composable
fun ArchitectureLabs() {
    var lab by remember { mutableStateOf(ArchitectureLab.Vision) }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SectionTitle("Practical Architectures", "Pixels, sequences and representations transformed live")
                Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    ArchitectureLab.entries.forEach { SegmentedOption(it.title, lab == it) { lab = it } }
                }
            }
        }
        when (lab) {
            ArchitectureLab.Vision -> VisionLab()
            ArchitectureLab.Sequence -> SequenceLab()
            ArchitectureLab.Autoencoder -> AutoencoderLab()
            ArchitectureLab.Embeddings -> EmbeddingLab()
            ArchitectureLab.Inspector -> ArchitectureInspector()
        }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(7.dp)) { SectionTitle("Continue to Modern", "Phase 4 is available in the Modern tab"); Text("Attention - Transformers - Vision Transformers - GNNs - VAEs - GANs - Diffusion", color = LabMuted, fontSize = 12.sp) } }
    }
}

@Composable
private fun VisionLab() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("ml_lab_phase_three", Context.MODE_PRIVATE) }
    var cnnKey by remember { mutableIntStateOf(0) }
    val cnn = remember(cnnKey) { TinyCnnClassifier() }
    var sampleIndex by remember { mutableIntStateOf(0) }
    var kernelIndex by remember { mutableIntStateOf(0) }
    var kernelCell by remember { mutableIntStateOf(4) }
    var kernelValue by remember { mutableFloatStateOf(PhaseThreeContent.kernels[0].values[4]) }
    var step by remember { mutableIntStateOf(0) }
    var stride by remember { mutableIntStateOf(1) }
    var padding by remember { mutableIntStateOf(0) }
    var maxPool by remember { mutableStateOf(true) }
    var normalize by remember { mutableStateOf(false) }
    var history by remember(cnnKey) { mutableStateOf(emptyList<CnnEpoch>()) }
    val sample = PhaseThreeContent.shapeDataset[sampleIndex]
    val kernel = PhaseThreeContent.kernels[kernelIndex].values.copyOf().also { it[kernelCell] = kernelValue }
    if (normalize) { val sum = kernel.sum(); if (abs(sum) > 1e-6f) kernel.indices.forEach { kernel[it] /= sum } }
    val explorer = remember(kernel.contentHashCode(), stride, padding) { Conv2D(1, 1, 3, stride, padding).also { kernel.copyInto(it.weights) } }
    val feature = explorer.forward(sample.image)
    val pool = Pool2D(2, 2, if (maxPool) PoolingType.Max else PoolingType.Average).forward(feature)
    val prediction = cnn.predict(sample.image)

    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Convolution Explorer", "The highlighted output is calculated from the active image and kernel")
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                PhaseThreeContent.kernels.forEachIndexed { index, item -> SegmentedOption(item.name, kernelIndex == index) { kernelIndex = index; kernelValue = item.values[kernelCell] } }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ImageGrid(sample.image, "Input 8 x 8", step = step, kernelSize = 3, modifier = Modifier.weight(1f))
                KernelGrid(kernel, kernelCell, Modifier.weight(0.72f)) { kernelCell = it; kernelValue = kernel[it] }
                ImageGrid(feature, "Feature ${feature.height} x ${feature.width}", selectedOutput = step, modifier = Modifier.weight(1f))
            }
            val outRow = step.coerceAtMost(feature.height * feature.width - 1) / feature.width
            val outColumn = step.coerceAtMost(feature.height * feature.width - 1) % feature.width
            val patchValues = buildList {
                for (kr in 0..2) for (kc in 0..2) {
                    val row = outRow * stride + kr - padding; val column = outColumn * stride + kc - padding
                    add(if (row in 0 until sample.image.height && column in 0 until sample.image.width) sample.image[0, row, column] else 0f)
                }
            }
            Text("Patch dot kernel + bias = ${patchValues.indices.joinToString(" + ") { "%.1fx%.1f".format(patchValues[it], kernel[it]) }} = %.3f".format(feature[0, outRow, outColumn]), color = LabText, fontSize = 12.sp)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                SegmentedOption("Previous", false, Modifier.weight(1f)) { step = (step - 1).coerceAtLeast(0) }
                SegmentedOption("Step", true, Modifier.weight(1f)) { step = (step + 1) % (feature.height * feature.width) }
                SegmentedOption("Reset", false, Modifier.weight(1f)) { step = 0 }
            }
            Text("Selected kernel cell ${kernelCell + 1}: %.2f".format(kernelValue), color = LabMuted)
            Slider(kernelValue, { kernelValue = it }, valueRange = -3f..3f)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("Normalize kernel", color = LabMuted); Switch(normalize, { normalize = it })
            }
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Shape, Padding and Pooling", "Output = floor((input + 2P - K) / S) + 1")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                MetricPill("Input", "8 x 8 x 1", LabCyan, Modifier.weight(1f))
                MetricPill("Conv", "${feature.width} x ${feature.height} x 1", LabPurple, Modifier.weight(1f))
                MetricPill("Pool", "${pool.output.width} x ${pool.output.height} x 1", LabGreen, Modifier.weight(1f))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                listOf(1, 2, 3).forEach { SegmentedOption("Stride $it", stride == it, Modifier.weight(1f)) { stride = it; step = 0 } }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                listOf(0, 1, 2).forEach { SegmentedOption("Pad $it", padding == it, Modifier.weight(1f)) { padding = it; step = 0 } }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                SegmentedOption("Max Pool", maxPool, Modifier.weight(1f)) { maxPool = true }
                SegmentedOption("Average Pool", !maxPool, Modifier.weight(1f)) { maxPool = false }
            }
            Text(if (maxPool) "Winner input index for first region: ${pool.winnerIndices.first()}; backward gradient returns only there." else "Average pooling distributes backward gradient equally across all four inputs.", color = LabMuted, fontSize = 12.sp)
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("CNN Playground", "Synthetic line shapes generated offline - trainable Conv, Pool, Dense and Softmax")
            Text("[8x8x1] -> [Conv 3x3x4] -> [ReLU] -> [MaxPool 2x2] -> [Flatten 64] -> [3 classes]", color = LabCyan, fontSize = 12.sp)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                MetricPill("Samples", "72", LabBlue, Modifier.weight(1f)); MetricPill("Parameters", cnn.parameterCount().toString(), LabPurple, Modifier.weight(1f)); MetricPill("Memory", "${cnn.memoryBytes() / 1024f} KB", LabOrange, Modifier.weight(1f))
            }
            ImageGrid(sample.image, "${PhaseThreeContent.classNames[sample.label]} sample", modifier = Modifier.fillMaxWidth())
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                SegmentedOption("Previous image", false, Modifier.weight(1f)) { sampleIndex = (sampleIndex - 1).coerceAtLeast(0) }
                SegmentedOption("Next image", false, Modifier.weight(1f)) { sampleIndex = (sampleIndex + 1) % PhaseThreeContent.shapeDataset.size }
            }
            FeatureGallery(prediction.featureMaps)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                PhaseThreeContent.classNames.forEachIndexed { index, name -> MetricPill(name, "%.1f%%".format(prediction.probabilities[index] * 100), if (index == prediction.predictedClass) LabGreen else LabMuted, Modifier.weight(1f)) }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                GradientButton("Train CNN", Modifier.weight(1f)) { history = cnn.train(PhaseThreeContent.shapeDataset, 24); prefs.edit().putFloat("cnn_accuracy", history.last().accuracy).apply() }
                SegmentedOption("Reset", false, Modifier.weight(1f)) { cnnKey++ }
                SegmentedOption("Save", false, Modifier.weight(1f)) { prefs.edit().putInt("cnn_parameters", cnn.parameterCount()).apply() }
            }
            if (history.isNotEmpty()) {
                val last = history.last(); Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                    MetricPill("Epoch", last.epoch.toString(), LabPurple, Modifier.weight(1f)); MetricPill("Loss", "%.4f".format(last.loss), LabPink, Modifier.weight(1f)); MetricPill("Accuracy", "%.0f%%".format(last.accuracy * 100), LabGreen, Modifier.weight(1f))
                }
                HistoryChart(history)
            }
            OcclusionMap(cnn, sample)
            Text("Occlusion sensitivity: brighter cells caused the largest drop in confidence when hidden.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun SequenceLab() {
    var kind by remember { mutableStateOf(SequenceKind.RNN) }
    var time by remember { mutableIntStateOf(0) }
    var runBptt by remember { mutableIntStateOf(0) }
    val sequence = PhaseThreeContent.sequence
    val rnn = remember { VanillaRnn(1, 4, 1) }
    val lstm = remember { LstmCell(1, 4) }
    val gru = remember { GruCell(1, 4) }
    val rnnTrace = remember(runBptt) { if (runBptt > 0) rnn.bptt(sequence, floatArrayOf(1f), 0.02f) else rnn.forward(sequence) }
    val lstmSteps = remember { lstm.forward(sequence) }
    val gruSteps = remember { gru.forward(sequence) }
    val safeTime = time.coerceIn(sequence.indices)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Sequence Learning Lab", "Follow shared state through time instead of seeing only the final output")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { SequenceKind.entries.forEach { SegmentedOption(it.name, kind == it, Modifier.weight(1f)) { kind = it } } }
            SequenceTimeline(sequence, safeTime)
            Text("Time step ${safeTime + 1} / ${sequence.size}", color = LabMuted); Slider(safeTime.toFloat(), { time = it.toInt() }, valueRange = 0f..sequence.lastIndex.toFloat())
            when (kind) {
                SequenceKind.RNN -> {
                    val step = rnnTrace.steps[safeTime]
                    VectorBars("Previous hidden", step.previousHidden, LabMuted); VectorBars("h_t = tanh(Wx*x_t + Wh*h_(t-1) + b)", step.hidden, LabCyan)
                    Text("Output ${step.output.joinToString { "%.4f".format(it) }} - BPTT loss %.5f".format(rnnTrace.loss), color = LabText)
                    GradientButton("Run BPTT Step", Modifier.fillMaxWidth()) { runBptt++ }
                    Text("Input-gradient norm %.5f - recurrent-gradient norm %.5f".format(norm(rnn.inputGradients), norm(rnn.recurrentGradients)), color = LabOrange)
                }
                SequenceKind.LSTM -> {
                    val step = lstmSteps[safeTime]
                    VectorBars("Forget gate", step.forget, LabPink); VectorBars("Input gate", step.inputGate, LabCyan); VectorBars("Candidate", step.candidate, LabOrange); VectorBars("Output gate", step.outputGate, LabPurple); VectorBars("Cell state", step.cell, LabGreen); VectorBars("Hidden state", step.hidden, LabBlue)
                    Text("Gate values near 1 pass information; values near 0 suppress it. Cell state carries memory across steps.", color = LabMuted, fontSize = 12.sp)
                }
                SequenceKind.GRU -> {
                    val step = gruSteps[safeTime]
                    VectorBars("Reset gate", step.reset, LabPink); VectorBars("Update gate", step.update, LabCyan); VectorBars("Candidate", step.candidate, LabOrange); VectorBars("Hidden state", step.hidden, LabGreen)
                }
            }
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("RNN vs LSTM vs GRU", "Same input and hidden size; architecture choice depends on the task")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                MetricPill("RNN", "${rnn.parameterCount()} params", LabCyan, Modifier.weight(1f)); MetricPill("LSTM", "${lstm.parameterCount()} params", LabPink, Modifier.weight(1f)); MetricPill("GRU", "${gru.parameterCount()} params", LabGreen, Modifier.weight(1f))
            }
            Text("Built-in tasks: next value, sine prediction, alternating-pattern classification, delayed memory and running sum. This trace uses [1, 0, 1, 1, 0].", color = LabMuted, fontSize = 12.sp)
            Text("Long RNN chains multiply recurrent derivatives repeatedly, which can make early gradients vanish. LSTM and GRU add gates that create more controlled memory paths.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun AutoencoderLab() {
    var latentSize by remember { mutableIntStateOf(2) }
    var modelKey by remember { mutableIntStateOf(0) }
    val autoencoder = remember(modelKey, latentSize) { DenseAutoencoder(8, latentSize) }
    var pattern by remember { mutableIntStateOf(0) }
    var history by remember(modelKey, latentSize) { mutableStateOf(emptyList<com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.EpochSnapshot>()) }
    var latentX by remember { mutableFloatStateOf(0f) }; var latentY by remember { mutableFloatStateOf(0f) }
    val input = PhaseThreeContent.reconstructionPatterns[pattern]
    val reconstruction = autoencoder.reconstruct(input)
    val latent = autoencoder.encode(input)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Autoencoder Reconstruction Lab", "Input -> compress -> latent -> decode -> reconstruction")
            Text("8 -> 4 -> $latentSize -> 4 -> 8", color = LabCyan, fontWeight = FontWeight.Bold)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) { BinaryStrip("Original", input, Modifier.weight(1f)); BinaryStrip("Reconstruction", reconstruction, Modifier.weight(1f)) }
            VectorBars("Latent representation", latent, LabPurple)
            val mse = input.indices.sumOf { ((input[it] - reconstruction[it]) * (input[it] - reconstruction[it])).toDouble() }.toFloat() / input.size
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                MetricPill("MSE", "%.5f".format(mse), LabPink, Modifier.weight(1f)); MetricPill("Bottleneck", "$latentSize / 8", LabCyan, Modifier.weight(1f)); MetricPill("Compression", "%.0f%%".format((1f - latentSize / 8f) * 100), LabGreen, Modifier.weight(1f))
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                SegmentedOption("Previous", false, Modifier.weight(1f)) { pattern = (pattern - 1).coerceAtLeast(0) }; SegmentedOption("Next", false, Modifier.weight(1f)) { pattern = (pattern + 1) % PhaseThreeContent.reconstructionPatterns.size }
                GradientButton("Train", Modifier.weight(1f)) { history = autoencoder.train(PhaseThreeContent.reconstructionPatterns, 600) }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { listOf(1, 2, 3, 4).forEach { SegmentedOption("z=$it", latentSize == it, Modifier.weight(1f)) { latentSize = it; modelKey++ } } }
            if (history.isNotEmpty()) Text("Reconstruction loss ${"%.5f".format(history.first().loss)} -> ${"%.5f".format(history.last().loss)}", color = LabGreen)
        }
    }
    if (latentSize >= 2) GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Latent Decoder", "Move through learned representation space and decode a new pattern")
            Text("z1 %.2f".format(latentX), color = LabMuted); Slider(latentX, { latentX = it }, valueRange = -1f..1f)
            Text("z2 %.2f".format(latentY), color = LabMuted); Slider(latentY, { latentY = it }, valueRange = -1f..1f)
            BinaryStrip("Decoded", autoencoder.decode(FloatArray(latentSize) { if (it == 0) latentX else if (it == 1) latentY else 0f }), Modifier.fillMaxWidth())
            Text("Denoising experiment: add input noise while keeping the clean pattern as the target. Large reconstruction errors can also flag anomalies.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun EmbeddingLab() {
    val embedding = remember { EmbeddingLayer(PhaseThreeContent.vocabulary, 4) }
    var first by remember { mutableIntStateOf(0) }; var second by remember { mutableIntStateOf(1) }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Embedding Explorer", "Token ID -> embedding table -> dense vector")
            Text("Word tokens: [deep, learning]   Character tokens: [d, e, e, p, ...]", color = LabMuted, fontSize = 12.sp)
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) { PhaseThreeContent.vocabulary.forEachIndexed { index, token -> SegmentedOption(token, first == index) { first = index } } }
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(7.dp)) { PhaseThreeContent.vocabulary.forEachIndexed { index, token -> SegmentedOption(token, second == index) { second = index } } }
            val firstToken = PhaseThreeContent.vocabulary[first]; val secondToken = PhaseThreeContent.vocabulary[second]
            VectorBars(firstToken, embedding.lookup(firstToken), LabCyan); VectorBars(secondToken, embedding.lookup(secondToken), LabPink)
            MetricPill("Cosine similarity", "%.4f".format(embedding.cosine(firstToken, secondToken)), LabPurple, Modifier.fillMaxWidth())
            EmbeddingPlot(embedding)
            Text("One-hot uses one dimension per vocabulary item. An embedding performs a trainable row lookup and can place related tokens near one another. The 2D view uses the first two coordinates of this tiny table; Phase 1 PCA remains available for larger vectors.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun ArchitectureInspector() {
    val cnn = remember { TinyCnnClassifier() }
    val sample = PhaseThreeContent.shapeDataset.first()
    val benchmarkNanos = remember { measureNanoTime { repeat(120) { cnn.predict(sample.image) } } }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Architecture Inspector", "Local backend - Kotlin FloatArray CPU")
            Text("Input 8x8x1 -> Conv2D(4, 3x3, same) -> ReLU -> MaxPool(2x2) -> Flatten(64) -> Dense(3) -> Softmax", color = LabCyan, fontSize = 12.sp)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                MetricPill("Parameters", cnn.parameterCount().toString(), LabPurple, Modifier.weight(1f)); MetricPill("FLOPs", cnn.flopEstimate().toString(), LabCyan, Modifier.weight(1f)); MetricPill("Memory", "%.2f KB".format(cnn.memoryBytes() / 1024f), LabOrange, Modifier.weight(1f))
            }
            MetricPill("Measured inference", "%.3f ms".format(benchmarkNanos / 120.0 / 1_000_000.0), LabGreen, Modifier.fillMaxWidth())
            Text("CameraFrame -> ImagePreprocessor -> ModelBackend -> InferenceResult", color = LabText, fontWeight = FontWeight.Bold)
            Text("No network access or native runtime is required. TFLite can later implement the same backend contract for imported optimized models without replacing the educational engine.", color = LabMuted, fontSize = 12.sp)
            Text("Saved model compatibility: Phase 3 stores architecture metadata and measured metrics locally. Full binary tensor checkpoints remain versioned future work.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun ImageGrid(image: TensorImage, title: String, modifier: Modifier = Modifier, step: Int? = null, kernelSize: Int = 0, selectedOutput: Int? = null) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Text(title, color = LabMuted, fontSize = 10.sp)
        Canvas(Modifier.fillMaxWidth().height(150.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
            val channel = 0; val cellW = size.width / image.width; val cellH = size.height / image.height
            val stats = image.stats(); val range = (stats.second - stats.first).coerceAtLeast(1e-5f)
            for (row in 0 until image.height) for (column in 0 until image.width) {
                val normalized = (image[channel, row, column] - stats.first) / range
                drawRect(Color(normalized, normalized * 0.55f, 1f - normalized * 0.55f, 1f), Offset(column * cellW, row * cellH), androidx.compose.ui.geometry.Size(cellW - 1f, cellH - 1f))
            }
            step?.let { index ->
                val outWidth = image.width - kernelSize + 1; val row = index / outWidth.coerceAtLeast(1); val column = index % outWidth.coerceAtLeast(1)
                drawRect(Color.White, Offset(column * cellW, row * cellH), androidx.compose.ui.geometry.Size(kernelSize * cellW, kernelSize * cellH), style = Stroke(4f))
            }
            selectedOutput?.let { index -> val row = index / image.width; val column = index % image.width; drawRect(Color.White, Offset(column * cellW, row * cellH), androidx.compose.ui.geometry.Size(cellW, cellH), style = Stroke(4f)) }
        }
    }
}

@Composable private fun KernelGrid(values: FloatArray, selected: Int, modifier: Modifier, onSelected: (Int) -> Unit) {
    Column(modifier, verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Text("Kernel 3 x 3", color = LabMuted, fontSize = 10.sp)
        Column(verticalArrangement = Arrangement.spacedBy(3.dp)) { repeat(3) { row -> Row(horizontalArrangement = Arrangement.spacedBy(3.dp)) { repeat(3) { column -> val index = row * 3 + column; SegmentedOption("%.1f".format(values[index]), selected == index, Modifier.weight(1f)) { onSelected(index) } } } } }
    }
}

@Composable private fun FeatureGallery(image: TensorImage) {
    val stats = image.stats()
    Text("Feature maps ${image.width}x${image.height}x${image.channels} - min %.3f max %.3f mean %.3f".format(stats.first, stats.second, stats.third), color = LabMuted, fontSize = 12.sp)
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        repeat(image.channels.coerceAtMost(4)) { channel ->
            val map = TensorImage(1, image.height, image.width, FloatArray(image.height * image.width) { image.values[channel * image.height * image.width + it] })
            ImageGrid(map, "Filter ${channel + 1}", Modifier.weight(1f))
        }
    }
}

@Composable private fun HistoryChart(history: List<CnnEpoch>) {
    Canvas(Modifier.fillMaxWidth().height(110.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        if (history.size < 2) return@Canvas
        val maxLoss = history.maxOf { it.loss }.coerceAtLeast(0.01f); val lossPath = Path(); val accuracyPath = Path()
        history.forEachIndexed { index, epoch ->
            val x = index * size.width / history.lastIndex; val lossY = size.height * (1f - epoch.loss / maxLoss); val accuracyY = size.height * (1f - epoch.accuracy)
            if (index == 0) { lossPath.moveTo(x, lossY); accuracyPath.moveTo(x, accuracyY) } else { lossPath.lineTo(x, lossY); accuracyPath.lineTo(x, accuracyY) }
        }
        drawPath(lossPath, LabPink, style = Stroke(4f)); drawPath(accuracyPath, LabGreen, style = Stroke(4f))
    }
}

@Composable private fun OcclusionMap(cnn: TinyCnnClassifier, sample: ImageSample) {
    val baseline = cnn.predict(sample.image).probabilities[sample.label]
    val values = FloatArray(16)
    repeat(4) { row -> repeat(4) { column ->
        val copy = sample.image.copy(); for (r in row * 2 until row * 2 + 2) for (c in column * 2 until column * 2 + 2) copy[0, r, c] = 0f
        values[row * 4 + column] = (baseline - cnn.predict(copy).probabilities[sample.label]).coerceAtLeast(0f)
    } }
    Canvas(Modifier.fillMaxWidth().height(115.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        val peak = values.maxOrNull()?.coerceAtLeast(1e-5f) ?: 1f; val cw = size.width / 4f; val ch = size.height / 4f
        values.forEachIndexed { index, value -> drawRect(LabOrange.copy(alpha = 0.1f + 0.9f * value / peak), Offset(index % 4 * cw, index / 4 * ch), androidx.compose.ui.geometry.Size(cw - 2f, ch - 2f)) }
    }
}

@Composable private fun SequenceTimeline(sequence: List<FloatArray>, selected: Int) { Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { sequence.forEachIndexed { index, item -> MetricPill("t${index + 1}", item[0].toInt().toString(), if (selected == index) LabPurple else LabBlue, Modifier.weight(1f)) } } }
@Composable private fun VectorBars(title: String, values: FloatArray, color: Color) { Column(verticalArrangement = Arrangement.spacedBy(4.dp)) { Text("$title  [${values.joinToString { "%.3f".format(it) }}]", color = LabMuted, fontSize = 11.sp); Row(Modifier.fillMaxWidth().height(9.dp)) { values.forEach { Box(Modifier.weight(1f).height(9.dp).padding(horizontal = 1.dp).background(color.copy(alpha = 0.2f + 0.8f * abs(it).coerceIn(0f, 1f)), RoundedCornerShape(3.dp))) } } } }
@Composable private fun BinaryStrip(title: String, values: FloatArray, modifier: Modifier) { Column(modifier, verticalArrangement = Arrangement.spacedBy(5.dp)) { Text(title, color = LabMuted, fontSize = 11.sp); Row(Modifier.fillMaxWidth().height(64.dp)) { values.forEach { Box(Modifier.weight(1f).height(64.dp).padding(horizontal = 1.dp).background(LabCyan.copy(alpha = 0.1f + 0.9f * it.coerceIn(0f, 1f)), RoundedCornerShape(3.dp))) } } } }
@Composable private fun EmbeddingPlot(embedding: EmbeddingLayer) { Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) { embedding.vocabulary.forEachIndexed { index, token -> val vector = embedding.lookup(token); val x = size.width * (vector[0] + 0.6f) / 1.2f; val y = size.height * (1f - (vector[1] + 0.6f) / 1.2f); drawCircle(if (index < 3) LabCyan else LabPink, 8f, Offset(x, y)) } } }
private fun norm(values: FloatArray) = kotlin.math.sqrt(values.sumOf { (it * it).toDouble() }.toFloat())
