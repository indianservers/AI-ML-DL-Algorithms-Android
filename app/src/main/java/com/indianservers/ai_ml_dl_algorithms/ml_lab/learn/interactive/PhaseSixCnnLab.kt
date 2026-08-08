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

private enum class CnnSection(val label: String) { Convolve("Convolve"), Feature("Feature Map"), Pool("Pooling"), Channels("Channels"), Architecture("Architecture"), Train("Tiny CNN") }

@Composable
fun PhaseSixCnnLab(
    topic: LearnTopic,
    concept: PhaseSixCnnConcept,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(defaultCnnSection(concept)) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("CNN Visual Lab - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "Phase 6", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                CnnSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            CnnSection.Convolve -> ConvolutionSection()
            CnnSection.Feature -> FeatureMapSection()
            CnnSection.Pool -> PoolingSection()
            CnnSection.Channels -> ChannelSection()
            CnnSection.Architecture -> ArchitectureSection()
            CnnSection.Train -> TinyCnnSection(onComplete)
        }
    }
}

@Composable
private fun ConvolutionSection() {
    var shape by remember { mutableStateOf(ShapeClass.Vertical) }
    var kernelPreset by remember { mutableStateOf(CnnKernelPreset.Vertical) }
    var stride by remember { mutableIntStateOf(1) }
    var paddingMode by remember { mutableStateOf(PaddingMode.Valid) }
    var step by remember { mutableIntStateOf(0) }
    val image = PhaseSixCnnEngines.presetImage(shape, 7)
    val kernel = PhaseSixCnnEngines.kernel(kernelPreset)
    val conv = PhaseSixCnnEngines.convolve(image, kernel, stride, if (paddingMode == PaddingMode.Same) 1 else 0, step)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Convolution Step Mode", "Kernel slides, multiplies, sums, and writes one feature-map value") }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelMatrixVisualizer(conv.padded, "Input", conv.current.row * stride, conv.current.col * stride, kernel.size, Modifier.weight(1f))
                PixelMatrixVisualizer(kernel, "Kernel", -1, -1, 0, Modifier.weight(1f))
            }
        }
        item { PixelMatrixVisualizer(conv.output, "Feature Map", conv.current.row, conv.current.col, 1, Modifier.fillMaxWidth()) }
        item { CalculationCard(conv.current) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        ShapeClass.entries.forEach { SegmentedOption(it.label, shape == it) { shape = it; step = 0 } }
                    }
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        CnnKernelPreset.entries.filter { it != CnnKernelPreset.Custom }.forEach { SegmentedOption(it.label, kernelPreset == it) { kernelPreset = it; step = 0 } }
                    }
                    SliderCnn("Stride", stride.toDouble(), 1.0, 3.0) { stride = it.toInt().coerceIn(1, 3); step = 0 }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        PaddingMode.entries.forEach { SegmentedOption(it.label, paddingMode == it, Modifier.weight(1f)) { paddingMode = it; step = 0 } }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        SegmentedOption("Next Position", true, Modifier.weight(1f)) { step++ }
                        SegmentedOption("Reset", false, Modifier.weight(1f)) { step = 0 }
                    }
                }
            }
        }
        item { InfoCnn("Output size", "floor((N + 2P - K) / S) + 1 = ${conv.output.size}x${conv.output.first().size}. Many CNN libraries use cross-correlation while calling it convolution.") }
    }
}

@Composable
private fun FeatureMapSection() {
    val image = PhaseSixCnnEngines.presetImage(ShapeClass.Vertical, 7)
    val maps = listOf(CnnKernelPreset.Vertical, CnnKernelPreset.Horizontal, CnnKernelPreset.Blur).map {
        it.label to PhaseSixCnnEngines.convolve(image, PhaseSixCnnEngines.kernel(it), padding = 1).output
    }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("What Does A Filter Detect?", "Same input, different filters, different feature maps") }
        item { PixelMatrixVisualizer(image, "Drawn image", -1, -1, 0, Modifier.fillMaxWidth()) }
        maps.forEach { (title, map) ->
            item { PixelMatrixVisualizer(map, title, -1, -1, 0, Modifier.fillMaxWidth()) }
        }
        item { InfoCnn("Learned filters", "Manual kernels help intuition. In a CNN, these filter weights are usually learned from data during training.") }
    }
}

@Composable
private fun PoolingSection() {
    var mode by remember { mutableStateOf(PoolMode.Max) }
    var poolSize by remember { mutableIntStateOf(2) }
    var stride by remember { mutableIntStateOf(2) }
    val raw = PhaseSixCnnEngines.convolve(PhaseSixCnnEngines.presetImage(ShapeClass.X, 8), PhaseSixCnnEngines.kernel(CnnKernelPreset.Vertical), padding = 1).output
    val relu = PhaseSixCnnEngines.relu(raw)
    val pool = PhaseSixCnnEngines.pool(relu, poolSize, stride, mode)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Pooling", "Reduce spatial size while preserving strong local evidence") }
        item { PixelMatrixVisualizer(relu, "ReLU Feature Map", 0, 0, poolSize, Modifier.fillMaxWidth()) }
        item { PixelMatrixVisualizer(pool.output, mode.label, 0, 0, 1, Modifier.fillMaxWidth()) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                        PoolMode.entries.forEach { SegmentedOption(it.label, mode == it, Modifier.weight(1f)) { mode = it } }
                    }
                    SliderCnn("Pool size", poolSize.toDouble(), 2.0, 3.0) { poolSize = it.toInt().coerceIn(2, 3) }
                    SliderCnn("Stride", stride.toDouble(), 1.0, 2.0) { stride = it.toInt().coerceIn(1, 2) }
                }
            }
        }
        item { InfoCnn("Selected region", "${pool.selectedRegion.flatten().joinToString { "%.1f".format(it) }} -> ${pool.mode.label} = %.2f".format(pool.selectedValue)) }
    }
}

@Composable
private fun ChannelSection() {
    val r = PhaseSixCnnEngines.presetImage(ShapeClass.Vertical, 5)
    val g = PhaseSixCnnEngines.presetImage(ShapeClass.Horizontal, 5)
    val b = PhaseSixCnnEngines.presetImage(ShapeClass.X, 5)
    val kernels = listOf(CnnKernelPreset.Vertical, CnnKernelPreset.Horizontal, CnnKernelPreset.Blur).map { PhaseSixCnnEngines.kernel(it) }
    val output = PhaseSixCnnEngines.multiChannelConvolution(listOf(r, g, b), kernels, bias = .1)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Multi-Channel Convolution", "Each channel contributes, then contributions are summed with bias") }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelMatrixVisualizer(r, "R", -1, -1, 0, Modifier.weight(1f))
                PixelMatrixVisualizer(g, "G", -1, -1, 0, Modifier.weight(1f))
                PixelMatrixVisualizer(b, "B", -1, -1, 0, Modifier.weight(1f))
            }
        }
        item { PixelMatrixVisualizer(output, "One Output Feature Map", -1, -1, 0, Modifier.fillMaxWidth()) }
        item { InfoCnn("Shape", "Input shape 5x5x3. One filter has one 3x3 kernel slice per channel. Multiple filters produce multiple output feature maps.") }
    }
}

@Composable
private fun ArchitectureSection() {
    var filters1 by remember { mutableIntStateOf(4) }
    var filters2 by remember { mutableIntStateOf(8) }
    val shapes = PhaseSixCnnEngines.architecture(filters1 = filters1, filters2 = filters2)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("CNN Architecture Builder", "Shape tracker and parameter count") }
        shapes.forEach { layer ->
            item {
                GlassPanel(Modifier.fillMaxWidth()) {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(layer.name, color = LabText, fontWeight = FontWeight.Bold)
                            Text(layer.explanation, color = LabMuted, fontSize = 12.sp)
                        }
                        MetricPill(layer.shape.toString(), "${layer.parameters} params", if (layer.parameters > 0) LabCyan else LabOrange)
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    SliderCnn("Conv1 filters", filters1.toDouble(), 1.0, 8.0) { filters1 = it.toInt().coerceIn(1, 8) }
                    SliderCnn("Conv2 filters", filters2.toDouble(), 1.0, 12.0) { filters2 = it.toInt().coerceIn(1, 12) }
                }
            }
        }
        item { InfoCnn("CNN vs Dense", "A dense 8x8 grayscale input to 16 units has ${8 * 8 * 16 + 16} parameters. A 3x3 CNN layer with $filters1 filters has ${PhaseSixCnnEngines.parameterCount(3, 3, 1, filters1)} parameters because filters are local and shared.") }
    }
}

@Composable
private fun TinyCnnSection(onComplete: () -> Unit) {
    var shape by remember { mutableStateOf(ShapeClass.X) }
    var noise by remember { mutableIntStateOf(0) }
    val image = PhaseSixCnnEngines.presetImage(shape, 8, noise / 100.0, 9)
    val prediction = PhaseSixCnnEngines.predictShape(image)
    val training = PhaseSixCnnEngines.trainTiny(epochs = 24)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Small Image Classification Lab", "Synthetic shapes, learned-ish filters, softmax prediction") }
        item { PixelMatrixVisualizer(image, "Selected Image", -1, -1, 0, Modifier.fillMaxWidth()) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ShapeClass.entries.forEach { SegmentedOption(it.label, shape == it, Modifier.weight(1f)) { shape = it } }
            }
        }
        item { SliderCnn("Add Noise", noise.toDouble(), 0.0, 45.0) { noise = it.toInt() } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                prediction.probabilities.forEachIndexed { i, p -> MetricPill(ShapeClass.entries[i].label, "%.0f%%".format(p * 100), if (i == prediction.predicted.ordinal) LabGreen else LabCyan, Modifier.weight(1f)) }
            }
        }
        item { PixelMatrixVisualizer(prediction.featureMaps.first(), "Feature Map Explorer: ${prediction.predicted.label}", -1, -1, 0, Modifier.fillMaxWidth()) }
        item { TrainingChartCnn(training) }
        item { InfoCnn("Selected kernel weight update", "Gradient %.4f, learning step produced selected weight %.4f".format(training.selectedGradient, training.updatedWeight)) }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun PixelMatrixVisualizer(matrix: List<List<Double>>, title: String, highlightRow: Int, highlightCol: Int, highlightSize: Int, modifier: Modifier) {
    GlassPanel(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp)
            Canvas(Modifier.fillMaxWidth().height(170.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
                val rows = matrix.size
                val cols = matrix.first().size
                val cellW = size.width / cols
                val cellH = size.height / rows
                val maxAbs = matrix.flatten().maxOf { kotlin.math.abs(it) }.coerceAtLeast(1.0)
                matrix.forEachIndexed { r, row ->
                    row.forEachIndexed { c, v ->
                        val selected = r in highlightRow until highlightRow + highlightSize && c in highlightCol until highlightCol + highlightSize
                        val alpha = (kotlin.math.abs(v) / maxAbs).toFloat().coerceIn(.08f, .9f)
                        drawRect((if (v >= 0) LabCyan else LabPink).copy(alpha = alpha), Offset(c * cellW, r * cellH), Size(cellW - 1f, cellH - 1f))
                        if (selected) drawRect(LabOrange, Offset(c * cellW, r * cellH), Size(cellW - 1f, cellH - 1f), style = Stroke(3f))
                    }
                }
            }
            Text(matrix.joinToString("  ") { row -> row.joinToString(" ") { "%.1f".format(it) } }, color = LabMuted, fontSize = 10.sp, maxLines = 3)
        }
    }
}

@Composable
private fun CalculationCard(step: ConvStepState) {
    val expression = step.patch.flatten().zip(step.kernel.flatten()).joinToString(" + ") { (a, b) -> "%.1fx%.1f".format(a, b) }
    InfoCnn("Current patch calculation", "$expression = %.2f. This value is written to output cell [${step.row}, ${step.col}].".format(step.sum))
}

@Composable
private fun TrainingChartCnn(state: CnnTrainingState) {
    Canvas(Modifier.fillMaxWidth().height(160.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val maxLoss = state.losses.max().coerceAtLeast(.001)
        val path = Path()
        state.losses.forEachIndexed { i, loss ->
            val x = size.width * i / state.losses.lastIndex
            val y = size.height * (loss / maxLoss).toFloat()
            if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
        val accPath = Path()
        state.accuracies.forEachIndexed { i, acc ->
            val x = size.width * i / state.accuracies.lastIndex
            val y = size.height * (1f - acc.toFloat())
            if (i == 0) accPath.moveTo(x, y) else accPath.lineTo(x, y)
        }
        drawPath(accPath, LabGreen, style = Stroke(4f, cap = StrokeCap.Round))
    }
}

@Composable
private fun SliderCnn(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: %.0f".format(value), color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun InfoCnn(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private fun defaultCnnSection(concept: PhaseSixCnnConcept) = when (concept) {
    PhaseSixCnnConcept.Convolution, PhaseSixCnnConcept.Filters, PhaseSixCnnConcept.FeatureMaps, PhaseSixCnnConcept.Padding, PhaseSixCnnConcept.Stride -> CnnSection.Convolve
    PhaseSixCnnConcept.Pooling -> CnnSection.Pool
    PhaseSixCnnConcept.MultiChannel -> CnnSection.Channels
    PhaseSixCnnConcept.Architecture -> CnnSection.Architecture
    PhaseSixCnnConcept.Training, PhaseSixCnnConcept.Classification -> CnnSection.Train
}
