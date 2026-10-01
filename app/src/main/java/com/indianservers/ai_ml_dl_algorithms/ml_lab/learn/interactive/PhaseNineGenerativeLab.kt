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
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPanelSoft
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic

private enum class GenerativeSection(val label: String) { Map("Map"), Autoencoder("AE"), Vae("VAE"), Gan("GAN"), Diffusion("Diffusion"), Compare("Compare"), Break("Break it") }

@Composable
fun PhaseNineGenerativeLab(
    topic: LearnTopic,
    concept: PhaseNineConcept,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit,
    embedded: Boolean = false
) {
    var section by remember(topic.id) { mutableStateOf(defaultGenerativeSection(concept)) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            if (!embedded) Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("Generative AI - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else depth.title, color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                GenerativeSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            GenerativeSection.Map -> GenerativeMapSection { section = it }
            GenerativeSection.Autoencoder -> AutoencoderSection(topic.title == "Denoising Autoencoder")
            GenerativeSection.Vae -> VaeSection()
            GenerativeSection.Gan -> GanSection()
            GenerativeSection.Diffusion -> DiffusionSection()
            GenerativeSection.Compare -> CompareSection(onComplete)
            GenerativeSection.Break -> BreakItSection()
        }
    }
}

@Composable
private fun GenerativeMapSection(open: (GenerativeSection) -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("One-Screen Generative Map", "Four ways to move from data, latent variables, or noise into samples") }
        item { FlowCard("AUTOENCODER", "Data -> Encode -> Latent -> Decode", "Compress and reconstruct", LabCyan) { open(GenerativeSection.Autoencoder) } }
        item { FlowCard("VAE", "Data -> mu/sigma -> Sample z -> Decode", "Make latent space smooth and sampleable", LabGreen) { open(GenerativeSection.Vae) } }
        item { FlowCard("GAN", "Noise -> Generator -> Fake <-> Discriminator", "Learn by adversarial feedback", LabOrange) { open(GenerativeSection.Gan) } }
        item { FlowCard("DIFFUSION", "Data -> Noise, then Noise -> Denoise -> Data", "Learn gradual noise removal", LabPurple) { open(GenerativeSection.Diffusion) } }
    }
}

@Composable
private fun AutoencoderSection(denoising: Boolean) {
    var shape by remember(denoising) { mutableStateOf(if (denoising) GenShape.Circle else GenShape.X) }
    var latentDims by remember(denoising) { mutableIntStateOf(if (denoising) 8 else 2) }
    val state = PhaseNineEngines.autoencoder(shape, latentDims)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle(if (denoising) "Denoising bottleneck" else "Autoencoder bottleneck",
            if (denoising) "Noisy input → compact code → clean estimate" else "Input → compact code → reconstruction") }
        item { Text("Fixed orthogonal shape-basis teaching projection; this screen does not train neural weights.",
            color = LabMuted, fontSize = 11.sp) }
        item { ShapePicker(shape) { shape = it } }
        if (denoising) item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelImage(state.noisy, "Noisy input", Modifier.weight(1f))
                PixelImage(state.denoised, "Decoded from noisy code", Modifier.weight(1f))
            }
        } else item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelImage(state.original, "Original", Modifier.weight(1f))
                PixelImage(state.reconstruction, "Decoded from code", Modifier.weight(1f))
            }
        }
        item { Slider9("Latent dimensions", latentDims.toDouble(), 1.0, 8.0) { latentDims = listOf(1, 2, 4, 8).minBy { k -> kotlin.math.abs(k - it.toInt()) } } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Vector9("Basis coefficients", state.hidden.take(8), Modifier.weight(1f))
                Vector9("Kept code", state.latent, Modifier.weight(1f))
            }
        }
        if (denoising) item { Info9("Denoising error",
            "Noisy MSE %.4f → decoded MSE %.4f against the clean reference. The decoder reads the noisy code only."
                .format(PhaseNineEngines.mse(state.original, state.noisy), PhaseNineEngines.mse(state.original, state.denoised))) }
        else item { Info9("Reconstruction error",
            "Selected pixel ${state.selectedPixel}: squared error %.4f, total MSE %.4f. More retained coefficients preserve more detail."
                .format(state.selectedPixelError, state.loss)) }
    }
}

@Composable
private fun VaeSection() {
    var shape by remember { mutableStateOf(GenShape.Circle) }
    var beta by remember { mutableFloatStateOf(1f) }
    var seed by remember { mutableIntStateOf(4) }
    val state = PhaseNineEngines.vae(shape, beta.toDouble(), seed)
    val prior = PhaseNineEngines.sampleVaePrior(seed + 9)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Variational Autoencoder", "Encoder outputs a distribution, then z = mu + sigma * epsilon") }
        item { Text("Fixed illustrative encoder parameters; beta changes this toy distribution and objective, without training weights.",
            color = LabMuted, fontSize = 11.sp) }
        item { ShapePicker(shape) { shape = it } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelImage(PhaseNineEngines.shapeImage(shape), "Input", Modifier.weight(1f))
                PixelImage(state.reconstruction, "Sampled output", Modifier.weight(1f))
            }
        }
        item { VaeLatentPlot(state) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Vector9("mu", state.mean, Modifier.weight(1f))
                Vector9("sigma", state.sigma, Modifier.weight(1f))
            }
        }
        item { Vector9("epsilon -> z", state.epsilon + state.z, Modifier.fillMaxWidth()) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Recon", "%.3f".format(state.reconstructionLoss), LabCyan, Modifier.weight(1f))
                MetricPill("KL", "%.3f".format(state.klLoss), LabOrange, Modifier.weight(1f))
                MetricPill("Total", "%.3f".format(state.totalLoss), LabGreen, Modifier.weight(1f))
            }
        }
        item { Slider9("beta KL weight", beta.toDouble(), .1, 3.0) { beta = it.toFloat() } }
        item { GradientButton("Sample again", Modifier.fillMaxWidth()) { seed += 1 } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                LatentMini(prior.point, "z ~ N(0,I)", Modifier.weight(1f))
                PixelImage(prior.reconstruction, "Prior sample", Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun GanSection() {
    var preset by remember { mutableStateOf(GanPreset.Balanced) }
    var steps by remember { mutableIntStateOf(12) }
    val state = PhaseNineEngines.gan(preset, steps)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("GAN Competition", "Real ring points vs generated points with a discriminator surface") }
        item { Text("Analytic teaching generator and discriminator; the progression does not train network weights.",
            color = LabMuted, fontSize = 11.sp) }
        item {
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                GanPreset.entries.forEach { SegmentedOption(it.label, preset == it) { preset = it } }
            }
        }
        item { GanPlot(state) }
        item { DiscriminatorField(state) }
        item { Slider9("Illustrative progression", steps.toDouble(), 1.0, 30.0) { steps = it.toInt() } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("D loss", "%.3f".format(state.discriminatorLoss), LabOrange, Modifier.weight(1f))
                MetricPill("G loss", "%.3f".format(state.generatorLoss), LabCyan, Modifier.weight(1f))
                MetricPill(state.phase.label, "active", LabGreen, Modifier.weight(1f))
            }
        }
        item { Vector9("Selected z -> G(z)", listOf(state.inspectedNoise.x, state.inspectedNoise.y, state.inspectedGenerated.x, state.inspectedGenerated.y), Modifier.fillMaxWidth()) }
        item { Info9("Discriminator confidence", "D(G(z)) = %.2f. GAN losses are useful signals, but lower loss is not a direct image-quality score.".format(state.inspectedConfidence)) }
        item { Info9("Timeline", state.timeline.joinToString(" -> ") { it.label }) }
    }
}

@Composable
private fun DiffusionSection() {
    var shape by remember { mutableStateOf(GenShape.Circle) }
    var step by remember { mutableIntStateOf(10) }
    var seed by remember { mutableIntStateOf(2) }
    var denoiseSteps by remember { mutableIntStateOf(6) }
    val state = PhaseNineEngines.diffusion(shape, step, seed = seed, denoiseSteps = denoiseSteps)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Diffusion noise equation", "Forward adds noise; a fixed basis estimates and subtracts it") }
        item { Text("Teaching predictor only: no trained diffusion network or multi-step sampler is connected.",
            color = LabMuted, fontSize = 11.sp) }
        item { ShapePicker(shape) { shape = it } }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelImage(state.clean, "Clean x0", Modifier.weight(1f))
                PixelImage(state.noisy, "Noisy xt", Modifier.weight(1f))
            }
        }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                PixelImage(state.predictedNoise, "Predicted noise", Modifier.weight(1f), signed = true)
                PixelImage(state.denoised, "One reverse step", Modifier.weight(1f))
            }
        }
        item { Slider9("Noise step t", step.toDouble(), 0.0, 23.0) { step = it.toInt() } }
        item { Slider9("Timeline frames", denoiseSteps.toDouble(), 3.0, 10.0) { denoiseSteps = it.toInt() } }
        item { DiffusionTimeline(state.reverseTimeline) }
        item { NoiseSchedule(state.step, state.totalSteps) }
        item { Info9("Pixel noise inspector", "x_t = sqrt(alphaBar) * x0 + sqrt(1-alphaBar) * eps. Pixel [${state.pixel.row}, ${state.pixel.col}]: x0 %.2f, eps %.2f, alphaBar %.2f, noisy %.2f. Noise-prediction loss %.4f.".format(state.pixel.original, state.pixel.noise, state.pixel.alphaBar, state.pixel.noisy, state.loss)) }
        item { GradientButton("Generate again", Modifier.fillMaxWidth()) { seed += 1 } }
    }
}

@Composable
private fun CompareSection(onComplete: () -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Autoencoder vs VAE vs GAN vs Diffusion", "No best-model ranking; each has a different mechanism") }
        item { CompareRow("Autoencoder", "Input -> compressed representation -> reconstruction", "Reconstruction MSE", "Bottleneck and noisy-input reconstruction") }
        item { CompareRow("VAE", "Distribution -> sample z -> decode", "Reconstruction + beta * KL", "Gaussian ellipse and prior sampling") }
        item { CompareRow("GAN", "Generator competes with Discriminator", "Binary adversarial losses", "Real/fake points and decision field") }
        item { CompareRow("Diffusion", "Estimate and remove noise", "Noise-prediction MSE", "Forward equation and staged subtraction") }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun BreakItSection() {
    val tiny = PhaseNineEngines.autoencoder(GenShape.X, 1)
    val collapsed = PhaseNineEngines.gan(GanPreset.ModeCollapse, 18)
    val few = PhaseNineEngines.diffusion(GenShape.Square, step = 22, denoiseSteps = 3)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Break It Experiments", "Failure modes made visible with the same tiny models") }
        item { PixelImage(tiny.reconstruction, "AE latent dim = 1, MSE %.3f".format(tiny.loss), Modifier.fillMaxWidth()) }
        item { Info9("VAE beta too high", "High beta pulls latent distributions toward the prior; in this educational model, KL grows in the total objective and samples become less tied to exact input details.") }
        item { GanPlot(collapsed) }
        item { Info9("Mode collapse", "Generated points gather near one region while real data covers the ring.") }
        item { DiffusionTimeline(few.reverseTimeline) }
        item { Info9("Coarse timeline", "Three frames skip intermediate views of the same predicted-noise subtraction; the final estimate is unchanged.") }
    }
}

@Composable
private fun FlowCard(title: String, flow: String, detail: String, color: Color, onOpen: () -> Unit) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text(title, color = color, fontWeight = FontWeight.Bold)
                SegmentedOption("Open", false, onClick = onOpen)
            }
            Text(flow, color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            Text(detail, color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun ShapePicker(shape: GenShape, onPick: (GenShape) -> Unit) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        GenShape.entries.forEach { SegmentedOption(it.label, shape == it) { onPick(it) } }
    }
}

@Composable
private fun PixelImage(matrix: List<List<Double>>, title: String, modifier: Modifier, signed: Boolean = false) {
    GlassPanel(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontSize = 12.sp, fontWeight = FontWeight.Bold, maxLines = 2)
            Canvas(Modifier.fillMaxWidth().height(145.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(6.dp)) {
                val cell = kotlin.math.min(size.width, size.height) / matrix.size
                val left = (size.width - cell * matrix.size) / 2f
                matrix.forEachIndexed { r, row ->
                    row.forEachIndexed { c, v ->
                        val color = if (signed && v < 0) LabPink.copy(alpha = kotlin.math.abs(v).toFloat().coerceIn(.08f, .95f)) else LabCyan.copy(alpha = v.toFloat().coerceIn(.06f, .95f))
                        drawRect(color, Offset(left + c * cell, r * cell), Size(cell - 2f, cell - 2f))
                    }
                }
            }
        }
    }
}

@Composable
private fun LatentPlot(points: List<LatentPointState>, dragged: Point2) {
    Canvas(Modifier.fillMaxWidth().height(210.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun map(p: Point2) = Offset(((p.x + 1.2) / 2.4).toFloat() * size.width, ((1.2 - p.y) / 2.4).toFloat() * size.height)
        drawLine(LabMuted.copy(alpha = .35f), Offset(0f, size.height / 2f), Offset(size.width, size.height / 2f))
        drawLine(LabMuted.copy(alpha = .35f), Offset(size.width / 2f, 0f), Offset(size.width / 2f, size.height))
        points.forEach { drawCircle(LabGreen, 8f, map(it.point)) }
        drawCircle(LabOrange, 12f, map(dragged), style = Stroke(3f))
    }
}

@Composable
private fun VaeLatentPlot(state: VaeState) {
    Canvas(Modifier.fillMaxWidth().height(210.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun map(p: Point2) = Offset(((p.x + 1.6) / 3.2).toFloat() * size.width, ((1.6 - p.y) / 3.2).toFloat() * size.height)
        val mean = map(Point2(state.mean[0], state.mean[1]))
        val z = map(Point2(state.z[0], state.z[1]))
        drawOval(LabPurple.copy(alpha = .22f), Offset(mean.x - (state.sigma[0] * 42).toFloat(), mean.y - (state.sigma[1] * 42).toFloat()), Size((state.sigma[0] * 84).toFloat(), (state.sigma[1] * 84).toFloat()))
        drawCircle(LabGreen, 8f, mean)
        drawCircle(LabOrange, 8f, z)
        drawLine(LabMuted, mean, z, 2f)
    }
}

@Composable
private fun LatentMini(point: Point2, title: String, modifier: Modifier) {
    GlassPanel(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            Text("[%.2f, %.2f]".format(point.x, point.y), color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun GanPlot(state: GanState) {
    Canvas(Modifier.fillMaxWidth().height(250.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun map(p: Point2) = Offset(((p.x + 1.2) / 2.4).toFloat() * size.width, ((1.2 - p.y) / 2.4).toFloat() * size.height)
        state.real.forEach { drawCircle(LabGreen, 5f, map(it), style = Stroke(2f)) }
        state.generated.forEach { drawRect(LabOrange.copy(alpha = .75f), map(it.second) - Offset(4f, 4f), Size(8f, 8f)) }
        drawCircle(LabPink, 9f, map(state.inspectedGenerated), style = Stroke(3f))
    }
}

@Composable
private fun DiscriminatorField(state: GanState) {
    Canvas(Modifier.fillMaxWidth().height(160.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val rows = state.discriminatorField.size
        val cols = state.discriminatorField.first().size
        val w = size.width / cols
        val h = size.height / rows
        state.discriminatorField.forEachIndexed { r, row ->
            row.forEachIndexed { c, p ->
                drawRect(LabCyan.copy(alpha = p.toFloat().coerceIn(.06f, .92f)), Offset(c * w, r * h), Size(w - 1f, h - 1f))
            }
        }
    }
}

@Composable
private fun DiffusionTimeline(images: List<List<List<Double>>>) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        images.forEachIndexed { i, image -> PixelImage(image, "t-$i", Modifier.size(116.dp)) }
    }
}

@Composable
private fun NoiseSchedule(step: Int, total: Int) {
    Canvas(Modifier.fillMaxWidth().height(110.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val points = List(total) { i -> Offset(i * size.width / (total - 1), (1.0 - PhaseNineEngines.alphaBar(i, total)).toFloat() * size.height) }
        points.zipWithNext().forEach { (a, b) -> drawLine(LabCyan, a, b, 3f) }
        val selected = points[step.coerceIn(0, total - 1)]
        drawCircle(LabOrange, 8f, selected)
    }
}

@Composable
private fun Vector9(title: String, values: List<Double>, modifier: Modifier) {
    GlassPanel(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text(title, color = LabText, fontSize = 12.sp, fontWeight = FontWeight.Bold)
            values.forEach { value -> Box(Modifier.fillMaxWidth().height(10.dp).background((if (value >= 0) LabCyan else LabPink).copy(alpha = kotlin.math.abs(value).toFloat().coerceIn(.08f, .9f)), RoundedCornerShape(4.dp))) }
            Text(values.joinToString { "%.2f".format(it) }, color = LabMuted, fontSize = 11.sp)
        }
    }
}

@Composable
private fun CompareRow(name: String, mechanism: String, objective: String, visual: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
            Text(name, color = LabText, fontWeight = FontWeight.Bold)
            Text(mechanism, color = LabCyan, fontSize = 12.sp)
            Text("Objective: $objective", color = LabMuted, fontSize = 12.sp)
            Text("Visual: $visual", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun Slider9(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        val shown = if (min % 1.0 == 0.0 && max % 1.0 == 0.0 && value % 1.0 == 0.0)
            value.toInt().toString() else "%.2f".format(value)
        Text("$label: $shown", color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun Info9(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private fun defaultGenerativeSection(concept: PhaseNineConcept) = when (concept) {
    PhaseNineConcept.Autoencoder -> GenerativeSection.Autoencoder
    PhaseNineConcept.Vae -> GenerativeSection.Vae
    PhaseNineConcept.Gan -> GenerativeSection.Gan
    PhaseNineConcept.Diffusion -> GenerativeSection.Diffusion
    PhaseNineConcept.GenerativeMap -> GenerativeSection.Map
}
