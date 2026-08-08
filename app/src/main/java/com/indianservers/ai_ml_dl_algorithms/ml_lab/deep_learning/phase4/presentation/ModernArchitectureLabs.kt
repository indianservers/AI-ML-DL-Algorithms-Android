package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.presentation

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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.Graph
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.GraphConvolution
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.GraphPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.CrossAttention
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.LearnedPositionEmbedding
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.Matrix
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.MatrixOps
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.MultiHeadAttention
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.PatchEmbedding
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.ScaledDotProductAttention
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TinyDiffusion
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TinyGan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TinyGcnNodeClassifier
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TinyAutoregressiveDecoder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TinyTransformerTask
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TinyVae
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TransformerEncoderBlock
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.layerNorm
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.attentionLinks
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.connect
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.graphPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.interpolateLatent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.probabilities
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.removeEdge
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.removeNode
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.sinusoidalPositionEncoding
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.twoCommunityGraph
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.system.measureNanoTime

private enum class ModernLab(val title: String) { Attention("Attn"), Vision("ViT"), Graphs("GNN"), Generative("Gen"), Compare("Compare") }
private enum class GeneratorLab { VAE, GAN, Diffusion }

@Composable
fun ModernArchitectureLabs() {
    var lab by remember { mutableStateOf(ModernLab.Attention) }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SectionTitle("Modern Architectures", "Expose every important internal representation")
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { ModernLab.entries.forEach { SegmentedOption(it.title, lab == it, Modifier.weight(1f)) { lab = it } } }
            }
        }
        when (lab) {
            ModernLab.Attention -> AttentionLab()
            ModernLab.Vision -> VisionTransformerLab()
            ModernLab.Graphs -> GraphLab()
            ModernLab.Generative -> GenerativeLab()
            ModernLab.Compare -> ComparisonLab()
        }
        GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                SectionTitle("Phase 5", "Practical AI deployment coming next")
                Text("Pretrained models - LiteRT/TFLite - ONNX Runtime - NNAPI/GPU - camera, audio and text inference - quantization - export", color = LabMuted, fontSize = 12.sp)
            }
        }
    }
}

@Composable
private fun AttentionLab() {
    val tokens = listOf("I", "love", "machine", "learning")
    val embeddings = remember { Matrix(4, 4, floatArrayOf(.8f, .1f, -.2f, .4f, .2f, .9f, .3f, -.1f, -.4f, .5f, .9f, .2f, .1f, .4f, .8f, .9f)) }
    var selectedToken by remember { mutableIntStateOf(2) }; var selectedKey by remember { mutableIntStateOf(0) }
    var causal by remember { mutableStateOf(false) }; var padding by remember { mutableStateOf(false) }; var stage by remember { mutableIntStateOf(0) }; var heads by remember { mutableIntStateOf(2) }
    var linkThreshold by remember { mutableFloatStateOf(.2f) }; var training by remember { mutableStateOf(emptyList<com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.TransformerTrainingMetric>()) }
    var temperature by remember { mutableFloatStateOf(1f) }; var topK by remember { mutableIntStateOf(3) }
    val mask = BooleanArray(tokens.size) { !padding || it != tokens.lastIndex }
    val attention = remember { ScaledDotProductAttention(4) }.forward(embeddings, causal, mask)
    val multiHead = remember(heads) { MultiHeadAttention(4, heads) }.forward(embeddings, causal, mask)
    val position = sinusoidalPositionEncoding(tokens.size, 4)
    val transformerInput = MatrixOps.add(embeddings, position)
    val transformer = remember(heads) { TransformerEncoderBlock(4, heads, 8) }.forward(transformerInput)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Attention Explorer", "Q, K and V are projected from the selected token embedding")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { tokens.forEachIndexed { index, token -> SegmentedOption(token, selectedToken == index, Modifier.weight(1f)) { selectedToken = index } } }
            VectorReadout("Embedding", embeddings.row(selectedToken), LabMuted); VectorReadout("Query", attention.query.row(selectedToken), LabCyan); VectorReadout("Key", attention.key.row(selectedToken), LabPink); VectorReadout("Value", attention.value.row(selectedToken), LabOrange)
            Text("Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V", color = LabText, fontWeight = FontWeight.Bold)
            val names = listOf("QK^T", "Scale", "Softmax", "Output")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { names.forEachIndexed { index, name -> SegmentedOption(name, stage == index, Modifier.weight(1f)) { stage = index } } }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { SegmentedOption("Previous", false, Modifier.weight(1f)) { stage = (stage - 1).coerceAtLeast(0) }; SegmentedOption("Step", false, Modifier.weight(1f)) { stage = (stage + 1).coerceAtMost(3) }; SegmentedOption("Reset", false, Modifier.weight(1f)) { stage = 0 } }
            MatrixHeatmap(if (stage < 2) attention.scores else attention.weights, tokens, selectedToken, selectedKey) { row, column -> selectedToken = row; selectedKey = column }
            Text("${tokens[selectedToken]} attends to ${tokens[selectedKey]} with weight %.4f".format(attention.weights[selectedToken, selectedKey]), color = LabCyan)
            VectorReadout("Final representation", attention.output.row(selectedToken), LabGreen)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text("Causal mask", color = LabMuted); Switch(causal, { causal = it }) }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text("Mask final token as <PAD>", color = LabMuted); Switch(padding, { padding = it }) }
            Text(if (causal) "Future-token cells are excluded before Softmax. Padding masks use the same exclusion path." else "Bidirectional self-attention allows every token to inspect every non-padding token.", color = LabMuted, fontSize = 12.sp)
            Text("Attention links above %.2f".format(linkThreshold), color = LabMuted); Slider(linkThreshold, { linkThreshold = it }, valueRange = 0f..1f)
            AttentionLinksCanvas(tokens, attention.weights, selectedToken, linkThreshold)
            val strongest = attention.weights.row(selectedToken).indices.maxBy { attention.weights[selectedToken, it] }
            Text("Text alternative: ${tokens[selectedToken]} attends most strongly to ${tokens[strongest]} with weight %.3f.".format(attention.weights[selectedToken, strongest]), color = LabCyan, fontSize = 12.sp)
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Multi-Head Transformer Encoder", "Attention -> residual + LayerNorm -> GELU FFN -> residual + LayerNorm")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { listOf(1, 2, 4).forEach { SegmentedOption("$it heads", heads == it, Modifier.weight(1f)) { heads = it } } }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                MetricPill("Parameters", TransformerEncoderBlock(4, heads, 8).parameterCount().toString(), LabPurple, Modifier.weight(1f)); MetricPill("Entropy", "%.4f".format(multiHead.heads.first().entropy), LabOrange, Modifier.weight(1f)); MetricPill("Heads", heads.toString(), LabCyan, Modifier.weight(1f))
            }
            multiHead.heads.forEachIndexed { index, head -> Text("Head ${index + 1}: ${head.weights.row(selectedToken).joinToString { "%.2f".format(it) }}", color = if (index % 2 == 0) LabCyan else LabPink, fontSize = 12.sp) }
            val norm = layerNorm(transformer.attentionResidual.row(selectedToken))
            Text("LayerNorm mean %.4f - variance %.4f".format(norm.mean, norm.variance), color = LabMuted)
            VectorReadout("Input + position", transformerInput.row(selectedToken), LabBlue); VectorReadout("Attention residual", transformer.attentionResidual.row(selectedToken), LabPurple); VectorReadout("FFN output", transformer.feedForward.row(selectedToken), LabOrange); VectorReadout("Encoder output", transformer.output.row(selectedToken), LabGreen)
            PositionHeatmap(position)
            val learned = remember { LearnedPositionEmbedding(8, 4) }.encoding(tokens.size)
            VectorReadout("Learned position ${selectedToken + 1}", learned.row(selectedToken), LabPink)
            Text("Sinusoidal positions preserve token order; learned positions are trainable alternatives. Neither is universally better.", color = LabMuted, fontSize = 12.sp)
            GradientButton("Train ABAB task", Modifier.fillMaxWidth()) { training = TinyTransformerTask().train() }
            if (training.isNotEmpty()) { val first = training.first(); val last = training.last(); Text("Epoch ${last.epoch} - loss %.3f -> %.3f - accuracy %.0f%% - gradient %.3f - attention entropy %.3f".format(first.loss, last.loss, last.accuracy * 100, last.gradientNorm, last.attentionEntropy), color = LabGreen, fontSize = 12.sp) }
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Decoder Sampling", "Tiny autoregressive controls, not a language model")
            Text("Logits: A 4.2   B 3.7   C 1.1", color = LabText)
            Text("Temperature %.2f".format(temperature), color = LabMuted); Slider(temperature, { temperature = it }, valueRange = 0.1f..2f)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { (1..3).forEach { SegmentedOption("Top $it", topK == it, Modifier.weight(1f)) { topK = it } } }
            val p = probabilities(floatArrayOf(4.2f, 3.7f, 1.1f), temperature, topK)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { listOf("A", "B", "C").forEachIndexed { index, token -> MetricPill(token, "%.1f%%".format(p[index] * 100), if (p[index] > 0f) LabCyan else LabMuted, Modifier.weight(1f)) } }
            val source = Matrix(3, 4, embeddings.values.copyOfRange(0, 12)); val target = Matrix(2, 4, embeddings.values.copyOfRange(8, 16)); val cross = CrossAttention(4).forward(target, source)
            Text("Cross-attention: decoder query 1 attends to source [A, B, C] as ${cross.attention.weights.row(0).joinToString { "%.2f".format(it) }}", color = LabCyan, fontSize = 12.sp)
            TinyAutoregressiveDecoder().generate(4, temperature, topK.coerceAtMost(5)).forEach { generated -> Text("Step ${generated.step + 1}: ${generated.inputTokens.joinToString(" ")} -> ${generated.selectedToken} (${generated.probabilities.joinToString { "%.2f".format(it) }})", color = LabMuted, fontSize = 11.sp) }
            Text("Decoder path: masked self-attention -> encoder cross-attention -> FFN. The tiny generator appends one selected token per step; it is not a language model.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun VisionTransformerLab() {
    var patchSize by remember { mutableIntStateOf(2) }; var sampleIndex by remember { mutableIntStateOf(0) }; var selectedPatch by remember { mutableIntStateOf(0) }
    val sample = PhaseThreeContent.shapeDataset[sampleIndex]
    val patcher = remember(patchSize) { PatchEmbedding(8, patchSize, 4) }; val patches = patcher.forward(sample.image)
    val clsAndPatches = Matrix(patches.embeddings.rows + 1, 4, FloatArray((patches.embeddings.rows + 1) * 4).also { patches.embeddings.values.copyInto(it, 4) })
    val transformerInput = MatrixOps.add(clsAndPatches, sinusoidalPositionEncoding(clsAndPatches.rows, 4))
    val encoder = remember { TransformerEncoderBlock(4, 2, 8, 207) }.forward(transformerInput)
    val clsAttention = encoder.attention.heads.first().weights.row(0).drop(1).toFloatArray()
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Vision Transformer Lab", "Image -> patches -> embeddings + position -> encoder -> CLS")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { listOf(2, 4).forEach { SegmentedOption("${it}x$it patches", patchSize == it, Modifier.weight(1f)) { patchSize = it; selectedPatch = 0 } }; SegmentedOption("Next image", false, Modifier.weight(1f)) { sampleIndex = (sampleIndex + 1) % PhaseThreeContent.shapeDataset.size } }
            PatchGrid(sample.image.values, 8, patchSize, selectedPatch, clsAttention) { selectedPatch = it }
            Text("Patch ${selectedPatch + 1} pixels: ${patches.patches[selectedPatch.coerceIn(patches.patches.indices)].joinToString { "%.2f".format(it) }}", color = LabMuted, fontSize = 11.sp)
            VectorReadout("Patch embedding", patches.embeddings.row(selectedPatch.coerceIn(patches.patches.indices)), LabCyan)
            VectorReadout("Final CLS representation", encoder.output.row(0), LabPurple)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { MetricPill("Tokens", clsAndPatches.rows.toString(), LabCyan, Modifier.weight(1f)); MetricPill("Patch params", patcher.parameterCount().toString(), LabOrange, Modifier.weight(1f)); MetricPill("Encoder params", TransformerEncoderBlock(4, 2, 8).parameterCount().toString(), LabPurple, Modifier.weight(1f)) }
            Text("Overlay brightness is CLS-to-patch attention weight. Attention is an internal routing weight, not a causal explanation.", color = LabPink, fontSize = 12.sp)
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("CNN vs ViT", "Different inductive biases on the same synthetic shapes"); Text("CNN: local filters, translation-friendly hierarchy, strong small-data bias. Tiny ViT: patch tokens, global attention, weaker locality assumptions and usually greater data appetite.", color = LabMuted, fontSize = 12.sp); Text("This educational ViT runs a real forward pass; the Phase 3 CNN remains the trainable phone-scale classifier.", color = LabCyan, fontSize = 12.sp) } }
}

@Composable
private fun GraphLab() {
    var graph by remember { mutableStateOf(twoCommunityGraph()) }; var selectedNode by remember { mutableIntStateOf(3) }; var layers by remember { mutableIntStateOf(1) }; var presetIndex by remember { mutableIntStateOf(GraphPreset.entries.lastIndex) }
    var classification by remember { mutableStateOf<com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.NodeClassificationResult?>(null) }
    val convolution = remember { GraphConvolution(2, 2) }
    var current = graph.features; var result = convolution.forward(graph)
    repeat((layers - 1).coerceAtLeast(0)) { current = result.output; result = convolution.forward(Graph(current, graph.edges, graph.labels)) }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionTitle("Graph Message Passing Lab", "Neighbor features -> normalized messages -> aggregate -> update")
            GraphCanvas(graph, result.output, selectedNode) { selectedNode = it }
            Text("Node ${selectedNode + 1} receives from ${graph.edges.flatMap { listOf(it.first to it.second, it.second to it.first) }.filter { it.second == selectedNode }.joinToString { "Node ${it.first + 1}" }} plus itself.", color = LabMuted, fontSize = 12.sp)
            VectorReadout("Input feature", graph.features.row(selectedNode), LabCyan); VectorReadout("Aggregated message", result.messages.row(selectedNode), LabOrange); VectorReadout("Updated embedding", result.output.row(selectedNode), LabGreen)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { (1..6).forEach { SegmentedOption("L$it", layers == it, Modifier.weight(1f)) { layers = it } } }
            val similarity = averagePairDistance(result.output)
            Text("Embedding separation %.4f. As layers deepen and this falls, node states oversmooth toward similar values.".format(similarity), color = LabPink)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                SegmentedOption("Add node", false, Modifier.weight(1f)) { graph = addNode(graph, selectedNode) }
                SegmentedOption("Connect next", false, Modifier.weight(1f)) { if (graph.features.rows > 1) graph = graph.connect(selectedNode, (selectedNode + 1) % graph.features.rows) }
                SegmentedOption("Remove edge", false, Modifier.weight(1f)) { if (graph.features.rows > 1) graph = graph.removeEdge(selectedNode, (selectedNode + 1) % graph.features.rows) }
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
                SegmentedOption("Remove node", false, Modifier.weight(1f)) { if (graph.features.rows > 2) { graph = graph.removeNode(selectedNode); selectedNode = selectedNode.coerceAtMost(graph.features.rows - 1) } }
                SegmentedOption("Next preset", false, Modifier.weight(1f)) { presetIndex = (presetIndex + 1) % GraphPreset.entries.size; graph = graphPreset(GraphPreset.entries[presetIndex]); selectedNode = 0; layers = 1 }
                SegmentedOption("Clear", false, Modifier.weight(1f)) { graph = graphPreset(GraphPreset.Chain).copy(edges = emptySet()); selectedNode = 0 }
            }
            Text("Preset ${GraphPreset.entries[presetIndex].name}; select nodes below the graph, then connect or remove the next numbered node.", color = LabMuted, fontSize = 11.sp)
        }
    }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("GCN Matrix Explorer", "H' = ReLU(D^-1/2 (A+I) D^-1/2 H W)")
            MatrixHeatmap(result.adjacency, List(graph.features.rows) { "${it + 1}" }, selectedNode, selectedNode) { row, _ -> selectedNode = row }
            Text("Self-loops preserve a node's own feature. Symmetric degree normalization prevents high-degree nodes from dominating the aggregate.", color = LabMuted, fontSize = 12.sp)
            Text("Graph attention concept: learned edge coefficients can weight neighbors unequally. They remain model internals, and do not guarantee human-interpretable importance.", color = LabMuted, fontSize = 12.sp)
            MetricPill("GCN parameters", convolution.parameterCount().toString(), LabPurple, Modifier.fillMaxWidth())
            GradientButton("Train node classifier", Modifier.fillMaxWidth()) { classification = TinyGcnNodeClassifier().train(graph) }
            classification?.let { result -> Text("Node classification loss %.3f - accuracy %.0f%% - probabilities [%s]".format(result.loss, result.accuracy * 100, result.probabilities.joinToString { "%.2f".format(it) }), color = LabGreen, fontSize = 12.sp) }
            Text("Limitations: deeper neighborhoods can oversmooth or oversquash information; graph construction, heterophily and neighborhood growth all affect results.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun GenerativeLab() {
    var lab by remember { mutableStateOf(GeneratorLab.VAE) }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { GeneratorLab.entries.forEach { SegmentedOption(it.name, lab == it, Modifier.weight(1f)) { lab = it } } }
    when (lab) { GeneratorLab.VAE -> VaeLab(); GeneratorLab.GAN -> GanLab(); GeneratorLab.Diffusion -> DiffusionLab() }
}

@Composable
private fun VaeLab() {
    var pattern by remember { mutableIntStateOf(0) }; var seed by remember { mutableIntStateOf(1) }; var latentX by remember { mutableFloatStateOf(0f) }; var latentY by remember { mutableFloatStateOf(0f) }; var interpolation by remember { mutableFloatStateOf(.5f) }
    val vae = remember { TinyVae() }; val input = PhaseThreeContent.reconstructionPatterns[pattern]; val sample = vae.sample(input, seed)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("VAE Latent Lab", "Encoder -> mean/log variance -> reparameterize -> decode")
            BinaryReadout("Input", input); VectorReadout("Mean", sample.mean, LabCyan); VectorReadout("Log variance", sample.logVariance, LabPink); VectorReadout("Epsilon ~ N(0,I)", sample.epsilon, LabOrange); VectorReadout("z = mean + sigma * epsilon", sample.latent, LabPurple); BinaryReadout("Reconstruction", sample.reconstruction)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { MetricPill("Reconstruction", "%.4f".format(sample.reconstructionLoss), LabCyan, Modifier.weight(1f)); MetricPill("KL", "%.4f".format(sample.klLoss), LabPink, Modifier.weight(1f)); MetricPill("Total", "%.4f".format(sample.totalLoss), LabGreen, Modifier.weight(1f)) }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { SegmentedOption("Next input", false, Modifier.weight(1f)) { pattern = (pattern + 1) % PhaseThreeContent.reconstructionPatterns.size }; SegmentedOption("Resample z", false, Modifier.weight(1f)) { seed++ } }
            Text("Latent x %.2f".format(latentX), color = LabMuted); Slider(latentX, { latentX = it }, valueRange = -2f..2f); Text("Latent y %.2f".format(latentY), color = LabMuted); Slider(latentY, { latentY = it }, valueRange = -2f..2f)
            BinaryReadout("Decoded latent", vae.decode(floatArrayOf(latentX, latentY)))
            val other = vae.sample(PhaseThreeContent.reconstructionPatterns[(pattern + 1) % PhaseThreeContent.reconstructionPatterns.size], seed + 1)
            Text("Interpolation %.0f%%".format(interpolation * 100), color = LabMuted); Slider(interpolation, { interpolation = it }, valueRange = 0f..1f)
            BinaryReadout("Decoded interpolation", vae.decode(interpolateLatent(sample.latent, other.latent, interpolation)))
            Text("Unlike a deterministic autoencoder point, the VAE encodes a distribution and regularizes it toward N(0,1).", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun GanLab() {
    var key by remember { mutableIntStateOf(0) }; val gan = remember(key) { TinyGan() }; var history by remember(key) { mutableStateOf(emptyList<com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.GanSnapshot>()) }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("GAN Distribution Lab", "Noise -> generator -> sample -> discriminator -> alternating updates")
            DistributionPlot(gan)
            Text("Generator: x = %.3f z %+.3f".format(gan.generatorWeight, gan.generatorBias), color = LabCyan); Text("Discriminator: sigmoid(%.3f x %+.3f)".format(gan.discriminatorWeight, gan.discriminatorBias), color = LabPink)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { GradientButton("Train 400", Modifier.weight(1f)) { history = gan.train(400) }; SegmentedOption("Reset", false, Modifier.weight(1f)) { key++ } }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { SegmentedOption("Step D", false, Modifier.weight(1f)) { history = history + gan.stepDiscriminator() }; SegmentedOption("Step G", false, Modifier.weight(1f)) { history = history + gan.stepGenerator() }; SegmentedOption("Collapse", false, Modifier.weight(1f)) { gan.modeCollapse(); history = history + gan.stepGenerator(0f) } }
            if (history.isNotEmpty()) { val last = history.last(); Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { MetricPill("D loss", "%.3f".format(last.discriminatorLoss), LabPink, Modifier.weight(1f)); MetricPill("G loss", "%.3f".format(last.generatorLoss), LabCyan, Modifier.weight(1f)); MetricPill("Generated", "%.2f +/- %.2f".format(last.generatedMean, last.generatedStd), LabGreen, Modifier.weight(1f)) } }
            Text("D(real) %.3f - D(fake) %.3f. Target is N(2, 0.6). Losses can oscillate; Collapse reduces generator variance to demonstrate missing modes.".format(gan.discriminate(2f), gan.discriminate(gan.generate(0f))), color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun DiffusionLab() {
    val diffusion = remember { TinyDiffusion() }; var step by remember { mutableIntStateOf(0) }; var trained by remember { mutableIntStateOf(0) }; var losses by remember { mutableStateOf(emptyList<Float>()) }; var trajectory by remember { mutableStateOf(emptyList<FloatArray>()) }
    val clean = floatArrayOf(-.65f, .45f); val sample = diffusion.forward(clean, step, 9); val predicted = diffusion.predictNoise(sample.noisy, step)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Diffusion Noise Lab", "Forward noising and reverse denoising on a tiny 2D point")
            DiffusionPlot(sample.clean, sample.noisy, predicted)
            NoiseSchedulePlot(diffusion.betas, diffusion.alphaBars, step)
            Text("Step $step / ${diffusion.steps - 1} - alpha_bar %.4f - beta %.4f".format(sample.alphaBar, diffusion.betas[step]), color = LabMuted); Slider(step.toFloat(), { step = it.toInt() }, valueRange = 0f..(diffusion.steps - 1).toFloat())
            VectorReadout("Clean x0", sample.clean, LabGreen); VectorReadout("Sampled epsilon", sample.noise, LabOrange); VectorReadout("x_t = sqrt(alpha_bar)x0 + sqrt(1-alpha_bar)epsilon", sample.noisy, LabPurple); VectorReadout("Predicted noise", predicted, LabCyan)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { GradientButton("Train denoiser", Modifier.weight(1f)) { losses = diffusion.train(diffusionPoints(), 500); trained++ }; SegmentedOption("Reverse all", false, Modifier.weight(1f)) { trajectory = diffusion.reverseTrajectory(sample.noisy, step.coerceAtLeast(1)) }; SegmentedOption("Reset", false, Modifier.weight(1f)) { step = 0; trajectory = emptyList() } }
            if (losses.isNotEmpty()) Text("Noise-prediction MSE ${"%.4f".format(losses.first())} -> ${"%.4f".format(losses.last())}", color = LabGreen)
            if (trajectory.isNotEmpty()) { DiffusionTrajectoryPlot(trajectory); Text("Reverse trajectory contains ${trajectory.size} actual denoising states; final point [${trajectory.last().joinToString { "%.3f".format(it) }}].", color = LabCyan, fontSize = 12.sp) }
            Text("GAN learns through a discriminator game; diffusion learns to predict noise across many noise levels and iteratively reverses the process.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun ComparisonLab() {
    val context = LocalContext.current
    val preferences = remember { context.getSharedPreferences("ml_lab_phase_four", Context.MODE_PRIVATE) }
    val attentionTime = remember { measureNanoTime { repeat(100) { ScaledDotProductAttention(4).forward(Matrix(4, 4, FloatArray(16) { it / 16f })) } } / 100.0 / 1_000_000.0 }
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Architecture Comparison", "Measured educational kernels, not production runtime claims")
            listOf("Transformer" to "Token relationships - O(n^2) attention memory", "ViT" to "Image patches and global token mixing", "GCN" to "Sparse neighborhood aggregation", "VAE" to "Regularized probabilistic latent generation", "GAN" to "Adversarial direct sampling", "Diffusion" to "Iterative denoising generation").forEach { (name, detail) -> Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) { Text(name, color = LabText, fontWeight = FontWeight.Bold); Text(detail, color = LabMuted, fontSize = 11.sp) } }
            MetricPill("Attention forward", "%.3f ms".format(attentionTime), LabCyan, Modifier.fillMaxWidth())
            SegmentedOption("Save Phase 4 experiment", false, Modifier.fillMaxWidth()) { preferences.edit().putFloat("attention_ms", attentionTime.toFloat()).putLong("saved_at", System.currentTimeMillis()).apply() }
            Text("Explainability caveat: attention, feature maps, graph coefficients and latent dimensions reveal model internals, but none alone proves why a prediction happened.", color = LabPink, fontSize = 12.sp)
            Text("All labs use local Kotlin FloatArray computation and deterministic seeds. Sequence length, graph size, patch count and generation steps are deliberately constrained for memory and thermal safety.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable private fun VectorReadout(title: String, values: FloatArray, color: Color) { Column(verticalArrangement = Arrangement.spacedBy(3.dp)) { Text("$title  [${values.joinToString { "%.3f".format(it) }}]", color = LabMuted, fontSize = 11.sp); Row(Modifier.fillMaxWidth().height(8.dp)) { values.forEach { Box(Modifier.weight(1f).height(8.dp).padding(horizontal = 1.dp).background(color.copy(alpha = .2f + .8f * abs(it).coerceIn(0f, 1f)), RoundedCornerShape(2.dp))) } } } }

@Composable private fun MatrixHeatmap(matrix: Matrix, labels: List<String>, selectedRow: Int, selectedColumn: Int, onCell: (Int, Int) -> Unit) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        val cw = size.width / matrix.columns; val ch = size.height / matrix.rows; val min = matrix.values.minOrNull() ?: 0f; val max = matrix.values.maxOrNull() ?: 1f; val range = (max - min).coerceAtLeast(1e-6f)
        for (row in 0 until matrix.rows) for (column in 0 until matrix.columns) { val v = (matrix[row, column] - min) / range; drawRect(LabPurple.copy(alpha = .12f + .85f * v), Offset(column * cw, row * ch), androidx.compose.ui.geometry.Size(cw - 2f, ch - 2f)); if (row == selectedRow && column == selectedColumn) drawRect(Color.White, Offset(column * cw, row * ch), androidx.compose.ui.geometry.Size(cw - 2f, ch - 2f), style = Stroke(4f)) }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) { labels.take(matrix.columns).forEachIndexed { column, label -> SegmentedOption(label, selectedColumn == column, Modifier.weight(1f)) { onCell(selectedRow.coerceIn(0, matrix.rows - 1), column) } } }
}

@Composable private fun PositionHeatmap(matrix: Matrix) { Canvas(Modifier.fillMaxWidth().height(100.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) { val cw = size.width / matrix.columns; val ch = size.height / matrix.rows; matrix.values.forEachIndexed { index, value -> drawRect(if (value >= 0) LabCyan.copy(alpha = .15f + .75f * abs(value)) else LabPink.copy(alpha = .15f + .75f * abs(value)), Offset(index % matrix.columns * cw, index / matrix.columns * ch), androidx.compose.ui.geometry.Size(cw - 2f, ch - 2f)) } } }

@Composable
private fun AttentionLinksCanvas(tokens: List<String>, weights: Matrix, selectedQuery: Int, threshold: Float) {
    val links = attentionLinks(weights, threshold).filter { it.query == selectedQuery }
    Canvas(Modifier.fillMaxWidth().height(120.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        val spacing = size.width / tokens.size
        val centers = List(tokens.size) { Offset(spacing * (it + .5f), size.height - 22f) }
        links.forEach { link ->
            val from = centers[link.query]; val to = centers[link.key]
            val path = Path().apply { moveTo(from.x, from.y); quadraticTo((from.x + to.x) / 2f, 12f, to.x, to.y) }
            drawPath(path, if (link.key == selectedQuery) LabPink else LabCyan, style = Stroke(1.5f + 12f * link.weight))
        }
        centers.forEachIndexed { index, center -> drawCircle(if (index == selectedQuery) LabOrange else LabPurple, if (index == selectedQuery) 11f else 8f, center) }
    }
}

@Composable private fun PatchGrid(values: FloatArray, imageSize: Int, patchSize: Int, selected: Int, attention: FloatArray, onSelect: (Int) -> Unit) { val patchesPerRow = imageSize / patchSize; Canvas(Modifier.fillMaxWidth().height(280.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) { val cw = size.width / imageSize; val ch = size.height / imageSize; for (row in 0 until imageSize) for (column in 0 until imageSize) { val patch = row / patchSize * patchesPerRow + column / patchSize; val value = values[row * imageSize + column].coerceIn(0f, 1f); drawRect(Color(value, .25f + value * .4f, 1f - value * .6f, 1f), Offset(column * cw, row * ch), androidx.compose.ui.geometry.Size(cw - 1f, ch - 1f)); drawRect(LabOrange.copy(alpha = (attention.getOrElse(patch) { 0f } * 2f).coerceIn(.05f, .8f)), Offset(column * cw, row * ch), androidx.compose.ui.geometry.Size(cw - 1f, ch - 1f)); if (patch == selected) drawRect(Color.White, Offset(column / patchSize * patchSize * cw, row / patchSize * patchSize * ch), androidx.compose.ui.geometry.Size(patchSize * cw, patchSize * ch), style = Stroke(4f)) } }; Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(4.dp)) { repeat(patchesPerRow * patchesPerRow) { SegmentedOption("P${it + 1}", selected == it) { onSelect(it) } } } }

@Composable private fun GraphCanvas(graph: Graph, embeddings: Matrix, selected: Int, onSelect: (Int) -> Unit) { Canvas(Modifier.fillMaxWidth().height(300.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) { val centers = List(graph.features.rows) { index -> val community = if (index < 4) 0 else 1; val local = index % 4; Offset(size.width * (if (community == 0) .27f else .73f) + (local % 2 - .5f) * 90f, size.height * (.32f + local / 2 * .34f)) }; graph.edges.forEach { (a, b) -> drawLine(if (a == selected || b == selected) LabOrange else LabMuted.copy(alpha = .35f), centers[a], centers[b], if (a == selected || b == selected) 6f else 3f) }; centers.forEachIndexed { index, center -> val color = if (graph.labels.getOrElse(index) { 0 } == 0) LabCyan else LabPink; drawCircle(if (index == selected) Color.White else color.copy(alpha = .3f), if (index == selected) 23f else 19f, center); drawCircle(color, 12f, center) } }; Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(4.dp)) { repeat(graph.features.rows) { SegmentedOption("N${it + 1}", selected == it) { onSelect(it) } } } }

@Composable private fun BinaryReadout(title: String, values: FloatArray) { Column(verticalArrangement = Arrangement.spacedBy(4.dp)) { Text(title, color = LabMuted, fontSize = 11.sp); Row(Modifier.fillMaxWidth().height(52.dp)) { values.forEach { Box(Modifier.weight(1f).height(52.dp).padding(horizontal = 1.dp).background(LabCyan.copy(alpha = .1f + .9f * it.coerceIn(0f, 1f)), RoundedCornerShape(2.dp))) } } } }

@Composable
private fun DistributionPlot(gan: TinyGan) {
    Canvas(Modifier.fillMaxWidth().height(170.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        fun scaleX(value: Float) = size.width * (value + 4f) / 9f
        drawLine(LabMuted, Offset(0f, size.height * .75f), Offset(size.width, size.height * .75f), 2f)
        repeat(40) { index ->
            val z = -2f + 4f * index / 39f
            val generated = gan.generate(z)
            drawCircle(LabCyan.copy(alpha = .65f), 5f, Offset(scaleX(generated), size.height * (.25f + (index % 5) * .08f)))
            val target = 2f + .6f * z
            drawCircle(LabPink.copy(alpha = .45f), 4f, Offset(scaleX(target), size.height * (.72f - (index % 5) * .06f)))
        }
    }
}

@Composable
private fun DiffusionPlot(clean: FloatArray, noisy: FloatArray, predicted: FloatArray) {
    Canvas(Modifier.fillMaxWidth().height(200.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        fun point(values: FloatArray) = Offset(size.width * (values[0] + 2f) / 4f, size.height * (1f - (values[1] + 2f) / 4f))
        drawLine(LabPurple, point(clean), point(noisy), 4f)
        drawCircle(LabGreen, 10f, point(clean))
        drawCircle(LabPurple, 10f, point(noisy))
        val end = floatArrayOf(noisy[0] - predicted[0], noisy[1] - predicted[1])
        drawLine(LabCyan, point(noisy), point(end), 5f)
        drawCircle(LabCyan, 7f, point(end))
    }
}

@Composable
private fun NoiseSchedulePlot(betas: FloatArray, alphaBars: FloatArray, selected: Int) {
    Canvas(Modifier.fillMaxWidth().height(110.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        fun point(index: Int, value: Float) = Offset(index * size.width / (betas.size - 1), size.height * (1f - value.coerceIn(0f, 1f)))
        val betaPath = Path(); val alphaPath = Path()
        betas.indices.forEach { index -> if (index == 0) { betaPath.moveTo(point(index, betas[index]).x, point(index, betas[index]).y); alphaPath.moveTo(point(index, alphaBars[index]).x, point(index, alphaBars[index]).y) } else { betaPath.lineTo(point(index, betas[index]).x, point(index, betas[index]).y); alphaPath.lineTo(point(index, alphaBars[index]).x, point(index, alphaBars[index]).y) } }
        drawPath(betaPath, LabOrange, style = Stroke(3f)); drawPath(alphaPath, LabPurple, style = Stroke(3f)); drawLine(Color.White.copy(alpha = .65f), Offset(point(selected, 0f).x, 0f), Offset(point(selected, 0f).x, size.height), 2f)
    }
    Text("Noise schedule: orange beta, purple cumulative alpha; white marker is the selected timestep.", color = LabMuted, fontSize = 11.sp)
}

@Composable
private fun DiffusionTrajectoryPlot(points: List<FloatArray>) {
    Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp))) {
        fun point(values: FloatArray) = Offset(size.width * (values[0] + 3f) / 6f, size.height * (1f - (values[1] + 3f) / 6f))
        points.zipWithNext().forEachIndexed { index, (from, to) -> drawLine(LabCyan.copy(alpha = .25f + .75f * index / points.size), point(from), point(to), 3f) }
        points.forEachIndexed { index, values -> drawCircle(if (index == points.lastIndex) LabGreen else LabPurple.copy(alpha = .45f), if (index == points.lastIndex) 8f else 3f, point(values)) }
    }
}

private fun addNode(graph: Graph, connectTo: Int): Graph { if (graph.features.rows >= 12) return graph; val rows = graph.features.rows + 1; val values = FloatArray(rows * graph.features.columns); graph.features.values.copyInto(values); values[(rows - 1) * graph.features.columns] = .5f; values[(rows - 1) * graph.features.columns + 1] = .5f; return Graph(Matrix(rows, graph.features.columns, values), graph.edges + (connectTo to rows - 1), graph.labels + (graph.labels.getOrElse(connectTo) { 0 })) }
private fun averagePairDistance(matrix: Matrix): Float {
    var total = 0f
    var count = 0
    for (a in 0 until matrix.rows) for (b in a + 1 until matrix.rows) {
        var squared = 0f
        for (column in 0 until matrix.columns) {
            val difference = matrix[a, column] - matrix[b, column]
            squared += difference * difference
        }
        total += sqrt(squared)
        count++
    }
    return total / count.coerceAtLeast(1)
}
private fun diffusionPoints() = List(40) { index -> val angle = index * .31f; floatArrayOf(kotlin.math.cos(angle), kotlin.math.sin(angle)) }
