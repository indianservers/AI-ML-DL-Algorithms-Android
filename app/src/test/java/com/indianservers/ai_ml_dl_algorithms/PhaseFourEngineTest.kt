package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.data.PhaseThreeContent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.GraphConvolution
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.CrossAttention
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.GraphPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.LearnedPositionEmbedding
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.Matrix
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.sinusoidalPositionEncoding
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.twoCommunityGraph
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.removeEdge
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine.removeNode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PhaseFourEngineTest {
    private val input = Matrix(3, 4, floatArrayOf(.8f, .1f, -.2f, .4f, .2f, .9f, .3f, -.1f, -.4f, .5f, .9f, .2f))

    @Test
    fun scaledAttentionNormalizesAndCausalMaskBlocksFuture() {
        val result = ScaledDotProductAttention(4, 3).forward(input, causal = true)
        repeat(result.weights.rows) { row -> assertEquals(1f, result.weights.row(row).sum(), 1e-5f) }
        assertEquals(0f, result.weights[0, 1], 0f)
        assertEquals(0f, result.weights[0, 2], 0f)
        assertTrue(result.output.values.all { it.isFinite() })
    }

    @Test
    fun multiHeadPositionLayerNormAndTransformerAreStable() {
        val heads = MultiHeadAttention(4, 2).forward(input)
        assertEquals(2, heads.heads.size)
        assertEquals(4, heads.output.columns)
        val position = sinusoidalPositionEncoding(3, 4)
        assertEquals(0f, position[0, 0], 0f)
        assertEquals(1f, position[0, 1], 0f)
        val normalized = layerNorm(floatArrayOf(2.4f, -.3f, 1.8f, .4f)).output
        assertEquals(0f, normalized.average().toFloat(), 1e-5f)
        assertTrue(TransformerEncoderBlock(4, 2, 8).forward(input).output.values.all { it.isFinite() })
    }

    @Test
    fun temperatureAndTopKProduceValidDistribution() {
        val distribution = probabilities(floatArrayOf(4.2f, 3.7f, 1.1f), .8f, 2)
        assertEquals(1f, distribution.sum(), 1e-6f)
        assertEquals(0f, distribution[2], 0f)
        assertTrue(distribution[0] > distribution[1])
    }

    @Test
    fun vitPatchesAndGcnMessagesUseActualInputs() {
        val image = PhaseThreeContent.shapeDataset.first().image
        val patches = PatchEmbedding(8, 2, 4).forward(image)
        assertEquals(16, patches.patches.size)
        assertEquals(16, patches.embeddings.rows)
        val graphResult = GraphConvolution(2, 2).forward(twoCommunityGraph())
        assertTrue(graphResult.normalizedAdjacency.values.all { it.isFinite() })
        assertTrue(graphResult.messages.values.any { abs(it) > 1e-5f })
        assertTrue(graphResult.output.values.all { it >= 0f })
    }

    @Test
    fun vaeReparameterizationAndKlAreFiniteAndDeterministic() {
        val vae = TinyVae()
        val first = vae.sample(PhaseThreeContent.reconstructionPatterns.first(), 9)
        val second = vae.sample(PhaseThreeContent.reconstructionPatterns.first(), 9)
        assertEquals(first.latent.toList(), second.latent.toList())
        assertTrue(first.klLoss >= 0f && first.klLoss.isFinite())
        assertTrue(first.reconstruction.all { it in 0f..1f })
    }

    @Test
    fun ganGeneratedMeanMovesTowardTarget() {
        val gan = TinyGan()
        val before = abs(gan.train(0).last().generatedMean - 2f)
        val after = abs(gan.train(1200).last().generatedMean - 2f)
        assertTrue("distance $before -> $after", after < before)
    }

    @Test
    fun diffusionTrainingImprovesNoisePredictionAndReverseIsFinite() {
        val diffusion = TinyDiffusion()
        val points = List(40) { index -> val angle = index * .31f; floatArrayOf(kotlin.math.cos(angle), kotlin.math.sin(angle)) }
        val history = diffusion.train(points, 800, .01f)
        assertTrue("loss ${history.first()} -> ${history.last()}", history.last() < history.first())
        val noisy = diffusion.forward(points.first(), 40, 7).noisy
        assertTrue(diffusion.reverseStep(noisy, 40).all { it.isFinite() })
    }

    @Test
    fun paddingMaskLinksAndCrossAttentionUseRealWeights() {
        val mask = booleanArrayOf(true, true, false)
        val self = ScaledDotProductAttention(4, 7).forward(input, paddingMask = mask)
        repeat(self.weights.rows) { assertEquals(0f, self.weights[it, 2], 0f) }
        assertTrue(attentionLinks(self.weights, .2f).all { it.weight >= .2f })
        val decoder = Matrix(1, 4, input.row(0))
        val cross = CrossAttention(4, 9).forward(decoder, input, mask)
        assertEquals(1f, cross.attention.weights.row(0).sum(), 1e-5f)
        assertEquals(0f, cross.attention.weights[0, 2], 0f)
    }

    @Test
    fun learnedPositionsGenerationAndTransformerTelemetryAreDeterministic() {
        val learned = LearnedPositionEmbedding(8, 4, 11).encoding(3)
        assertEquals(3, learned.rows)
        assertEquals(32, LearnedPositionEmbedding(8, 4).parameterCount())
        val generation = TinyAutoregressiveDecoder(13).generate(4, .8f, 3)
        assertTrue(generation.isNotEmpty())
        generation.forEach { assertEquals(1f, it.probabilities.sum(), 1e-5f) }
        val metrics = TinyTransformerTask(15).train(100)
        assertTrue("loss ${metrics.first().loss} -> ${metrics.last().loss}", metrics.last().loss < metrics.first().loss)
        assertTrue("accuracy ${metrics.last().accuracy}", metrics.last().accuracy >= .75f)
        assertTrue(metrics.all { it.loss.isFinite() && it.attentionEntropy.isFinite() })
    }

    @Test
    fun graphEditingAndNodeClassificationRemainValid() {
        val chain = graphPreset(GraphPreset.Chain)
        val connected = chain.connect(0, 2)
        assertTrue(connected.edges.any { it == (0 to 2) })
        val disconnected = connected.removeEdge(0, 2)
        assertTrue(disconnected.edges.none { it == (0 to 2) })
        val smaller = disconnected.removeNode(1)
        assertEquals(chain.features.rows - 1, smaller.features.rows)
        assertTrue(smaller.edges.all { it.first in 0 until smaller.features.rows && it.second in 0 until smaller.features.rows })
        val result = TinyGcnNodeClassifier(17).train(twoCommunityGraph(), 200)
        assertTrue(result.loss.isFinite())
        assertTrue(result.accuracy >= .75f)
    }

    @Test
    fun interpolationGanStepsAndDiffusionTrajectoryAreFinite() {
        assertEquals(listOf(.5f, 1f), interpolateLatent(floatArrayOf(0f, 0f), floatArrayOf(1f, 2f), .5f).toList())
        val gan = TinyGan(19)
        val generatorBefore = gan.generatorWeight
        gan.stepDiscriminator()
        assertEquals(generatorBefore, gan.generatorWeight, 0f)
        gan.stepGenerator()
        assertTrue(gan.generatorWeight.isFinite())
        gan.modeCollapse()
        assertEquals(.01f, gan.generatorWeight, 0f)
        val diffusion = TinyDiffusion(20)
        val start = diffusion.forward(floatArrayOf(.2f, -.4f), 15, 3).noisy
        val trajectory = diffusion.reverseTrajectory(start, 15)
        assertEquals(16, trajectory.size)
        assertTrue(trajectory.all { point -> point.all { it.isFinite() } })
    }
}
