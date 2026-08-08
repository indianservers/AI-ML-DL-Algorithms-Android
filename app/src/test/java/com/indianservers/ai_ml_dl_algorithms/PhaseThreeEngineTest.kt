package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.data.PhaseThreeContent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.Conv2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.DenseAutoencoder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.EmbeddingLayer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.GruCell
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.LstmCell
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.Pool2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.PoolingType
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.TensorImage
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.TinyCnnClassifier
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.VanillaRnn
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class PhaseThreeEngineTest {
    @Test
    fun convolutionForwardMatchesKnownValues() {
        val input = TensorImage(1, 3, 3, floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f))
        val conv = Conv2D(1, 1, 2).also { floatArrayOf(1f, 0f, 0f, -1f).copyInto(it.weights); it.biases[0] = 0f }
        val output = conv.forward(input)
        assertEquals(2, output.width)
        output.values.forEach { assertEquals(-4f, it, 1e-6f) }
    }

    @Test
    fun convolutionWeightGradientMatchesFiniteDifference() {
        val input = TensorImage(1, 3, 3, floatArrayOf(1f, .2f, .3f, .4f, .8f, .1f, .7f, .5f, .9f))
        val conv = Conv2D(1, 1, 2, seed = 4)
        val output = conv.forward(input)
        conv.backward(TensorImage(1, output.height, output.width, FloatArray(output.values.size) { 1f }))
        val analytic = conv.weightGradients[0]
        val original = conv.weights[0]; val epsilon = 1e-3f
        conv.weights[0] = original + epsilon; val plus = conv.forward(input).values.sum()
        conv.weights[0] = original - epsilon; val minus = conv.forward(input).values.sum()
        conv.weights[0] = original
        assertTrue("analytic=$analytic numerical=${(plus - minus) / (2f * epsilon)}", abs(analytic - (plus - minus) / (2f * epsilon)) < 2e-3f)
    }

    @Test
    fun poolingTracksWinnerAndBackwardRoute() {
        val input = TensorImage(1, 2, 2, floatArrayOf(4f, 1f, 2f, 8f))
        val pool = Pool2D(2, 2, PoolingType.Max)
        val result = pool.forward(input)
        assertEquals(8f, result.output.values[0], 0f)
        assertEquals(3, result.winnerIndices[0])
        val gradient = pool.backward(TensorImage(1, 1, 1, floatArrayOf(2f)))
        assertEquals(listOf(0f, 0f, 0f, 2f), gradient.values.toList())
    }

    @Test
    fun tinyCnnLearnsSyntheticShapes() {
        val cnn = TinyCnnClassifier()
        val before = cnn.train(PhaseThreeContent.shapeDataset, 0).last()
        val after = cnn.train(PhaseThreeContent.shapeDataset, 30).last()
        assertTrue("loss ${before.loss} -> ${after.loss}", after.loss < before.loss * 0.55f)
        assertTrue("accuracy ${after.accuracy}", after.accuracy >= 0.85f)
    }

    @Test
    fun recurrentCellsProduceRealFiniteStateAndBpttGradients() {
        val sequence = PhaseThreeContent.sequence
        val rnn = VanillaRnn(1, 3, 1)
        val trace = rnn.bptt(sequence, floatArrayOf(1f))
        assertEquals(sequence.size, trace.steps.size)
        assertTrue(rnn.recurrentGradients.any { abs(it) > 1e-7f })
        assertTrue(LstmCell(1, 3).forward(sequence).last().cell.all { it.isFinite() })
        assertTrue(GruCell(1, 3).forward(sequence).last().hidden.all { it.isFinite() })
    }

    @Test
    fun autoencoderConvergesAndEmbeddingLookupIsStable() {
        val autoencoder = DenseAutoencoder(8, 2)
        val history = autoencoder.train(PhaseThreeContent.reconstructionPatterns, 800)
        assertTrue("loss ${history.first().loss} -> ${history.last().loss}", history.last().loss < history.first().loss * 0.35f)
        val embeddings = EmbeddingLayer(PhaseThreeContent.vocabulary, 4)
        assertEquals(4, embeddings.lookup("cat").size)
        assertEquals(1f, embeddings.cosine("cat", "cat"), 1e-5f)
    }
}
