package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseEightEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseEightTransformerEngineTest {
    @Test
    fun embeddingsAndQkvProjectionHaveExpectedShapes() {
        val tokens = listOf("A", "B", "C")
        val x = PhaseEightEngines.embeddings(tokens, dim = 4)
        val q = PhaseEightEngines.projection(x, 11)
        assertEquals(3, x.size)
        assertEquals(4, x.first().size)
        assertEquals(x.size, q.size)
        assertEquals(4, q.first().size)
    }

    @Test
    fun knownAttentionFixtureComputesWeightsAndWeightedValues() {
        val (weights, output) = PhaseEightEngines.knownAttention()
        assertEquals(1.0, weights.sum(), 1e-12)
        assertTrue(weights[0] > weights[1])
        assertEquals(weights[0] * 2.0, output[0], 1e-12)
        assertEquals(weights[1] * 3.0, output[1], 1e-12)
    }

    @Test
    fun causalMaskPreventsFutureProbabilityMass() {
        val state = PhaseEightEngines.attention(listOf("I", "love", "ML"), causal = true)
        state.cells.forEachIndexed { row, cells ->
            cells.forEachIndexed { col, cell ->
                if (col > row) {
                    assertTrue(cell.masked)
                    assertEquals(0.0, cell.weight, 0.0)
                }
            }
            assertEquals(1.0, cells.sumOf { it.weight }, 1e-12)
        }
    }

    @Test
    fun multiHeadSplitConcatAndProjectionAreConsistent() {
        val state = PhaseEightEngines.multiHead(listOf("A", "B", "C"), heads = 3, dim = 4)
        assertEquals(3, state.heads.size)
        assertEquals(12, state.concatenated.first().size)
        assertEquals(4, state.projected.first().size)
    }

    @Test
    fun positionalEncodingAndLayerNormAreNumericallyValid() {
        val pos = PhaseEightEngines.positional(position = 2, dim = 6)
        assertEquals(6, pos.size)
        val norm = PhaseEightEngines.layerNorm(listOf(1.0, 2.0, 3.0, 4.0))
        assertEquals(0.0, norm.average(), 1e-6)
        assertTrue(norm.all { it.isFinite() })
    }

    @Test
    fun encoderBlockAndFeedForwardProduceTokenRepresentations() {
        val block = PhaseEightEngines.encoderBlock(listOf("The", "cat", "sat"), dim = 4)
        assertEquals(3, block.output.size)
        assertEquals(4, block.output.first().size)
        assertTrue(block.feedForward.flatten().all { it.isFinite() })
    }

    @Test
    fun tinyTokenPredictionProducesSoftmaxLossAndSnapshots() {
        val prediction = PhaseEightEngines.tokenPrediction(listOf("A", "B"), expected = "C")
        assertEquals(1.0, prediction.probabilities.sum(), 1e-12)
        assertTrue(prediction.loss.isFinite())
        assertEquals(2, prediction.snapshots.size)
    }

    @Test
    fun parameterCountIncludesTransformerParts() {
        val counts = PhaseEightEngines.parameterCount(vocabSize = 8, dim = 4, heads = 2, ffHidden = 8)
        assertTrue(counts.getValue("Total") > counts.getValue("Embedding"))
        assertEquals(3 * 4 * 4, counts.getValue("QKV"))
    }
}
