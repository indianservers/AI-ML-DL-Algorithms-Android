package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.CnnKernelPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PaddingMode
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSixCnnEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PoolMode
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.ShapeClass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseSixCnnEngineTest {
    @Test
    fun knownConvolutionExampleMatchesExpectedValues() {
        val input = listOf(
            listOf(1.0, 2.0, 3.0),
            listOf(4.0, 5.0, 6.0),
            listOf(7.0, 8.0, 9.0)
        )
        val kernel = listOf(
            listOf(1.0, 0.0),
            listOf(0.0, -1.0)
        )
        val output = PhaseSixCnnEngines.convolve(input, kernel).output
        assertEquals(listOf(listOf(-4.0, -4.0), listOf(-4.0, -4.0)), output)
    }

    @Test
    fun stridePaddingAndOutputDimensionsAreCorrect() {
        assertEquals(4, PhaseSixCnnEngines.outputSize(input = 7, kernel = 3, padding = 1, stride = 2))
        val image = PhaseSixCnnEngines.presetImage(ShapeClass.Vertical, 7)
        val same = PhaseSixCnnEngines.convolve(image, PhaseSixCnnEngines.kernel(CnnKernelPreset.Blur), stride = 1, padding = 1)
        val validStride = PhaseSixCnnEngines.convolve(image, PhaseSixCnnEngines.kernel(CnnKernelPreset.Blur), stride = 2, padding = 0)
        assertEquals(7, same.output.size)
        assertEquals(3, validStride.output.size)
        assertEquals(9, same.current.patch.flatten().size)
    }

    @Test
    fun reluMaxPoolAndAveragePoolUseRealValues() {
        val input = listOf(listOf(-1.0, 5.0), listOf(3.0, 2.0))
        val relu = PhaseSixCnnEngines.relu(input)
        assertEquals(listOf(listOf(0.0, 5.0), listOf(3.0, 2.0)), relu)
        assertEquals(5.0, PhaseSixCnnEngines.pool(relu, 2, 2, PoolMode.Max).selectedValue, 0.0)
        assertEquals(2.5, PhaseSixCnnEngines.pool(relu, 2, 2, PoolMode.Average).selectedValue, 0.0)
    }

    @Test
    fun multiChannelConvolutionSumsChannelContributions() {
        val channel = listOf(listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, 1.0), listOf(1.0, 1.0, 1.0))
        val kernel = listOf(listOf(1.0, 0.0), listOf(0.0, 1.0))
        val output = PhaseSixCnnEngines.multiChannelConvolution(listOf(channel, channel), listOf(kernel, kernel), bias = .5)
        assertEquals(4.5, output.first().first(), 0.0)
    }

    @Test
    fun parameterCountsAndArchitectureShapesAreTracked() {
        assertEquals(80, PhaseSixCnnEngines.parameterCount(3, 3, 1, 8))
        val architecture = PhaseSixCnnEngines.architecture(filters1 = 4, filters2 = 8)
        assertEquals("Input", architecture.first().name)
        assertTrue(architecture.any { it.parameters > 0 })
        assertEquals(4, architecture[1].shape.channels)
    }

    @Test
    fun tinyClassifierProducesSoftmaxPredictionAndFeatureMaps() {
        val image = PhaseSixCnnEngines.presetImage(ShapeClass.X, 8, noise = .02, seed = 4)
        val prediction = PhaseSixCnnEngines.predictShape(image)
        assertEquals(1.0, prediction.probabilities.sum(), 1e-9)
        assertEquals(4, prediction.featureMaps.size)
        assertTrue(prediction.predicted in ShapeClass.entries)
    }

    @Test
    fun tinyTrainingUpdateProducesFiniteLossAndWeight() {
        val state = PhaseSixCnnEngines.trainTiny(seed = 5, epochs = 12, learningRate = .05)
        assertEquals(12, state.losses.size)
        assertTrue(state.losses.last().isFinite())
        assertTrue(state.updatedWeight.isFinite())
    }
}
