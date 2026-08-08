package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.DlActivation
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.DlLossKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.DlOptimizerKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFiveEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseFiveDeepLearningEngineTest {
    @Test
    fun neuronWeightedSumAndActivationsAreCorrect() {
        val calc = PhaseFiveEngines.neuron(listOf(.8, -.4), listOf(.7, .3), .2, DlActivation.Relu)
        assertEquals(.64, calc.z, 1e-12)
        assertEquals(.64, calc.output, 1e-12)
        assertEquals(0.5, PhaseFiveEngines.activate(0.0, DlActivation.Sigmoid), 1e-12)
        assertEquals(0.0, PhaseFiveEngines.activate(-1.0, DlActivation.Relu), 0.0)
        assertTrue(PhaseFiveEngines.activate(-1.0, DlActivation.LeakyRelu) < 0.0)
    }

    @Test
    fun softmaxAndLossFunctionsAreNumericallyValid() {
        val probs = PhaseFiveEngines.softmax(listOf(2.1, 1.2, .3))
        assertEquals(1.0, probs.sum(), 1e-12)
        assertTrue(probs[0] > probs[1] && probs[1] > probs[2])
        assertEquals(4.0, PhaseFiveEngines.loss(3.0, 5.0, DlLossKind.Mse), 0.0)
        assertTrue(PhaseFiveEngines.loss(.95, 1.0, DlLossKind.Bce) < PhaseFiveEngines.loss(.05, 1.0, DlLossKind.Bce))
        assertEquals(-kotlin.math.ln(.7), PhaseFiveEngines.loss(.7, 1.0, DlLossKind.Cce), 1e-12)
    }

    @Test
    fun forwardPropagationProducesLayerActivationsAndLoss() {
        val (weights, biases) = PhaseFiveEngines.initialize(listOf(2, 3, 1), 4)
        val state = PhaseFiveEngines.forward(listOf(.5, .8), weights, biases, target = 1.0)
        assertEquals(2, state.layers.size)
        assertEquals(3, state.layers.first().activations.size)
        assertTrue(state.prediction in 0.0..1.0)
        assertTrue(state.loss.isFinite())
    }

    @Test
    fun backpropGradientMatchesFiniteDifference() {
        val (analytical, numerical) = PhaseFiveEngines.finiteDifferenceGradient()
        assertEquals(numerical, analytical, 1e-4)
    }

    @Test
    fun sgdAdamAndL2UpdatesAreFinite() {
        val sgd = PhaseFiveEngines.sgd(weight = 1.0, gradient = .5, learningRate = .1, l2 = .2)
        assertEquals(.93, sgd.after, 1e-12)
        val adam = PhaseFiveEngines.adam(weight = 1.0, gradient = .5, learningRate = .1, t = 1)
        assertTrue(adam.after.isFinite())
        val (weights, _) = PhaseFiveEngines.initialize(listOf(2, 2, 1), 3)
        assertTrue(PhaseFiveEngines.l2Penalty(weights, .01) > 0.0)
    }

    @Test
    fun dropoutMaskAndInitializationAreDeterministic() {
        val a = PhaseFiveEngines.dropoutMask(8, .25, 7)
        val b = PhaseFiveEngines.dropoutMask(8, .25, 7)
        assertEquals(a, b)
        assertTrue(a.any { it } && a.any { !it })
        val initA = PhaseFiveEngines.initialize(listOf(2, 2, 1), 9)
        val initB = PhaseFiveEngines.initialize(listOf(2, 2, 1), 9)
        assertEquals(initA, initB)
    }

    @Test
    fun xorHiddenLayerImprovesNonlinearSeparation() {
        val data = PhaseFiveEngines.xorData()
        val linearCorrect = data.count { (PhaseFiveEngines.xorPrediction(it.x, it.y, hidden = false) >= .5).toInt() == it.label }
        val hiddenCorrect = data.count { (PhaseFiveEngines.xorPrediction(it.x, it.y, hidden = true) >= .5).toInt() == it.label }
        assertTrue(hiddenCorrect >= linearCorrect)
    }

    @Test
    fun optimizerTrainingTracesDecreaseLoss() {
        val sgd = PhaseFiveEngines.trainingTrace(DlOptimizerKind.Sgd, epochs = 30, learningRate = .08)
        val adam = PhaseFiveEngines.trainingTrace(DlOptimizerKind.Adam, epochs = 30, learningRate = .08)
        assertTrue(sgd.losses.last() < sgd.losses.first())
        assertTrue(adam.losses.last() < adam.losses.first())
        assertEquals(30, sgd.losses.size)
    }

    private fun Boolean.toInt(): Int = if (this) 1 else 0
}
