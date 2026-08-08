package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.RecurrentModelKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.SequencePreset
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseSevenSequenceEngineTest {
    @Test
    fun rnnStepUsesInputAndPreviousHiddenState() {
        val step = PhaseSevenEngines.rnnStep(input = .8, previousHidden = .3)
        assertEquals(.56, step.inputContribution, 1e-12)
        assertEquals(.12, step.memoryContribution, 1e-12)
        assertEquals(.73, step.z, 1e-12)
        assertTrue(step.hidden in -1.0..1.0)
        assertTrue(step.output in 0.0..1.0)
    }

    @Test
    fun rnnSequenceForwardProducesOneStepPerInputAndLoss() {
        val inputs = PhaseSevenEngines.sequence(SequencePreset.Increasing, 6)
        val state = PhaseSevenEngines.rnnForward(inputs, target = 1.0)
        assertEquals(inputs.size, state.steps.size)
        assertTrue(state.loss.isFinite())
        assertEquals(state.steps.last().output, state.prediction, 0.0)
    }

    @Test
    fun bpttGradientMatchesFiniteDifference() {
        val (analytical, numerical) = PhaseSevenEngines.finiteDifferenceRecurrentGradient()
        assertEquals(numerical, analytical, 1e-4)
    }

    @Test
    fun gradientClippingBoundsRecurrentGradient() {
        val state = PhaseSevenEngines.bptt(List(10) { 1.0 }, target = 1.0, wh = 1.2, clip = .25)
        assertTrue(kotlin.math.abs(state.clippedGradient) <= .25 + 1e-12)
        assertEquals(10, state.gradients.size)
    }

    @Test
    fun lstmGatesUpdateCellAndHiddenState() {
        val step = PhaseSevenEngines.lstmStep(input = .6, previousHidden = .2, previousCell = .8, forget = .5, inputGate = .25, outputGate = .67)
        assertEquals(.4 + .25 * step.candidate, step.cell, 1e-12)
        assertEquals(.67 * kotlin.math.tanh(step.cell), step.hidden, 1e-12)
        assertTrue(step.forget in 0.0..1.0)
    }

    @Test
    fun gruResetUpdateAndHiddenStateAreValid() {
        val step = PhaseSevenEngines.gruStep(input = .6, previousHidden = .7, reset = .45, update = .8)
        assertEquals(.45, step.resetGate, 0.0)
        assertEquals(.8, step.updateGate, 0.0)
        assertEquals((1.0 - .8) * step.candidate + .8 * .7, step.hidden, 1e-12)
    }

    @Test
    fun parameterCountsScaleByGateSets() {
        val rnn = PhaseSevenEngines.parameterCount(RecurrentModelKind.Rnn, inputSize = 1, hiddenSize = 4, outputSize = 1)
        val lstm = PhaseSevenEngines.parameterCount(RecurrentModelKind.Lstm, inputSize = 1, hiddenSize = 4, outputSize = 1)
        val gru = PhaseSevenEngines.parameterCount(RecurrentModelKind.Gru, inputSize = 1, hiddenSize = 4, outputSize = 1)
        assertTrue(lstm > gru)
        assertTrue(gru > rnn)
    }

    @Test
    fun trainingTracesAndOneHotStateAreDeterministic() {
        val trace = PhaseSevenEngines.train(RecurrentModelKind.Lstm, SequencePreset.Delayed, epochs = 12, hiddenSize = 3)
        assertEquals(12, trace.losses.size)
        assertTrue(trace.losses.last() < trace.losses.first())
        val oneHot = PhaseSevenEngines.oneHot(listOf("A", "B", "C", "A"))
        assertEquals(listOf(1, 0, 0), oneHot.getValue("A"))
        assertEquals(1.0, PhaseSevenEngines.softmax(listOf(1.0, 2.0, 3.0)).sum(), 1e-12)
    }
}
