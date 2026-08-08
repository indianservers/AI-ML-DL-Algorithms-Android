package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.tanh

enum class PhaseSevenConcept(val displayName: String) {
    Rnn("Recurrent Neural Network - RNN"),
    HiddenState("Hidden State"),
    Bptt("Backpropagation Through Time - BPTT"),
    Gradients("Vanishing / Exploding Gradients"),
    Lstm("LSTM"),
    LstmGates("LSTM Gates"),
    Gru("GRU"),
    SequencePrediction("Sequence Prediction Lab")
}

enum class SequencePreset(val label: String) { Increasing("Increasing"), Alternating("Alternating"), Sine("Sine"), Symbols("A B C"), Delayed("Delayed Memory") }
enum class RecurrentModelKind { Rnn, Lstm, Gru }

data class RnnStepState(
    val timestep: Int,
    val input: Double,
    val previousHidden: Double,
    val inputContribution: Double,
    val memoryContribution: Double,
    val bias: Double,
    val z: Double,
    val hidden: Double,
    val output: Double
)

data class SequenceState(val inputs: List<Double>, val steps: List<RnnStepState>, val prediction: Double, val target: Double, val loss: Double)
data class BpttState(val gradients: List<Double>, val recurrentGradient: Double, val clippedGradient: Double, val warning: String?)
data class LstmStepState(val input: Double, val previousHidden: Double, val previousCell: Double, val forget: Double, val inputGate: Double, val candidate: Double, val cell: Double, val outputGate: Double, val hidden: Double)
data class GruStepState(val input: Double, val previousHidden: Double, val resetGate: Double, val updateGate: Double, val candidate: Double, val hidden: Double)
data class SequenceTrainingState(val model: RecurrentModelKind, val losses: List<Double>, val finalPrediction: Double, val parameterCount: Int, val memoryTrace: List<Double>)

object PhaseSevenTopicMatcher {
    fun kindFor(title: String, domain: String): PhaseSevenConcept? = if (domain != "Deep Learning") null else when (title) {
        "Recurrent Neural Network", "Bidirectional RNN" -> PhaseSevenConcept.Rnn
        "LSTM" -> PhaseSevenConcept.Lstm
        "GRU" -> PhaseSevenConcept.Gru
        "Sequence-to-Sequence" -> PhaseSevenConcept.SequencePrediction
        "Backpropagation" -> null
        else -> null
    }
}

object PhaseSevenEngines {
    fun sequence(preset: SequencePreset, length: Int = 6): List<Double> = when (preset) {
        SequencePreset.Increasing -> List(length) { (it + 1) / length.toDouble() }
        SequencePreset.Alternating -> List(length) { if (it % 2 == 0) 1.0 else 0.0 }
        SequencePreset.Sine -> List(length) { sin(it * .55) }
        SequencePreset.Symbols -> List(length) { (it % 3) / 2.0 }
        SequencePreset.Delayed -> listOf(1.0) + List((length - 2).coerceAtLeast(1)) { 0.0 } + listOf(1.0)
    }

    fun rnnStep(input: Double, previousHidden: Double, wx: Double = .7, wh: Double = .4, bias: Double = .05, timestep: Int = 1): RnnStepState {
        val inputContribution = wx * input
        val memoryContribution = wh * previousHidden
        val z = inputContribution + memoryContribution + bias
        val hidden = tanh(z)
        val output = sigmoid(1.1 * hidden - .1)
        return RnnStepState(timestep, input, previousHidden, inputContribution, memoryContribution, bias, z, hidden, output)
    }

    fun rnnForward(inputs: List<Double>, target: Double = inputs.lastOrNull() ?: 0.0, wh: Double = .4): SequenceState {
        var h = 0.0
        val steps = inputs.mapIndexed { i, x ->
            rnnStep(x, h, wh = wh, timestep = i + 1).also { h = it.hidden }
        }
        val prediction = steps.last().output
        val loss = -(target * ln(prediction.coerceIn(1e-9, 1.0)) + (1.0 - target) * ln((1.0 - prediction).coerceIn(1e-9, 1.0)))
        return SequenceState(inputs, steps, prediction, target, loss)
    }

    fun bptt(inputs: List<Double>, target: Double = 1.0, wh: Double = .4, clip: Double = 1.0): BpttState {
        val forward = rnnForward(inputs, target, wh)
        var grad = (forward.prediction - target) * 1.1
        val grads = mutableListOf<Double>()
        var recurrent = 0.0
        forward.steps.asReversed().forEach { step ->
            val dz = grad * (1.0 - step.hidden * step.hidden)
            recurrent += dz * step.previousHidden
            grads += abs(dz)
            grad = dz * wh
        }
        val ordered = grads.asReversed()
        val clipped = recurrent.coerceIn(-clip, clip)
        val warning = when {
            ordered.zipWithNext().all { it.first < it.second } && ordered.first() < ordered.last() * .2 -> "Early timesteps receive much smaller gradients."
            ordered.max() > 2.0 -> "Gradient magnitude is rapidly increasing."
            else -> null
        }
        return BpttState(ordered, recurrent, clipped, warning)
    }

    fun lstmStep(input: Double, previousHidden: Double, previousCell: Double, forget: Double? = null, inputGate: Double? = null, outputGate: Double? = null): LstmStepState {
        val f = forget ?: sigmoid(.9 * input + .6 * previousHidden + .4)
        val i = inputGate ?: sigmoid(.7 * input - .2 * previousHidden)
        val candidate = tanh(.8 * input + .3 * previousHidden)
        val cell = f * previousCell + i * candidate
        val o = outputGate ?: sigmoid(.5 * input + .4 * previousHidden)
        val hidden = o * tanh(cell)
        return LstmStepState(input, previousHidden, previousCell, f, i, candidate, cell, o, hidden)
    }

    fun lstmForward(inputs: List<Double>): List<LstmStepState> {
        var h = 0.0
        var c = 0.0
        return inputs.map {
            lstmStep(it, h, c).also { step -> h = step.hidden; c = step.cell }
        }
    }

    fun gruStep(input: Double, previousHidden: Double, reset: Double? = null, update: Double? = null): GruStepState {
        val r = reset ?: sigmoid(.6 * input + .3 * previousHidden)
        val z = update ?: sigmoid(.5 * input - .4 * previousHidden)
        val candidate = tanh(.8 * input + r * .7 * previousHidden)
        val hidden = (1.0 - z) * candidate + z * previousHidden
        return GruStepState(input, previousHidden, r, z, candidate, hidden)
    }

    fun gruForward(inputs: List<Double>): List<GruStepState> {
        var h = 0.0
        return inputs.map { gruStep(it, h).also { step -> h = step.hidden } }
    }

    fun parameterCount(model: RecurrentModelKind, inputSize: Int, hiddenSize: Int, outputSize: Int): Int {
        val base = inputSize * hiddenSize + hiddenSize * hiddenSize + hiddenSize
        val recurrent = when (model) {
            RecurrentModelKind.Rnn -> base
            RecurrentModelKind.Lstm -> base * 4
            RecurrentModelKind.Gru -> base * 3
        }
        return recurrent + hiddenSize * outputSize + outputSize
    }

    fun train(model: RecurrentModelKind, preset: SequencePreset, epochs: Int, hiddenSize: Int): SequenceTrainingState {
        val inputs = sequence(preset, 8)
        val base = when (model) {
            RecurrentModelKind.Rnn -> .92
            RecurrentModelKind.Lstm -> .84
            RecurrentModelKind.Gru -> .87
        }
        val losses = List(epochs.coerceAtLeast(1)) { epoch -> base.pow(epoch + 1) + .03 / hiddenSize.coerceAtLeast(1) }
        val memory = when (model) {
            RecurrentModelKind.Rnn -> rnnForward(inputs).steps.map { it.hidden }
            RecurrentModelKind.Lstm -> lstmForward(inputs).map { it.cell }
            RecurrentModelKind.Gru -> gruForward(inputs).map { it.hidden }
        }
        return SequenceTrainingState(model, losses, memory.last(), parameterCount(model, 1, hiddenSize, 1), memory)
    }

    fun oneHot(tokens: List<String>): Map<String, List<Int>> {
        val vocab = tokens.distinct().sorted()
        return vocab.associateWith { token -> vocab.map { if (it == token) 1 else 0 } }
    }

    fun softmax(values: List<Double>): List<Double> {
        val m = values.max()
        val exps = values.map { exp(it - m) }
        return exps.map { it / exps.sum().coerceAtLeast(1e-12) }
    }

    fun finiteDifferenceRecurrentGradient(): Pair<Double, Double> {
        val inputs = listOf(.2, .4, .6)
        val target = 1.0
        val wh = .4
        val analytical = bptt(inputs, target, wh).recurrentGradient
        fun f(delta: Double) = rnnForward(inputs, target, wh + delta).loss
        val eps = 1e-5
        return analytical to (f(eps) - f(-eps)) / (2.0 * eps)
    }

    private fun sigmoid(x: Double): Double = 1.0 / (1.0 + exp(-x))
}
