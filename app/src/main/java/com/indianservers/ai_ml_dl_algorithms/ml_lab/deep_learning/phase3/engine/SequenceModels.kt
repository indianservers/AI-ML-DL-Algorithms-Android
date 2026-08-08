package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine

import kotlin.math.exp
import kotlin.math.tanh
import kotlin.random.Random

private fun sigmoid(x: Float) = if (x >= 0f) 1f / (1f + exp(-x)) else exp(x) / (1f + exp(x))
private fun randomArray(size: Int, random: Random, scale: Float = 0.35f) = FloatArray(size) { (random.nextFloat() * 2f - 1f) * scale }
private fun affine(weights: FloatArray, rows: Int, cols: Int, input: FloatArray, bias: FloatArray): FloatArray = FloatArray(rows) { row ->
    bias[row] + (0 until cols).sumOf { (weights[row * cols + it] * input[it]).toDouble() }.toFloat()
}
private fun concat(a: FloatArray, b: FloatArray) = a + b

data class RnnStep(val input: FloatArray, val previousHidden: FloatArray, val hidden: FloatArray, val output: FloatArray)
data class RnnTrace(val steps: List<RnnStep>, val loss: Float = 0f)

class VanillaRnn(val inputSize: Int, val hiddenSize: Int, val outputSize: Int, seed: Int = 31) {
    private val random = Random(seed)
    val inputWeights = randomArray(hiddenSize * inputSize, random)
    val recurrentWeights = randomArray(hiddenSize * hiddenSize, random)
    val hiddenBias = FloatArray(hiddenSize)
    val outputWeights = randomArray(outputSize * hiddenSize, random)
    val outputBias = FloatArray(outputSize)
    val inputGradients = FloatArray(inputWeights.size)
    val recurrentGradients = FloatArray(recurrentWeights.size)

    fun forward(sequence: List<FloatArray>): RnnTrace {
        var hidden = FloatArray(hiddenSize)
        val steps = sequence.map { input ->
            val previous = hidden.copyOf()
            hidden = FloatArray(hiddenSize) { h ->
                val fromInput = (0 until inputSize).sumOf { (inputWeights[h * inputSize + it] * input[it]).toDouble() }.toFloat()
                val recurrent = (0 until hiddenSize).sumOf { (recurrentWeights[h * hiddenSize + it] * previous[it]).toDouble() }.toFloat()
                tanh(fromInput + recurrent + hiddenBias[h])
            }
            val output = affine(outputWeights, outputSize, hiddenSize, hidden, outputBias)
            RnnStep(input.copyOf(), previous, hidden.copyOf(), output)
        }
        return RnnTrace(steps)
    }

    fun bptt(sequence: List<FloatArray>, target: FloatArray, learningRate: Float = 0f): RnnTrace {
        val trace = forward(sequence)
        inputGradients.fill(0f); recurrentGradients.fill(0f)
        val output = trace.steps.last().output
        val dOutput = FloatArray(outputSize) { 2f * (output[it] - target[it]) / outputSize }
        val outputGrad = FloatArray(outputWeights.size)
        var hiddenGradient = FloatArray(hiddenSize)
        for (o in 0 until outputSize) for (h in 0 until hiddenSize) {
            outputGrad[o * hiddenSize + h] = dOutput[o] * trace.steps.last().hidden[h]
            hiddenGradient[h] += outputWeights[o * hiddenSize + h] * dOutput[o]
        }
        for (time in trace.steps.indices.reversed()) {
            val step = trace.steps[time]
            val local = FloatArray(hiddenSize) { hiddenGradient[it] * (1f - step.hidden[it] * step.hidden[it]) }
            val previousGradient = FloatArray(hiddenSize)
            for (h in 0 until hiddenSize) {
                hiddenBias[h] -= learningRate * local[h]
                for (i in 0 until inputSize) inputGradients[h * inputSize + i] += local[h] * step.input[i]
                for (p in 0 until hiddenSize) {
                    recurrentGradients[h * hiddenSize + p] += local[h] * step.previousHidden[p]
                    previousGradient[p] += recurrentWeights[h * hiddenSize + p] * local[h]
                }
            }
            hiddenGradient = previousGradient
        }
        if (learningRate > 0f) {
            inputWeights.indices.forEach { inputWeights[it] -= learningRate * inputGradients[it] }
            recurrentWeights.indices.forEach { recurrentWeights[it] -= learningRate * recurrentGradients[it] }
            outputWeights.indices.forEach { outputWeights[it] -= learningRate * outputGrad[it] }
            outputBias.indices.forEach { outputBias[it] -= learningRate * dOutput[it] }
        }
        val loss = output.indices.sumOf { ((output[it] - target[it]) * (output[it] - target[it])).toDouble() }.toFloat() / output.size
        return RnnTrace(trace.steps, loss)
    }

    fun parameterCount() = inputWeights.size + recurrentWeights.size + hiddenBias.size + outputWeights.size + outputBias.size
}

data class LstmStep(val forget: FloatArray, val inputGate: FloatArray, val candidate: FloatArray, val outputGate: FloatArray, val cell: FloatArray, val hidden: FloatArray)
class LstmCell(val inputSize: Int, val hiddenSize: Int, seed: Int = 41) {
    private val combined = inputSize + hiddenSize
    private val random = Random(seed)
    private val wf = randomArray(hiddenSize * combined, random); private val wi = randomArray(hiddenSize * combined, random)
    private val wg = randomArray(hiddenSize * combined, random); private val wo = randomArray(hiddenSize * combined, random)
    private val bf = FloatArray(hiddenSize) { 1f }; private val bi = FloatArray(hiddenSize); private val bg = FloatArray(hiddenSize); private val bo = FloatArray(hiddenSize)
    fun forward(sequence: List<FloatArray>): List<LstmStep> {
        var hidden = FloatArray(hiddenSize); var cell = FloatArray(hiddenSize)
        return sequence.map { input ->
            val joined = concat(input, hidden)
            val forget = affine(wf, hiddenSize, combined, joined, bf).map(::sigmoid).toFloatArray()
            val inputGate = affine(wi, hiddenSize, combined, joined, bi).map(::sigmoid).toFloatArray()
            val candidate = affine(wg, hiddenSize, combined, joined, bg).map { tanh(it) }.toFloatArray()
            val outputGate = affine(wo, hiddenSize, combined, joined, bo).map(::sigmoid).toFloatArray()
            cell = FloatArray(hiddenSize) { forget[it] * cell[it] + inputGate[it] * candidate[it] }
            hidden = FloatArray(hiddenSize) { outputGate[it] * tanh(cell[it]) }
            LstmStep(forget, inputGate, candidate, outputGate, cell.copyOf(), hidden.copyOf())
        }
    }
    fun parameterCount() = 4 * (hiddenSize * combined + hiddenSize)
}

data class GruStep(val reset: FloatArray, val update: FloatArray, val candidate: FloatArray, val hidden: FloatArray)
class GruCell(val inputSize: Int, val hiddenSize: Int, seed: Int = 51) {
    private val combined = inputSize + hiddenSize; private val random = Random(seed)
    private val wr = randomArray(hiddenSize * combined, random); private val wz = randomArray(hiddenSize * combined, random); private val wn = randomArray(hiddenSize * combined, random)
    private val br = FloatArray(hiddenSize); private val bz = FloatArray(hiddenSize); private val bn = FloatArray(hiddenSize)
    fun forward(sequence: List<FloatArray>): List<GruStep> {
        var hidden = FloatArray(hiddenSize)
        return sequence.map { input ->
            val joined = concat(input, hidden)
            val reset = affine(wr, hiddenSize, combined, joined, br).map(::sigmoid).toFloatArray()
            val update = affine(wz, hiddenSize, combined, joined, bz).map(::sigmoid).toFloatArray()
            val candidateInput = concat(input, FloatArray(hiddenSize) { reset[it] * hidden[it] })
            val candidate = affine(wn, hiddenSize, combined, candidateInput, bn).map { tanh(it) }.toFloatArray()
            hidden = FloatArray(hiddenSize) { (1f - update[it]) * candidate[it] + update[it] * hidden[it] }
            GruStep(reset, update, candidate, hidden.copyOf())
        }
    }
    fun parameterCount() = 3 * (hiddenSize * combined + hiddenSize)
}
