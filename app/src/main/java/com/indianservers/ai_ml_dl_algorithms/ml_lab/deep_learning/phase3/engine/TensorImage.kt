package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine

import kotlin.math.sqrt
import kotlin.random.Random

data class TensorImage(val channels: Int, val height: Int, val width: Int, val values: FloatArray) {
    init { require(values.size == channels * height * width) }
    operator fun get(channel: Int, row: Int, column: Int): Float = values[(channel * height + row) * width + column]
    operator fun set(channel: Int, row: Int, column: Int, value: Float) { values[(channel * height + row) * width + column] = value }
    fun copy() = TensorImage(channels, height, width, values.copyOf())
    fun stats(): Triple<Float, Float, Float> = Triple(values.minOrNull() ?: 0f, values.maxOrNull() ?: 0f, values.average().toFloat())

    companion object { fun zeros(channels: Int, height: Int, width: Int) = TensorImage(channels, height, width, FloatArray(channels * height * width)) }
}

data class ConvTrace(val input: TensorImage, val output: TensorImage)

class Conv2D(
    val inputChannels: Int,
    val outputChannels: Int,
    val kernelSize: Int,
    val stride: Int = 1,
    val padding: Int = 0,
    seed: Int = 19
) {
    val weights = FloatArray(outputChannels * inputChannels * kernelSize * kernelSize)
    val biases = FloatArray(outputChannels)
    val weightGradients = FloatArray(weights.size)
    val biasGradients = FloatArray(biases.size)
    private var lastInput: TensorImage? = null

    init {
        val random = Random(seed)
        val scale = sqrt(6f / (inputChannels * kernelSize * kernelSize + outputChannels))
        weights.indices.forEach { weights[it] = (random.nextFloat() * 2f - 1f) * scale }
    }

    fun outputSize(input: Int): Int = ((input + 2 * padding - kernelSize) / stride) + 1
    private fun wi(out: Int, input: Int, row: Int, column: Int) = (((out * inputChannels + input) * kernelSize + row) * kernelSize + column)

    fun forward(input: TensorImage): TensorImage {
        require(input.channels == inputChannels)
        lastInput = input.copy()
        val outH = outputSize(input.height)
        val outW = outputSize(input.width)
        val output = TensorImage.zeros(outputChannels, outH, outW)
        for (out in 0 until outputChannels) for (row in 0 until outH) for (column in 0 until outW) {
            var sum = biases[out]
            for (channel in 0 until inputChannels) for (kr in 0 until kernelSize) for (kc in 0 until kernelSize) {
                val ir = row * stride + kr - padding
                val ic = column * stride + kc - padding
                if (ir in 0 until input.height && ic in 0 until input.width) sum += input[channel, ir, ic] * weights[wi(out, channel, kr, kc)]
            }
            output[out, row, column] = sum
        }
        return output
    }

    fun backward(outputGradient: TensorImage): TensorImage {
        val input = requireNotNull(lastInput) { "forward must run before backward" }
        require(outputGradient.channels == outputChannels)
        weightGradients.fill(0f); biasGradients.fill(0f)
        val inputGradient = TensorImage.zeros(inputChannels, input.height, input.width)
        for (out in 0 until outputChannels) for (row in 0 until outputGradient.height) for (column in 0 until outputGradient.width) {
            val gradient = outputGradient[out, row, column]
            biasGradients[out] += gradient
            for (channel in 0 until inputChannels) for (kr in 0 until kernelSize) for (kc in 0 until kernelSize) {
                val ir = row * stride + kr - padding
                val ic = column * stride + kc - padding
                if (ir in 0 until input.height && ic in 0 until input.width) {
                    val index = wi(out, channel, kr, kc)
                    weightGradients[index] += input[channel, ir, ic] * gradient
                    inputGradient[channel, ir, ic] = inputGradient[channel, ir, ic] + weights[index] * gradient
                }
            }
        }
        return inputGradient
    }

    fun step(learningRate: Float) {
        weights.indices.forEach { weights[it] -= learningRate * weightGradients[it] }
        biases.indices.forEach { biases[it] -= learningRate * biasGradients[it] }
    }

    fun parameterCount() = weights.size + biases.size
}

enum class PoolingType { Max, Average }
data class PoolResult(val output: TensorImage, val winnerIndices: IntArray)

class Pool2D(val size: Int = 2, val stride: Int = 2, val type: PoolingType = PoolingType.Max) {
    private var lastInput: TensorImage? = null
    private var winners = IntArray(0)

    fun forward(input: TensorImage): PoolResult {
        lastInput = input.copy()
        val outH = (input.height - size) / stride + 1
        val outW = (input.width - size) / stride + 1
        val output = TensorImage.zeros(input.channels, outH, outW)
        winners = IntArray(output.values.size) { -1 }
        for (channel in 0 until input.channels) for (row in 0 until outH) for (column in 0 until outW) {
            var selected = if (type == PoolingType.Max) -Float.MAX_VALUE else 0f
            var winner = -1
            for (pr in 0 until size) for (pc in 0 until size) {
                val ir = row * stride + pr; val ic = column * stride + pc
                val value = input[channel, ir, ic]
                if (type == PoolingType.Max && value > selected) { selected = value; winner = (channel * input.height + ir) * input.width + ic }
                if (type == PoolingType.Average) selected += value / (size * size)
            }
            output[channel, row, column] = selected
            winners[(channel * outH + row) * outW + column] = winner
        }
        return PoolResult(output, winners.copyOf())
    }

    fun backward(outputGradient: TensorImage): TensorImage {
        val input = requireNotNull(lastInput)
        val gradient = TensorImage.zeros(input.channels, input.height, input.width)
        outputGradient.values.indices.forEach { outIndex ->
            if (type == PoolingType.Max) gradient.values[winners[outIndex]] += outputGradient.values[outIndex]
            else {
                val channelArea = outputGradient.height * outputGradient.width
                val channel = outIndex / channelArea
                val local = outIndex % channelArea
                val row = local / outputGradient.width
                val column = local % outputGradient.width
                for (pr in 0 until size) for (pc in 0 until size) {
                    val index = (channel * input.height + row * stride + pr) * input.width + column * stride + pc
                    gradient.values[index] += outputGradient.values[outIndex] / (size * size)
                }
            }
        }
        return gradient
    }
}

fun relu(image: TensorImage) = TensorImage(image.channels, image.height, image.width, FloatArray(image.values.size) { image.values[it].coerceAtLeast(0f) })
