package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine

import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.random.Random

class DenseLayer(
    val inputSize: Int,
    val outputSize: Int,
    var activation: Activation,
    initializer: Initializer,
    random: Random
) {
    val weights = Parameter(NeuralMath.initialized(inputSize * outputSize, inputSize, outputSize, initializer, random))
    val biases = Parameter(FloatArray(outputSize))
    private var lastInput = FloatArray(inputSize)
    private var lastWeighted = FloatArray(outputSize)
    var lastOutput = FloatArray(outputSize)
        private set

    fun forward(input: FloatArray): FloatArray {
        require(input.size == inputSize)
        lastInput = input.copyOf()
        lastWeighted = NeuralMath.matVec(weights.values, outputSize, inputSize, input)
        lastWeighted.indices.forEach { lastWeighted[it] += biases.values[it] }
        lastOutput = activation.vector(lastWeighted)
        return lastOutput.copyOf()
    }

    fun backward(upstream: FloatArray): FloatArray {
        val delta = if (activation == Activation.Softmax) {
            FloatArray(outputSize) { i ->
                lastOutput[i] * (upstream[i] - NeuralMath.dot(upstream, lastOutput))
            }
        } else FloatArray(outputSize) { upstream[it] * activation.derivative(lastWeighted[it]) }

        val inputGradient = FloatArray(inputSize)
        for (outIndex in 0 until outputSize) {
            biases.gradients[outIndex] += delta[outIndex]
            for (inIndex in 0 until inputSize) {
                val index = outIndex * inputSize + inIndex
                weights.gradients[index] += delta[outIndex] * lastInput[inIndex]
                inputGradient[inIndex] += weights.values[index] * delta[outIndex]
            }
        }
        return inputGradient
    }

    fun trace() = LayerTrace(lastInput.copyOf(), lastWeighted.copyOf(), lastOutput.copyOf())
    fun parameters() = listOf(weights, biases)
}

class NeuralNetwork(
    val sizes: List<Int>,
    hiddenActivation: Activation = Activation.Tanh,
    val outputActivation: Activation = Activation.Sigmoid,
    val lossFunction: LossFunction = LossFunction.BinaryCrossEntropy,
    initializer: Initializer = Initializer.Xavier,
    seed: Int = 42
) {
    val layers: List<DenseLayer>
    private val optimizers = mutableMapOf<OptimizerType, NetworkOptimizer>()

    init {
        require(sizes.size >= 2 && sizes.all { it in 1..32 })
        val random = Random(seed)
        layers = sizes.zipWithNext().mapIndexed { index, (input, output) ->
            DenseLayer(input, output, if (index == sizes.size - 2) outputActivation else hiddenActivation, initializer, random)
        }
    }

    fun forward(input: FloatArray): FloatArray = layers.fold(input.copyOf()) { value, layer -> layer.forward(value) }
    fun predict(input: FloatArray) = forward(input)
    fun parameters() = layers.flatMap { it.parameters() }
    fun zeroGrad() = parameters().forEach(Parameter::zeroGrad)

    fun backward(target: FloatArray): FloatArray {
        val prediction = layers.last().lastOutput
        var gradient = lossFunction.gradient(prediction, target)
        layers.asReversed().forEach { gradient = it.backward(gradient) }
        return gradient
    }

    fun trainBatch(
        samples: List<NeuralSample>,
        learningRate: Float,
        optimizerType: OptimizerType,
        clipNorm: Float? = null,
        l1: Float = 0f,
        l2: Float = 0f
    ): Float {
        if (samples.isEmpty()) return 0f
        zeroGrad()
        var loss = 0f
        samples.forEach { sample ->
            val prediction = forward(sample.input)
            loss += lossFunction.value(prediction, sample.target)
            backward(sample.target)
        }
        parameters().forEach { parameter ->
            parameter.gradients.indices.forEach { index ->
                parameter.gradients[index] /= samples.size
                parameter.gradients[index] += l2 * parameter.values[index]
                if (l1 > 0f) parameter.gradients[index] += if (parameter.values[index] >= 0f) l1 else -l1
            }
        }
        clipNorm?.let { clipGradients(it) }
        optimizers.getOrPut(optimizerType) { NetworkOptimizer(optimizerType) }.step(parameters(), learningRate)
        return loss / samples.size
    }

    fun train(
        samples: List<NeuralSample>,
        epochs: Int,
        learningRate: Float,
        optimizerType: OptimizerType,
        batchSize: Int = samples.size,
        clipNorm: Float? = 5f,
        snapshotEvery: Int = 1
    ): List<EpochSnapshot> {
        val snapshots = mutableListOf<EpochSnapshot>()
        repeat(epochs + 1) { epoch ->
            if (epoch > 0) samples.shuffled(Random(epoch + 91)).chunked(batchSize.coerceAtLeast(1)).forEach {
                trainBatch(it, learningRate, optimizerType, clipNorm)
            }
            if (epoch % snapshotEvery == 0 || epoch == epochs) {
                val metrics = evaluate(samples)
                snapshots += EpochSnapshot(epoch, metrics.first, metrics.second, gradientNorm(), weightNorm())
            }
        }
        return snapshots
    }

    fun trace(sample: NeuralSample): NetworkTrace {
        val prediction = forward(sample.input)
        return NetworkTrace(layers.map { it.trace() }, prediction, sample.target, lossFunction.value(prediction, sample.target))
    }

    fun evaluate(samples: List<NeuralSample>): Pair<Float, Float> {
        var loss = 0f
        var correct = 0
        samples.forEach {
            val prediction = predict(it.input)
            loss += lossFunction.value(prediction, it.target)
            val predictedClass = if (prediction.size == 1) if (prediction[0] >= 0.5f) 1 else 0 else prediction.indices.maxBy { i -> prediction[i] }
            val targetClass = if (it.target.size == 1) if (it.target[0] >= 0.5f) 1 else 0 else it.target.indices.maxBy { i -> it.target[i] }
            if (predictedClass == targetClass) correct++
        }
        return loss / samples.size.coerceAtLeast(1) to correct.toFloat() / samples.size.coerceAtLeast(1)
    }

    fun gradientHealth(): List<GradientHealth> = layers.map { layer ->
        val gradients = layer.weights.gradients
        val outputs = layer.lastOutput
        GradientHealth(
            gradients.average().toFloat(), gradients.minOrNull() ?: 0f, gradients.maxOrNull() ?: 0f,
            NeuralMath.norm(gradients), outputs.count { it == 0f }, outputs.count { abs(it) > 0.98f }
        )
    }

    fun parameterCount() = parameters().sumOf { it.values.size }
    fun flopEstimate() = layers.sumOf { it.inputSize * it.outputSize * 2 + it.outputSize }
    fun weightNorm() = sqrt(parameters().sumOf { p -> p.values.sumOf { (it * it).toDouble() } }.toFloat())
    fun gradientNorm() = sqrt(parameters().sumOf { p -> p.gradients.sumOf { (it * it).toDouble() } }.toFloat())

    private fun clipGradients(limit: Float) {
        val norm = gradientNorm()
        if (norm > limit && norm > 0f) {
            val scale = limit / norm
            parameters().forEach { p -> p.gradients.indices.forEach { p.gradients[it] *= scale } }
        }
    }
}

private class NetworkOptimizer(private val type: OptimizerType) {
    private val first = mutableMapOf<Parameter, FloatArray>()
    private val second = mutableMapOf<Parameter, FloatArray>()
    private var step = 0

    fun step(parameters: List<Parameter>, learningRate: Float) {
        step++
        parameters.forEach { parameter ->
            val m = first.getOrPut(parameter) { FloatArray(parameter.values.size) }
            val v = second.getOrPut(parameter) { FloatArray(parameter.values.size) }
            parameter.values.indices.forEach { i ->
                val g = parameter.gradients[i]
                val update = when (type) {
                    OptimizerType.BatchGD, OptimizerType.SGD, OptimizerType.MiniBatch -> g
                    OptimizerType.Momentum -> { m[i] = 0.9f * m[i] + g; m[i] }
                    OptimizerType.RMSProp -> { v[i] = 0.9f * v[i] + 0.1f * g * g; g / (sqrt(v[i]) + 1e-7f) }
                    OptimizerType.Adam -> {
                        m[i] = 0.9f * m[i] + 0.1f * g
                        v[i] = 0.999f * v[i] + 0.001f * g * g
                        val mHat = m[i] / (1f - 0.9f.powInt(step))
                        val vHat = v[i] / (1f - 0.999f.powInt(step))
                        mHat / (sqrt(vHat) + 1e-7f)
                    }
                }
                parameter.values[i] -= learningRate * update
            }
        }
    }

    private fun Float.powInt(power: Int): Float {
        var result = 1f
        repeat(power) { result *= this }
        return result
    }
}
