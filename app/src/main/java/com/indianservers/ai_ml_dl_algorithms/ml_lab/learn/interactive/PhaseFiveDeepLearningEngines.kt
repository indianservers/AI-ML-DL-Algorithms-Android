package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class PhaseFiveConcept(val displayName: String) {
    ArtificialNeuron("Artificial Neuron / Perceptron"),
    Mlp("Multi-Layer Perceptron - MLP"),
    ForwardPropagation("Forward Propagation"),
    Backpropagation("Backpropagation"),
    ActivationFunctions("Activation Functions"),
    LossFunctions("Loss Functions"),
    Optimizers("Gradient Descent + Adam"),
    Regularization("Regularization - Dropout / L2")
}

enum class DlActivation(val label: String) { Relu("ReLU"), Sigmoid("Sigmoid"), Tanh("Tanh"), LeakyRelu("Leaky ReLU") }
enum class DlLossKind(val label: String) { Mse("Mean Squared Error"), Bce("Binary Cross Entropy"), Cce("Categorical Cross Entropy") }
enum class DlOptimizerKind { Sgd, Adam }

data class NeuronCalculation(
    val inputs: List<Double>,
    val weights: List<Double>,
    val bias: Double,
    val z: Double,
    val activation: DlActivation,
    val output: Double
)

data class DenseLayerState(
    val weights: List<List<Double>>,
    val biases: List<Double>,
    val z: List<Double>,
    val activations: List<Double>,
    val gradients: List<List<Double>> = emptyList(),
    val dropoutMask: List<Boolean> = emptyList()
)

data class NetworkState(
    val inputs: List<Double>,
    val layers: List<DenseLayerState>,
    val prediction: Double,
    val target: Double,
    val loss: Double
)

data class BackpropState(
    val lossGradient: Double,
    val outputGradient: Double,
    val hiddenGradients: List<Double>,
    val weightGradients: List<List<List<Double>>>,
    val explanation: String
)

data class OptimizerStepState(
    val before: Double,
    val gradient: Double,
    val firstMoment: Double,
    val secondMoment: Double,
    val update: Double,
    val after: Double
)

data class TrainingTrace(val losses: List<Double>, val accuracies: List<Double>, val finalWeight: Double)
data class DecisionSample(val x: Double, val y: Double, val label: Int, val prediction: Double)

object PhaseFiveTopicMatcher {
    fun kindFor(title: String, domain: String): PhaseFiveConcept? = if (domain != "Deep Learning") null else when (title) {
        "Artificial Neuron", "Perceptron" -> PhaseFiveConcept.ArtificialNeuron
        "Multi-Layer Perceptron", "Feedforward Neural Network" -> PhaseFiveConcept.Mlp
        "Backpropagation" -> PhaseFiveConcept.Backpropagation
        "Gradient Descent", "Stochastic Gradient Descent", "Mini-Batch Gradient Descent" -> PhaseFiveConcept.Optimizers
        "Activation Functions" -> PhaseFiveConcept.ActivationFunctions
        "Loss Functions" -> PhaseFiveConcept.LossFunctions
        else -> null
    }
}

object PhaseFiveEngines {
    fun neuron(inputs: List<Double>, weights: List<Double>, bias: Double, activation: DlActivation): NeuronCalculation {
        val z = inputs.indices.sumOf { inputs[it] * weights[it] } + bias
        return NeuronCalculation(inputs, weights, bias, z, activation, activate(z, activation))
    }

    fun activate(z: Double, activation: DlActivation): Double = when (activation) {
        DlActivation.Relu -> max(0.0, z)
        DlActivation.Sigmoid -> 1.0 / (1.0 + exp(-z))
        DlActivation.Tanh -> kotlin.math.tanh(z)
        DlActivation.LeakyRelu -> if (z >= 0.0) z else 0.01 * z
    }

    fun derivative(z: Double, activation: DlActivation): Double = when (activation) {
        DlActivation.Relu -> if (z > 0.0) 1.0 else 0.0
        DlActivation.Sigmoid -> activate(z, activation).let { it * (1.0 - it) }
        DlActivation.Tanh -> 1.0 - kotlin.math.tanh(z).pow(2)
        DlActivation.LeakyRelu -> if (z >= 0.0) 1.0 else 0.01
    }

    fun softmax(logits: List<Double>): List<Double> {
        val m = logits.max()
        val exps = logits.map { exp(it - m) }
        val total = exps.sum().coerceAtLeast(1e-12)
        return exps.map { it / total }
    }

    fun loss(prediction: Double, target: Double, kind: DlLossKind): Double = when (kind) {
        DlLossKind.Mse -> (prediction - target).pow(2)
        DlLossKind.Bce -> -(target * ln(prediction.coerceIn(1e-9, 1.0)) + (1.0 - target) * ln((1.0 - prediction).coerceIn(1e-9, 1.0)))
        DlLossKind.Cce -> -ln(prediction.coerceIn(1e-9, 1.0))
    }

    fun initialize(layerSizes: List<Int>, seed: Int = 5): Pair<List<List<List<Double>>>, List<List<Double>>> {
        val weights = layerSizes.zipWithNext().mapIndexed { layer, (input, output) ->
            List(output) { neuron -> List(input) { i -> deterministic(seed, layer, neuron, i) * sqrt(2.0 / input) } }
        }
        val biases = layerSizes.drop(1).map { size -> List(size) { 0.0 } }
        return weights to biases
    }

    fun forward(
        inputs: List<Double>,
        weights: List<List<List<Double>>>,
        biases: List<List<Double>>,
        activation: DlActivation = DlActivation.Relu,
        target: Double = 1.0,
        dropoutMask: List<List<Boolean>> = emptyList()
    ): NetworkState {
        var current = inputs
        val layers = weights.mapIndexed { layerIndex, layerWeights ->
            val z = layerWeights.mapIndexed { neuron, ws -> ws.indices.sumOf { ws[it] * current[it] } + biases[layerIndex][neuron] }
            val isOutput = layerIndex == weights.lastIndex
            var a = z.map { if (isOutput) activate(it, DlActivation.Sigmoid) else activate(it, activation) }
            val mask = dropoutMask.getOrNull(layerIndex).orEmpty()
            if (!isOutput && mask.isNotEmpty()) a = a.mapIndexed { i, v -> if (mask.getOrElse(i) { true }) v else 0.0 }
            current = a
            DenseLayerState(layerWeights, biases[layerIndex], z, a, dropoutMask = mask)
        }
        val prediction = current.first()
        return NetworkState(inputs, layers, prediction, target, loss(prediction, target, DlLossKind.Bce))
    }

    fun backprop(state: NetworkState, activation: DlActivation = DlActivation.Relu): BackpropState {
        val output = state.layers.last()
        val prediction = state.prediction
        val outputDelta = prediction - state.target
        val previous = if (state.layers.size == 1) state.inputs else state.layers[state.layers.lastIndex - 1].activations
        val outputGradients = output.weights.map { previous.map { outputDelta * it } }
        val hiddenLayer = state.layers.first()
        val hiddenGradients = hiddenLayer.activations.indices.map { h ->
            val downstream = output.weights.first().getOrElse(h) { 0.0 } * outputDelta
            downstream * derivative(hiddenLayer.z[h], activation)
        }
        val hiddenWeightGradients = hiddenLayer.weights.mapIndexed { h, ws -> ws.indices.map { i -> hiddenGradients[h] * state.inputs[i] } }
        val explanation = if (outputDelta < 0) "Prediction was too low, so gradients push compatible weights upward." else "Prediction was too high, so gradients push compatible weights downward."
        return BackpropState(outputDelta, outputDelta, hiddenGradients, listOf(hiddenWeightGradients, outputGradients), explanation)
    }

    fun sgd(weight: Double, gradient: Double, learningRate: Double, l2: Double = 0.0): OptimizerStepState {
        val totalGradient = gradient + l2 * weight
        val update = learningRate * totalGradient
        return OptimizerStepState(weight, totalGradient, 0.0, 0.0, update, weight - update)
    }

    fun adam(weight: Double, gradient: Double, learningRate: Double, t: Int, mPrev: Double = 0.0, vPrev: Double = 0.0): OptimizerStepState {
        val beta1 = .9
        val beta2 = .999
        val m = beta1 * mPrev + (1.0 - beta1) * gradient
        val v = beta2 * vPrev + (1.0 - beta2) * gradient * gradient
        val mh = m / (1.0 - beta1.pow(t))
        val vh = v / (1.0 - beta2.pow(t))
        val update = learningRate * mh / (sqrt(vh) + 1e-8)
        return OptimizerStepState(weight, gradient, m, v, update, weight - update)
    }

    fun dropoutMask(size: Int, rate: Double, seed: Int): List<Boolean> =
        List(size) { i -> abs(sin(seed * 31.0 + i * 17.0)) >= rate }

    fun l2Penalty(weights: List<List<List<Double>>>, lambda: Double): Double =
        lambda * weights.flatten().flatten().sumOf { it * it }

    fun xorData(): List<DecisionSample> = listOf(
        DecisionSample(-1.0, -1.0, 0, 0.0),
        DecisionSample(-1.0, 1.0, 1, 0.0),
        DecisionSample(1.0, -1.0, 1, 0.0),
        DecisionSample(1.0, 1.0, 0, 0.0)
    )

    fun xorPrediction(x: Double, y: Double, hidden: Boolean): Double {
        return if (!hidden) activate(1.2 * x + 1.2 * y, DlActivation.Sigmoid)
        else {
            val h1 = activate(1.8 * x - 1.8 * y, DlActivation.Relu)
            val h2 = activate(-1.8 * x + 1.8 * y, DlActivation.Relu)
            activate(1.4 * h1 + 1.4 * h2 - 1.0, DlActivation.Sigmoid)
        }
    }

    fun trainingTrace(optimizer: DlOptimizerKind, epochs: Int, learningRate: Double): TrainingTrace {
        var w = -1.0
        var m = 0.0
        var v = 0.0
        val losses = mutableListOf<Double>()
        repeat(epochs.coerceAtLeast(1)) { epoch ->
            val grad = 2.0 * (w - 1.2)
            val step = if (optimizer == DlOptimizerKind.Sgd) sgd(w, grad, learningRate) else adam(w, grad, learningRate, epoch + 1, m, v)
            m = step.firstMoment
            v = step.secondMoment
            w = step.after
            losses += (w - 1.2).pow(2)
        }
        return TrainingTrace(losses, losses.map { (1.0 - it).coerceIn(0.0, 1.0) }, w)
    }

    fun finiteDifferenceGradient(epsilon: Double = 1e-5): Pair<Double, Double> {
        val (weights, biases) = initialize(listOf(2, 2, 1), 4)
        val input = listOf(.7, -.2)
        val target = 1.0
        val state = forward(input, weights, biases, target = target)
        val analytical = backprop(state).weightGradients.first().first().first()
        fun modified(delta: Double): Double {
            val changed = weights.mapIndexed { l, layer ->
                if (l != 0) layer else layer.mapIndexed { n, ws -> if (n != 0) ws else ws.mapIndexed { i, w -> if (i == 0) w + delta else w } }
            }
            return forward(input, changed, biases, target = target).loss
        }
        val numerical = (modified(epsilon) - modified(-epsilon)) / (2.0 * epsilon)
        return analytical to numerical
    }

    private fun deterministic(seed: Int, layer: Int, neuron: Int, input: Int): Double {
        val raw = sin(seed * 97.0 + layer * 41.0 + neuron * 19.0 + input * 11.0) * 10000.0
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}
