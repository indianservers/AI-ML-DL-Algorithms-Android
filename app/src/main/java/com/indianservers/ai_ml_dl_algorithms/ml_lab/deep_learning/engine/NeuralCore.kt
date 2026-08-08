package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.random.Random

enum class Activation(val label: String) {
    Linear("Linear"), Sigmoid("Sigmoid"), Tanh("Tanh"), ReLU("ReLU"),
    LeakyReLU("Leaky ReLU"), ELU("ELU"), Softplus("Softplus"), Softmax("Softmax");

    fun apply(x: Float): Float = when (this) {
        Linear -> x
        Sigmoid -> if (x >= 0f) 1f / (1f + exp(-x)) else exp(x) / (1f + exp(x))
        Tanh -> kotlin.math.tanh(x)
        ReLU -> max(0f, x)
        LeakyReLU -> if (x >= 0f) x else 0.01f * x
        ELU -> if (x >= 0f) x else exp(x) - 1f
        Softplus -> if (x > 20f) x else ln(1f + exp(x))
        Softmax -> x
    }

    fun derivative(x: Float): Float = when (this) {
        Linear -> 1f
        Sigmoid -> apply(x) * (1f - apply(x))
        Tanh -> 1f - apply(x).pow(2)
        ReLU -> if (x > 0f) 1f else 0f
        LeakyReLU -> if (x > 0f) 1f else 0.01f
        ELU -> if (x >= 0f) 1f else exp(x)
        Softplus -> Sigmoid.apply(x)
        Softmax -> 1f
    }

    fun vector(values: FloatArray): FloatArray {
        if (this != Softmax) return FloatArray(values.size) { apply(values[it]) }
        val peak = values.maxOrNull() ?: 0f
        val exps = FloatArray(values.size) { exp(values[it] - peak) }
        val total = exps.sum().coerceAtLeast(1e-7f)
        return FloatArray(values.size) { exps[it] / total }
    }

    val formula: String get() = when (this) {
        Linear -> "f(x) = x"
        Sigmoid -> "f(x) = 1 / (1 + exp(-x))"
        Tanh -> "f(x) = tanh(x)"
        ReLU -> "f(x) = max(0, x)"
        LeakyReLU -> "f(x) = max(0.01x, x)"
        ELU -> "f(x) = x if x>0, exp(x)-1 otherwise"
        Softplus -> "f(x) = log(1 + exp(x))"
        Softmax -> "f(x_i) = exp(x_i) / sum(exp(x_j))"
    }
}

enum class LossFunction(val label: String) {
    MSE("Mean Squared Error"), MAE("Mean Absolute Error"), Huber("Huber"),
    BinaryCrossEntropy("Binary Cross Entropy"), CategoricalCrossEntropy("Categorical Cross Entropy");

    fun value(prediction: FloatArray, target: FloatArray): Float {
        val n = prediction.size.coerceAtLeast(1)
        return when (this) {
            MSE -> prediction.indices.sumOf { (prediction[it] - target[it]).pow(2).toDouble() }.toFloat() / n
            MAE -> prediction.indices.sumOf { abs(prediction[it] - target[it]).toDouble() }.toFloat() / n
            Huber -> prediction.indices.sumOf {
                val e = abs(prediction[it] - target[it])
                (if (e <= 1f) 0.5f * e * e else e - 0.5f).toDouble()
            }.toFloat() / n
            BinaryCrossEntropy -> prediction.indices.sumOf {
                val p = prediction[it].coerceIn(1e-7f, 1f - 1e-7f)
                (-(target[it] * ln(p) + (1f - target[it]) * ln(1f - p))).toDouble()
            }.toFloat() / n
            CategoricalCrossEntropy -> prediction.indices.sumOf {
                (-target[it] * ln(prediction[it].coerceAtLeast(1e-7f))).toDouble()
            }.toFloat()
        }
    }

    fun gradient(prediction: FloatArray, target: FloatArray): FloatArray {
        val n = prediction.size.coerceAtLeast(1).toFloat()
        return FloatArray(prediction.size) { i ->
            val error = prediction[i] - target[i]
            when (this) {
                MSE -> 2f * error / n
                MAE -> if (error >= 0f) 1f / n else -1f / n
                Huber -> if (abs(error) <= 1f) error / n else if (error > 0f) 1f / n else -1f / n
                BinaryCrossEntropy -> {
                    val p = prediction[i].coerceIn(1e-7f, 1f - 1e-7f)
                    ((p - target[i]) / (p * (1f - p))) / n
                }
                CategoricalCrossEntropy -> -target[i] / prediction[i].coerceAtLeast(1e-7f)
            }
        }
    }
}

enum class Initializer(val label: String) { Zero("Zero"), SmallRandom("Small random"), Uniform("Uniform"), Xavier("Xavier"), He("He") }
enum class OptimizerType(val label: String) { BatchGD("Batch GD"), SGD("SGD"), MiniBatch("Mini-batch"), Momentum("Momentum"), RMSProp("RMSProp"), Adam("Adam") }

class Parameter(val values: FloatArray) {
    val gradients = FloatArray(values.size)
    fun zeroGrad() = gradients.fill(0f)
}

object NeuralMath {
    fun matVec(matrix: FloatArray, rows: Int, cols: Int, vector: FloatArray): FloatArray =
        FloatArray(rows) { row -> (0 until cols).sumOf { (matrix[row * cols + it] * vector[it]).toDouble() }.toFloat() }

    fun dot(a: FloatArray, b: FloatArray): Float = a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
    fun norm(values: FloatArray): Float = sqrt(values.sumOf { (it * it).toDouble() }.toFloat())
    fun clip(values: FloatArray, limit: Float) { values.indices.forEach { values[it] = values[it].coerceIn(-limit, limit) } }

    fun initialized(size: Int, fanIn: Int, fanOut: Int, kind: Initializer, random: Random): FloatArray {
        val scale = when (kind) {
            Initializer.Zero -> 0f
            Initializer.SmallRandom -> 0.05f
            Initializer.Uniform -> 0.5f
            Initializer.Xavier -> sqrt(6f / (fanIn + fanOut))
            Initializer.He -> sqrt(6f / fanIn)
        }
        return FloatArray(size) { if (kind == Initializer.Zero) 0f else (random.nextFloat() * 2f - 1f) * scale }
    }
}

data class NeuralSample(val input: FloatArray, val target: FloatArray)
data class LayerTrace(val input: FloatArray, val weighted: FloatArray, val output: FloatArray)
data class NetworkTrace(val layers: List<LayerTrace>, val prediction: FloatArray, val target: FloatArray, val loss: Float)
data class EpochSnapshot(val epoch: Int, val loss: Float, val accuracy: Float, val gradientNorm: Float, val weightNorm: Float)
data class GradientHealth(val mean: Float, val min: Float, val max: Float, val norm: Float, val deadReluCount: Int, val saturatedCount: Int)
