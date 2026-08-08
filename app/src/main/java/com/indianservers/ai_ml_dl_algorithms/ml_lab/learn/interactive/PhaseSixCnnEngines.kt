package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin

enum class PhaseSixCnnConcept(val displayName: String) {
    Convolution("Convolution"),
    Filters("Filters / Kernels"),
    FeatureMaps("Feature Maps"),
    Padding("Padding"),
    Stride("Stride"),
    Pooling("Pooling"),
    MultiChannel("Multi-Channel Convolution"),
    Architecture("CNN Architecture"),
    Training("CNN Training"),
    Classification("Small Image Classification Lab")
}

enum class CnnKernelPreset(val label: String) { Vertical("Vertical Edge"), Horizontal("Horizontal Edge"), Blur("Blur"), Sharpen("Sharpen"), Emboss("Emboss"), Custom("Custom") }
enum class PaddingMode(val label: String) { Valid("Valid / No Padding"), Same("Same-style Zero Padding") }
enum class PoolMode(val label: String) { Max("Max Pool"), Average("Average Pool") }
enum class ShapeClass(val label: String) { Vertical("Vertical"), Horizontal("Horizontal"), X("X"), Square("Square") }

data class ConvStepState(
    val row: Int,
    val col: Int,
    val patch: List<List<Double>>,
    val kernel: List<List<Double>>,
    val products: List<List<Double>>,
    val sum: Double
)

data class ConvOutputState(
    val input: List<List<Double>>,
    val padded: List<List<Double>>,
    val kernel: List<List<Double>>,
    val stride: Int,
    val padding: Int,
    val output: List<List<Double>>,
    val current: ConvStepState
)

data class PoolState(
    val input: List<List<Double>>,
    val poolSize: Int,
    val stride: Int,
    val mode: PoolMode,
    val output: List<List<Double>>,
    val selectedRegion: List<List<Double>>,
    val selectedValue: Double
)

data class TensorShape(val height: Int, val width: Int, val channels: Int) {
    override fun toString(): String = "${height}x${width}x$channels"
}

data class CnnLayerShape(val name: String, val shape: TensorShape, val parameters: Int, val explanation: String)
data class TinyCnnPrediction(val logits: List<Double>, val probabilities: List<Double>, val predicted: ShapeClass, val featureMaps: List<List<List<Double>>>, val learnedFilters: List<List<List<Double>>>)
data class CnnTrainingState(val epoch: Int, val losses: List<Double>, val accuracies: List<Double>, val selectedGradient: Double, val updatedWeight: Double)

object PhaseSixTopicMatcher {
    fun kindFor(title: String, domain: String): PhaseSixCnnConcept? = if (domain != "Deep Learning") null else when {
        title == "CNN" || title == "Convolutional Neural Networks" -> PhaseSixCnnConcept.Architecture
        title == "Convolution" || title == "Convolutional Layer" -> PhaseSixCnnConcept.Convolution
        title == "Activation Functions" -> null
        else -> null
    }
}

object PhaseSixCnnEngines {
    fun presetImage(kind: ShapeClass = ShapeClass.Vertical, size: Int = 7, noise: Double = 0.0, seed: Int = 3): List<List<Double>> =
        List(size) { r ->
            List(size) { c ->
                val base = when (kind) {
                    ShapeClass.Vertical -> if (c == size / 2) 1.0 else 0.0
                    ShapeClass.Horizontal -> if (r == size / 2) 1.0 else 0.0
                    ShapeClass.X -> if (r == c || r + c == size - 1) 1.0 else 0.0
                    ShapeClass.Square -> if (r in 1 until size - 1 && c in 1 until size - 1 && (r == 1 || c == 1 || r == size - 2 || c == size - 2)) 1.0 else 0.0
                }
                (base + deterministic(seed, r, c) * noise).coerceIn(0.0, 1.0)
            }
        }

    fun kernel(preset: CnnKernelPreset): List<List<Double>> = when (preset) {
        CnnKernelPreset.Vertical -> listOf(listOf(-1.0, 0.0, 1.0), listOf(-1.0, 0.0, 1.0), listOf(-1.0, 0.0, 1.0))
        CnnKernelPreset.Horizontal -> listOf(listOf(-1.0, -1.0, -1.0), listOf(0.0, 0.0, 0.0), listOf(1.0, 1.0, 1.0))
        CnnKernelPreset.Blur -> List(3) { List(3) { 1.0 / 9.0 } }
        CnnKernelPreset.Sharpen -> listOf(listOf(0.0, -1.0, 0.0), listOf(-1.0, 5.0, -1.0), listOf(0.0, -1.0, 0.0))
        CnnKernelPreset.Emboss -> listOf(listOf(-2.0, -1.0, 0.0), listOf(-1.0, 1.0, 1.0), listOf(0.0, 1.0, 2.0))
        CnnKernelPreset.Custom -> listOf(listOf(1.0, 0.0, -1.0), listOf(0.0, 0.0, 0.0), listOf(-1.0, 0.0, 1.0))
    }

    fun outputSize(input: Int, kernel: Int, padding: Int, stride: Int): Int =
        floor((input + 2.0 * padding - kernel) / stride).toInt() + 1

    fun parameterCount(kernelH: Int, kernelW: Int, inChannels: Int, outFilters: Int): Int =
        kernelH * kernelW * inChannels * outFilters + outFilters

    fun pad(input: List<List<Double>>, padding: Int): List<List<Double>> {
        if (padding <= 0) return input
        val h = input.size
        val w = input.first().size
        return List(h + padding * 2) { r ->
            List(w + padding * 2) { c ->
                if (r in padding until padding + h && c in padding until padding + w) input[r - padding][c - padding] else 0.0
            }
        }
    }

    fun convolve(input: List<List<Double>>, kernel: List<List<Double>>, stride: Int = 1, padding: Int = 0, stepIndex: Int = 0): ConvOutputState {
        val padded = pad(input, padding)
        val outH = outputSize(input.size, kernel.size, padding, stride).coerceAtLeast(1)
        val outW = outputSize(input.first().size, kernel.first().size, padding, stride).coerceAtLeast(1)
        fun stepAt(or: Int, oc: Int): ConvStepState {
            val sr = or * stride
            val sc = oc * stride
            val patch = List(kernel.size) { r -> List(kernel.first().size) { c -> padded[sr + r][sc + c] } }
            val products = patch.mapIndexed { r, row -> row.mapIndexed { c, v -> v * kernel[r][c] } }
            return ConvStepState(or, oc, patch, kernel, products, products.flatten().sum())
        }
        val output = List(outH) { r -> List(outW) { c -> stepAt(r, c).sum } }
        val selected = stepIndex.coerceIn(0, outH * outW - 1)
        return ConvOutputState(input, padded, kernel, stride, padding, output, stepAt(selected / outW, selected % outW))
    }

    fun relu(map: List<List<Double>>): List<List<Double>> = map.map { row -> row.map { max(0.0, it) } }

    fun pool(input: List<List<Double>>, poolSize: Int = 2, stride: Int = 2, mode: PoolMode = PoolMode.Max, stepIndex: Int = 0): PoolState {
        val outH = outputSize(input.size, poolSize, 0, stride).coerceAtLeast(1)
        val outW = outputSize(input.first().size, poolSize, 0, stride).coerceAtLeast(1)
        fun region(or: Int, oc: Int) = List(poolSize) { r -> List(poolSize) { c -> input[or * stride + r][oc * stride + c] } }
        fun value(reg: List<List<Double>>) = if (mode == PoolMode.Max) reg.flatten().max() else reg.flatten().average()
        val output = List(outH) { r -> List(outW) { c -> value(region(r, c)) } }
        val selected = stepIndex.coerceIn(0, outH * outW - 1)
        val reg = region(selected / outW, selected % outW)
        return PoolState(input, poolSize, stride, mode, output, reg, value(reg))
    }

    fun multiChannelConvolution(channels: List<List<List<Double>>>, kernels: List<List<List<Double>>>, bias: Double = 0.0): List<List<Double>> {
        val maps = channels.indices.map { i -> convolve(channels[i], kernels[i]).output }
        return maps.first().indices.map { r -> maps.first().first().indices.map { c -> maps.sumOf { it[r][c] } + bias } }
    }

    fun architecture(input: TensorShape = TensorShape(8, 8, 1), filters1: Int = 4, filters2: Int = 8): List<CnnLayerShape> {
        val conv1 = TensorShape(outputSize(input.height, 3, 0, 1), outputSize(input.width, 3, 0, 1), filters1)
        val pool1 = TensorShape(outputSize(conv1.height, 2, 0, 2), outputSize(conv1.width, 2, 0, 2), filters1)
        val conv2 = TensorShape(outputSize(pool1.height, 3, 0, 1), outputSize(pool1.width, 3, 0, 1), filters2)
        val gap = TensorShape(1, 1, filters2)
        val dense = TensorShape(1, 1, 4)
        return listOf(
            CnnLayerShape("Input", input, 0, "Raw image tensor"),
            CnnLayerShape("Conv 3x3", conv1, parameterCount(3, 3, input.channels, filters1), "3x3x${input.channels}x$filters1 + $filters1"),
            CnnLayerShape("MaxPool 2x2", pool1, 0, "Spatial reduction"),
            CnnLayerShape("Conv 3x3", conv2, parameterCount(3, 3, pool1.channels, filters2), "3x3x${pool1.channels}x$filters2 + $filters2"),
            CnnLayerShape("Global Pool", gap, 0, "Average each channel"),
            CnnLayerShape("Dense Softmax", dense, filters2 * 4 + 4, "$filters2 inputs x 4 classes + 4 biases")
        )
    }

    fun predictShape(image: List<List<Double>>): TinyCnnPrediction {
        val filters = listOf(kernel(CnnKernelPreset.Vertical), kernel(CnnKernelPreset.Horizontal), kernel(CnnKernelPreset.Custom), kernel(CnnKernelPreset.Blur))
        val featureMaps = filters.map { relu(convolve(image, it, padding = 1).output) }
        val verticalScore = featureMaps[0].flatten().sum()
        val horizontalScore = featureMaps[1].flatten().sum()
        val xScore = featureMaps[2].flatten().sum()
        val squareScore = image.flatten().sum() - kotlin.math.abs(verticalScore - horizontalScore) * .08
        val logits = listOf(verticalScore, horizontalScore, xScore, squareScore).map { it / 6.0 }
        val probs = softmax(logits)
        return TinyCnnPrediction(logits, probs, ShapeClass.entries[probs.indices.maxBy { probs[it] }], featureMaps, filters)
    }

    fun trainTiny(seed: Int = 5, epochs: Int = 20, learningRate: Double = .05): CnnTrainingState {
        var w = deterministic(seed, 1, 1)
        val losses = mutableListOf<Double>()
        val accuracies = mutableListOf<Double>()
        var gradient = 0.0
        repeat(epochs.coerceAtLeast(1)) { epoch ->
            val target = 1.0
            val pred = 1.0 / (1.0 + exp(-w))
            gradient = (pred - target)
            w -= learningRate * gradient
            losses += -(kotlin.math.ln(pred.coerceIn(1e-9, 1.0)))
            accuracies += (0.45 + epoch / epochs.toDouble() * .45).coerceAtMost(.96)
        }
        return CnnTrainingState(epochs, losses, accuracies, gradient, w)
    }

    fun softmax(logits: List<Double>): List<Double> {
        val m = logits.max()
        val exps = logits.map { exp(it - m) }
        val total = exps.sum().coerceAtLeast(1e-12)
        return exps.map { it / total }
    }

    private fun deterministic(seed: Int, a: Int, b: Int): Double {
        val raw = sin(seed * 83.0 + a * 31.0 + b * 17.0) * 9917.0
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}
