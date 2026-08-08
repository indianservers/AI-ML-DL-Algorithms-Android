package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

data class AttentionLink(val query: Int, val key: Int, val weight: Float)

fun attentionLinks(weights: Matrix, threshold: Float): List<AttentionLink> = buildList {
    for (query in 0 until weights.rows) for (key in 0 until weights.columns) {
        val weight = weights[query, key]
        if (weight >= threshold) add(AttentionLink(query, key, weight))
    }
}

class LearnedPositionEmbedding(
    val maximumLength: Int,
    val dimensions: Int,
    seed: Int = 211,
) {
    private val values = MatrixOps.random(maximumLength, dimensions, seed)

    fun encoding(length: Int): Matrix {
        require(length in 1..maximumLength)
        return Matrix(length, dimensions, values.values.copyOfRange(0, length * dimensions))
    }

    fun parameterCount() = maximumLength * dimensions
}

data class CrossAttentionResult(
    val decoderQuery: Matrix,
    val encoderKey: Matrix,
    val encoderValue: Matrix,
    val attention: AttentionResult,
)

class CrossAttention(private val dimensions: Int, seed: Int = 221) {
    private val queryProjection = MatrixOps.random(dimensions, dimensions, seed)
    private val keyProjection = MatrixOps.random(dimensions, dimensions, seed + 1)
    private val valueProjection = MatrixOps.random(dimensions, dimensions, seed + 2)

    fun forward(decoder: Matrix, encoder: Matrix, encoderPaddingMask: BooleanArray? = null): CrossAttentionResult {
        require(decoder.columns == dimensions && encoder.columns == dimensions)
        require(encoderPaddingMask == null || encoderPaddingMask.size == encoder.rows)
        val query = MatrixOps.matmul(decoder, queryProjection)
        val key = MatrixOps.matmul(encoder, keyProjection)
        val value = MatrixOps.matmul(encoder, valueProjection)
        return CrossAttentionResult(query, key, value, ScaledDotProductAttention(dimensions).attend(query, key, value, paddingMask = encoderPaddingMask))
    }
}

data class GenerationStep(
    val step: Int,
    val inputTokens: List<String>,
    val logits: FloatArray,
    val probabilities: FloatArray,
    val selectedToken: String,
)

class TinyAutoregressiveDecoder(seed: Int = 231) {
    val vocabulary = listOf("<BOS>", "A", "B", "C", "<EOS>")
    private val transitions = MatrixOps.random(vocabulary.size, vocabulary.size, seed).also { matrix ->
        matrix[0, 1] += 2.5f
        matrix[1, 2] += 2.5f
        matrix[2, 3] += 2.5f
        matrix[3, 4] += 2.5f
    }

    fun generate(maxTokens: Int = 4, temperature: Float = 1f, topK: Int = vocabulary.size): List<GenerationStep> {
        require(maxTokens in 1..12)
        val tokens = mutableListOf("<BOS>")
        return buildList {
            repeat(maxTokens) { step ->
                val last = vocabulary.indexOf(tokens.last()).coerceAtLeast(0)
                val logits = transitions.row(last)
                val distribution = probabilities(logits, temperature, topK)
                val selected = distribution.indices.maxBy { distribution[it] }
                add(GenerationStep(step, tokens.toList(), logits, distribution, vocabulary[selected]))
                tokens += vocabulary[selected]
                if (selected == vocabulary.lastIndex) return@buildList
            }
        }
    }
}

data class TransformerTrainingMetric(
    val epoch: Int,
    val loss: Float,
    val accuracy: Float,
    val gradientNorm: Float,
    val attentionEntropy: Float,
)

/** Trains a small classification head on real Transformer representations. */
class TinyTransformerTask(seed: Int = 241) {
    private val encoder = TransformerEncoderBlock(4, 2, 8, seed)
    private val random = Random(seed)
    private var classifier = FloatArray(16) { (random.nextFloat() - .5f) * .2f }
    private var bias = 0f

    fun train(epochs: Int = 80, learningRate: Float = .12f): List<TransformerTrainingMetric> {
        require(epochs in 1..400)
        val samples = sequenceSamples()
        return buildList {
            repeat(epochs + 1) { epoch ->
                var loss = 0f
                var correct = 0
                var entropy = 0f
                val classifierGradient = FloatArray(classifier.size)
                var biasGradient = 0f
                samples.forEach { (input, label) ->
                    val trace = encoder.forward(input)
                    // Preserve sequence order for this order-sensitive ABAB task.
                    val representation = trace.output.values.copyOf()
                    val logit = representation.indices.sumOf { (representation[it] * classifier[it]).toDouble() }.toFloat() + bias
                    val prediction = sigmoid(logit)
                    val safe = prediction.coerceIn(1e-6f, 1f - 1e-6f)
                    loss += -(label * ln(safe) + (1f - label) * ln(1f - safe))
                    if ((prediction >= .5f) == (label == 1f)) correct++
                    val error = prediction - label
                    representation.indices.forEach { index ->
                        classifierGradient[index] += error * representation[index] / samples.size
                    }
                    biasGradient += error / samples.size
                    entropy += trace.attention.heads.sumOf { it.entropy.toDouble() }.toFloat() / trace.attention.heads.size
                }
                val gradientNorm = sqrt(classifierGradient.sumOf { (it * it).toDouble() }.toFloat() + biasGradient * biasGradient)
                classifier.indices.forEach { classifier[it] -= learningRate * classifierGradient[it] }
                bias -= learningRate * biasGradient
                if (epoch % 5 == 0 || epoch == epochs) add(TransformerTrainingMetric(epoch, loss / samples.size, correct.toFloat() / samples.size, gradientNorm, entropy / samples.size))
            }
        }
    }

    private fun sequenceSamples(): List<Pair<Matrix, Float>> = listOf(
        tokens(0, 1, 0, 1) to 1f, tokens(1, 0, 1, 0) to 1f,
        tokens(0, 0, 1, 1) to 0f, tokens(1, 1, 0, 0) to 0f,
    )

    private fun tokens(vararg values: Int) = Matrix(values.size, 4, FloatArray(values.size * 4) { index ->
        val token = values[index / 4]
        val dimension = index % 4
        if (dimension == token) 1f else if (dimension == 2) index / 4f / values.size else 0f
    })

    private fun sigmoid(value: Float) = if (value >= 0f) 1f / (1f + exp(-value)) else exp(value) / (1f + exp(value))
}

enum class GraphPreset { Chain, Cycle, Star, Grid, TwoCommunities }

fun graphPreset(preset: GraphPreset): Graph {
    if (preset == GraphPreset.TwoCommunities) return twoCommunityGraph()
    val count = if (preset == GraphPreset.Grid) 9 else 6
    val features = Matrix(count, 2, FloatArray(count * 2) { index -> if (index % 2 == (index / 2) % 2) 1f else .15f })
    val edges = when (preset) {
        GraphPreset.Chain -> (0 until count - 1).map { it to it + 1 }.toSet()
        GraphPreset.Cycle -> ((0 until count - 1).map { it to it + 1 } + (count - 1 to 0)).toSet()
        GraphPreset.Star -> (1 until count).map { 0 to it }.toSet()
        GraphPreset.Grid -> buildSet {
            for (row in 0..2) for (column in 0..2) {
                val node = row * 3 + column
                if (column < 2) add(node to node + 1)
                if (row < 2) add(node to node + 3)
            }
        }
        GraphPreset.TwoCommunities -> emptySet()
    }
    return Graph(features, edges, IntArray(count) { if (it < count / 2) 0 else 1 })
}

fun Graph.connect(first: Int, second: Int): Graph {
    require(first in 0 until features.rows && second in 0 until features.rows && first != second)
    return copy(edges = edges + normalizedEdge(first, second))
}

fun Graph.removeEdge(first: Int, second: Int) = copy(edges = edges - normalizedEdge(first, second) - (second to first))

fun Graph.removeNode(node: Int): Graph {
    require(node in 0 until features.rows)
    val values = FloatArray((features.rows - 1) * features.columns)
    var target = 0
    for (row in 0 until features.rows) if (row != node) for (column in 0 until features.columns) values[target++] = features[row, column]
    val newEdges = edges.filter { it.first != node && it.second != node }.map {
        (if (it.first > node) it.first - 1 else it.first) to (if (it.second > node) it.second - 1 else it.second)
    }.toSet()
    return Graph(Matrix(features.rows - 1, features.columns, values), newEdges, labels.filterIndexed { index, _ -> index != node }.toIntArray())
}

private fun normalizedEdge(first: Int, second: Int) = if (first < second) first to second else second to first

data class NodeClassificationResult(val probabilities: FloatArray, val loss: Float, val accuracy: Float)

class TinyGcnNodeClassifier(seed: Int = 251) {
    private val convolution = GraphConvolution(2, 2, seed)
    private var classifier = floatArrayOf(.15f, -.15f)
    private var bias = 0f

    fun train(graph: Graph, epochs: Int = 100, learningRate: Float = .15f): NodeClassificationResult {
        require(epochs in 1..500)
        val embeddings = convolution.forward(graph).output
        repeat(epochs) {
            for (node in 0 until graph.features.rows) {
                val p = sigmoid(embeddings[node, 0] * classifier[0] + embeddings[node, 1] * classifier[1] + bias)
                val error = p - graph.labels[node]
                classifier[0] -= learningRate * error * embeddings[node, 0] / graph.features.rows
                classifier[1] -= learningRate * error * embeddings[node, 1] / graph.features.rows
                bias -= learningRate * error / graph.features.rows
            }
        }
        val probabilities = FloatArray(graph.features.rows) { node -> sigmoid(embeddings[node, 0] * classifier[0] + embeddings[node, 1] * classifier[1] + bias) }
        val loss = probabilities.indices.sumOf { index ->
            val p = probabilities[index].coerceIn(1e-6f, 1f - 1e-6f)
            val label = graph.labels[index].toFloat()
            (-(label * ln(p) + (1f - label) * ln(1f - p))).toDouble()
        }.toFloat() / probabilities.size
        val accuracy = probabilities.indices.count { (probabilities[it] >= .5f) == (graph.labels[it] == 1) }.toFloat() / probabilities.size
        return NodeClassificationResult(probabilities, loss, accuracy)
    }

    private fun sigmoid(value: Float) = 1f / (1f + exp(-value.coerceIn(-30f, 30f)))
}

fun interpolateLatent(from: FloatArray, to: FloatArray, amount: Float): FloatArray {
    require(from.size == to.size)
    val t = amount.coerceIn(0f, 1f)
    return FloatArray(from.size) { from[it] * (1f - t) + to[it] * t }
}
