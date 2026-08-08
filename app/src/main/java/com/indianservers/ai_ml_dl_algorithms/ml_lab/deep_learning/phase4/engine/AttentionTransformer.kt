package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

data class Matrix(val rows: Int, val columns: Int, val values: FloatArray) {
    init {
        require(rows > 0 && columns > 0)
        require(values.size == rows * columns)
        require(values.all { it.isFinite() })
    }
    operator fun get(row: Int, column: Int) = values[row * columns + column]
    operator fun set(row: Int, column: Int, value: Float) { values[row * columns + column] = value }
    fun row(index: Int) = FloatArray(columns) { get(index, it) }
    fun copy() = Matrix(rows, columns, values.copyOf())
    companion object { fun zeros(rows: Int, columns: Int) = Matrix(rows, columns, FloatArray(rows * columns)) }
}

object MatrixOps {
    fun matmul(a: Matrix, b: Matrix): Matrix {
        require(a.columns == b.rows)
        return Matrix(a.rows, b.columns, FloatArray(a.rows * b.columns) { index ->
            val row = index / b.columns; val column = index % b.columns
            (0 until a.columns).sumOf { (a[row, it] * b[it, column]).toDouble() }.toFloat()
        })
    }
    fun transpose(matrix: Matrix) = Matrix(matrix.columns, matrix.rows, FloatArray(matrix.values.size) { index -> matrix[index % matrix.rows, index / matrix.rows] })
    fun add(a: Matrix, b: Matrix) = Matrix(a.rows, a.columns, FloatArray(a.values.size) { a.values[it] + b.values[it] })
    fun random(rows: Int, columns: Int, seed: Int): Matrix { val random = Random(seed); val scale = sqrt(6f / (rows + columns)); return Matrix(rows, columns, FloatArray(rows * columns) { (random.nextFloat() * 2f - 1f) * scale }) }
}

data class AttentionResult(val query: Matrix, val key: Matrix, val value: Matrix, val scores: Matrix, val weights: Matrix, val output: Matrix, val entropy: Float)

class ScaledDotProductAttention(val embeddingDimension: Int, seed: Int = 101) {
    val queryProjection = MatrixOps.random(embeddingDimension, embeddingDimension, seed)
    val keyProjection = MatrixOps.random(embeddingDimension, embeddingDimension, seed + 1)
    val valueProjection = MatrixOps.random(embeddingDimension, embeddingDimension, seed + 2)

    fun forward(input: Matrix, causal: Boolean = false, paddingMask: BooleanArray? = null): AttentionResult {
        require(input.columns == embeddingDimension)
        require(input.rows > 0)
        require(paddingMask == null || paddingMask.size == input.rows)
        val query = MatrixOps.matmul(input, queryProjection); val key = MatrixOps.matmul(input, keyProjection); val value = MatrixOps.matmul(input, valueProjection)
        return attend(query, key, value, causal, paddingMask)
    }

    fun attend(query: Matrix, key: Matrix, value: Matrix, causal: Boolean = false, paddingMask: BooleanArray? = null): AttentionResult {
        require(query.columns > 0 && query.columns == key.columns)
        require(key.rows == value.rows)
        require(paddingMask == null || paddingMask.size == key.rows)
        val raw = MatrixOps.matmul(query, MatrixOps.transpose(key)); val scale = sqrt(query.columns.toFloat())
        val scores = Matrix(raw.rows, raw.columns, FloatArray(raw.values.size) { raw.values[it] / scale })
        val weights = Matrix.zeros(scores.rows, scores.columns)
        var entropy = 0f
        for (row in 0 until scores.rows) {
            val valid = (0 until scores.columns).filter { column -> (!causal || column <= row) && (paddingMask?.getOrNull(column) != false) }
            val peak = valid.maxOfOrNull { scores[row, it] } ?: 0f
            val exps = FloatArray(scores.columns) { column -> if (column in valid) exp(scores[row, column] - peak) else 0f }
            val total = exps.sum().coerceAtLeast(1e-7f)
            for (column in 0 until scores.columns) {
                weights[row, column] = exps[column] / total
                val p = weights[row, column]
                if (p > 0f) entropy -= p * ln(p)
            }
        }
        return AttentionResult(query, key, value, scores, weights, MatrixOps.matmul(weights, value), entropy / scores.rows)
    }
}

data class MultiHeadResult(val heads: List<AttentionResult>, val concatenated: Matrix, val output: Matrix)
class MultiHeadAttention(val embeddingDimension: Int, val headCount: Int, seed: Int = 111) {
    init { require(embeddingDimension % headCount == 0) }
    private val headDimension = embeddingDimension / headCount
    private val q = MatrixOps.random(embeddingDimension, embeddingDimension, seed)
    private val k = MatrixOps.random(embeddingDimension, embeddingDimension, seed + 1)
    private val v = MatrixOps.random(embeddingDimension, embeddingDimension, seed + 2)
    private val outputProjection = MatrixOps.random(embeddingDimension, embeddingDimension, seed + 3)
    fun forward(input: Matrix, causal: Boolean = false, paddingMask: BooleanArray? = null): MultiHeadResult {
        require(paddingMask == null || paddingMask.size == input.rows)
        val fullQ = MatrixOps.matmul(input, q); val fullK = MatrixOps.matmul(input, k); val fullV = MatrixOps.matmul(input, v)
        val heads = (0 until headCount).map { head ->
            fun slice(matrix: Matrix) = Matrix(matrix.rows, headDimension, FloatArray(matrix.rows * headDimension) { index -> matrix[index / headDimension, head * headDimension + index % headDimension] })
            ScaledDotProductAttention(headDimension).attend(slice(fullQ), slice(fullK), slice(fullV), causal, paddingMask)
        }
        val concatenated = Matrix(input.rows, embeddingDimension, FloatArray(input.rows * embeddingDimension) { index ->
            val row = index / embeddingDimension; val column = index % embeddingDimension; val head = column / headDimension
            heads[head].output[row, column % headDimension]
        })
        return MultiHeadResult(heads, concatenated, MatrixOps.matmul(concatenated, outputProjection))
    }
    fun parameterCount() = 4 * embeddingDimension * embeddingDimension
}

fun sinusoidalPositionEncoding(length: Int, dimensions: Int): Matrix = Matrix(length, dimensions, FloatArray(length * dimensions) { index ->
    val position = index / dimensions; val dimension = index % dimensions
    val angle = position / 10000.0.pow((2 * (dimension / 2)).toDouble() / dimensions).toFloat()
    if (dimension % 2 == 0) sin(angle) else cos(angle)
})

data class LayerNormResult(val mean: Float, val variance: Float, val output: FloatArray)
fun layerNorm(vector: FloatArray, gamma: FloatArray = FloatArray(vector.size) { 1f }, beta: FloatArray = FloatArray(vector.size), epsilon: Float = 1e-5f): LayerNormResult {
    val mean = vector.average().toFloat(); val variance = vector.sumOf { ((it - mean) * (it - mean)).toDouble() }.toFloat() / vector.size
    return LayerNormResult(mean, variance, FloatArray(vector.size) { gamma[it] * (vector[it] - mean) / sqrt(variance + epsilon) + beta[it] })
}

fun gelu(x: Float): Float = (0.5f * x * (1f + kotlin.math.tanh(sqrt(2f / PI.toFloat()) * (x + 0.044715f * x * x * x))))

data class TransformerTrace(val input: Matrix, val attention: MultiHeadResult, val attentionResidual: Matrix, val attentionNormalized: Matrix, val feedForward: Matrix, val output: Matrix)
class TransformerEncoderBlock(val dimension: Int, heads: Int, val feedForwardDimension: Int, seed: Int = 121) {
    val attention = MultiHeadAttention(dimension, heads, seed)
    private val first = MatrixOps.random(dimension, feedForwardDimension, seed + 10)
    private val second = MatrixOps.random(feedForwardDimension, dimension, seed + 11)
    fun forward(input: Matrix): TransformerTrace {
        val attentionResult = attention.forward(input)
        val residual = MatrixOps.add(input, attentionResult.output)
        val normalized = normalizeRows(residual)
        val hidden = MatrixOps.matmul(normalized, first).also { it.values.indices.forEach { index -> it.values[index] = gelu(it.values[index]) } }
        val feedForward = MatrixOps.matmul(hidden, second)
        val output = normalizeRows(MatrixOps.add(normalized, feedForward))
        return TransformerTrace(input, attentionResult, residual, normalized, feedForward, output)
    }
    private fun normalizeRows(matrix: Matrix) = Matrix(matrix.rows, matrix.columns, FloatArray(matrix.values.size).also { output ->
        for (row in 0 until matrix.rows) layerNorm(matrix.row(row)).output.copyInto(output, row * matrix.columns)
    })
    fun parameterCount() = attention.parameterCount() + 2 * dimension * feedForwardDimension + 4 * dimension
}

fun probabilities(logits: FloatArray, temperature: Float = 1f, topK: Int = logits.size): FloatArray {
    val allowed = logits.indices.sortedByDescending { logits[it] }.take(topK.coerceIn(1, logits.size)).toSet()
    val peak = allowed.maxOf { logits[it] / temperature.coerceAtLeast(0.05f) }
    val exps = FloatArray(logits.size) { if (it in allowed) exp(logits[it] / temperature.coerceAtLeast(0.05f) - peak) else 0f }
    val total = exps.sum().coerceAtLeast(1e-7f); return FloatArray(logits.size) { exps[it] / total }
}
