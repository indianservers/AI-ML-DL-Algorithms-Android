package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.TensorImage
import kotlin.math.sqrt

data class PatchResult(val patches: List<FloatArray>, val embeddings: Matrix)
class PatchEmbedding(val imageSize: Int, val patchSize: Int, val embeddingDimension: Int, seed: Int = 141) {
    val patchPixels = patchSize * patchSize
    private val projection = MatrixOps.random(patchPixels, embeddingDimension, seed)
    fun forward(image: TensorImage): PatchResult {
        require(image.channels == 1 && image.height == imageSize && image.width == imageSize && imageSize % patchSize == 0)
        val patches = buildList {
            for (row in 0 until imageSize step patchSize) for (column in 0 until imageSize step patchSize) {
                add(FloatArray(patchPixels) { index -> image[0, row + index / patchSize, column + index % patchSize] })
            }
        }
        val patchMatrix = Matrix(patches.size, patchPixels, patches.flatMap { it.asIterable() }.toFloatArray())
        return PatchResult(patches, MatrixOps.matmul(patchMatrix, projection))
    }
    fun parameterCount() = patchPixels * embeddingDimension
}

data class Graph(val features: Matrix, val edges: Set<Pair<Int, Int>>, val labels: IntArray = IntArray(features.rows))
data class GraphConvolutionResult(val adjacency: Matrix, val normalizedAdjacency: Matrix, val messages: Matrix, val output: Matrix)
class GraphConvolution(val inputDimension: Int, val outputDimension: Int, seed: Int = 151) {
    val weights = MatrixOps.random(inputDimension, outputDimension, seed)
    fun forward(graph: Graph): GraphConvolutionResult {
        val count = graph.features.rows; val adjacency = Matrix.zeros(count, count)
        for (node in 0 until count) adjacency[node, node] = 1f
        graph.edges.forEach { (a, b) -> adjacency[a, b] = 1f; adjacency[b, a] = 1f }
        val degrees = FloatArray(count) { row -> (0 until count).sumOf { adjacency[row, it].toDouble() }.toFloat() }
        val normalized = Matrix(count, count, FloatArray(count * count) { index ->
            val row = index / count; val column = index % count
            if (adjacency[row, column] == 0f) 0f else adjacency[row, column] / sqrt(degrees[row] * degrees[column]).coerceAtLeast(1e-7f)
        })
        val messages = MatrixOps.matmul(normalized, graph.features)
        val output = MatrixOps.matmul(messages, weights).also { matrix -> matrix.values.indices.forEach { matrix.values[it] = matrix.values[it].coerceAtLeast(0f) } }
        return GraphConvolutionResult(adjacency, normalized, messages, output)
    }
    fun parameterCount() = inputDimension * outputDimension
}

fun twoCommunityGraph(): Graph {
    val features = Matrix(8, 2, floatArrayOf(1f, .1f, .9f, .2f, .8f, .1f, .9f, .3f, .1f, 1f, .2f, .9f, .1f, .8f, .3f, .9f))
    val edges = setOf(0 to 1, 1 to 2, 2 to 3, 3 to 0, 0 to 2, 4 to 5, 5 to 6, 6 to 7, 7 to 4, 5 to 7, 3 to 4)
    return Graph(features, edges, intArrayOf(0, 0, 0, 0, 1, 1, 1, 1))
}
