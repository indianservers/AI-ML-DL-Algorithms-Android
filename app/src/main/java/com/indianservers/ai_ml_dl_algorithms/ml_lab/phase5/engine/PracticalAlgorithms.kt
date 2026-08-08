package com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

data class Detection(val classId: Int, val score: Float, val left: Float, val top: Float, val right: Float, val bottom: Float)

object PracticalAlgorithms {
    fun softmax(logits: FloatArray): FloatArray {
        require(logits.isNotEmpty() && logits.all { it.isFinite() })
        val peak = logits.max()
        val values = FloatArray(logits.size) { exp(logits[it] - peak) }
        val total = values.sum().coerceAtLeast(1e-12f)
        return FloatArray(values.size) { values[it] / total }
    }

    fun quantize(values: FloatArray, scale: Float, zeroPoint: Int, signed: Boolean = true): ByteArray {
        require(scale > 0f && scale.isFinite())
        val low = if (signed) -128 else 0
        val high = if (signed) 127 else 255
        return ByteArray(values.size) { (((values[it] / scale).toInt() + zeroPoint).coerceIn(low, high) and 0xff).toByte() }
    }

    fun dequantize(values: ByteArray, scale: Float, zeroPoint: Int, signed: Boolean = true) = FloatArray(values.size) {
        val quantized = if (signed) values[it].toInt() else values[it].toInt() and 0xff
        (quantized - zeroPoint) * scale
    }

    fun meanAbsoluteError(first: FloatArray, second: FloatArray): Float {
        require(first.size == second.size && first.isNotEmpty())
        return first.indices.sumOf { kotlin.math.abs(first[it] - second[it]).toDouble() }.toFloat() / first.size
    }

    fun iou(first: Detection, second: Detection): Float {
        val intersectionWidth = (min(first.right, second.right) - max(first.left, second.left)).coerceAtLeast(0f)
        val intersectionHeight = (min(first.bottom, second.bottom) - max(first.top, second.top)).coerceAtLeast(0f)
        val intersection = intersectionWidth * intersectionHeight
        val firstArea = (first.right - first.left).coerceAtLeast(0f) * (first.bottom - first.top).coerceAtLeast(0f)
        val secondArea = (second.right - second.left).coerceAtLeast(0f) * (second.bottom - second.top).coerceAtLeast(0f)
        return intersection / (firstArea + secondArea - intersection).coerceAtLeast(1e-7f)
    }

    fun nonMaximumSuppression(detections: List<Detection>, threshold: Float): List<Detection> {
        require(threshold in 0f..1f)
        val pending = detections.sortedByDescending { it.score }.toMutableList()
        val kept = mutableListOf<Detection>()
        while (pending.isNotEmpty()) {
            val candidate = pending.removeAt(0)
            kept += candidate
            pending.removeAll { it.classId == candidate.classId && iou(candidate, it) > threshold }
        }
        return kept
    }

    fun spectrogram(samples: FloatArray, windowSize: Int = 64, hop: Int = 32): Array<FloatArray> {
        require(windowSize in 8..512 && hop in 1..windowSize && samples.size >= windowSize)
        val frames = 1 + (samples.size - windowSize) / hop
        return Array(frames) { frame ->
            FloatArray(windowSize / 2 + 1) { bin ->
                var real = 0.0; var imaginary = 0.0
                for (index in 0 until windowSize) {
                    val window = .5 - .5 * cos(2.0 * PI * index / (windowSize - 1))
                    val angle = 2.0 * PI * bin * index / windowSize
                    val value = samples[frame * hop + index] * window
                    real += value * cos(angle); imaginary -= value * sin(angle)
                }
                ln(1f + sqrt((real * real + imaginary * imaginary).toFloat()))
            }
        }
    }

    fun tokenize(text: String, vocabulary: Map<String, Int>, unknownId: Int = 1): List<Pair<String, Int>> {
        return Regex("[A-Za-z0-9']+|[^\\sA-Za-z0-9]").findAll(text.lowercase()).map { match ->
            val token = match.value
            token to (vocabulary[token] ?: unknownId)
        }.toList()
    }

    fun cosineSimilarity(first: FloatArray, second: FloatArray): Float {
        require(first.size == second.size && first.isNotEmpty())
        var dot = 0f; var firstNorm = 0f; var secondNorm = 0f
        first.indices.forEach { dot += first[it] * second[it]; firstNorm += first[it] * first[it]; secondNorm += second[it] * second[it] }
        return dot / sqrt(firstNorm * secondNorm).coerceAtLeast(1e-8f)
    }

    fun percentile(values: List<Double>, percentile: Double): Double {
        require(values.isNotEmpty() && percentile in 0.0..1.0)
        val sorted = values.sorted()
        val position = percentile * (sorted.size - 1)
        val lower = position.toInt(); val upper = (lower + 1).coerceAtMost(sorted.lastIndex)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower)
    }

    fun benchmark(coldMillis: Double, warmMillis: List<Double>): BenchmarkSummary {
        require(coldMillis >= 0 && warmMillis.isNotEmpty() && warmMillis.all { it >= 0 })
        val mean = warmMillis.average()
        return BenchmarkSummary(coldMillis, warmMillis, percentile(warmMillis, .5), percentile(warmMillis, .9), percentile(warmMillis, .95), mean, if (mean == 0.0) Double.POSITIVE_INFINITY else 1000.0 / mean)
    }

    fun confusionMatrix(labels: IntArray, predictions: IntArray, classes: Int): Array<IntArray> {
        require(labels.size == predictions.size && classes > 1)
        return Array(classes) { IntArray(classes) }.also { matrix -> labels.indices.forEach { matrix[labels[it]][predictions[it]]++ } }
    }

    fun expectedCalibrationError(confidences: FloatArray, correct: BooleanArray, bins: Int = 10): Float {
        require(confidences.size == correct.size && bins > 0)
        var error = 0f
        repeat(bins) { bin ->
            val low = bin.toFloat() / bins; val high = (bin + 1f) / bins
            val indices = confidences.indices.filter { confidences[it] >= low && (confidences[it] < high || bin == bins - 1 && confidences[it] <= high) }
            if (indices.isNotEmpty()) {
                val confidence = indices.map { confidences[it] }.average().toFloat()
                val accuracy = indices.count { correct[it] }.toFloat() / indices.size
                error += indices.size.toFloat() / confidences.size * kotlin.math.abs(accuracy - confidence)
            }
        }
        return error
    }
}

class EmbeddingClassifier(private val dimensions: Int) {
    private val samples = mutableListOf<Pair<FloatArray, String>>()
    fun add(vector: FloatArray, label: String) { require(vector.size == dimensions && label.isNotBlank()); samples += vector.copyOf() to label }
    fun predict(vector: FloatArray): Pair<String, Float> {
        require(vector.size == dimensions && samples.isNotEmpty())
        return samples.map { (sample, label) -> label to PracticalAlgorithms.cosineSimilarity(sample, vector) }.maxBy { it.second }
    }
    fun size() = samples.size
}

data class QLearningSnapshot(val episode: Int, val reward: Float, val epsilon: Float)
class GridWorld(private val size: Int = 4, seed: Int = 301) {
    private val random = Random(seed)
    val qValues = Array(size * size) { FloatArray(4) }
    fun train(episodes: Int = 200, alpha: Float = .25f, gamma: Float = .92f): List<QLearningSnapshot> {
        val gridSize = size
        return buildList {
        repeat(episodes) { episode ->
            var state = 0; var rewardTotal = 0f; val epsilon = (.9f * (1f - episode.toFloat() / episodes)).coerceAtLeast(.05f)
            for (step in 0 until gridSize * gridSize * 2) {
                val action = if (random.nextFloat() < epsilon) random.nextInt(4) else qValues[state].indices.maxBy { qValues[state][it] }
                val row = state / gridSize; val column = state % gridSize
                val nextRow = (row + when (action) { 0 -> -1; 2 -> 1; else -> 0 }).coerceIn(0, gridSize - 1)
                val nextColumn = (column + when (action) { 1 -> 1; 3 -> -1; else -> 0 }).coerceIn(0, gridSize - 1)
                val next = nextRow * gridSize + nextColumn; val reward = if (next == gridSize * gridSize - 1) 1f else -.02f
                qValues[state][action] += alpha * (reward + gamma * qValues[next].max() - qValues[state][action])
                rewardTotal += reward; state = next
                if (state == gridSize * gridSize - 1) break
            }
            if (episode % 10 == 0 || episode == episodes - 1) add(QLearningSnapshot(episode, rewardTotal, epsilon))
        }
        }
    }
}

fun movingAverageForecast(values: FloatArray, window: Int): Float {
    require(window in 1..values.size && values.all { it.isFinite() })
    return values.takeLast(window).average().toFloat()
}
