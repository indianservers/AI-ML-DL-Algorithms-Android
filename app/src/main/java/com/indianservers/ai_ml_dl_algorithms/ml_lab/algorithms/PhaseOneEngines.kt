package com.indianservers.ai_ml_dl_algorithms.ml_lab.algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Point2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.RegressionState
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.TrainingSnapshot
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

object PhaseOneEngines {
    fun trainLinearRegression(
        points: List<Point2D>,
        epochs: Int,
        learningRate: Float,
        startWeight: Float = -0.45f,
        startBias: Float = 0.25f
    ): RegressionState {
        var weight = startWeight
        var bias = startBias
        val snapshots = mutableListOf<TrainingSnapshot>()
        repeat(epochs + 1) { epoch ->
            var loss = 0f
            var gradW = 0f
            var gradB = 0f
            points.forEach { point ->
                val prediction = weight * point.x + bias
                val error = prediction - point.y
                loss += error * error
                gradW += 2f * error * point.x
                gradB += 2f * error
            }
            loss /= points.size
            gradW /= points.size
            gradB /= points.size
            snapshots += TrainingSnapshot(epoch, loss, weight, bias, gradW, gradB)
            weight -= learningRate * gradW
            bias -= learningRate * gradB
        }
        return RegressionState(points, snapshots, snapshots.lastIndex)
    }

    fun regressionMetrics(points: List<Point2D>, weight: Float, bias: Float): Map<String, Float> {
        val mean = points.map { it.y }.average().toFloat()
        var mse = 0f
        var mae = 0f
        var total = 0f
        points.forEach {
            val prediction = weight * it.x + bias
            val error = prediction - it.y
            mse += error * error
            mae += abs(error)
            total += (it.y - mean) * (it.y - mean)
        }
        mse /= points.size
        mae /= points.size
        val r2 = if (total == 0f) 1f else 1f - (mse * points.size / total)
        return mapOf("MSE" to mse, "MAE" to mae, "RMSE" to sqrt(mse), "R2" to r2)
    }

    fun logisticProbability(point: Point2D, weightX: Float = 3.2f, weightY: Float = 2.4f, bias: Float = -0.05f): Float {
        val z = weightX * point.x + weightY * point.y + bias
        return (1f / (1f + exp(-z))).coerceIn(0f, 1f)
    }

    fun classifyLogistic(point: Point2D, threshold: Float): Int =
        if (logisticProbability(point) >= threshold) 1 else 0

    fun knnPredict(points: List<Point2D>, query: Point2D, k: Int, manhattan: Boolean): Pair<Int, List<Point2D>> {
        val neighbours = points.sortedBy {
            if (manhattan) abs(it.x - query.x) + abs(it.y - query.y)
            else sqrt((it.x - query.x).pow(2) + (it.y - query.y).pow(2))
        }.take(k.coerceAtLeast(1))
        val vote = neighbours.groupingBy { it.label }.eachCount().maxByOrNull { it.value }?.key ?: 0
        return vote to neighbours
    }

    fun gaussianNaiveBayes(points: List<Point2D>, query: Point2D): Pair<Int, Map<Int, Float>> {
        val byClass = points.groupBy { it.label }
        val scores = byClass.mapValues { (_, classPoints) ->
            val prior = classPoints.size.toFloat() / points.size
            prior * gaussian(query.x, classPoints.map { it.x }) * gaussian(query.y, classPoints.map { it.y })
        }
        val total = scores.values.sum().takeIf { it > 0f } ?: 1f
        val probabilities = scores.mapValues { it.value / total }
        return (probabilities.maxByOrNull { it.value }?.key ?: 0) to probabilities
    }

    fun kMeans(points: List<Point2D>, k: Int = 3, iterations: Int = 8): List<Point2D> {
        if (points.isEmpty()) return emptyList()
        var centers = points.take(k).mapIndexed { index, point -> point.copy(label = index) }
        repeat(iterations) {
            val grouped = points.groupBy { point ->
                centers.indices.minBy { index ->
                    val center = centers[index]
                    (point.x - center.x).pow(2) + (point.y - center.y).pow(2)
                }
            }
            centers = centers.mapIndexed { index, center ->
                val cluster = grouped[index].orEmpty()
                if (cluster.isEmpty()) center else Point2D(
                    cluster.map { it.x }.average().toFloat(),
                    cluster.map { it.y }.average().toFloat(),
                    index
                )
            }
        }
        return centers
    }

    fun pcaDirection(points: List<Point2D>): Point2D {
        val meanX = points.map { it.x }.average().toFloat()
        val meanY = points.map { it.y }.average().toFloat()
        val centered = points.map { it.x - meanX to it.y - meanY }
        val xx = centered.sumOf { (x, _) -> (x * x).toDouble() }.toFloat()
        val xy = centered.sumOf { (x, y) -> (x * y).toDouble() }.toFloat()
        val yy = centered.sumOf { (_, y) -> (y * y).toDouble() }.toFloat()
        val angle = 0.5f * kotlin.math.atan2(2f * xy, xx - yy)
        return Point2D(kotlin.math.cos(angle), kotlin.math.sin(angle))
    }

    fun polynomialPreview(x: Float): Float = 0.75f * x * x - 0.18f
    fun ridgePenalty(weight: Float, lambda: Float): Float = lambda * weight * weight
    fun lassoPenalty(weight: Float, lambda: Float): Float = lambda * abs(weight)

    private fun gaussian(value: Float, values: List<Float>): Float {
        val mean = values.average().toFloat()
        val variance = (values.sumOf { (it - mean).pow(2).toDouble() } / values.size).toFloat().coerceAtLeast(0.001f)
        val exponent = -((value - mean).pow(2)) / (2f * variance)
        return (1f / sqrt(2f * Math.PI.toFloat() * variance)) * exp(exponent)
    }
}
