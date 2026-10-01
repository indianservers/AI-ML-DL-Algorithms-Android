package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sign
import kotlin.math.sqrt
import kotlin.random.Random

internal data class LinearEstimate(val intercept: Double, val slope: Double) {
    fun predict(x: Double) = intercept + slope * x
}

internal data class UncertainPrediction(val x: Double, val mean: Double, val standardDeviation: Double)
internal data class GaussianPosterior(
    val predictions: List<UncertainPrediction>,
    val sampledFunctions: List<List<Double>>
)

internal data class GaussianClassModel(
    val points: List<LabPoint>,
    val lengthScale: Double,
    val alpha: DoubleArray,
    val inverse: Array<DoubleArray>
) {
    private fun kernel(x: Double, y: Double, point: LabPoint): Double =
        exp(-((x - point.x).pow(2) + (y - point.y).pow(2)) /
            (2 * lengthScale * lengthScale))

    fun predict(x: Double, y: Double): Pair<Double, Double> {
        val vector = DoubleArray(points.size) { kernel(x, y, points[it]) }
        val latentMean = vector.indices.sumOf { vector[it] * alpha[it] }
        val variance = (1.0 - vector.indices.sumOf { i ->
            vector[i] * vector.indices.sumOf { j -> inverse[i][j] * vector[j] }
        }).coerceIn(1e-8, 1.0)
        val probability = 1.0 / (1.0 + exp(-(latentMean /
            sqrt(1.0 + kotlin.math.PI * variance / 8.0)).coerceIn(-30.0, 30.0)))
        return probability to sqrt(variance)
    }
}

internal data class MarginModel(
    val weightX: Double, val weightY: Double, val bias: Double,
    val supportIndices: List<Int>, val hingeLoss: Double
) {
    fun score(x: Double, y: Double) = weightX * x + weightY * y + bias
}

internal data class KernelMarginModel(
    val points: List<LabPoint>,
    val coefficients: DoubleArray,
    val bias: Double,
    val kernel: KernelType,
    val gamma: Double,
    val degree: Int,
    val hingeLoss: Double
) {
    val supportIndices: List<Int> get() = coefficients.indices.filter { coefficients[it] > 1e-4 }
    fun score(x: Double, y: Double): Double = bias + points.indices.sumOf { i ->
        val dot = points[i].x * x + points[i].y * y
        val distance = (points[i].x - x).pow(2) + (points[i].y - y).pow(2)
        val value = when (kernel) {
            KernelType.Linear -> dot
            KernelType.Polynomial -> (1.0 + gamma * dot).pow(degree)
            KernelType.Rbf -> exp(-gamma * distance)
        }
        coefficients[i] * (if (points[i].label == 1) 1.0 else -1.0) * value
    }
}

internal data class SvrModel(
    val centers: List<Double>, val coefficients: List<Double>, val bias: Double,
    val gamma: Double, val kernel: KernelType, val epsilon: Double,
    val supportIndices: List<Int>
) {
    fun predict(x: Double): Double = bias + centers.indices.sumOf { i ->
        coefficients[i] * when (kernel) {
            KernelType.Linear -> centers[i] * x
            KernelType.Polynomial -> (1.0 + gamma * centers[i] * x).pow(3)
            KernelType.Rbf -> exp(-gamma * (centers[i] - x).pow(2))
        }
    }
}

internal object SupervisedStatisticalEngine {
    fun quantile(points: List<LabPoint>, q: Double): LinearEstimate {
        require(points.size >= 2)
        val quantile = q.coerceIn(0.01, 0.99)
        val start = PhaseOneEngines.fitSimpleLinear(points)
        var slope = start.weights.first()
        var intercept = start.bias
        repeat(1800) { epoch ->
            var gradSlope = 0.0
            var gradIntercept = 0.0
            points.forEach {
                val residual = it.target - (intercept + slope * it.x)
                val derivative = if (residual >= 0) -quantile else 1.0 - quantile
                gradSlope += derivative * it.x
                gradIntercept += derivative
            }
            val rate = 0.2 / (1.0 + epoch / 300.0) / points.size
            slope -= rate * gradSlope
            intercept -= rate * gradIntercept
        }
        return LinearEstimate(intercept, slope)
    }

    fun robustHuber(points: List<LabPoint>, delta: Double = 0.18): LinearEstimate {
        require(points.size >= 2)
        val initial = PhaseOneEngines.fitSimpleLinear(points)
        var slope = initial.weights.first()
        var intercept = initial.bias
        repeat(35) {
            val weights = points.map { point ->
                val residual = abs(point.target - (intercept + slope * point.x))
                if (residual <= delta) 1.0 else delta / residual.coerceAtLeast(1e-9)
            }
            val sum = weights.sum()
            val meanX = points.indices.sumOf { weights[it] * points[it].x } / sum
            val meanY = points.indices.sumOf { weights[it] * points[it].target } / sum
            val covariance = points.indices.sumOf { weights[it] * (points[it].x - meanX) * (points[it].target - meanY) }
            val variance = points.indices.sumOf { weights[it] * (points[it].x - meanX).pow(2) }.coerceAtLeast(1e-9)
            slope = covariance / variance
            intercept = meanY - slope * meanX
        }
        return LinearEstimate(intercept, slope)
    }

    fun bayesianLinear(points: List<LabPoint>, noise: Double = 0.12, grid: List<Double>): GaussianPosterior {
        require(points.isNotEmpty())
        val precision = 1.0 / noise.coerceAtLeast(0.01).pow(2)
        val a = 1.0 + points.size * precision
        val b = points.sumOf { it.x } * precision
        val d = 1.0 + points.sumOf { it.x * it.x } * precision
        val determinant = (a * d - b * b).coerceAtLeast(1e-9)
        val covariance00 = d / determinant
        val covariance01 = -b / determinant
        val covariance11 = a / determinant
        val target0 = points.sumOf { it.target } * precision
        val target1 = points.sumOf { it.x * it.target } * precision
        val intercept = covariance00 * target0 + covariance01 * target1
        val slope = covariance01 * target0 + covariance11 * target1
        val predictions = grid.map { x ->
            val variance = covariance00 + 2 * x * covariance01 + x * x * covariance11
            UncertainPrediction(x, intercept + slope * x, sqrt(max(variance, 0.0)))
        }
        val random = Random(51)
        val samples = List(4) {
            val z0 = random.nextDouble(-2.0, 2.0)
            val z1 = random.nextDouble(-2.0, 2.0)
            val sampledIntercept = intercept + sqrt(covariance00) * z0
            val conditionalSlope = covariance01 / sqrt(covariance00) * z0 +
                sqrt(max(covariance11 - covariance01.pow(2) / covariance00, 0.0)) * z1
            grid.map { x -> sampledIntercept + (slope + conditionalSlope) * x }
        }
        return GaussianPosterior(predictions, samples)
    }

    fun gaussianProcess(
        points: List<LabPoint>, lengthScale: Double, noise: Double, grid: List<Double>
    ): GaussianPosterior {
        require(points.isNotEmpty())
        val n = points.size
        val length = lengthScale.coerceAtLeast(0.05)
        fun kernel(a: Double, b: Double) = exp(-0.5 * ((a - b) / length).pow(2))
        val matrix = Array(n) { i -> DoubleArray(n) { j ->
            kernel(points[i].x, points[j].x) + if (i == j) noise.coerceAtLeast(0.01).pow(2) + 1e-6 else 0.0
        } }
        val alpha = solve(matrix, DoubleArray(n) { points[it].target })
        val predictions = grid.map { x ->
            val k = DoubleArray(n) { kernel(x, points[it].x) }
            val mean = k.indices.sumOf { k[it] * alpha[it] }
            val solved = solve(matrix, k)
            val variance = (1.0 - k.indices.sumOf { k[it] * solved[it] }).coerceAtLeast(1e-8)
            UncertainPrediction(x, mean, sqrt(variance))
        }
        val covariance = Array(grid.size) { i -> DoubleArray(grid.size) { j ->
            val ki = DoubleArray(n) { kernel(grid[i], points[it].x) }
            val kj = DoubleArray(n) { kernel(grid[j], points[it].x) }
            val solved = solve(matrix, kj)
            kernel(grid[i], grid[j]) - ki.indices.sumOf { ki[it] * solved[it] } +
                if (i == j) 1e-7 else 0.0
        } }
        val lower = Array(grid.size) { DoubleArray(grid.size) }
        for (i in grid.indices) for (j in 0..i) {
            val remainder = covariance[i][j] - (0 until j).sumOf { lower[i][it] * lower[j][it] }
            lower[i][j] = if (i == j) sqrt(remainder.coerceAtLeast(1e-10))
                else remainder / lower[j][j].coerceAtLeast(1e-10)
        }
        val random = Random(63)
        fun normal(): Double {
            val u1 = random.nextDouble().coerceAtLeast(1e-10)
            val u2 = random.nextDouble()
            return sqrt(-2.0 * ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
        }
        val samples = List(4) {
            val z = DoubleArray(grid.size) { normal() }
            predictions.indices.map { i ->
                predictions[i].mean + (0..i).sumOf { j -> lower[i][j] * z[j] }
            }
        }
        return GaussianPosterior(predictions, samples)
    }

    // Kernel GP latent regression with Bernoulli probabilities approximated
    // through a logistic link. Variance is computed from the GP posterior,
    // so unseen regions become uncertain instead of borrowing KNN votes.
    fun gaussianProcessClassifier(
        points: List<LabPoint>, lengthScale: Double, noise: Double = .35
    ): GaussianClassModel {
        require(points.any { it.label == 0 } && points.any { it.label == 1 })
        val scale = lengthScale.coerceAtLeast(.05)
        val n = points.size
        val matrix = Array(n) { i -> DoubleArray(n) { j ->
            exp(-((points[i].x - points[j].x).pow(2) +
                (points[i].y - points[j].y).pow(2)) / (2 * scale * scale)) +
                if (i == j) noise * noise + 1e-6 else 0.0
        } }
        val targets = DoubleArray(n) { if (points[it].label == 1) 2.0 else -2.0 }
        val alpha = solve(matrix, targets)
        val inverse = Array(n) { column ->
            solve(matrix, DoubleArray(n) { if (it == column) 1.0 else 0.0 })
        }.let { columns -> Array(n) { row -> DoubleArray(n) { col -> columns[col][row] } } }
        return GaussianClassModel(points, scale, alpha, inverse)
    }

    fun linearSvm(points: List<LabPoint>, c: Double): MarginModel {
        require(points.any { it.label == 0 } && points.any { it.label == 1 })
        var wx = 0.0
        var wy = 0.0
        var bias = 0.0
        val penalty = c.coerceAtLeast(0.01)
        repeat(500) { epoch ->
            val rate = 0.18 / (1.0 + epoch / 80.0)
            for (point in points) {
                val label = if (point.label == 1) 1.0 else -1.0
                val margin = label * (wx * point.x + wy * point.y + bias)
                wx *= 1.0 - rate / points.size
                wy *= 1.0 - rate / points.size
                if (margin < 1.0) {
                    wx += rate * penalty * label * point.x / points.size
                    wy += rate * penalty * label * point.y / points.size
                    bias += rate * penalty * label / points.size
                }
            }
        }
        val margins = points.map {
            (if (it.label == 1) 1.0 else -1.0) * (wx * it.x + wy * it.y + bias)
        }
        val support = margins.indices.filter { margins[it] <= 1.1 }
        return MarginModel(wx, wy, bias, support, margins.sumOf { max(0.0, 1.0 - it) } / points.size)
    }

    fun kernelSvm(
        points: List<LabPoint>, c: Double, kernel: KernelType,
        gamma: Double, degree: Int = 3
    ): KernelMarginModel {
        require(points.any { it.label == 0 } && points.any { it.label == 1 })
        val n = points.size
        val labels = DoubleArray(n) { if (points[it].label == 1) 1.0 else -1.0 }
        val penalty = c.coerceAtLeast(.01)
        val alpha = DoubleArray(n)
        var bias = 0.0
        val gram = Array(n) { i -> DoubleArray(n) { j ->
            val dot = points[i].x * points[j].x + points[i].y * points[j].y
            val distance = (points[i].x - points[j].x).pow(2) +
                (points[i].y - points[j].y).pow(2)
            when (kernel) {
                KernelType.Linear -> dot
                KernelType.Polynomial -> (1.0 + gamma * dot).pow(degree)
                KernelType.Rbf -> exp(-gamma * distance)
            }
        } }
        fun score(index: Int): Double = bias + (0 until n).sumOf {
            alpha[it] * labels[it] * gram[it][index]
        }
        repeat(60) {
            for (i in 0 until n) {
                val errorI = score(i) - labels[i]
                if (!((labels[i] * errorI < -.002 && alpha[i] < penalty - 1e-6) ||
                    (labels[i] * errorI > .002 && alpha[i] > 1e-6))) continue
                val j = (0 until n).filter { it != i }
                    .maxBy { kotlin.math.abs(errorI - (score(it) - labels[it])) }
                val errorJ = score(j) - labels[j]
                val oldI = alpha[i]
                val oldJ = alpha[j]
                val lower = if (labels[i] != labels[j]) max(0.0, oldJ - oldI)
                    else max(0.0, oldI + oldJ - penalty)
                val upper = if (labels[i] != labels[j]) kotlin.math.min(penalty, penalty + oldJ - oldI)
                    else kotlin.math.min(penalty, oldI + oldJ)
                if (upper - lower < 1e-8) continue
                val eta = 2 * gram[i][j] - gram[i][i] - gram[j][j]
                if (eta >= -1e-10) continue
                alpha[j] = (oldJ - labels[j] * (errorI - errorJ) / eta).coerceIn(lower, upper)
                if (abs(alpha[j] - oldJ) < 1e-7) { alpha[j] = oldJ; continue }
                alpha[i] = oldI + labels[i] * labels[j] * (oldJ - alpha[j])
                val b1 = bias - errorI - labels[i] * (alpha[i] - oldI) * gram[i][i] -
                    labels[j] * (alpha[j] - oldJ) * gram[i][j]
                val b2 = bias - errorJ - labels[i] * (alpha[i] - oldI) * gram[i][j] -
                    labels[j] * (alpha[j] - oldJ) * gram[j][j]
                bias = when {
                    alpha[i] > 1e-6 && alpha[i] < penalty - 1e-6 -> b1
                    alpha[j] > 1e-6 && alpha[j] < penalty - 1e-6 -> b2
                    else -> (b1 + b2) / 2
                }
            }
        }
        val model = KernelMarginModel(points, alpha, bias, kernel, gamma, degree, 0.0)
        val loss = points.indices.sumOf {
            max(0.0, 1.0 - labels[it] * model.score(points[it].x, points[it].y))
        } / n
        return model.copy(hingeLoss = loss)
    }

    fun epsilonSvr(
        points: List<LabPoint>, c: Double, epsilon: Double,
        kernel: KernelType, gamma: Double
    ): SvrModel {
        require(points.size >= 2)
        val centers = points.map { it.x }
        val n = points.size
        val coefficients = DoubleArray(n)
        var bias = points.map { it.target }.average()
        fun kernelValue(a: Double, b: Double): Double = when (kernel) {
            KernelType.Linear -> a * b
            KernelType.Polynomial -> (1.0 + gamma * a * b).pow(3)
            KernelType.Rbf -> exp(-gamma * (a - b).pow(2))
        }
        repeat(400) { epoch ->
            val rate = 0.12 / (1.0 + epoch / 120.0)
            for (i in points.indices) {
                val predicted = bias + coefficients.indices.sumOf { j -> coefficients[j] * kernelValue(centers[j], points[i].x) }
                val error = predicted - points[i].target
                val gradient = if (abs(error) <= epsilon) 0.0 else sign(error)
                coefficients[i] = (coefficients[i] * (1.0 - rate * 0.05) - rate * c * gradient / n).coerceIn(-2.0, 2.0)
                bias -= rate * c * gradient / n
            }
        }
        val model = SvrModel(centers, coefficients.toList(), bias, gamma, kernel, epsilon, emptyList())
        return model.copy(supportIndices = points.indices.filter {
            abs(model.predict(points[it].x) - points[it].target) >= epsilon * 0.95
        })
    }

    private fun solve(matrix: Array<DoubleArray>, right: DoubleArray): DoubleArray {
        val n = right.size
        val data = Array(n) { i -> matrix[i].clone() }
        val result = right.clone()
        for (column in 0 until n) {
            val pivot = (column until n).maxBy { abs(data[it][column]) }
            if (pivot != column) {
                val row = data[column]; data[column] = data[pivot]; data[pivot] = row
                val value = result[column]; result[column] = result[pivot]; result[pivot] = value
            }
            val pivotValue = data[column][column]
            val divisor = if (abs(pivotValue) < 1e-10)
                if (pivotValue < 0.0) -1e-10 else 1e-10
            else pivotValue
            for (j in column until n) data[column][j] /= divisor
            result[column] /= divisor
            for (row in 0 until n) if (row != column) {
                val factor = data[row][column]
                for (j in column until n) data[row][j] -= factor * data[column][j]
                result[row] -= factor * result[column]
            }
        }
        return result
    }
}
