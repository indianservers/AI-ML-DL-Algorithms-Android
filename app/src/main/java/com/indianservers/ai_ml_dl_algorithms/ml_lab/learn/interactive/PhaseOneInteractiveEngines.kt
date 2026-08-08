package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sqrt

enum class PhaseOneAlgorithmKind(val displayName: String, val family: String) {
    SimpleLinearRegression("Simple Linear Regression", "Supervised Learning - Regression"),
    MultipleLinearRegression("Multiple Linear Regression", "Supervised Learning - Regression"),
    PolynomialRegression("Polynomial Regression", "Supervised Learning - Regression"),
    RidgeRegression("Ridge Regression", "Supervised Learning - Regression"),
    LassoRegression("Lasso Regression", "Supervised Learning - Regression"),
    ElasticNetRegression("Elastic Net Regression", "Supervised Learning - Regression"),
    LogisticRegression("Logistic Regression", "Supervised Learning - Classification"),
    Knn("K-Nearest Neighbors", "Supervised Learning - Classification"),
    Perceptron("Perceptron", "Supervised Learning - Classification"),
    DecisionTreeClassification("Decision Tree - Classification", "Supervised Learning - Tree Models"),
    DecisionTreeRegression("Decision Tree - Regression", "Supervised Learning - Tree Models"),
    BatchGradientDescent("Batch Gradient Descent", "Optimization Foundation"),
    StochasticGradientDescent("Stochastic Gradient Descent", "Optimization Foundation"),
    MiniBatchGradientDescent("Mini-Batch Gradient Descent", "Optimization Foundation")
}

enum class DatasetPreset(val label: String) {
    LinearNoise("Linear + noise"),
    PerfectLinear("Perfect linear"),
    NegativeCorrelation("Negative correlation"),
    Outliers("Outliers"),
    Polynomial("Non-linear curve"),
    TwoClusters("Two clusters"),
    OverlappingClasses("Overlapping classes"),
    XorLike("XOR-like"),
    Circular("Circular"),
    Imbalanced("Imbalanced")
}

enum class DistanceMetric(val label: String) { Euclidean("Euclidean"), Manhattan("Manhattan") }
enum class SplitCriterion(val label: String) { Gini("Gini"), Entropy("Entropy") }
enum class TrainingMode { Batch, Stochastic, MiniBatch }

data class LabPoint(
    val x: Double,
    val y: Double,
    val label: Int = 0,
    val target: Double = y,
    val train: Boolean = true
)

data class ParameterSpec(
    val id: String,
    val label: String,
    val value: Double,
    val range: ClosedFloatingPointRange<Double>,
    val steps: Int = 0
)

data class RegressionFit(
    val weights: List<Double>,
    val bias: Double,
    val predictions: List<Double>,
    val mse: Double,
    val mae: Double,
    val rmse: Double,
    val r2: Double,
    val penalty: Double = 0.0
)

data class ClassificationMetrics(
    val accuracy: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val tp: Int,
    val tn: Int,
    val fp: Int,
    val fn: Int
)

data class TrainingStep(
    val iteration: Int,
    val description: String,
    val loss: Double,
    val parameters: Map<String, Double>,
    val highlightedSamples: List<Int>,
    val explanation: String
)

data class GradientPathPoint(
    val weight: Double,
    val bias: Double,
    val loss: Double,
    val gradW: Double,
    val gradB: Double,
    val sampleIndices: List<Int>
)

data class TreeSplit(
    val feature: String,
    val threshold: Double,
    val impurity: Double,
    val leftCount: Int,
    val rightCount: Int,
    val explanation: String
)

object PhaseOneTopicMatcher {
    fun kindFor(title: String, section: String): PhaseOneAlgorithmKind? = when {
        title == "Simple Linear Regression" -> PhaseOneAlgorithmKind.SimpleLinearRegression
        title == "Multiple Linear Regression" -> PhaseOneAlgorithmKind.MultipleLinearRegression
        title == "Polynomial Regression" -> PhaseOneAlgorithmKind.PolynomialRegression
        title == "Ridge Regression" -> PhaseOneAlgorithmKind.RidgeRegression
        title == "Lasso Regression" -> PhaseOneAlgorithmKind.LassoRegression
        title == "Elastic Net Regression" -> PhaseOneAlgorithmKind.ElasticNetRegression
        title == "Logistic Regression" -> PhaseOneAlgorithmKind.LogisticRegression
        title == "K-Nearest Neighbors" -> PhaseOneAlgorithmKind.Knn
        title == "Perceptron" -> PhaseOneAlgorithmKind.Perceptron
        title == "Decision Tree" && section == "Classification" -> PhaseOneAlgorithmKind.DecisionTreeClassification
        title == "Decision Tree Regression" -> PhaseOneAlgorithmKind.DecisionTreeRegression
        title == "Batch Gradient Descent" -> PhaseOneAlgorithmKind.BatchGradientDescent
        title == "Stochastic Gradient Descent" -> PhaseOneAlgorithmKind.StochasticGradientDescent
        title == "Mini-Batch Gradient Descent" -> PhaseOneAlgorithmKind.MiniBatchGradientDescent
        else -> null
    }
}

object PhaseOneDatasets {
    fun generate(
        preset: DatasetPreset,
        samples: Int = 36,
        noise: Double = 0.12,
        seed: Int = 7,
        trainRatio: Double = 0.8
    ): List<LabPoint> {
        val count = samples.coerceIn(10, 200)
        return List(count) { index ->
            val train = index < (count * trainRatio).toInt()
            val base = -1.0 + 2.0 * index / (count - 1).coerceAtLeast(1)
            val jitterX = centeredNoise(seed, index, 0) * 0.08
            val eps = centeredNoise(seed, index, 1) * noise
            when (preset) {
                DatasetPreset.PerfectLinear -> LabPoint(base, 0.72 * base - 0.12, train = train)
                DatasetPreset.LinearNoise -> {
                    val x = (base + jitterX).coerceIn(-1.0, 1.0)
                    LabPoint(x, 0.68 * x - 0.05 + eps, train = train)
                }
                DatasetPreset.NegativeCorrelation -> {
                    val x = (base + jitterX).coerceIn(-1.0, 1.0)
                    LabPoint(x, -0.75 * x + 0.04 + eps, train = train)
                }
                DatasetPreset.Outliers -> {
                    val x = (base + jitterX).coerceIn(-1.0, 1.0)
                    val bump = if (index % 11 == 0) 0.85 * if (index % 2 == 0) 1 else -1 else 0.0
                    LabPoint(x, 0.52 * x + eps + bump, train = train)
                }
                DatasetPreset.Polynomial -> {
                    val x = (base + jitterX).coerceIn(-1.0, 1.0)
                    LabPoint(x, 0.92 * x * x - 0.38 + eps, train = train)
                }
                DatasetPreset.TwoClusters -> clusterPoint(index, count, seed, noise, overlap = 0.0, train)
                DatasetPreset.OverlappingClasses -> clusterPoint(index, count, seed, noise + 0.24, overlap = 0.26, train)
                DatasetPreset.XorLike -> xorPoint(index, count, seed, noise, train)
                DatasetPreset.Circular -> circularPoint(index, count, seed, noise, train)
                DatasetPreset.Imbalanced -> imbalancedPoint(index, count, seed, noise, train)
            }
        }
    }

    private fun clusterPoint(index: Int, count: Int, seed: Int, noise: Double, overlap: Double, train: Boolean): LabPoint {
        val label = index % 2
        val centerX = if (label == 0) -0.48 + overlap else 0.48 - overlap
        val centerY = if (label == 0) -0.24 else 0.28
        val x = (centerX + centeredNoise(seed, index, 2) * (0.18 + noise)).coerceIn(-1.0, 1.0)
        val y = (centerY + centeredNoise(seed, index, 3) * (0.18 + noise)).coerceIn(-1.0, 1.0)
        return LabPoint(x, y, label, label.toDouble(), train)
    }

    private fun xorPoint(index: Int, count: Int, seed: Int, noise: Double, train: Boolean): LabPoint {
        val quadrant = index % 4
        val signX = if (quadrant < 2) -1.0 else 1.0
        val signY = if (quadrant % 2 == 0) -1.0 else 1.0
        val label = if (signX * signY > 0) 1 else 0
        return LabPoint(
            signX * (0.36 + abs(centeredNoise(seed, index, 4)) * 0.36),
            signY * (0.36 + abs(centeredNoise(seed, index, 5)) * 0.36),
            label,
            label.toDouble(),
            train
        )
    }

    private fun circularPoint(index: Int, count: Int, seed: Int, noise: Double, train: Boolean): LabPoint {
        val angle = 2.0 * Math.PI * index / count
        val inner = index % 3 == 0
        val radius = if (inner) 0.24 else 0.68
        val r = radius + centeredNoise(seed, index, 6) * noise
        val label = if (inner) 1 else 0
        return LabPoint(r * kotlin.math.cos(angle), r * kotlin.math.sin(angle), label, label.toDouble(), train)
    }

    private fun imbalancedPoint(index: Int, count: Int, seed: Int, noise: Double, train: Boolean): LabPoint {
        val label = if (index < count * 0.76) 0 else 1
        val centerX = if (label == 0) -0.22 else 0.62
        val centerY = if (label == 0) -0.08 else 0.44
        return LabPoint(
            (centerX + centeredNoise(seed, index, 7) * (0.26 + noise)).coerceIn(-1.0, 1.0),
            (centerY + centeredNoise(seed, index, 8) * (0.26 + noise)).coerceIn(-1.0, 1.0),
            label,
            label.toDouble(),
            train
        )
    }

    private fun centeredNoise(seed: Int, index: Int, salt: Int): Double {
        val raw = kotlin.math.sin((seed * 97 + index * 31 + salt * 13).toDouble()) * 43758.5453123
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}

object PhaseOneEngines {
    fun fitSimpleLinear(points: List<LabPoint>, weight: Double? = null, bias: Double? = null): RegressionFit {
        val train = points.filter { it.train }.ifEmpty { points }
        val resolved = if (weight != null && bias != null) weight to bias else {
            val meanX = train.map { it.x }.average()
            val meanY = train.map { it.target }.average()
            val variance = train.sumOf { (it.x - meanX).pow(2) }.coerceAtLeast(1e-9)
            val covariance = train.sumOf { (it.x - meanX) * (it.target - meanY) }
            val w = covariance / variance
            w to meanY - w * meanX
        }
        return regressionFit(points, listOf(resolved.first), resolved.second) { p, w -> w[0] * p.x + resolved.second }
    }

    fun fitMultiple(points: List<LabPoint>, w1: Double? = null, w2: Double? = null, bias: Double? = null): RegressionFit {
        val weights = if (w1 != null && w2 != null && bias != null) listOf(w1, w2, bias) else {
            solveLeastSquares(points.filter { it.train }.ifEmpty { points }, degree = 1, extraFeature = true, ridge = 0.0)
        }
        return regressionFit(points, weights.take(2), weights[2]) { p, w -> w[0] * p.x + w[1] * p.y + weights[2] }
    }

    fun fitPolynomial(points: List<LabPoint>, degree: Int): RegressionFit {
        val coefficients = solveLeastSquares(points.filter { it.train }.ifEmpty { points }, degree.coerceIn(1, 8), extraFeature = false, ridge = 0.0)
        val bias = coefficients.first()
        val weights = coefficients.drop(1)
        return regressionFit(points, weights, bias) { p, _ ->
            coefficients.indices.sumOf { power -> coefficients[power] * p.x.pow(power) }
        }
    }

    fun fitRidge(points: List<LabPoint>, alpha: Double): RegressionFit {
        val coefficients = solveLeastSquares(points.filter { it.train }.ifEmpty { points }, 3, extraFeature = false, ridge = alpha)
        val penalty = alpha * coefficients.drop(1).sumOf { it * it }
        return regressionFit(points, coefficients.drop(1), coefficients.first(), penalty) { p, _ ->
            coefficients.indices.sumOf { power -> coefficients[power] * p.x.pow(power) }
        }
    }

    fun fitLasso(points: List<LabPoint>, alpha: Double): RegressionFit {
        val coefficients = coordinateDescent(points.filter { it.train }.ifEmpty { points }, alpha, 0.0)
        val penalty = alpha * coefficients.drop(1).sumOf { abs(it) }
        return regressionFit(points, coefficients.drop(1), coefficients.first(), penalty) { p, _ ->
            coefficients.indices.sumOf { power -> coefficients[power] * p.x.pow(power) }
        }
    }

    fun fitElasticNet(points: List<LabPoint>, alpha: Double, l1Ratio: Double): RegressionFit {
        val coefficients = coordinateDescent(points.filter { it.train }.ifEmpty { points }, alpha * l1Ratio, alpha * (1.0 - l1Ratio))
        val penalty = alpha * (l1Ratio * coefficients.drop(1).sumOf { abs(it) } + (1.0 - l1Ratio) * coefficients.drop(1).sumOf { it * it })
        return regressionFit(points, coefficients.drop(1), coefficients.first(), penalty) { p, _ ->
            coefficients.indices.sumOf { power -> coefficients[power] * p.x.pow(power) }
        }
    }

    fun gradientPath(
        points: List<LabPoint>,
        learningRate: Double,
        iterations: Int,
        mode: TrainingMode,
        batchSize: Int = 8,
        startWeight: Double = -0.85,
        startBias: Double = 0.35
    ): List<GradientPathPoint> {
        val train = points.filter { it.train }.ifEmpty { points }
        var weight = startWeight
        var bias = startBias
        return List(iterations.coerceAtLeast(1) + 1) { iteration ->
            val indices = when (mode) {
                TrainingMode.Batch -> train.indices.toList()
                TrainingMode.Stochastic -> listOf(iteration % train.size)
                TrainingMode.MiniBatch -> List(batchSize.coerceIn(1, train.size)) { (iteration * batchSize + it) % train.size }
            }
            val batch = indices.map { train[it] }
            val currentLoss = meanSquaredError(train, weight, bias)
            var gradW = 0.0
            var gradB = 0.0
            batch.forEach {
                val error = weight * it.x + bias - it.target
                gradW += 2.0 * error * it.x
                gradB += 2.0 * error
            }
            gradW /= batch.size
            gradB /= batch.size
            val point = GradientPathPoint(weight, bias, currentLoss, gradW, gradB, indices)
            weight -= learningRate * gradW
            bias -= learningRate * gradB
            point
        }
    }

    fun sigmoid(z: Double): Double = 1.0 / (1.0 + exp(-z))

    fun logisticProbability(point: LabPoint, w1: Double, w2: Double, bias: Double): Double =
        sigmoid(w1 * point.x + w2 * point.y + bias)

    fun logisticMetrics(points: List<LabPoint>, w1: Double, w2: Double, bias: Double, threshold: Double): ClassificationMetrics {
        var tp = 0
        var tn = 0
        var fp = 0
        var fn = 0
        points.forEach {
            val prediction = if (logisticProbability(it, w1, w2, bias) >= threshold) 1 else 0
            when {
                it.label == 1 && prediction == 1 -> tp++
                it.label == 0 && prediction == 0 -> tn++
                it.label == 0 && prediction == 1 -> fp++
                it.label == 1 && prediction == 0 -> fn++
            }
        }
        val total = (tp + tn + fp + fn).coerceAtLeast(1)
        val precision = tp / (tp + fp).coerceAtLeast(1).toDouble()
        val recall = tp / (tp + fn).coerceAtLeast(1).toDouble()
        val f1 = if (precision + recall == 0.0) 0.0 else 2.0 * precision * recall / (precision + recall)
        return ClassificationMetrics((tp + tn) / total.toDouble(), precision, recall, f1, tp, tn, fp, fn)
    }

    fun knn(points: List<LabPoint>, query: LabPoint, k: Int, metric: DistanceMetric): Pair<Int, List<Pair<LabPoint, Double>>> {
        val neighbours = points.map { it to distance(it, query, metric) }.sortedBy { it.second }.take(k.coerceAtLeast(1))
        val prediction = neighbours.groupingBy { it.first.label }.eachCount().maxWithOrNull(
            compareBy<Map.Entry<Int, Int>> { it.value }.thenByDescending { -it.key }
        )?.key ?: 0
        return prediction to neighbours
    }

    fun perceptronStep(points: List<LabPoint>, iteration: Int, weightX: Double, weightY: Double, bias: Double, rate: Double): TrainingStep {
        val index = iteration % points.size
        val sample = points[index]
        val expected = if (sample.label == 1) 1 else -1
        val score = weightX * sample.x + weightY * sample.y + bias
        val prediction = if (score >= 0.0) 1 else -1
        val error = expected - prediction
        val nextWx = weightX + rate * error * sample.x
        val nextWy = weightY + rate * error * sample.y
        val nextB = bias + rate * error
        val loss = if (error == 0) 0.0 else 1.0
        val explanation = if (error == 0) "The sample is correctly classified, so the boundary stays in place." else "The sample is misclassified, so weights move toward the correct side of the point."
        return TrainingStep(
            iteration + 1,
            "Perceptron checks one labelled sample and updates only on mistakes.",
            loss,
            mapOf("w1" to nextWx, "w2" to nextWy, "bias" to nextB),
            listOf(index),
            explanation
        )
    }

    fun bestClassificationSplit(points: List<LabPoint>, criterion: SplitCriterion): TreeSplit {
        return candidateSplits(points).minBy { split ->
            weightedImpurity(points, split.first, split.second, criterion)
        }.let { (feature, threshold) ->
            val impurity = weightedImpurity(points, feature, threshold, criterion)
            val left = points.count { featureValue(it, feature) <= threshold }
            TreeSplit(feature, threshold, impurity, left, points.size - left, "Lower impurity means child nodes contain cleaner class groups.")
        }
    }

    fun bestRegressionSplit(points: List<LabPoint>): TreeSplit {
        return candidateSplits(points).minBy { split ->
            weightedVariance(points, split.first, split.second)
        }.let { (feature, threshold) ->
            val impurity = weightedVariance(points, feature, threshold)
            val left = points.count { featureValue(it, feature) <= threshold }
            TreeSplit(feature, threshold, impurity, left, points.size - left, "The split is chosen because the leaf means reduce squared target error.")
        }
    }

    fun impurity(labels: List<Int>, criterion: SplitCriterion): Double {
        if (labels.isEmpty()) return 0.0
        val probabilities = labels.groupingBy { it }.eachCount().values.map { it / labels.size.toDouble() }
        return when (criterion) {
            SplitCriterion.Gini -> 1.0 - probabilities.sumOf { it * it }
            SplitCriterion.Entropy -> probabilities.sumOf { p -> if (p == 0.0) 0.0 else -p * ln(p) / ln(2.0) }
        }
    }

    private fun regressionFit(
        points: List<LabPoint>,
        weights: List<Double>,
        bias: Double,
        penalty: Double = 0.0,
        predictor: (LabPoint, List<Double>) -> Double
    ): RegressionFit {
        val predictions = points.map { predictor(it, weights) }
        val mean = points.map { it.target }.average()
        val errors = points.zip(predictions).map { (point, prediction) -> prediction - point.target }
        val mse = errors.sumOf { it * it } / errors.size.coerceAtLeast(1)
        val mae = errors.sumOf { abs(it) } / errors.size.coerceAtLeast(1)
        val total = points.sumOf { (it.target - mean).pow(2) }
        val r2 = if (total <= 1e-12) 1.0 else 1.0 - errors.sumOf { it * it } / total
        return RegressionFit(weights, bias, predictions, mse + penalty, mae, sqrt(mse), r2, penalty)
    }

    private fun solveLeastSquares(points: List<LabPoint>, degree: Int, extraFeature: Boolean, ridge: Double): List<Double> {
        val rows = points.map { p ->
            if (extraFeature) listOf(p.x, p.y, 1.0) else List(degree + 1) { power -> p.x.pow(power) }
        }
        val cols = rows.first().size
        val matrix = Array(cols) { r -> DoubleArray(cols) { c -> rows.sumOf { it[r] * it[c] } } }
        val target = DoubleArray(cols) { r -> rows.indices.sumOf { i -> rows[i][r] * points[i].target } }
        for (i in 0 until cols) if (!(extraFeature && i == cols - 1) && i != 0) matrix[i][i] += ridge
        return gaussianSolve(matrix, target).toList()
    }

    private fun coordinateDescent(points: List<LabPoint>, l1: Double, l2: Double): List<Double> {
        val degree = 3
        val features = points.map { p -> DoubleArray(degree + 1) { power -> p.x.pow(power) } }
        val coefficients = DoubleArray(degree + 1)
        repeat(90) {
            for (j in coefficients.indices) {
                var numerator = 0.0
                var denominator = 0.0
                for (i in points.indices) {
                    val residual = points[i].target - coefficients.indices.filter { it != j }.sumOf { k -> coefficients[k] * features[i][k] }
                    numerator += features[i][j] * residual
                    denominator += features[i][j] * features[i][j]
                }
                coefficients[j] = if (j == 0) {
                    numerator / denominator.coerceAtLeast(1e-9)
                } else {
                    softThreshold(numerator, l1) / (denominator + l2).coerceAtLeast(1e-9)
                }
            }
        }
        return coefficients.toList()
    }

    private fun gaussianSolve(matrix: Array<DoubleArray>, target: DoubleArray): DoubleArray {
        val n = target.size
        for (pivot in 0 until n) {
            val best = (pivot until n).maxBy { abs(matrix[it][pivot]) }
            val row = matrix[pivot]
            matrix[pivot] = matrix[best]
            matrix[best] = row
            val targetSwap = target[pivot]
            target[pivot] = target[best]
            target[best] = targetSwap
            val divisor = matrix[pivot][pivot].takeIf { abs(it) > 1e-9 } ?: 1e-9
            for (col in pivot until n) matrix[pivot][col] /= divisor
            target[pivot] /= divisor
            for (rowIndex in 0 until n) {
                if (rowIndex == pivot) continue
                val factor = matrix[rowIndex][pivot]
                for (col in pivot until n) matrix[rowIndex][col] -= factor * matrix[pivot][col]
                target[rowIndex] -= factor * target[pivot]
            }
        }
        return target
    }

    private fun meanSquaredError(points: List<LabPoint>, weight: Double, bias: Double): Double =
        points.sumOf { (weight * it.x + bias - it.target).pow(2) } / points.size.coerceAtLeast(1)

    private fun distance(a: LabPoint, b: LabPoint, metric: DistanceMetric): Double = when (metric) {
        DistanceMetric.Euclidean -> sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
        DistanceMetric.Manhattan -> abs(a.x - b.x) + abs(a.y - b.y)
    }

    private fun candidateSplits(points: List<LabPoint>): List<Pair<String, Double>> =
        listOf("x", "y").flatMap { feature ->
            points.map { featureValue(it, feature) }.distinct().sorted().zipWithNext { a, b -> feature to (a + b) / 2.0 }
        }

    private fun weightedImpurity(points: List<LabPoint>, feature: String, threshold: Double, criterion: SplitCriterion): Double {
        val left = points.filter { featureValue(it, feature) <= threshold }
        val right = points - left.toSet()
        return (left.size * impurity(left.map { it.label }, criterion) + right.size * impurity(right.map { it.label }, criterion)) / points.size
    }

    private fun weightedVariance(points: List<LabPoint>, feature: String, threshold: Double): Double {
        val left = points.filter { featureValue(it, feature) <= threshold }
        val right = points - left.toSet()
        fun sse(group: List<LabPoint>): Double {
            if (group.isEmpty()) return 0.0
            val mean = group.map { it.target }.average()
            return group.sumOf { (it.target - mean).pow(2) }
        }
        return (sse(left) + sse(right)) / points.size
    }

    private fun featureValue(point: LabPoint, feature: String): Double = if (feature == "x") point.x else point.y
    private fun softThreshold(value: Double, alpha: Double): Double = when {
        value > alpha -> value - alpha
        value < -alpha -> value + alpha
        else -> 0.0
    }
}
