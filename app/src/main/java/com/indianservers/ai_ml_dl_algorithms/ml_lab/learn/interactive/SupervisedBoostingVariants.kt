package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.random.Random

private fun probability(score: Double) = 1.0 / (1.0 + exp(-score.coerceIn(-30.0, 30.0)))

internal object SupervisedBoostingVariants {
    fun fit(
        points: List<LabPoint>, regression: Boolean, variant: BoostingVariant,
        stages: Int, rate: Double
    ): BoostingResult = when (variant) {
        BoostingVariant.Ada -> ada(points, regression, stages, rate)
        BoostingVariant.Xgboost -> xgboost(points, regression, stages, rate)
        BoostingVariant.Lightgbm -> lightgbm(points, regression, stages, rate)
        BoostingVariant.Catboost -> catboost(points, regression, stages, rate)
        BoostingVariant.Gradient -> error("Gradient boosting is handled by SupervisedBoostingEngine")
    }

    private fun targets(points: List<LabPoint>, regression: Boolean) =
        points.map { if (regression) it.target else it.label.toDouble() }

    private fun initial(labels: List<Double>, regression: Boolean): Double {
        if (regression) return labels.average()
        val prior = labels.average().coerceIn(.001, .999)
        return ln(prior / (1 - prior))
    }

    private fun feature(point: LabPoint, index: Int) = if (index == 0) point.x else point.y

    private fun leaf(
        indices: List<Int>, depth: Int, prediction: Double, impurity: Double = 0.0
    ) = SupervisedTreeNode(indices, depth, prediction, impurity, emptyMap())

    private fun weightedStump(
        points: List<LabPoint>, weights: List<Double>
    ): Pair<SupervisedTreeNode, Double> {
        var best: SupervisedTreeNode? = null
        var lowestError = Double.POSITIVE_INFINITY
        for (axis in 0..1) {
            val candidates = points.map { feature(it, axis) }.distinct().sorted()
                .zipWithNext { a, b -> (a + b) / 2.0 }
            for (threshold in candidates) {
                val left = points.indices.filter { feature(points[it], axis) <= threshold }
                val right = points.indices.filter { feature(points[it], axis) > threshold }
                if (left.isEmpty() || right.isEmpty()) continue
                fun vote(indices: List<Int>): Double =
                    if (indices.sumOf { weights[it] * if (points[it].label == 1) 1.0 else -1.0 } >= 0) 1.0 else -1.0
                fun impurity(indices: List<Int>): Double {
                    val total = indices.sumOf { weights[it] }.coerceAtLeast(1e-10)
                    val positive = indices.sumOf {
                        if (points[it].label == 1) weights[it] else 0.0
                    } / total
                    return 2 * positive * (1 - positive)
                }
                val leftVote = vote(left)
                val rightVote = vote(right)
                val errors = points.indices.sumOf { i ->
                    val predicted = if (i in left) leftVote else rightVote
                    if (predicted != (if (points[i].label == 1) 1.0 else -1.0)) weights[i] else 0.0
                }
                if (errors < lowestError) {
                    lowestError = errors
                    best = leaf(points.indices.toList(), 0, 0.0,
                        impurity(points.indices.toList())).copy(
                        feature = axis, threshold = threshold, gain = .5 - errors,
                        left = leaf(left, 1, leftVote, impurity(left)),
                        right = leaf(right, 1, rightVote, impurity(right))
                    )
                }
            }
        }
        return (best ?: leaf(points.indices.toList(), 0, 1.0)) to lowestError
    }

    private fun weightedSample(
        points: List<LabPoint>, weights: List<Double>, random: Random
    ): List<LabPoint> = List(points.size) {
        val draw = random.nextDouble()
        var sum = 0.0
        val index = weights.indices.firstOrNull { i ->
            sum += weights[i]
            sum >= draw
        } ?: weights.lastIndex
        points[index]
    }

    private fun ada(points: List<LabPoint>, regression: Boolean, stageCount: Int, rate: Double): BoostingResult {
        val labels = targets(points, regression)
        var weights = List(points.size) { 1.0 / points.size }
        val stages = mutableListOf<SupervisedBoostStage>()
        repeat(stageCount.coerceIn(1, 12)) { index ->
            val tree: SupervisedTreeNode
            val normalizedErrors: List<Double>
            val weightedError: Double
            if (regression) {
                val sample = weightedSample(points, weights, Random(119 + index * 71))
                tree = SupervisedTreeEngine.fit(sample, true, 2, 2, seed = index + 17)
                val errors = points.map { abs(it.target - tree.predict(it.x, it.y)) }
                val scale = errors.maxOrNull()?.coerceAtLeast(1e-9) ?: 1.0
                normalizedErrors = errors.map { it / scale }
                weightedError = normalizedErrors.indices.sumOf { weights[it] * normalizedErrors[it] }
            } else {
                val result = weightedStump(points, weights)
                tree = result.first
                weightedError = result.second
                normalizedErrors = points.map {
                    if (tree.predict(it.x, it.y) == if (it.label == 1) 1.0 else -1.0) 0.0 else 1.0
                }
            }
            val error = weightedError.coerceIn(1e-6, .499)
            val beta = error / (1.0 - error)
            val contribution = rate * (if (regression) ln(1.0 / beta) else .5 * ln(1.0 / beta))
            val updated = if (regression) weights.indices.map {
                weights[it] * beta.pow(rate * (1.0 - normalizedErrors[it]))
            } else weights.indices.map {
                weights[it] * exp(-contribution *
                    (if (points[it].label == 1) 1.0 else -1.0) *
                    tree.predict(points[it].x, points[it].y))
            }
            val total = updated.sum().coerceAtLeast(1e-12)
            weights = updated.map { it / total }
            stages += SupervisedBoostStage(
                tree, points.map { tree.predict(it.x, it.y) },
                normalizedErrors, weights, tree.gain,
                if (regression) "Weighted median uses a depth-2 learner; error weights focus the next sample."
                else "Weighted stump; misclassified samples receive more weight next round.",
                contribution
            )
        }
        return BoostingResult(if (regression) labels.average() else 0.0,
            stages, regression, rate, BoostingVariant.Ada, points = points)
    }

    private fun newtonTree(
        points: List<LabPoint>, gradients: List<Double>, hessians: List<Double>,
        regression: Boolean, maxDepth: Int = 2, lambda: Double = 1.0
    ): SupervisedTreeNode {
        fun build(indices: List<Int>, depth: Int): SupervisedTreeNode {
            val g = indices.sumOf { gradients[it] }
            val h = indices.sumOf { hessians[it] }
            val base = leaf(indices, depth, -g / (h + lambda))
            if (depth >= maxDepth || indices.size < 4) return base
            var bestGain = 0.0
            var bestFeature = -1
            var bestThreshold = 0.0
            var bestLeft = emptyList<Int>()
            var bestRight = emptyList<Int>()
            for (feature in 0..if (regression) 0 else 1) {
                val sorted = indices.map { this.feature(points[it], feature) }.distinct().sorted()
                for (threshold in sorted.zipWithNext { a, b -> (a + b) / 2.0 }) {
                    val left = indices.filter { this.feature(points[it], feature) <= threshold }
                    val right = indices.filter { this.feature(points[it], feature) > threshold }
                    if (left.size < 2 || right.size < 2) continue
                    val gl = left.sumOf { gradients[it] }; val hl = left.sumOf { hessians[it] }
                    val gr = g - gl; val hr = h - hl
                    val gain = .5 * (gl * gl / (hl + lambda) + gr * gr / (hr + lambda) -
                        g * g / (h + lambda)) - .005
                    if (gain > bestGain) {
                        bestGain = gain
                        bestFeature = feature
                        bestThreshold = threshold
                        bestLeft = left
                        bestRight = right
                    }
                }
            }
            return if (bestFeature < 0) base else base.copy(
                feature = bestFeature, threshold = bestThreshold, gain = bestGain,
                left = build(bestLeft, depth + 1), right = build(bestRight, depth + 1)
            )
        }
        return build(points.indices.toList(), 0)
    }

    private fun xgboost(
        points: List<LabPoint>, regression: Boolean, count: Int, rate: Double
    ): BoostingResult {
        val labels = targets(points, regression)
        val start = initial(labels, regression)
        val scores = MutableList(points.size) { start }
        val stages = mutableListOf<SupervisedBoostStage>()
        repeat(count.coerceIn(1, 12)) { index ->
            val gradients = points.indices.map { i ->
                if (regression) scores[i] - labels[i] else probability(scores[i]) - labels[i]
            }
            val hessians = points.indices.map { i ->
                if (regression) 1.0 else {
                    val p = probability(scores[i])
                    (p * (1.0 - p)).coerceAtLeast(.01)
                }
            }
            val tree = newtonTree(points, gradients, hessians, regression)
            points.indices.forEach { i -> scores[i] += rate * tree.predict(points[i].x, points[i].y) }
            stages += SupervisedBoostStage(tree, scores.toList(), gradients.map { -it },
                List(points.size) { 1.0 / points.size }, tree.gain,
                "Regularized split gain uses gradients and Hessians; leaves apply Newton updates.",
                gradients = gradients, hessians = hessians)
        }
        return BoostingResult(start, stages, regression, rate, BoostingVariant.Xgboost, points = points)
    }

    private class GrowingNode(val indices: List<Int>, val depth: Int, val prediction: Double) {
        var splitFeature: Int? = null
        var threshold: Double? = null
        var gain = 0.0
        var left: GrowingNode? = null
        var right: GrowingNode? = null
    }

    private fun leafWiseTree(
        points: List<LabPoint>, residuals: List<Double>, regression: Boolean
    ): SupervisedTreeNode {
        fun mean(indices: List<Int>) = indices.map { residuals[it] }.average()
        fun error(indices: List<Int>): Double {
            val average = mean(indices)
            return indices.sumOf { (residuals[it] - average).pow(2) }
        }
        val root = GrowingNode(points.indices.toList(), 0, residuals.average())
        val leaves = mutableListOf(root)
        repeat(5) {
            var bestLeaf: GrowingNode? = null
            var bestFeature = -1
            var bestThreshold = 0.0
            var bestGain = 0.0
            var bestLeft = emptyList<Int>()
            var bestRight = emptyList<Int>()
            leaves.filter { it.depth < 4 && it.indices.size >= 5 }.forEach { candidate ->
                val parentError = error(candidate.indices)
                for (feature in 0..if (regression) 0 else 1) {
                    val values = candidate.indices.map { this.feature(points[it], feature) }
                    val minimum = values.minOrNull() ?: continue
                    val maximum = values.maxOrNull() ?: continue
                    for (bin in 1..12) {
                        val threshold = minimum + (maximum - minimum) * bin / 13.0
                        val left = candidate.indices.filter { this.feature(points[it], feature) <= threshold }
                        val right = candidate.indices.filter { this.feature(points[it], feature) > threshold }
                        if (left.size < 2 || right.size < 2) continue
                        val gain = parentError - error(left) - error(right)
                        if (gain > bestGain + 1e-7) {
                            bestLeaf = candidate
                            bestFeature = feature
                            bestThreshold = threshold
                            bestGain = gain
                            bestLeft = left
                            bestRight = right
                        }
                    }
                }
            }
            val chosen = bestLeaf ?: return@repeat
            chosen.splitFeature = bestFeature
            chosen.threshold = bestThreshold
            chosen.gain = bestGain
            chosen.left = GrowingNode(bestLeft, chosen.depth + 1, mean(bestLeft))
            chosen.right = GrowingNode(bestRight, chosen.depth + 1, mean(bestRight))
            leaves.remove(chosen)
            leaves += chosen.left!!
            leaves += chosen.right!!
        }
        fun freeze(node: GrowingNode): SupervisedTreeNode = leaf(
            node.indices, node.depth, node.prediction,
            error(node.indices) / node.indices.size
        ).copy(feature = node.splitFeature, threshold = node.threshold, gain = node.gain,
            left = node.left?.let(::freeze), right = node.right?.let(::freeze))
        return freeze(root)
    }

    private fun lightgbm(
        points: List<LabPoint>, regression: Boolean, count: Int, rate: Double
    ): BoostingResult {
        val labels = targets(points, regression)
        val start = initial(labels, regression)
        val scores = MutableList(points.size) { start }
        val stages = mutableListOf<SupervisedBoostStage>()
        repeat(count.coerceIn(1, 12)) {
            val residuals = points.indices.map { i ->
                if (regression) labels[i] - scores[i] else labels[i] - probability(scores[i])
            }
            val tree = leafWiseTree(points, residuals, regression)
            points.indices.forEach { i -> scores[i] += rate * tree.predict(points[i].x, points[i].y) }
            stages += SupervisedBoostStage(tree, scores.toList(), residuals,
                List(points.size) { 1.0 / points.size }, tree.gain,
                "Histogram thresholds; the highest-gain leaf grows next.")
        }
        return BoostingResult(start, stages, regression, rate, BoostingVariant.Lightgbm, points = points)
    }

    private fun symmetricTree(
        points: List<LabPoint>, residuals: List<Double>
    ): SupervisedTreeNode {
        var groups = listOf(points.indices.toList())
        val splits = mutableListOf<Pair<Int, Double>>()
        repeat(2) {
            var bestGain = 0.0
            var best: Pair<Int, Double>? = null
            for (feature in 0..1) {
                val sorted = points.map { this.feature(it, feature) }.distinct().sorted()
                for (threshold in sorted.zipWithNext { a, b -> (a + b) / 2.0 }) {
                    var gain = 0.0
                    groups.forEach { indices ->
                        if (indices.size < 4) return@forEach
                        val left = indices.filter { this.feature(points[it], feature) <= threshold }
                        val right = indices.filter { this.feature(points[it], feature) > threshold }
                        if (left.size < 2 || right.size < 2) return@forEach
                        fun sse(items: List<Int>): Double {
                            val average = items.map { residuals[it] }.average()
                            return items.sumOf { (residuals[it] - average).pow(2) }
                        }
                        gain += sse(indices) - sse(left) - sse(right)
                    }
                    if (gain > bestGain + 1e-8) {
                        bestGain = gain
                        best = feature to threshold
                    }
                }
            }
            val split = best ?: return@repeat
            splits += split
            groups = groups.flatMap { indices ->
                listOf(
                    indices.filter { this.feature(points[it], split.first) <= split.second },
                    indices.filter { this.feature(points[it], split.first) > split.second }
                )
            }
        }
        fun build(indices: List<Int>, depth: Int, parent: Double): SupervisedTreeNode {
            val average = if (indices.isEmpty()) parent else indices.map { residuals[it] }.average()
            val impurity = if (indices.isEmpty()) 0.0 else
                indices.sumOf { (residuals[it] - average).pow(2) } / indices.size
            val base = leaf(indices, depth, average, impurity)
            val split = splits.getOrNull(depth) ?: return base
            val left = indices.filter { this.feature(points[it], split.first) <= split.second }
            val right = indices.filter { this.feature(points[it], split.first) > split.second }
            return base.copy(feature = split.first, threshold = split.second,
                left = build(left, depth + 1, average),
                right = build(right, depth + 1, average))
        }
        return build(points.indices.toList(), 0, residuals.average())
    }

    private fun catboost(
        points: List<LabPoint>, regression: Boolean, count: Int, rate: Double
    ): BoostingResult {
        val labels = targets(points, regression)
        val start = initial(labels, regression)
        val order = points.indices.shuffled(Random(43))
        val sums = mutableMapOf<Double, Double>()
        val counts = mutableMapOf<Double, Int>()
        val ordered = DoubleArray(points.size)
        order.forEach { index ->
            val category = points[index].x
            ordered[index] = ((sums[category] ?: 0.0) + 2 * labels.average()) /
                ((counts[category] ?: 0) + 2)
            sums[category] = (sums[category] ?: 0.0) + labels[index]
            counts[category] = (counts[category] ?: 0) + 1
        }
        val encoding = sums.mapValues { (category, sum) ->
            (sum + 2 * labels.average()) / ((counts[category] ?: 0) + 2)
        }
        val encoded = points.indices.map { points[it].copy(x = ordered[it]) }
        val scores = MutableList(points.size) { start }
        val stages = mutableListOf<SupervisedBoostStage>()
        repeat(count.coerceIn(1, 12)) {
            val residuals = points.indices.map { i ->
                if (regression) labels[i] - scores[i] else labels[i] - probability(scores[i])
            }
            val tree = symmetricTree(encoded, residuals)
            points.indices.forEach { i -> scores[i] += rate * tree.predict(encoded[i].x, encoded[i].y) }
            stages += SupervisedBoostStage(tree, scores.toList(), residuals,
                List(points.size) { 1.0 / points.size }, tree.gain,
                "Ordered category statistics feed a symmetric depth-2 tree.")
        }
        return BoostingResult(start, stages, regression, rate, BoostingVariant.Catboost, encoding, points)
    }
}
