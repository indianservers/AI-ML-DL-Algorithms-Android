package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.ln
import kotlin.math.pow
import kotlin.random.Random

internal data class SupervisedTreeNode(
    val indices: List<Int>,
    val depth: Int,
    val prediction: Double,
    val impurity: Double,
    val counts: Map<Int, Int>,
    val feature: Int? = null,
    val threshold: Double? = null,
    val gain: Double = 0.0,
    val left: SupervisedTreeNode? = null,
    val right: SupervisedTreeNode? = null
) {
    val isLeaf: Boolean get() = feature == null
    fun predict(x: Double, y: Double): Double {
        val f = feature ?: return prediction
        val child = if ((if (f == 0) x else y) <= (threshold ?: 0.0)) left else right
        return child?.predict(x, y) ?: prediction
    }
    fun nodes(): List<SupervisedTreeNode> = listOf(this) + left.orEmptyNodes() + right.orEmptyNodes()
    private fun SupervisedTreeNode?.orEmptyNodes() = this?.nodes().orEmpty()
}

internal object SupervisedTreeEngine {
    fun fit(
        points: List<LabPoint>,
        regression: Boolean,
        maxDepth: Int = 3,
        minSamples: Int = 3,
        criterion: SplitCriterion = SplitCriterion.Gini,
        randomThresholds: Boolean = false,
        featureSubset: Int = 2,
        seed: Int = 41
    ): SupervisedTreeNode {
        require(points.isNotEmpty())
        val random = Random(seed)
        fun value(index: Int, feature: Int) = if (feature == 0) points[index].x else points[index].y
        fun target(index: Int) = if (regression) points[index].target else points[index].label.toDouble()
        fun impurity(indices: List<Int>): Double {
            if (indices.isEmpty()) return 0.0
            if (regression) {
                val mean = indices.sumOf(::target) / indices.size
                return indices.sumOf { (target(it) - mean).pow(2) } / indices.size
            }
            val frequencies = indices.groupingBy { points[it].label }.eachCount()
            return when (criterion) {
                SplitCriterion.Gini -> 1.0 - frequencies.values.sumOf { (it / indices.size.toDouble()).pow(2) }
                SplitCriterion.Entropy -> -frequencies.values.sumOf {
                    val p = it / indices.size.toDouble()
                    p * ln(p) / ln(2.0)
                }
            }
        }
        fun build(indices: List<Int>, depth: Int): SupervisedTreeNode {
            val parentImpurity = impurity(indices)
            val counts = indices.groupingBy { points[it].label }.eachCount()
            val prediction = if (regression) indices.sumOf(::target) / indices.size
                else counts.maxWithOrNull(compareBy<Map.Entry<Int, Int>> { it.value }.thenBy { -it.key })!!.key.toDouble()
            val leaf = SupervisedTreeNode(indices, depth, prediction, parentImpurity, counts)
            if (depth >= maxDepth || indices.size < minSamples * 2 || parentImpurity < 1e-10) return leaf

            val features = if (regression) listOf(0)
                else if (featureSubset >= 2) listOf(0, 1) else listOf(random.nextInt(2))
            // A zero-gain split can still expose a useful split at the next
            // level (the classic XOR example). Never accept negative gain.
            var bestGain = -1e-8
            var bestFeature = -1
            var bestThreshold = 0.0
            var bestLeft = emptyList<Int>()
            var bestRight = emptyList<Int>()
            for (feature in features) {
                val sorted = indices.map { value(it, feature) }.distinct().sorted()
                val candidates = if (randomThresholds) {
                    if (sorted.size < 2) emptyList() else List(4) {
                        sorted.first() + random.nextDouble() * (sorted.last() - sorted.first())
                    }
                } else sorted.zipWithNext { a, b -> (a + b) / 2.0 }
                    .let { values -> if (values.size <= 32) values else values.filterIndexed { i, _ -> i % (values.size / 32 + 1) == 0 } }
                for (threshold in candidates) {
                    val left = indices.filter { value(it, feature) <= threshold }
                    val right = indices.filter { value(it, feature) > threshold }
                    if (left.size < minSamples || right.size < minSamples) continue
                    val weighted = (left.size * impurity(left) + right.size * impurity(right)) / indices.size
                    val gain = parentImpurity - weighted
                    if (gain > bestGain + 1e-10 ||
                        (kotlin.math.abs(gain - bestGain) <= 1e-10 &&
                            minOf(left.size, right.size) > minOf(bestLeft.size, bestRight.size))) {
                        bestGain = gain
                        bestFeature = feature
                        bestThreshold = threshold
                        bestLeft = left
                        bestRight = right
                    }
                }
            }
            if (bestFeature < 0) return leaf
            return leaf.copy(
                feature = bestFeature, threshold = bestThreshold, gain = bestGain,
                left = build(bestLeft, depth + 1), right = build(bestRight, depth + 1)
            )
        }
        return build(points.indices.toList(), 0)
    }

    fun forest(
        points: List<LabPoint>,
        regression: Boolean,
        trees: Int,
        maxDepth: Int,
        extraTrees: Boolean,
        featureSubset: Int = 1,
        seed: Int = 17,
        minSamples: Int = 2,
        bootstrap: Boolean = !extraTrees
    ): List<SupervisedTreeNode> {
        require(points.isNotEmpty())
        return List(trees.coerceIn(1, 30)) { member ->
            val random = Random(seed + member * 997)
            val training = if (!bootstrap) points else List(points.size) { points[random.nextInt(points.size)] }
            fit(training, regression, maxDepth, minSamples = minSamples,
                randomThresholds = extraTrees, featureSubset = featureSubset, seed = seed + member * 37)
        }
    }

    fun forestPrediction(
        forest: List<SupervisedTreeNode>, x: Double, y: Double, regression: Boolean
    ): Double {
        require(forest.isNotEmpty())
        val outputs = forest.map { it.predict(x, y) }
        return if (regression) outputs.average()
        else outputs.map { it.toInt() }.groupingBy { it }.eachCount()
            .maxWithOrNull(compareBy<Map.Entry<Int, Int>> { it.value }.thenBy { -it.key })!!.key.toDouble()
    }
}
