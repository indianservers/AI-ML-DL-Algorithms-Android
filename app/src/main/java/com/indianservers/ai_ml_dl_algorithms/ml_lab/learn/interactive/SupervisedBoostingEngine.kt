package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.exp
import kotlin.math.ln

private fun boostedSigmoid(value: Double) = 1.0 / (1.0 + exp(-value.coerceIn(-30.0, 30.0)))

internal enum class BoostingVariant { Gradient, Ada, Xgboost, Lightgbm, Catboost }

internal data class SupervisedBoostStage(
    val tree: SupervisedTreeNode,
    val predictions: List<Double>,
    val residuals: List<Double>,
    val sampleWeights: List<Double>,
    val gain: Double,
    val description: String,
    val contribution: Double = 1.0,
    val gradients: List<Double> = emptyList(),
    val hessians: List<Double> = emptyList()
)

internal data class BoostingResult(
    val initial: Double,
    val stages: List<SupervisedBoostStage>,
    val regression: Boolean,
    val rate: Double,
    val variant: BoostingVariant,
    val categoryEncoding: Map<Double, Double> = emptyMap(),
    val points: List<LabPoint> = emptyList()
) {
    fun predict(x: Double, y: Double, stageCount: Int = stages.size): Double {
        val active = stages.take(stageCount)
        if (active.isEmpty()) return if (regression) initial else boostedSigmoid(initial)
        val feature = if (variant == BoostingVariant.Catboost && categoryEncoding.isNotEmpty())
            categoryEncoding.minBy { kotlin.math.abs(it.key - x) }.value else x
        if (variant == BoostingVariant.Ada && regression) {
            val predictions = active.map { it.tree.predict(feature, y) to it.contribution }
                .sortedBy { it.first }
            val midpoint = predictions.sumOf { it.second } / 2.0
            var cumulative = 0.0
            return predictions.firstOrNull {
                cumulative += it.second
                cumulative >= midpoint
            }?.first ?: initial
        }
        val score = initial + if (variant == BoostingVariant.Ada)
            active.sumOf { it.contribution * it.tree.predict(feature, y) }
        else rate * active.sumOf { it.tree.predict(feature, y) }
        return if (regression) score else boostedSigmoid(if (variant == BoostingVariant.Ada) score * 2 else score)
    }
}

internal object SupervisedBoostingEngine {
    fun fit(
        points: List<LabPoint>,
        regression: Boolean,
        variant: BoostingVariant,
        stages: Int,
        rate: Double
    ): BoostingResult {
        require(points.size >= 4)
        if (variant != BoostingVariant.Gradient)
            return SupervisedBoostingVariants.fit(points, regression, variant, stages, rate)
        val labels = points.map { if (regression) it.target else it.label.toDouble() }
        val prior = labels.average().coerceIn(0.001, 0.999)
        val initial = if (regression) labels.average() else ln(prior / (1.0 - prior))
        val scores = MutableList(points.size) { initial }
        val result = mutableListOf<SupervisedBoostStage>()
        repeat(stages.coerceIn(1, 12)) { index ->
            val residuals = points.indices.map { i ->
                if (regression) labels[i] - scores[i] else labels[i] - boostedSigmoid(scores[i])
            }
            val training = points.indices.map { i -> points[i].copy(target = residuals[i]) }
            val tree = SupervisedTreeEngine.fit(training, regression = true, maxDepth = 1,
                minSamples = 2, seed = 71 + index)
            for (i in points.indices) scores[i] += rate * tree.predict(points[i].x, points[i].y)
            result += SupervisedBoostStage(tree, scores.toList(), residuals,
                List(points.size) { 1.0 / points.size }, tree.gain,
                "Tree ${index + 1} fits the current residuals.")
        }
        return BoostingResult(initial, result, regression, rate, variant, points = points)
    }

}
