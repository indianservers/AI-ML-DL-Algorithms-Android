package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.DatasetPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.DistanceMetric
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.LabPoint
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseOneDatasets
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseOneEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.SplitCriterion
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.TrainingMode
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseOneInteractiveEngineTest {
    @Test
    fun linearRegressionRecoversKnownCoefficients() {
        val points = listOf(
            LabPoint(-1.0, -2.0),
            LabPoint(0.0, 1.0),
            LabPoint(1.0, 4.0),
            LabPoint(2.0, 7.0)
        )
        val fit = PhaseOneEngines.fitSimpleLinear(points)
        assertEquals(3.0, fit.weights.single(), 1e-6)
        assertEquals(1.0, fit.bias, 1e-6)
        assertEquals(0.0, fit.mse, 1e-6)
    }

    @Test
    fun logisticSigmoidAndConfusionMetricsAreCorrect() {
        assertEquals(0.5, PhaseOneEngines.sigmoid(0.0), 1e-12)
        assertTrue(PhaseOneEngines.sigmoid(4.0) > PhaseOneEngines.sigmoid(-4.0))
        val points = listOf(
            LabPoint(-1.0, -1.0, 0, 0.0),
            LabPoint(-0.8, -0.4, 0, 0.0),
            LabPoint(0.8, 0.4, 1, 1.0),
            LabPoint(1.0, 1.0, 1, 1.0)
        )
        val metrics = PhaseOneEngines.logisticMetrics(points, 3.0, 3.0, 0.0, 0.5)
        assertEquals(1.0, metrics.accuracy, 0.0)
        assertEquals(2, metrics.tp)
        assertEquals(2, metrics.tn)
    }

    @Test
    fun knnClassifiesFromNearestNeighborsWithBothDistances() {
        val points = listOf(
            LabPoint(-1.0, 0.0, 0),
            LabPoint(-0.8, 0.1, 0),
            LabPoint(0.9, 0.1, 1),
            LabPoint(1.0, 0.0, 1)
        )
        val query = LabPoint(0.85, 0.0)
        assertEquals(1, PhaseOneEngines.knn(points, query, 3, DistanceMetric.Euclidean).first)
        assertEquals(1, PhaseOneEngines.knn(points, query, 3, DistanceMetric.Manhattan).first)
    }

    @Test
    fun decisionTreeImpurityAndSplitPreferCleanerChildren() {
        assertEquals(0.5, PhaseOneEngines.impurity(listOf(0, 0, 1, 1), SplitCriterion.Gini), 1e-9)
        assertEquals(1.0, PhaseOneEngines.impurity(listOf(0, 0, 1, 1), SplitCriterion.Entropy), 1e-9)
        val points = listOf(
            LabPoint(-0.8, 0.0, 0),
            LabPoint(-0.5, 0.1, 0),
            LabPoint(0.4, 0.0, 1),
            LabPoint(0.9, 0.1, 1)
        )
        val split = PhaseOneEngines.bestClassificationSplit(points, SplitCriterion.Gini)
        assertEquals("x", split.feature)
        assertTrue(split.impurity < 0.01)
    }

    @Test
    fun gradientDescentUpdateLowersLossOnLinearDataset() {
        val points = PhaseOneDatasets.generate(DatasetPreset.PerfectLinear, samples = 24, noise = 0.0)
        val path = PhaseOneEngines.gradientPath(points, learningRate = 0.08, iterations = 30, mode = TrainingMode.Batch)
        assertTrue(path.last().loss < path.first().loss)
        assertTrue(path.all { it.loss.isFinite() && it.gradW.isFinite() && it.gradB.isFinite() })
    }

    @Test
    fun regularizationPenaltiesAndCoefficientBehaviorAreDeterministic() {
        val points = PhaseOneDatasets.generate(DatasetPreset.Polynomial, samples = 40, noise = 0.03)
        val ridgeLow = PhaseOneEngines.fitRidge(points, 0.0)
        val ridgeHigh = PhaseOneEngines.fitRidge(points, 4.0)
        val lasso = PhaseOneEngines.fitLasso(points, 8.0)
        val elastic = PhaseOneEngines.fitElasticNet(points, 1.5, 0.5)
        assertTrue(ridgeHigh.penalty >= ridgeLow.penalty)
        assertTrue(ridgeHigh.weights.sumOf { kotlin.math.abs(it) } <= ridgeLow.weights.sumOf { kotlin.math.abs(it) } + 1e-6)
        assertTrue(lasso.weights.any { kotlin.math.abs(it) < 0.05 })
        assertTrue(elastic.penalty > 0.0)
    }

    @Test
    fun visualizationStateContainsExpectedLineAndNeighbors() {
        val points = PhaseOneDatasets.generate(DatasetPreset.LinearNoise, samples = 20, noise = 0.0)
        val fit = PhaseOneEngines.fitSimpleLinear(points, weight = 0.5, bias = 0.1)
        assertEquals(points.size, fit.predictions.size)
        assertEquals(0.5 * points.first().x + 0.1, fit.predictions.first(), 1e-9)
        val classes = PhaseOneDatasets.generate(DatasetPreset.TwoClusters, samples = 20, noise = 0.0)
        val (_, neighbours) = PhaseOneEngines.knn(classes, LabPoint(-0.5, -0.2), 5, DistanceMetric.Euclidean)
        assertEquals(5, neighbours.size)
        assertTrue(neighbours.all { it.second >= 0.0 })
    }
}
