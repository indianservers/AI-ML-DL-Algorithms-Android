package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.*
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs
import kotlin.math.pow

class SupervisedVisualizationEngineTest {
    private val xor = listOf(
        LabPoint(-.7, -.7, 1), LabPoint(-.7, .7, 0),
        LabPoint(.7, -.7, 0), LabPoint(.7, .7, 1),
        LabPoint(-.6, -.6, 1), LabPoint(-.6, .6, 0),
        LabPoint(.6, -.6, 0), LabPoint(.6, .6, 1)
    )

    @Test fun recursiveTreePartitionsXorIntoFourLeaves() {
        val tree = SupervisedTreeEngine.fit(xor, false, maxDepth = 3, minSamples = 1)
        assertTrue(tree.nodes().count { it.isLeaf } >= 4)
        xor.forEach { assertEquals(it.label.toDouble(), tree.predict(it.x, it.y), 0.0) }
        assertTrue(tree.pathFor(-.7, -.7) != tree.pathFor(.7, -.7))
    }

    @Test fun entropyAndGiniReactToClassPurity() {
        val pure = listOf(LabPoint(-.7, -.7, 0), LabPoint(-.5, -.5, 0),
            LabPoint(.5, .5, 1), LabPoint(.7, .7, 1))
        val gini = SupervisedTreeEngine.fit(pure, false, 2, 1, SplitCriterion.Gini)
        val entropy = SupervisedTreeEngine.fit(pure, false, 2, 1, SplitCriterion.Entropy)
        assertTrue(gini.gain > 0.0)
        assertTrue(entropy.gain > 0.0)
        assertEquals(0.0, gini.left!!.impurity, 0.0)
        assertEquals(0.0, entropy.right!!.impurity, 0.0)
    }

    @Test fun regressionTreeIsPiecewiseConstantAndForestAggregatesMembers() {
        val data = (-8..8).map {
            val x = it / 10.0
            LabPoint(x, if (x < 0) -.6 else .6)
        }
        val tree = SupervisedTreeEngine.fit(data, true, 3, 2)
        assertTrue(tree.nodes().count { it.isLeaf } >= 2)
        assertTrue(tree.predict(-.5, 0.0) < -.3)
        assertTrue(tree.predict(.5, 0.0) > .3)
        val forest = SupervisedTreeEngine.forest(data, true, 5, 3, false)
        val expected = forest.map { it.predict(.5, 0.0) }.average()
        assertEquals(expected, SupervisedTreeEngine.forestPrediction(forest, .5, 0.0, true), 1e-9)
    }

    @Test fun quantilesSeparateAndHuberResistsOutlier() {
        val data = (-10..10).map {
            val x = it / 10.0
            LabPoint(x, .45 * x + if (it % 2 == 0) .12 else -.12)
        } + LabPoint(.8, -.95)
        val low = SupervisedStatisticalEngine.quantile(data, .1)
        val high = SupervisedStatisticalEngine.quantile(data, .9)
        assertTrue(low.predict(0.0) < high.predict(0.0))
        val ols = PhaseOneEngines.fitSimpleLinear(data)
        val robust = SupervisedStatisticalEngine.robustHuber(data)
        assertTrue(abs(robust.slope - .45) < abs(ols.weights.first() - .45))
    }

    @Test fun posteriorUncertaintyContractsNearObservation() {
        val sparse = listOf(LabPoint(-.8, -.5), LabPoint(.8, .5))
        val dense = sparse + (-4..4).map { LabPoint(it / 10.0, it / 16.0) }
        val grid = listOf(0.0)
        val sparseGp = SupervisedStatisticalEngine.gaussianProcess(sparse, .3, .1, grid)
        val denseGp = SupervisedStatisticalEngine.gaussianProcess(dense, .3, .1, grid)
        assertTrue(denseGp.predictions.first().standardDeviation < sparseGp.predictions.first().standardDeviation)
        val sparseBayes = SupervisedStatisticalEngine.bayesianLinear(sparse, .1, grid)
        val denseBayes = SupervisedStatisticalEngine.bayesianLinear(dense, .1, grid)
        assertTrue(denseBayes.predictions.first().standardDeviation < sparseBayes.predictions.first().standardDeviation)
    }

    @Test fun gaussianClassifierUsesKernelPosteriorAndUncertainty() {
        val data = listOf(
            LabPoint(-.8, -.7, 0), LabPoint(-.7, -.8, 0),
            LabPoint(.8, .7, 1), LabPoint(.7, .8, 1)
        )
        val model = SupervisedStatisticalEngine.gaussianProcessClassifier(data, .28)
        val nearA = model.predict(-.8, -.7)
        val nearB = model.predict(.8, .7)
        val unseen = model.predict(0.0, 0.0)
        assertTrue(nearA.first < .5)
        assertTrue(nearB.first > .5)
        assertTrue(unseen.second > nearA.second)
    }

    @Test fun gaussianRegressionFollowsSmoothTrainingDataWithoutSpikes() {
        val data = PhaseOneDatasets.generate(DatasetPreset.LinearNoise, 28, .12, seed = 17)
        val grid = (0..80).map { -1.0 + it / 40.0 }
        val model = SupervisedStatisticalEngine.gaussianProcess(data, .38, .12, grid)
        assertTrue("range=${model.predictions.minOf { it.mean }}..${model.predictions.maxOf { it.mean }}",
            model.predictions.all { it.mean.isFinite() && abs(it.mean) < 1.25 })
    }

    @Test fun svmMarginAndSvrEpsilonUseFittedData() {
        val classes = PhaseOneDatasets.generate(DatasetPreset.TwoClusters, 30)
        val svm = SupervisedStatisticalEngine.linearSvm(classes, 1.0)
        val mean0 = classes.filter { it.label == 0 }.map { svm.score(it.x, it.y) }.average()
        val mean1 = classes.filter { it.label == 1 }.map { svm.score(it.x, it.y) }.average()
        assertTrue(mean1 > mean0)
        val regression = PhaseOneDatasets.generate(DatasetPreset.LinearNoise, 20)
        val svr = SupervisedStatisticalEngine.epsilonSvr(regression, 1.0, .1, KernelType.Rbf, 2.0)
        assertTrue(svr.predict(.5).isFinite())
        assertTrue(svr.supportIndices.isNotEmpty())
    }

    @Test fun kernelSvmFitsNonlinearXorAndExposesSupportVectors() {
        val model = SupervisedStatisticalEngine.kernelSvm(xor, 4.0, KernelType.Rbf, 3.0)
        assertTrue(model.supportIndices.isNotEmpty())
        val correct = xor.count {
            (model.score(it.x, it.y) >= 0.0) == (it.label == 1)
        }
        assertTrue("RBF should separate XOR", correct >= 7)
    }

    @Test fun boostingVariantsExposeDistinctMechanisms() {
        val regression = PhaseOneDatasets.generate(DatasetPreset.LinearNoise, 28, .08, 17)
        val baseline = regression.map { it.target }.average()
        val xgb = SupervisedBoostingEngine.fit(regression, true, BoostingVariant.Xgboost, 4, .4)
        val baselineMse = regression.map { (it.target - baseline) * (it.target - baseline) }.average()
        val fittedMse = regression.map {
            (it.target - xgb.predict(it.x, it.y)).pow(2)
        }.average()
        assertTrue(fittedMse < baselineMse)
        assertTrue(xgb.stages.any { it.hessians.isNotEmpty() && it.gain > 0 })

        val ada = SupervisedBoostingEngine.fit(
            PhaseOneDatasets.generate(DatasetPreset.TwoClusters, 28), false,
            BoostingVariant.Ada, 4, .3
        )
        assertTrue(ada.stages.all { abs(it.sampleWeights.sum() - 1.0) < 1e-8 })
        assertTrue(ada.stages.all { it.tree.nodes().count { node -> node.isLeaf } <= 2 })
        assertTrue(ada.stages.all { it.contribution > 0 })
        val adaRegression = SupervisedBoostingEngine.fit(regression, true, BoostingVariant.Ada, 4, .3)
        assertTrue(adaRegression.predict(.25, 0.0).isFinite())
        assertTrue(adaRegression.stages.all { it.contribution > 0 })
        val slowerAda = SupervisedBoostingEngine.fit(regression, true, BoostingVariant.Ada, 4, .1)
        assertTrue(abs(adaRegression.stages.first().contribution -
            slowerAda.stages.first().contribution) > 1e-6)

        val light = SupervisedBoostingEngine.fit(regression, true, BoostingVariant.Lightgbm, 2, .3)
        assertTrue(light.stages.first().tree.nodes().count { it.isLeaf } >= 2)

        val categories = List(30) { index ->
            val category = index % 3
            LabPoint(listOf(-.7, 0.0, .7)[category], (index % 5 - 2) / 5.0,
                label = if (category == 2) 1 else 0)
        }
        val cat = SupervisedBoostingEngine.fit(categories, false, BoostingVariant.Catboost, 3, .4)
        assertEquals(3, cat.categoryEncoding.size)
        assertTrue(cat.stages.first().tree.nodes().any { !it.isLeaf })
        val root = cat.stages.first().tree
        if (!root.left!!.isLeaf && !root.right!!.isLeaf) {
            assertEquals(root.left!!.feature, root.right!!.feature)
            assertEquals(root.left!!.threshold, root.right!!.threshold)
        }
        assertTrue(cat.predict(.7, 0.0).isFinite())
    }
}
