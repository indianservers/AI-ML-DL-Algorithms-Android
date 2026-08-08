package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.KernelType
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.LabPoint
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoAlgorithmKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoDatasetPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoDatasets
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseTwoInteractiveEngineTest {
    @Test
    fun gaussianNaiveBayesProducesNormalizedPosterior() {
        val points = PhaseTwoDatasets.generate(PhaseTwoDatasetPreset.WellSeparatedGaussian, samples = 48, classes = 3, noise = 0.02, seed = 3)
        val (prediction, posterior) = PhaseTwoEngines.gaussianNaiveBayes(points, points.first())
        assertEquals(1.0, posterior.sumOf { it.posterior }, 1e-9)
        assertTrue(prediction in 0..2)
        assertTrue(posterior.all { it.prior > 0.0 && it.likelihoodX > 0.0 && it.likelihoodY > 0.0 })
    }

    @Test
    fun multinomialAndBernoulliNaiveBayesReactToEvidence() {
        val spammy = PhaseTwoEngines.multinomialNaiveBayes(mapOf("offer" to 2, "money" to 2, "free" to 1))
        val work = PhaseTwoEngines.multinomialNaiveBayes(mapOf("meeting" to 2, "project" to 1, "report" to 2))
        assertNotEquals(spammy.prediction, work.prediction)
        val bernoulli = PhaseTwoEngines.bernoulliNaiveBayes(mapOf("offer" to true, "money" to true, "meeting" to false, "project" to false, "free" to true, "report" to false))
        assertEquals(spammy.prediction, bernoulli.prediction)
    }

    @Test
    fun ldaAndQdaClassifiersReturnFiniteState() {
        val points = PhaseTwoDatasets.generate(PhaseTwoDatasetPreset.DifferentVariances, samples = 72, classes = 3, noise = 0.05, seed = 9)
        val lda = PhaseTwoEngines.ldaPredict(points, points[4])
        val qda = PhaseTwoEngines.qdaPredict(points, points[4])
        assertTrue(lda.first in 0..2)
        assertTrue(lda.second.isFinite() && lda.second >= 0.0)
        assertEquals(1.0, qda.second.sumOf { it.posterior }, 1e-9)
    }

    @Test
    fun svmSupportVectorsMarginsAndHingeLossAreExposed() {
        val points = PhaseTwoDatasets.generate(PhaseTwoDatasetPreset.PerfectlySeparable, samples = 40, classes = 2, noise = 0.02, seed = 11)
        val svm = PhaseTwoEngines.svmState(points, c = 2.0)
        assertTrue(svm.marginWidth > 0.0)
        assertTrue(svm.supportVectorIndices.isNotEmpty())
        assertTrue(svm.hingeLoss.isFinite())
        assertEquals(0.0, PhaseTwoEngines.hingeLoss(1, 2.0), 0.0)
        assertTrue(PhaseTwoEngines.hingeLoss(1, -0.5) > 1.0)
    }

    @Test
    fun kernelCalculationsAreDeterministicAndRbfSelfSimilarityIsOne() {
        val a = LabPoint(0.2, -0.1, 0)
        val b = LabPoint(-0.4, 0.5, 1)
        assertEquals(1.0, PhaseTwoEngines.kernel(a, a, KernelType.Rbf, gamma = 3.0), 1e-12)
        assertEquals(PhaseTwoEngines.kernel(a, b, KernelType.Linear), PhaseTwoEngines.kernel(b, a, KernelType.Linear), 1e-12)
        assertTrue(PhaseTwoEngines.kernel(a, b, KernelType.Polynomial, gamma = 0.5, degree = 3).isFinite())
        assertTrue(PhaseTwoEngines.kernelState(KernelType.Rbf, gamma = 8.0, degree = 3).warning != null)
    }

    @Test
    fun bootstrapSamplingTracksRepeatedAndOutOfBagSamples() {
        val points = PhaseTwoDatasets.generate(PhaseTwoDatasetPreset.NoisyEnsemble, samples = 36, classes = 2, noise = 0.1, seed = 2)
        val boot = PhaseTwoEngines.bootstrap(points, seed = 7, memberId = 1)
        assertEquals(points.size, boot.frequencies.size)
        assertEquals(points.size, boot.frequencies.sum())
        assertTrue(boot.frequencies.any { it == 0 })
        assertTrue(boot.frequencies.any { it > 1 })
    }

    @Test
    fun randomForestVotesAndExtraTreesRandomizationAreVisible() {
        val points = PhaseTwoDatasets.generate(PhaseTwoDatasetPreset.NoisyEnsemble, samples = 50, classes = 2, noise = 0.12, seed = 6)
        val query = LabPoint(0.1, 0.1, 0)
        val forest = PhaseTwoEngines.ensemble(points, query, trees = 9, kind = PhaseTwoAlgorithmKind.RandomForestClassifier, seed = 4)
        val extra = PhaseTwoEngines.ensemble(points, query, trees = 9, kind = PhaseTwoAlgorithmKind.ExtraTreesClassifier, seed = 4)
        assertEquals(9, forest.members.size)
        assertEquals(1.0, forest.voteDistribution.values.sum(), 1e-9)
        assertNotEquals(forest.bootstrapStates.map { it.threshold }, extra.bootstrapStates.map { it.threshold })
    }

    @Test
    fun adaBoostUpdatesSampleWeightsAfterMistakes() {
        val points = PhaseTwoDatasets.generate(PhaseTwoDatasetPreset.LabelNoise, samples = 44, classes = 2, noise = 0.08, seed = 13)
        val rounds = PhaseTwoEngines.adaBoost(points, rounds = 4, seed = 3)
        assertEquals(4, rounds.size)
        assertTrue(rounds.all { it.weightedError in 0.0..0.5 })
        assertEquals(1.0, rounds.last().sampleWeights.sum(), 1e-9)
        assertTrue(rounds.last().sampleWeights.max() > rounds.last().sampleWeights.min())
    }

    @Test
    fun hardSoftVotingAndStackingMetaFeaturesAreDeterministic() {
        assertEquals(1, PhaseTwoEngines.hardVoting(listOf(1, 0, 1)))
        val (prediction, averaged) = PhaseTwoEngines.softVoting(listOf(mapOf(0 to .2, 1 to .8), mapOf(0 to .6, 1 to .4)))
        assertEquals(1.0, averaged.values.sum(), 1e-9)
        assertEquals(1, prediction)
        val features = PhaseTwoEngines.stackingMetaFeatures(listOf(.8, .6, .7))
        assertEquals(listOf(.8, .6, .7), features)
        assertTrue(PhaseTwoEngines.stackingMetaProbability(features) in 0.0..1.0)
    }

    @Test
    fun multiclassMetricsExposeMacroAndPerClassValues() {
        val metrics = PhaseTwoEngines.multiclassMetrics(
            actual = listOf(0, 0, 1, 1, 2, 2),
            predicted = listOf(0, 1, 1, 1, 2, 0)
        )
        assertEquals(listOf(0, 1, 2), metrics.classes)
        assertEquals(3, metrics.confusion.size)
        assertEquals(3, metrics.perClass.size)
        assertTrue(metrics.accuracy in 0.0..1.0)
        assertTrue(metrics.macroF1 in 0.0..1.0)
    }
}
