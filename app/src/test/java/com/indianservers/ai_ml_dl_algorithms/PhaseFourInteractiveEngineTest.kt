package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.AnomalyPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourData
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseFourInteractiveEngineTest {
    @Test
    fun isolationLofAndEnvelopeExposeAnomalyState() {
        val points = PhaseFourData.anomalies(AnomalyPreset.SingleOutlier, samples = 60, contamination = .08, seed = 3)
        val isolation = PhaseFourEngines.isolationForest(points, points.lastIndex, trees = 6)
        val lof = PhaseFourEngines.lof(points, points.lastIndex, k = 5)
        val envelope = PhaseFourEngines.envelope(points)
        assertEquals(points.size, isolation.scores.size)
        assertTrue(isolation.selectedPaths.isNotEmpty())
        assertTrue(lof.lof.isFinite() && lof.kDistance >= 0.0)
        assertEquals(points.size, envelope.distances.size)
        assertTrue(envelope.threshold > 0.0)
    }

    @Test
    fun zScoreAndIqrOutlierRulesAreDeterministic() {
        val values = listOf(1.0, 1.1, .9, 1.2, 10.0)
        assertTrue(PhaseFourEngines.zScores(values, 1.5).last())
        val (_, outliers) = PhaseFourEngines.iqrOutliers(values)
        assertTrue(outliers.last())
    }

    @Test
    fun associationMetricsUseConsistentSupportDenominators() {
        val baskets = PhaseFourData.baskets
        val rule = PhaseFourEngines.associationRule(baskets, setOf("Bread"), setOf("Milk"))
        assertEquals(PhaseFourEngines.support(baskets, setOf("Bread", "Milk")), rule.support, 1e-12)
        assertTrue(rule.confidence in 0.0..1.0)
        assertTrue(rule.lift > 0.0)
        val levels = PhaseFourEngines.apriori(baskets, .3)
        assertTrue(levels.first().frequent.isNotEmpty())
        assertTrue(PhaseFourEngines.fpTree(baskets).children.isNotEmpty())
        assertTrue(PhaseFourEngines.eclatTidsets(baskets).getValue("Bread").isNotEmpty())
    }

    @Test
    fun recommenderSimilarityPredictionsAndFactorsAreFinite() {
        val data = PhaseFourData.ratings
        assertEquals(1.0, PhaseFourEngines.cosine(listOf(1.0, 2.0), listOf(1.0, 2.0)), 1e-12)
        assertTrue(PhaseFourEngines.popularity(data).first().score.isFinite())
        assertTrue(PhaseFourEngines.userCf(data, 0, 2).score.isFinite())
        assertTrue(PhaseFourEngines.itemCf(data, 0, 4).score.isFinite())
        val factor = PhaseFourEngines.factorState(data, 0, 2, factors = 2)
        assertTrue(factor.prediction.isFinite())
        assertEquals(2, factor.userFactors.first().size)
    }

    @Test
    fun bayesBetaMleAndMapAreCorrect() {
        val bayes = PhaseFourEngines.bayes(prior = .01, sensitivity = .9, specificity = .95)
        assertEquals(.009 / (.009 + .0495), bayes.posterior, 1e-12)
        val beta = PhaseFourEngines.betaBernoulli(2.0, 2.0, heads = 3, tails = 1)
        assertEquals(5.0, beta.alpha, 0.0)
        assertEquals(3.0, beta.beta, 0.0)
        assertEquals(.75, PhaseFourEngines.mleTheta(3, 1), 0.0)
        assertTrue(PhaseFourEngines.mapTheta(3, 1, 2.0, 2.0) in 0.0..1.0)
    }

    @Test
    fun hmmForwardAndViterbiProduceSequenceState() {
        val hmm = PhaseFourEngines.hmm(listOf("Walk", "Shop", "Clean"))
        assertEquals(3, hmm.forward.size)
        assertEquals(3, hmm.viterbi.size)
        assertTrue(hmm.forward.all { kotlin.math.abs(it.values.sum() - 1.0) < 1e-9 })
    }

    @Test
    fun gaussianProcessUncertaintyShrinksNearObservations() {
        val gp = PhaseFourEngines.gaussianProcess(listOf(0.0 to 1.0), lengthScale = .25, noise = .05)
        val centerIndex = gp.xs.indices.minBy { kotlin.math.abs(gp.xs[it]) }
        assertTrue(gp.variance[centerIndex] < gp.variance.first())
        assertEquals(gp.xs.size, gp.mean.size)
    }

    @Test
    fun mcmcGibbsAndVariationalStatesAreBounded() {
        val mh = PhaseFourEngines.metropolis(current = .2, proposalStd = .4, step = 5)
        assertTrue(mh.ratio in 0.0..1.0)
        assertTrue(mh.u in 0.0..1.0)
        val gibbs = PhaseFourEngines.gibbs(6)
        assertEquals(7, gibbs.path.size)
        val vi = PhaseFourEngines.variational(12)
        assertTrue(vi.approximateVariance > 0.0)
        assertTrue(vi.elboProxy.isFinite())
    }
}
