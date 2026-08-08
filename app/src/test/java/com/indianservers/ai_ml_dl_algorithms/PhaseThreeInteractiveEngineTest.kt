package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.ClusterPoint
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.ClusterPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.LinkageMethod
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeAlgorithmKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeDatasets
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseThreeInteractiveEngineTest {
    @Test
    fun kMeansAssignsPointsAndUpdatesCentroids() {
        val points = listOf(
            ClusterPoint(-.9, -.9),
            ClusterPoint(-.8, -.7),
            ClusterPoint(.8, .7),
            ClusterPoint(.9, .9)
        )
        val state = PhaseThreeEngines.kMeans(points, k = 2, iterations = 6, seed = 1)
        assertEquals(points.size, state.assignments.size)
        assertEquals(2, state.centers.size)
        assertTrue(state.inertia >= 0.0)
        assertTrue(state.history.last() <= state.history.first() + 1e-9)
    }

    @Test
    fun kMeansPlusPlusProbabilitiesNormalizeAndFavorFarPoints() {
        val points = listOf(ClusterPoint(0.0, 0.0), ClusterPoint(.1, 0.0), ClusterPoint(.9, .9))
        val probabilities = PhaseThreeEngines.kMeansPlusPlusProbabilities(points, listOf(points.first()))
        assertEquals(1.0, probabilities.sum(), 1e-9)
        assertTrue(probabilities[2] > probabilities[1])
    }

    @Test
    fun dbscanMarksCoreBorderAndNoise() {
        val points = PhaseThreeDatasets.clusters(ClusterPreset.Blobs, samples = 60, groups = 2, noise = .01, seed = 4) +
            listOf(ClusterPoint(1.0, -1.0))
        val state = PhaseThreeEngines.dbscan(points, eps = .18, minPts = 4)
        assertTrue(state.core.isNotEmpty())
        assertTrue(state.noise.isNotEmpty())
        assertEquals(points.size, state.labels.size)
    }

    @Test
    fun hierarchicalProducesMergeSequence() {
        val points = PhaseThreeDatasets.clusters(ClusterPreset.Blobs, samples = 12, groups = 3, noise = .02, seed = 6)
        val state = PhaseThreeEngines.hierarchical(points, LinkageMethod.Average)
        assertEquals(points.size - 1, state.merges.size)
        assertTrue(state.clusterCount >= 1)
        assertTrue(state.merges.all { it.height >= 0.0 })
    }

    @Test
    fun gmmResponsibilitiesNormalizeAndEmUpdatesComponents() {
        val points = PhaseThreeDatasets.clusters(ClusterPreset.Elongated, samples = 50, groups = 2, noise = .03, seed = 8)
        val state = PhaseThreeEngines.gmm(points, k = 2, iterations = 4)
        assertEquals(2, state.components.size)
        assertTrue(state.logLikelihood.isFinite())
        assertTrue(state.responsibilities.all { row -> kotlin.math.abs(row.sum() - 1.0) < 1e-6 })
    }

    @Test
    fun pcaFindsFiniteProjectionAndReconstructionError() {
        val points = PhaseThreeDatasets.clusters(ClusterPreset.Elongated, samples = 80, groups = 1, noise = .03, seed = 10)
        val pca = PhaseThreeEngines.pca(points)
        assertTrue(pca.variance1 >= 0.0)
        assertTrue(pca.variance2 >= 0.0)
        assertEquals(points.size, pca.projected.size)
        assertTrue(pca.reconstructionError >= 0.0)
    }

    @Test
    fun truncatedSvdErrorDecreasesWithRank() {
        val rank1 = PhaseThreeEngines.truncatedSvd(1).second
        val rank3 = PhaseThreeEngines.truncatedSvd(3).second
        assertTrue(rank3 < rank1)
    }

    @Test
    fun gradientBoostingStagesUseActualResidualUpdates() {
        val points = PhaseThreeDatasets.regression(samples = 30, seed = 12)
        val state = PhaseThreeEngines.gradientBoosting(points, stages = 5, learningRate = .2)
        assertEquals(5, state.stages.size)
        val first = state.stages.first()
        assertEquals(first.predictionBefore + first.learningRate * first.contribution, first.predictionAfter, 1e-12)
        assertTrue(state.trainError.isFinite())
    }

    @Test
    fun xgBoostGainSelectsBestCandidateAndRespondsToRegularization() {
        val lowPenalty = PhaseThreeEngines.xgBoostGain(lambda = .1, gamma = 0.0)
        val highPenalty = PhaseThreeEngines.xgBoostGain(lambda = 10.0, gamma = 1.0)
        assertTrue(lowPenalty.gainA.isFinite() && lowPenalty.gainB.isFinite())
        assertTrue(highPenalty.gainA < lowPenalty.gainA)
        assertTrue(lowPenalty.selected in listOf("A", "B"))
    }

    @Test
    fun graphEmbeddingAndSilhouetteExposeVisualState() {
        val points = PhaseThreeDatasets.clusters(ClusterPreset.HighDim, samples = 60, groups = 3, noise = .04, seed = 15)
        val graph = PhaseThreeEngines.similarityGraph(points, neighbors = 4)
        val embedding = PhaseThreeEngines.embedding(points, PhaseThreeAlgorithmKind.Umap, seed = 4, neighbors = 5)
        val km = PhaseThreeEngines.kMeans(points, 3, 5)
        val silhouette = PhaseThreeEngines.silhouette(points, km.assignments)
        assertTrue(graph.edges.isNotEmpty())
        assertEquals(points.size, embedding.embedding.size)
        assertTrue(embedding.neighborOverlap in 0..5)
        assertTrue(silhouette.overall == null || silhouette.overall in -1.0..1.0)
    }
}
