package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmVisualizationRegistry
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.DeepVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.ForecastVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LanguageVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.NeuralLanguageVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeCnnArchitectureEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeFundamentalsEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeGraphEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisionVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationRoute
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageThreeCoverageTest {
    @Test fun everyPhaseThreePlacementHasNativeRoute() {
        val domains = setOf("Deep Learning", "Natural Language Processing", "Computer Vision", "Time-Series Algorithms")
        val topics = LearnCatalog.topics.filter { it.domain in domains }
        assertEquals(116, topics.size)
        topics.forEach { topic ->
            val registration = AlgorithmVisualizationRegistry.forTopic(topic)
            assertTrue("${topic.domain} / ${topic.section} / ${topic.title}: ${registration.route}",
                registration.route !in setOf(VisualizationRoute.LegacyGeneral, VisualizationRoute.ExistingWeb))
        }
    }

    @Test fun explicitBindingsMatchCatalogPlacements() {
        assertEquals(17, ForecastVisualization.entries.size)
        assertEquals(11, LanguageVisualization.entries.size)
        assertEquals(9, NeuralLanguageVisualization.entries.size)
        assertEquals(17, VisionVisualization.entries.size)
        assertTrue(DeepVisualization.entries.size >= 30)
    }

    @Test fun normalizationAxesDiffer() {
        val data = StageThreeFundamentalsEngine.activations
        val batch = StageThreeFundamentalsEngine.batchNorm(data)
        val layer = StageThreeFundamentalsEngine.layerNorm(data)
        assertTrue(batch != layer)
        batch.first().indices.forEach { c -> assertEquals(0.0, batch.map { it[c] }.average(), 1e-8) }
        layer.forEach { row -> assertEquals(0.0, row.average(), 1e-8) }
    }

    @Test fun residualDenseAndDepthwiseComputationsDiffer() {
        val resnet = StageThreeCnnArchitectureEngine.frame(DeepVisualization.Resnet, 5)
        val densenet = StageThreeCnnArchitectureEngine.frame(DeepVisualization.Densenet, 5)
        val mobile = StageThreeCnnArchitectureEngine.frame(DeepVisualization.Mobilenet, 5)
        assertTrue(resnet.equation.contains("+ x"))
        assertTrue(densenet.distinction.contains("CONCATENATION"))
        assertTrue(mobile.numbers.last().second > 0.0)
    }

    @Test fun graphWeightsAndSamplingAreComputed() {
        val node = 2
        assertTrue(StageThreeGraphEngine.neighbors(node).isNotEmpty())
        assertEquals(1.0, StageThreeGraphEngine.attention(node).sumOf { it.second }, 1e-9)
        assertEquals(2, StageThreeGraphEngine.sampled(node, 2).size)
        assertTrue(StageThreeGraphEngine.linkProbability(0, 4) in 0.0..1.0)
    }
}
