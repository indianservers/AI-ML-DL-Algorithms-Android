package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmVisualizationRegistry
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationRoute
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationStatus
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.hasLiveTrainer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FinalHardeningTest {
    @Test fun everyVisiblePlacementResolvesWithoutGenericOrWebFallback() {
        val placements = LearnCatalog.domains.flatMap { it.sections }.flatMap { it.topics }
        assertEquals(327, placements.size)
        assertEquals(320, placements.map { it.id }.toSet().size)
        assertEquals(LearnCatalog.topics.map { it.id }.toSet(), AlgorithmVisualizationRegistry.entries.keys)
        placements.forEach { topic ->
            val registration = AlgorithmVisualizationRegistry.forTopic(topic)
            assertEquals(topic.id, registration.topicId)
            assertTrue("${topic.id}: ${registration.route}", registration.route !in setOf(
                VisualizationRoute.ExistingWeb, VisualizationRoute.LegacyGeneral))
            assertTrue("${topic.id}: ${registration.status}", registration.status !in setOf(
                VisualizationStatus.Generic, VisualizationStatus.Incorrect, VisualizationStatus.Missing))
            assertTrue("${topic.id}: empty implementation", registration.implementation.isNotBlank())
            val bound = when (registration.route) {
                VisualizationRoute.SupervisedNative -> registration.supervised != null
                VisualizationRoute.StageTwoNative -> registration.stageTwo != null
                VisualizationRoute.StageThreeForecastNative -> registration.forecast != null
                VisualizationRoute.StageThreeLanguageNative -> registration.language != null
                VisualizationRoute.StageThreeNeuralLanguageNative -> registration.neuralLanguage != null
                VisualizationRoute.StageThreeVisionNative -> registration.vision != null
                VisualizationRoute.StageThreeDeepNative -> registration.deep != null
                VisualizationRoute.StageFourNative -> registration.stageFour
                VisualizationRoute.ExistingNative -> registration.implementation.contains("AlgorithmLab/")
                VisualizationRoute.ExistingWeb, VisualizationRoute.LegacyGeneral -> false
            }
            assertTrue("${topic.id}: declared route lacks a screen binding", bound)
        }
    }

    @Test fun trainingControlsAppearOnlyForConnectedEngines() {
        val placements = LearnCatalog.domains.flatMap { it.sections }.flatMap { it.topics }
        val live = placements.filter(::hasLiveTrainer)
        assertEquals(8, live.size)
        assertEquals(8, live.map { it.id }.toSet().size)
        assertTrue(live.all { it.domain == "Supervised Learning" })
        val fundamentals = LearnCatalog.domains.first { it.title == "Reinforcement Learning" }
            .sections.first { it.title == "Fundamentals" }.topics
        assertEquals(8, fundamentals.size)
        assertTrue(fundamentals.none(::hasLiveTrainer))
        assertFalse(placements.first { it.title == "CNN" }.let(::hasLiveTrainer))
    }
}
