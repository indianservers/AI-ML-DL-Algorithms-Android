package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LearnCatalogTest {
    @Test
    fun catalogContainsCompleteUniqueTaxonomy() {
        assertEquals(14, LearnCatalog.domains.size)
        assertTrue(LearnCatalog.topics.size > 240)
        assertEquals(LearnCatalog.topics.size, LearnCatalog.topics.map { it.id }.toSet().size)
        assertTrue(LearnCatalog.domains.all { domain ->
            domain.sections.isNotEmpty() && domain.sections.all { it.topics.isNotEmpty() }
        })
    }

    @Test
    fun everyTopicProducesACompleteLearningProfile() {
        LearnCatalog.topics.forEach { topic ->
            LearningDepth.entries.forEach { depth ->
                val profile = LearnCatalog.profile(topic, depth)
                assertTrue(profile.definition.isNotBlank())
                assertTrue(profile.steps.size >= 5)
                assertTrue(profile.equation.isNotBlank())
                assertTrue(profile.advantages.isNotEmpty())
                assertTrue(profile.limitations.isNotEmpty())
                assertTrue(profile.hyperparameters.isNotEmpty())
            }
        }
    }

    @Test
    fun flagshipAlgorithmsUseSpecificVisualizations() {
        fun kind(title: String) = LearnCatalog.profile(
            LearnCatalog.topics.first { it.title == title },
            LearningDepth.University
        ).kind

        assertEquals(VisualizationKind.Regression, kind("Simple Linear Regression"))
        assertEquals(VisualizationKind.Clustering, kind("K-Means"))
        assertEquals(VisualizationKind.Density, kind("DBSCAN"))
        assertEquals(VisualizationKind.Attention, kind("Transformer"))
        assertEquals(VisualizationKind.Reinforcement, kind("Q-Learning"))
    }
}
