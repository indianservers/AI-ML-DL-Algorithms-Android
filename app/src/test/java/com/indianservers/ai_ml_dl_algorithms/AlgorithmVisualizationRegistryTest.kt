package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmVisualizationRegistry
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.SupervisedVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationRoute
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class AlgorithmVisualizationRegistryTest {
    @Test fun everyUniqueTopicHasAnExplicitRegistration() {
        assertEquals(320, LearnCatalog.topics.size)
        assertEquals(LearnCatalog.topics.map { it.id }.toSet(), AlgorithmVisualizationRegistry.entries.keys)
        val supervised = LearnCatalog.domains.first { it.title == "Supervised Learning" }
            .sections.flatMap { it.topics }
        assertEquals(40, supervised.size)
        supervised.forEach {
            val entry = AlgorithmVisualizationRegistry.forTopic(it)
            assertEquals(VisualizationRoute.SupervisedNative, entry.route)
            assertNotNull(entry.supervised)
        }
        assertEquals(SupervisedVisualization.ClassificationTree,
            AlgorithmVisualizationRegistry.forTopic(supervised.first { it.title == "Decision Tree" }).supervised)
    }

    @Test fun emitAuditRowsFromRuntimeRegistry() {
        LearnCatalog.topics.forEach { topic ->
            val entry = AlgorithmVisualizationRegistry.forTopic(topic)
            println(listOf(
                "VIS_AUDIT", topic.id, topic.title, topic.domain, topic.section,
                entry.implementation, entry.family, entry.controls, entry.dataset,
                entry.interaction, entry.animation, entry.graphic, entry.status.name
            ).joinToString("\t"))
        }
    }
}
