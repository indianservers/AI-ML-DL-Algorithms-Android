package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmVisualizationRegistry
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourEvolutionEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourOptimizationSpecs
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourRlSpecs
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourSearchSpecs
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationRoute
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageFourCoverageTest {
    @Test fun everyRemainingPlacementHasExplicitNativeRoute() {
        val placements=LearnCatalog.domains.flatMap{it.sections}.flatMap{it.topics}
        assertEquals(327,placements.size)
        assertEquals(320,LearnCatalog.topics.size)
        val remaining=placements.filter(StageFourVisualization::supports)
        assertEquals(111,remaining.size)
        remaining.forEach{topic->
            val registered=AlgorithmVisualizationRegistry.forTopic(topic)
            assertEquals("${topic.domain}/${topic.section}/${topic.title}",VisualizationRoute.StageFourNative,registered.route)
            assertTrue(registered.stageFour)
            assertTrue(registered.implementation.contains(topic.title))
        }
    }

    @Test fun topicSpecificImplementationsCoverEveryRemainingTitle() {
        val remaining=LearnCatalog.domains.flatMap{it.sections}.flatMap{it.topics}.filter(StageFourVisualization::supports)
        remaining.forEach{topic->
            when(topic.domain){
                "Reinforcement Learning"->assertTrue(StageFourRlSpecs.forTitle(topic.title).flow.isNotEmpty())
                "Optimization Algorithms"->assertTrue(StageFourOptimizationSpecs.forTitle(topic.title).flow.isNotEmpty())
                "Evolutionary Algorithms"->assertTrue(StageFourEvolutionEngine.frame(topic.title,1,3).flow.isNotEmpty())
                "AI Algorithms"->when(topic.section){
                    "Search"->assertTrue(StageFourSearchSpecs.forTitle(topic.title).flow.isNotEmpty())
                    "Dynamic Programming"->assertTrue(StageFourRlSpecs.forTitle(topic.title).flow.isNotEmpty())
                    else->Unit
                }
                else->Unit
            }
        }
    }

    @Test fun entireCatalogHasNoGenericOrWebVisualizationRoute() {
        val placements=LearnCatalog.domains.flatMap{it.sections}.flatMap{it.topics}
        placements.forEach { topic ->
            val route=AlgorithmVisualizationRegistry.forTopic(topic).route
            assertTrue("${topic.domain}/${topic.section}/${topic.title}: $route",route in setOf(
                VisualizationRoute.SupervisedNative,VisualizationRoute.StageTwoNative,
                VisualizationRoute.StageThreeForecastNative,VisualizationRoute.StageThreeLanguageNative,
                VisualizationRoute.StageThreeNeuralLanguageNative,VisualizationRoute.StageThreeVisionNative,
                VisualizationRoute.StageThreeDeepNative,VisualizationRoute.StageFourNative,
                VisualizationRoute.ExistingNative))
        }
    }

    @Test fun emitVisibleCatalogPlacementRowsForFinalAudit() {
        LearnCatalog.domains.forEach { domain -> domain.sections.forEach { section -> section.topics.forEach { topic ->
            val registration=AlgorithmVisualizationRegistry.forTopic(topic)
            println(listOf("VIS_PLACE",domain.title,section.title,topic.id,topic.title,
                topic.domain,topic.section,registration.route.name,registration.implementation,
                registration.controls,registration.dataset,registration.status.name).joinToString("\t"))
        } } }
    }
}
