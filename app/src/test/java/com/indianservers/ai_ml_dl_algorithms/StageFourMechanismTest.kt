package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourEvolutionEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourExplanationEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourOptimizationEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourProbabilityEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourRecommendationEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourRlEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageFourSearchEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.SearchCandidate
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageFourMechanismTest {
    @Test fun rlTargetsUseSampleMaxAndExpectationDifferently() {
        val values=listOf(.2,.8,.4)
        val probabilities=listOf(.6,.1,.3)
        val sarsa=StageFourRlEngine.sarsaTarget(-.1,values[0],.9)
        val q=StageFourRlEngine.qTarget(-.1,values,.9)
        val expected=StageFourRlEngine.expectedSarsaTarget(-.1,values,probabilities,.9)
        assertEquals(.08,sarsa,1e-9)
        assertEquals(.62,q,1e-9)
        assertEquals(.188,expected,1e-9)
        assertTrue(sarsa<expected&&expected<q)
        assertNotEquals(q,StageFourRlEngine.doubleDqnTarget(-.1,values,listOf(.9,.1,.2),.9),1e-9)
    }

    @Test fun policyMethodsAndValueBackupsHaveRealMath() {
        assertEquals(.72,StageFourRlEngine.mcReturn(listOf(-.1,1.0),.82),1e-9)
        assertEquals(.6,StageFourRlEngine.duelingQ(.5,listOf(.1,.3))[1],1e-9)
        assertEquals(1.2,StageFourRlEngine.ppoClipped(2.0,1.0,.2),1e-9)
        assertTrue(StageFourRlEngine.entropy(listOf(.5,.5))>StageFourRlEngine.entropy(listOf(.9,.1)))
        assertTrue(StageFourRlEngine.valueIteration(6)[0]>StageFourRlEngine.valueIteration(0)[0])
    }

    @Test fun probabilityUpdatesAndBayesianGraphAreDistinct() {
        assertEquals(.5,StageFourProbabilityEngine.likelihood(1,1,.5)/.5,1e-9)
        assertEquals(1.5,StageFourProbabilityEngine.betaDensity(2.0,2.0,.5),1e-9)
        assertTrue(StageFourProbabilityEngine.wetPosterior(true)>StageFourProbabilityEngine.wetPosterior(false))
        assertTrue(StageFourProbabilityEngine.acyclic(listOf("A" to "B","B" to "C")))
        assertFalse(StageFourProbabilityEngine.acyclic(listOf("A" to "B","B" to "A")))
    }

    @Test fun optimizersTakeDifferentPathsAndCurvatureConverges() {
        val batch=StageFourOptimizationEngine.path("Batch Gradient Descent",.08,8)
        val stochastic=StageFourOptimizationEngine.path("Stochastic Gradient Descent",.08,8)
        val adam=StageFourOptimizationEngine.path("Adam",.08,8)
        val adamw=StageFourOptimizationEngine.path("AdamW",.08,8)
        val newton=StageFourOptimizationEngine.path("Newton's Method",.08,2)
        val lbfgs=StageFourOptimizationEngine.path("L-BFGS",.08,8)
        assertNotEquals(batch.last().x,stochastic.last().x,1e-6)
        assertNotEquals(adam.last().x,adamw.last().x,1e-6)
        assertTrue(newton.last().loss<1e-9)
        assertTrue(lbfgs.last().memory in 1..3)
        assertTrue(listOf(batch,stochastic,adam,adamw,newton,lbfgs).all{it.all{point->point.loss.isFinite()}})
    }

    @Test fun evolutionaryOperatorsAreCalculated() {
        assertEquals("111000",StageFourEvolutionEngine.crossover("111111","000000",3))
        assertEquals("101",StageFourEvolutionEngine.mutate("111",1))
        assertTrue(StageFourEvolutionEngine.gpPopulation(1).maxOf(StageFourEvolutionEngine::gpFitness) >
            StageFourEvolutionEngine.gpPopulation(0).maxOf(StageFourEvolutionEngine::gpFitness))
        val es=StageFourEvolutionEngine.esState(3,4)
        assertEquals(4,es.offspring.size)
        assertTrue(es.sigma in .08..1.4)
        val sa=StageFourEvolutionEngine.saState(4,4)
        assertEquals(if(sa.accepted)sa.candidate else sa.current,sa.next,1e-9)
        assertEquals("Scout",StageFourEvolutionEngine.abcState(2).role)
        assertNotEquals(StageFourEvolutionEngine.abcState(0).sources,
            StageFourEvolutionEngine.abcState(3).sources)
        assertEquals(2.0,StageFourEvolutionEngine.deMutant(1.0 to 0.0,2.0 to 0.0,0.0 to 0.0,.5).first,1e-9)
        assertTrue(StageFourEvolutionEngine.annealProbability(2.0,1.0,2.0)>StageFourEvolutionEngine.annealProbability(2.0,1.0,.1))
        assertTrue(StageFourEvolutionEngine.pso(3).all{it.first.isFinite()&&it.second.isFinite()})
    }

    @Test fun recommendationAndExplanationScoresUseData() {
        val rec=StageFourRecommendationEngine
        assertEquals(5,rec.popularity().size)
        assertNotEquals(rec.userNeighbors(0,2).map{it.first},rec.itemNeighbors(0,2).map{it.first})
        assertTrue(rec.factorize(15).loss<rec.factorize(0).loss)
        assertTrue(rec.als(4).loss.isFinite())
        assertTrue(rec.singularValues().zipWithNext().all{it.first>=it.second-1e-6})
        assertNotEquals(rec.neuralScore(rec.factorize(3),0,2),rec.deepScore(rec.factorize(3),0,2),1e-9)
        val x=StageFourExplanationEngine.data[0]
        assertTrue(StageFourExplanationEngine.permutationDrop(0)>0)
        val baselineLogit=StageFourExplanationEngine.data.map { row ->
            StageFourExplanationEngine.coefficients.zip(row).sumOf { it.first*it.second }-.2
        }.average()
        val predictionLogit=StageFourExplanationEngine.coefficients.zip(x).sumOf { it.first*it.second }-.2
        assertEquals(predictionLogit,baselineLogit+StageFourExplanationEngine.exactLinearContributions(x).sum(),1e-9)
        assertEquals(21,StageFourExplanationEngine.pdp(0).size)
        assertEquals(3,StageFourExplanationEngine.lime(x).size)
        val image=StageFourExplanationEngine.pixels
        val changedImage=image.map{it.toMutableList()}.toMutableList()
        changedImage[2][1]+=.0001
        assertEquals(StageFourExplanationEngine.saliency[2][1],
            (StageFourExplanationEngine.imageScore(changedImage)-StageFourExplanationEngine.imageScore(image))/.0001,1e-6)
        val (changed,_)=StageFourExplanationEngine.counterfactual(x)
        assertNotEquals(StageFourExplanationEngine.score(x)>=.5,StageFourExplanationEngine.score(changed)>=.5)
    }

    @Test fun searchFrontiersAndPruningUseDifferentRules() {
        val bfs=StageFourSearchEngine.trace("Breadth First Search")
        val dfs=StageFourSearchEngine.trace("AI Depth First Search")
        val ucs=StageFourSearchEngine.trace("Uniform Cost Search")
        val astar=StageFourSearchEngine.trace("A Star Search")
        val beam=StageFourSearchEngine.trace("Beam Search",2)
        val bidirectional=StageFourSearchEngine.bidirectional()
        assertEquals("G",bfs.last().current)
        assertNotEquals(bfs.map{it.current},dfs.map{it.current})
        assertEquals(7.0,StageFourSearchEngine.priority("A Star Search",SearchCandidate("A",listOf("S","A"),2.0)),1e-9)
        assertEquals("G",ucs.last().current)
        assertEquals("G",astar.last().current)
        assertTrue(beam.any{it.note.startsWith("Level 1")})
        assertEquals("G",bidirectional.last().path.last())
        assertTrue(StageFourSearchEngine.hillPath(5).zipWithNext().all { (a,b) -> b.second >= a.second })
        assertTrue(StageFourSearchEngine.localBeam(4,3).all { StageFourSearchEngine.localFitness(it).isFinite() })
        assertEquals(3,StageFourSearchEngine.minimax(listOf(3,5,2,9)))
        val (alpha,pruned)=StageFourSearchEngine.alphaBeta(listOf(3,5,2,9))
        assertEquals(3,alpha)
        assertTrue(3 in pruned)
    }
}
