package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.DeepVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.ActivationKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeActivationEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeFundamentalsEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeGenerativeVariantEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeOptimizationEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.swinWindow
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseEightEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageThreeDeepMechanismTest {
    @Test fun teacherForcedDecoderDependsOnContextAndPriorToken() {
        val baseline = PhaseSevenEngines.teacherForcedDecoder(.2, listOf(0.0, .4, .6))
        val changedContext = PhaseSevenEngines.teacherForcedDecoder(.8, listOf(0.0, .4, .6))
        val changedPriorToken = PhaseSevenEngines.teacherForcedDecoder(.2, listOf(0.0, .9, .6))
        assertNotEquals(baseline, changedContext)
        assertEquals(baseline[0], changedPriorToken[0], 1e-9)
        assertNotEquals(baseline[1], changedPriorToken[1])
    }
    @Test fun activationAndSoftmaxMathIsFinite() {
        ActivationKind.entries.forEach { kind ->
            assertTrue(kind.name,StageThreeActivationEngine.output(kind,.4).isFinite())
            assertTrue(kind.name,StageThreeActivationEngine.derivative(kind,.4).isFinite())
        }
        assertEquals(1.0,StageThreeActivationEngine.softmax(listOf(1.0,2.0,3.0)).sum(),1e-9)
        assertEquals(0.0,StageThreeActivationEngine.output(ActivationKind.Relu,-2.0),1e-9)
    }
    @Test fun gradientStrategiesUseDifferentSamplesAndPaths() {
        val batch = StageThreeOptimizationEngine.path(DeepVisualization.GradientDescent,.1,4,5)
        val stochastic = StageThreeOptimizationEngine.path(DeepVisualization.Sgd,.1,4,5)
        val mini = StageThreeOptimizationEngine.path(DeepVisualization.MiniBatch,.1,4,5)
        assertNotEquals(batch,stochastic)
        assertNotEquals(batch,mini)
        assertNotEquals(stochastic,mini)
    }

    @Test fun lossFunctionsHaveDifferentShapes() {
        assertEquals(.25,StageThreeOptimizationEngine.loss(0,.5),1e-9)
        assertEquals(.5,StageThreeOptimizationEngine.loss(1,.5),1e-9)
        assertTrue(StageThreeOptimizationEngine.loss(3,.5)>0)
    }

    @Test fun dropoutMaskAndPenaltiesRespondToControls() {
        val low=StageThreeFundamentalsEngine.mask(.1,0)
        val high=StageThreeFundamentalsEngine.mask(.8,0)
        assertTrue(low.count{it}>high.count{it})
        val weights=listOf(2.0,.5)
        assertNotEquals(StageThreeFundamentalsEngine.penalty(weights,.2,true),
            StageThreeFundamentalsEngine.penalty(weights,.2,false))
    }

    @Test fun swinShiftChangesAttentionNeighborhood() {
        assertNotEquals(swinWindow(10,false),swinWindow(10,true))
        assertTrue(10 in swinWindow(10,true))
    }

    @Test fun attentionHeadsHaveDistinctLearnedProjections() {
        val heads=PhaseEightEngines.multiHead(listOf("the","cat","sat"),heads=3,dim=4).heads
        assertNotEquals(heads[0].queries,heads[1].queries)
        assertNotEquals(heads[1].keys,heads[2].keys)
    }

    @Test fun generativeVariantsExposeDistinctMechanisms() {
        val sparse=StageThreeGenerativeVariantEngine.sparseActivations(4)
        assertTrue(sparse.any{it>0})
        assertTrue(sparse.any{it==0.0})
        val cycle=StageThreeGenerativeVariantEngine.frame(DeepVisualization.CycleGan,5)
        val wgan=StageThreeGenerativeVariantEngine.frame(DeepVisualization.Wgan,5)
        val latent=StageThreeGenerativeVariantEngine.frame(DeepVisualization.LatentDiffusion,5)
        assertTrue(cycle.values.any{it.first.contains("Cycle")})
        assertTrue(wgan.values.any{it.first.contains("Score gap")})
        assertTrue(latent.flow.contains("Compact latent z"))
    }
}
