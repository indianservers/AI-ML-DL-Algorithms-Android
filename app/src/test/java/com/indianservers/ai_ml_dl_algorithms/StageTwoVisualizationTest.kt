package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.*
import org.junit.Assert.*
import org.junit.Test
import kotlin.math.abs

class StageTwoVisualizationTest {
    private val blobs=PhaseThreeDatasets.clusters(ClusterPreset.Blobs,30,3,.02,7)

    @Test fun allStageTwoPlacementsHaveExplicitNativeBindings() {
        val topics=LearnCatalog.domains.filter { it.title in setOf(
            "Unsupervised Learning","Semi-Supervised Learning","Ensemble Learning"
        ) }.flatMap { it.sections }.flatMap { it.topics }
        assertEquals(60,topics.size)
        assertEquals(60,StageTwoVisualization.entries.size)
        topics.forEach { topic ->
            val registration=AlgorithmVisualizationRegistry.forTopic(topic)
            assertEquals(VisualizationRoute.StageTwoNative,registration.route)
            assertNotNull(registration.stageTwo)
            assertTrue(registration.graphic.isNotBlank())
        }
    }
    @Test fun kMeansTraceAlternatesAssignmentAndCentroidRecomputation() {
        val frames=StageTwoClusteringEngine.kMeansTrace(blobs,3,12,false,null,3)
        assertTrue(frames.any { it.explanation.startsWith("Assign") })
        assertTrue(frames.any { it.explanation.startsWith("Recompute") })
        assertTrue(frames.last().inertia<=frames.first { it.inertia>0 }.inertia+1e-8)
    }
    @Test fun plusPlusProbabilitiesNormalizeAndMiniBatchUsesSubset() {
        val frames=StageTwoClusteringEngine.kMeansTrace(blobs,3,12,true,null,1)
        val probs=frames.first().probabilities
        assertEquals(1.0,probs.sum(),1e-8)
        assertTrue(StageTwoClusteringEngine.kMeansTrace(blobs,3,12,false,null,1).first().probabilities.isEmpty())
        val mini=StageTwoClusteringEngine.kMeansTrace(blobs,3,12,false,5,1)
        assertTrue(mini.any { it.active.size in 1..5 })
    }
    @Test fun hierarchyAndDivisiveTracesChangeClusterCount() {
        val hierarchy=StageTwoClusteringEngine.hierarchy(blobs.take(9),LinkageMethod.Ward)
        assertEquals(8,hierarchy.merges.size)
        assertEquals(9,StageTwoClusteringEngine.hierarchyLabels(9,hierarchy.merges,0).distinct().size)
        assertEquals(1,StageTwoClusteringEngine.hierarchyLabels(9,hierarchy.merges,8).distinct().size)
        val divisive=StageTwoClusteringEngine.divisiveTrace(blobs.take(12),4)
        assertEquals(1,divisive.first().distinct().size)
        assertTrue(divisive.last().distinct().size>1)
    }
    @Test fun densityMethodsProduceCoreNoiseAndReachabilityOrder() {
        val points=PhaseThreeDatasets.clusters(ClusterPreset.Noise,40,2,.02,10)
        val db=PhaseThreeEngines.dbscan(points,.13,4)
        assertTrue(db.core.isNotEmpty())
        assertTrue(db.noise.isNotEmpty())
        val optics=StageTwoClusteringEngine.optics(points,.4,4)
        assertEquals(points.indices.toSet(),optics.order.toSet())
        val h=StageTwoClusteringEngine.densityHierarchy(points,4,.6)
        assertEquals(points.size-1,h.edges.size)
    }
    @Test fun dbscanTraceExpandsClustersAndMatchesFinalModel() {
        val points=PhaseThreeDatasets.clusters(ClusterPreset.Noise,40,2,.02,10)
        val trace=StageTwoClusteringEngine.dbscanTrace(points,.13,4)
        assertTrue(trace.map { it.phase }.toSet().containsAll(
            setOf("Select","Count","Core","Expand")))
        assertTrue(trace.first().labels.any { it == -99 })
        assertEquals(PhaseThreeEngines.dbscan(points,.13,4).labels,trace.last().labels)
    }
    @Test fun fuzzyAndGmmMembershipsNormalize() {
        val fuzzy=StageTwoClusteringEngine.fuzzy(blobs,3,2.0,6)
        fuzzy.membership.forEach { assertEquals(1.0,it.sum(),1e-6) }
        val gmm=PhaseThreeEngines.gmm(blobs,3,4)
        gmm.responsibilities.forEach { assertEquals(1.0,it.sum(),1e-6) }
    }
    @Test fun gmmFitsOffDiagonalCovarianceForTiltedData() {
        val tilted=(0..20).map { index ->
            val x=(index-10)/20.0
            ClusterPoint(x,x*.8+if (index%2==0) .015 else -.015)
        }
        val component=PhaseThreeEngines.gmm(tilted,1,3).components.single()
        assertTrue(component.covarianceXY>.01)
        assertTrue(component.varianceX*component.varianceY>
            component.covarianceXY*component.covarianceXY)
    }
    @Test fun pcaKernelPcaAndSvdHaveDerivedNumericalState() {
        val pca=PhaseThreeEngines.pca(blobs)
        assertTrue(pca.variance1>=pca.variance2-1e-9)
        val kernel=StageTwoReductionEngine.kernelPca(blobs.take(12),3.0)
        assertEquals(12,kernel.embedded.size)
        assertTrue(kernel.embedded.all { it.x.isFinite() && it.y.isFinite() })
        val matrix=listOf(listOf(3.0,1.0),listOf(1.0,2.0))
        assertTrue(StageTwoReductionEngine.truncatedSvd(matrix,1).error>0)
        assertEquals(0.0,StageTwoReductionEngine.truncatedSvd(matrix,2).error,1e-7)
    }
    @Test fun mdsIsomapLleAndEmbeddingsAreFinite() {
        val sample=blobs.take(12)
        val mds=StageTwoReductionEngine.mds(sample)
        assertTrue(mds.metric<1e-5)
        listOf(StageTwoReductionEngine.isomap(sample,4),
            StageTwoReductionEngine.lle(sample,4),
            StageTwoReductionEngine.tsne(sample,4.0,5,.5),
            StageTwoReductionEngine.umap(sample,4,.2,5)).forEach { state ->
            assertEquals(sample.size,state.embedded.size)
            assertTrue(state.embedded.all { it.x.isFinite() && it.y.isFinite() })
        }
    }
    @Test fun associationSupportAprioriFpAndEclatAgree() {
        val baskets=StageTwoAssociationEngine.baskets
        assertEquals(5.0/10,StageTwoAssociationEngine.support(baskets,setOf("Bread","Milk")),1e-8)
        val frequent=StageTwoAssociationEngine.frequent(baskets,.2)
        val eclat=StageTwoAssociationEngine.eclat(baskets,.2)
        assertEquals(frequent.map { it.items }.toSet(),eclat.map { it.items }.toSet())
        val root=StageTwoAssociationEngine.fpTree(baskets,.2,4)
        assertTrue(root.children.isNotEmpty())
    }
    @Test fun associationConfidenceAndLiftUseTransactionCounts() {
        val rules=StageTwoAssociationEngine.rules(StageTwoAssociationEngine.baskets,.2,.5)
        assertTrue(rules.isNotEmpty())
        rules.forEach {
            assertEquals(it.support/
                StageTwoAssociationEngine.support(StageTwoAssociationEngine.baskets,it.premise),
                it.confidence,1e-8)
            assertEquals(it.confidence/
                StageTwoAssociationEngine.support(StageTwoAssociationEngine.baskets,it.conclusion),
                it.lift,1e-8)
        }
    }
    @Test fun anomalyScoresDependOnPathDensityAndCovariance() {
        val dense=listOf(ClusterPoint(0.0,0.0),ClusterPoint(.02,.01),
            ClusterPoint(-.02,.01),ClusterPoint(.01,-.02),ClusterPoint(.03,.02),
            ClusterPoint(.8,.8))
        val tree=StageTwoAnomalyEngine.isolationTree(dense,0,4)
        assertTrue(StageTwoAnomalyEngine.path(tree,dense.last())>=0)
        val lof=StageTwoAnomalyEngine.lof(dense,2)
        assertTrue(lof.last()>1.0)
        val ellipse=StageTwoAnomalyEngine.robustEllipse(dense)
        assertTrue(ellipse.distance(dense.last())>ellipse.distance(dense.first()))
    }
    @Test fun oneClassSvmAndStatisticalFencesRespondToInput() {
        val normal=blobs.take(12)
        val svm=StageTwoAnomalyEngine.oneClassSvm(normal,.2,2.0)
        assertEquals(12,svm.alpha.size)
        assertEquals(1.0,svm.alpha.sum(),1e-5)
        val q=StageTwoAnomalyEngine.quartiles(listOf(1.0,2.0,3.0,4.0,20.0))
        assertTrue(q.first<q.second)
    }
    @Test fun pseudoLabelsAndGraphDiffusionDoNotRevealHiddenLabels() {
        val points=PhaseThreeDatasets.clusters(ClusterPreset.Blobs,24,2,.04,12)
        val base=StageTwoSemiSupervisedEngine.selfTraining(points,0,.6,false)
        val advanced=StageTwoSemiSupervisedEngine.selfTraining(points,3,.6,false)
        assertTrue(advanced.labeled.size>=base.labeled.size)
        val hard=StageTwoSemiSupervisedEngine.graphPropagation(points,5,false,.8,4)
        for(i in 0..3) assertEquals(points[i].hiddenLabel.toDouble(),hard.probabilities[i],1e-9)
        assertTrue(hard.probabilities.all { it in 0.0..1.0 })
    }
    @Test fun teacherEmaFixMatchAndMixMatchAreNumerical() {
        assertEquals(.3,StageTwoSemiSupervisedEngine.teacherEma(.8,.2,.8333333333),1e-6)
        val points=PhaseThreeDatasets.clusters(ClusterPreset.Blobs,24,2,.04,12)
        val gated=StageTwoSemiSupervisedEngine.consistency(points,8,.5,.99,true)
        if(!gated.accepted) assertEquals(0.0,gated.loss,0.0)
        val mix=StageTwoSemiSupervisedEngine.mixMatch(points,8,.5,.6)
        assertEquals(5,mix.size)
        assertTrue(mix.all { it in 0.0..1.0 })
    }
    @Test fun baggingVotesAndMetaModelsUseRealBaseOutputs() {
        val points=PhaseThreeDatasets.clusters(ClusterPreset.Blobs,30,2,.05,11)
            .map { LabPoint(it.x,it.y,it.hiddenLabel) }
        val query=LabPoint(.1,.1)
        val bag=StageTwoEnsembleEngine.bootstrap(points,6,6,query)
        assertEquals(6,bag.size)
        assertTrue(bag.any { it.indices.toSet().size<points.size })
        val vote=StageTwoEnsembleEngine.votes(points,query)
        assertEquals(vote.probabilities.average(),vote.softVote,1e-9)
        val stack=StageTwoEnsembleEngine.stacking(points,query)
        assertEquals(points.size,stack.metaRows.size)
        assertEquals(4,stack.metaWeights.size)
        val blend=StageTwoEnsembleEngine.blending(points,query)
        assertTrue(blend.metaRows.size<points.size)
    }
}
