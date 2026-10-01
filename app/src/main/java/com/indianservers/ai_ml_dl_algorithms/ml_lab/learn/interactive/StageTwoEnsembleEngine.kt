package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.*
import kotlin.random.Random

internal data class BootstrapLearner(
    val indices: List<Int>, val tree: SupervisedTreeNode, val probability: Double
)
internal data class CombinedModel(
    val probabilities: List<Double>, val hardVotes: List<Int>,
    val softVote: Double, val hardVote: Int,
    val metaWeights: List<Double> = emptyList(),
    val metaPrediction: Double = 0.0,
    val metaRows: List<List<Double>> = emptyList(),
    val metaLabels: List<Int> = emptyList()
)

internal object StageTwoEnsembleEngine {
    private fun sigmoid(x: Double)=1.0/(1.0+exp(-x.coerceIn(-30.0,30.0)))
    fun bootstrap(points: List<LabPoint>,trees: Int,seed: Int,query: LabPoint): List<BootstrapLearner> =
        List(trees.coerceIn(1,30)) { round ->
            val random=Random(seed+round*127)
            val indices=List(points.size) { random.nextInt(points.size) }
            val tree=SupervisedTreeEngine.fit(indices.map { points[it] },false,3,2,seed=seed+round)
            BootstrapLearner(indices,tree,tree.predict(query.x,query.y))
        }
    private fun logistic(points: List<LabPoint>): (LabPoint)->Double {
        var b=0.0; var wx=0.0; var wy=0.0
        repeat(240) {
            var gb=0.0; var gx=0.0; var gy=0.0
            points.forEach { p ->
                val error=sigmoid(b+wx*p.x+wy*p.y)-p.label
                gb+=error; gx+=error*p.x; gy+=error*p.y
            }
            val rate=.35/points.size.coerceAtLeast(1)
            b-=rate*gb; wx-=rate*gx; wy-=rate*gy
        }
        return { p -> sigmoid(b+wx*p.x+wy*p.y) }
    }
    fun baseProbabilities(train: List<LabPoint>,query: LabPoint): List<Double> {
        if(train.isEmpty()) return listOf(.5,.5,.5)
        val linear=logistic(train)(query)
        val nearest=train.sortedBy { hypot(it.x-query.x,it.y-query.y) }.take(5)
        val knn=nearest.sumOf { p ->
            p.label/(hypot(p.x-query.x,p.y-query.y)+.05)
        }/nearest.sumOf { 1.0/(hypot(it.x-query.x,it.y-query.y)+.05) }
        var leaf=SupervisedTreeEngine.fit(train,false,3,2)
        while (!leaf.isLeaf) {
            val value=if(leaf.feature==0) query.x else query.y
            leaf=(if(value<=(leaf.threshold ?: 0.0)) leaf.left else leaf.right) ?: leaf
            if(leaf.left==null && leaf.right==null) break
        }
        val positive=leaf.counts[1] ?: 0
        val total=leaf.counts.values.sum()
        val tree=(positive+1.0)/(total+2.0)
        return listOf(linear,knn,tree)
    }
    fun votes(points: List<LabPoint>,query: LabPoint): CombinedModel {
        val probabilities=baseProbabilities(points,query)
        val hard=probabilities.map { if(it>=.5) 1 else 0 }
        return CombinedModel(probabilities,hard,probabilities.average(),
            if(hard.sum()*2>=hard.size) 1 else 0)
    }
    private fun fitMeta(rows: List<List<Double>>,labels: List<Int>): List<Double> {
        var weights=DoubleArray(4)
        repeat(300) {
            val gradient=DoubleArray(4)
            rows.indices.forEach { i ->
                val x=listOf(1.0)+rows[i]
                val error=sigmoid((0..3).sumOf { j -> weights[j]*x[j] })-labels[i]
                for(j in 0..3) gradient[j]+=error*x[j]
            }
            for(j in 0..3) weights[j]-=.35*gradient[j]/rows.size.coerceAtLeast(1)
        }
        return weights.toList()
    }
    fun stacking(points: List<LabPoint>,query: LabPoint,folds: Int=3): CombinedModel {
        val foldCount=folds.coerceIn(2,5)
        val rows=points.indices.map { i ->
            val train=points.filterIndexed { j,_ -> j%foldCount!=i%foldCount }
            baseProbabilities(train,points[i])
        }
        val labels=points.map { it.label }
        val meta=fitMeta(rows,labels)
        val queryFeatures=baseProbabilities(points,query)
        val prediction=sigmoid(meta[0]+queryFeatures.indices.sumOf { meta[it+1]*queryFeatures[it] })
        val vote=votes(points,query)
        return vote.copy(metaWeights=meta,metaPrediction=prediction,
            metaRows=rows,metaLabels=labels)
    }
    fun blending(points: List<LabPoint>,query: LabPoint,holdout: Double=.25): CombinedModel {
        val split=((1-holdout.coerceIn(.15,.45))*points.size).toInt().coerceIn(5,points.size-2)
        val baseTrain=points.take(split)
        val heldout=points.drop(split)
        val rows=heldout.map { baseProbabilities(baseTrain,it) }
        val labels=heldout.map { it.label }
        val meta=fitMeta(rows,labels)
        val features=baseProbabilities(baseTrain,query)
        val prediction=sigmoid(meta[0]+features.indices.sumOf { meta[it+1]*features[it] })
        val vote=votes(baseTrain,query)
        return vote.copy(metaWeights=meta,metaPrediction=prediction,
            metaRows=rows,metaLabels=labels)
    }
}
