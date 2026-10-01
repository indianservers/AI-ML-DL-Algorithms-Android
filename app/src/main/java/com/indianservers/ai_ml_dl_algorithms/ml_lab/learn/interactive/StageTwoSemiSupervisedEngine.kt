package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.*

internal data class SemiState(
    val probabilities: List<Double>, val labeled: Set<Int>,
    val newlyLabeled: Set<Int>, val confidence: Double,
    val detail: String
)
internal data class ConsistencyState(
    val weak: ClusterPoint, val strong: ClusterPoint,
    val weakProbability: Double, val strongProbability: Double,
    val accepted: Boolean, val loss: Double
)

internal object StageTwoSemiSupervisedEngine {
    private fun sigmoid(x: Double) = 1.0/(1.0+exp(-x.coerceIn(-30.0,30.0)))
    private fun distance(a: ClusterPoint,b: ClusterPoint) = hypot(a.x-b.x,a.y-b.y)
    private fun initialLabels(points: List<ClusterPoint>): Set<Int> =
        setOf(0,1,2,3).filter { it in points.indices }.toSet()
    fun knnProbability(points: List<ClusterPoint>,labeled: Map<Int,Int>,index: Int,
                       view: Int=2): Double = knnProbabilityAt(points,labeled,points[index],view)
    fun knnProbabilityAt(points: List<ClusterPoint>,labeled: Map<Int,Int>,query: ClusterPoint,
                         view: Int=2): Double {
        val neighbors=labeled.keys.sortedBy { i ->
            when(view) {
                0 -> abs(points[i].x-query.x)
                1 -> abs(points[i].y-query.y)
                else -> distance(points[i],query)
            }
        }.take(5)
        val weight=neighbors.map { i ->
            val d=when(view) {
                0 -> abs(points[i].x-query.x)
                1 -> abs(points[i].y-query.y)
                else -> distance(points[i],query)
            }
            1.0/(d+.04)
        }
        return neighbors.indices.sumOf { if(labeled[neighbors[it]]==1) weight[it] else 0.0 }/
            weight.sum().coerceAtLeast(1e-9)
    }
    fun selfTraining(points: List<ClusterPoint>,step: Int,threshold: Double,
                     allAtOnce: Boolean): SemiState {
        val labeled=initialLabels(points).associateWith { points[it].hiddenLabel }.toMutableMap()
        var newest=emptySet<Int>()
        repeat(step.coerceIn(0,points.size)) {
            val candidate=points.indices.filter { it !in labeled }.map { i ->
                i to knnProbability(points,labeled,i)
            }.filter { (_,p) -> max(p,1-p)>=threshold }
            val added=if(allAtOnce) candidate
                else listOfNotNull(candidate.maxByOrNull { (_,p) -> max(p,1-p) })
            newest=added.map { it.first }.toSet()
            added.forEach { (i,p) -> labeled[i]=if(p>=.5) 1 else 0 }
        }
        val probabilities=points.indices.map { i ->
            if(i in initialLabels(points)) points[i].hiddenLabel.toDouble()
            else knnProbability(points,labeled,i)
        }
        val confidence=points.indices.filter { it !in initialLabels(points) }
            .map { max(probabilities[it],1-probabilities[it]) }.average()
        return SemiState(probabilities,labeled.keys,newest,confidence,
            if(allAtOnce) "Add every prediction above the confidence gate, then retrain."
            else "Admit the single most confident unlabeled prediction and refit.")
    }
    fun graphPropagation(points: List<ClusterPoint>,step: Int,soft: Boolean,
                         alpha: Double,neighbors: Int): SemiState {
        val n=points.size; val seeds=initialLabels(points)
        val edges=PhaseThreeEngines.similarityGraph(points,neighbors).edges
        val weights=Array(n) { DoubleArray(n) }
        edges.forEach { (a,b) ->
            val w=exp(-distance(points[a],points[b]).pow(2)/.12)
            weights[a][b]=w; weights[b][a]=w
        }
        val original=DoubleArray(n) { if(it in seeds) points[it].hiddenLabel.toDouble() else .5 }
        var current=original.copyOf()
        repeat(step.coerceIn(0,40)) {
            val next=DoubleArray(n) { i ->
                val total=weights[i].sum()
                val spread=if(total>0) (0 until n).sumOf { j -> weights[i][j]*current[j] }/total
                    else current[i]
                when {
                    !soft && i in seeds -> original[i]
                    soft -> alpha*spread+(1-alpha)*original[i]
                    else -> spread
                }
            }
            current=next
        }
        return SemiState(current.toList(),seeds,emptySet(),current.map { max(it,1-it) }.average(),
            if(soft) "Soft clamping blends each diffusion step with the original label distribution."
            else "Labeled seeds remain fixed while probabilities diffuse over weighted graph edges.")
    }
    fun coTraining(points: List<ClusterPoint>,step: Int,threshold: Double): Pair<SemiState,SemiState> {
        val labelsA=initialLabels(points).associateWith { points[it].hiddenLabel }.toMutableMap()
        val labelsB=initialLabels(points).associateWith { points[it].hiddenLabel }.toMutableMap()
        var newlyA=emptySet<Int>(); var newlyB=emptySet<Int>()
        repeat(step.coerceIn(0,12)) {
            fun selected(from: Map<Int,Int>,view: Int): Pair<Int,Double>? =
                points.indices.filter { it !in labelsA && it !in labelsB }
                    .map { it to knnProbability(points,from,it,view) }
                    .filter { (_,p) -> max(p,1-p)>=threshold }
                    .maxByOrNull { (_,p) -> max(p,1-p) }
            val a=selected(labelsA,0); val b=selected(labelsB,1)
            newlyB=if(a==null) emptySet() else setOf(a.first)
            newlyA=if(b==null) emptySet() else setOf(b.first)
            if(a!=null) labelsB[a.first]=if(a.second>=.5) 1 else 0
            if(b!=null) labelsA[b.first]=if(b.second>=.5) 1 else 0
        }
        val pa=points.indices.map { knnProbability(points,labelsA,it,0) }
        val pb=points.indices.map { knnProbability(points,labelsB,it,1) }
        return SemiState(pa,labelsA.keys,newlyA,pa.map { max(it,1-it) }.average(),
            "View A uses X1 and accepts View B's confident pseudo-labels.") to
            SemiState(pb,labelsB.keys,newlyB,pb.map { max(it,1-it) }.average(),
                "View B uses X2 and accepts View A's confident pseudo-labels.")
    }
    fun consistency(points: List<ClusterPoint>,index: Int,strength: Double,threshold: Double,
                    strong: Boolean): ConsistencyState {
        val p=points[index]
        val weak=ClusterPoint((p.x+strength*.12).coerceIn(-1.0,1.0),
            (p.y-strength*.08).coerceIn(-1.0,1.0))
        val amplified=if(strong) 2.8 else 1.0
        val altered=ClusterPoint((p.x-strength*.25*amplified).coerceIn(-1.0,1.0),
            (p.y+strength*.19*amplified).coerceIn(-1.0,1.0))
        val labeled=initialLabels(points).associateWith { points[it].hiddenLabel }
        val weakP=knnProbabilityAt(points,labeled,weak)
        val strongP=knnProbabilityAt(points,labeled,altered)
        val accepted=max(weakP,1-weakP)>=threshold
        return ConsistencyState(weak,altered,weakP,strongP,accepted,
            if(accepted || !strong) (weakP-strongP).pow(2) else 0.0)
    }
    fun teacherEma(student: Double,teacher: Double,decay: Double): Double =
        decay*teacher+(1-decay)*student
    fun teacherTrace(steps: Int,decay: Double): List<Pair<Double,Double>> {
        var student=.25; var teacher=.25
        return buildList {
            add(student to teacher)
            repeat(steps.coerceIn(0,40)) { t ->
                val target=.7+sin(t*.8)*.1
                student+=.28*(target-student)
                teacher=teacherEma(student,teacher,decay)
                add(student to teacher)
            }
        }
    }
    fun sharpen(probability: Double,temperature: Double): Double {
        val a=probability.coerceIn(1e-6,1.0-1e-6).pow(1.0/temperature)
        val b=(1-probability).coerceIn(1e-6,1.0-1e-6).pow(1.0/temperature)
        return a/(a+b)
    }
    fun mixMatch(points: List<ClusterPoint>,index: Int,temperature: Double,mix: Double): List<Double> {
        val labeled=initialLabels(points).associateWith { points[it].hiddenLabel }
        val p1=knnProbability(points,labeled,index)
        val p2=(p1+.05*sin(index.toDouble())).coerceIn(0.0,1.0)
        val guess=(p1+p2)/2
        val sharp=sharpen(guess,temperature)
        val partner=points[labeled.keys.first()]
        val mixed=mix*sharp+(1-mix)*partner.hiddenLabel
        return listOf(p1,p2,guess,sharp,mixed)
    }
}
