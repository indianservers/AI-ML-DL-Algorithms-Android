package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.*
import kotlin.random.Random

internal data class IsolationNode(
    val axis: Int, val threshold: Double, val count: Int,
    val left: IsolationNode? = null, val right: IsolationNode? = null
)
internal data class EllipseFit(
    val centerX: Double, val centerY: Double,
    val xx: Double, val xy: Double, val yy: Double
) {
    fun distance(point: ClusterPoint): Double {
        val dx = point.x-centerX; val dy = point.y-centerY
        val det = (xx*yy-xy*xy).coerceAtLeast(1e-8)
        return sqrt(((yy*dx*dx-2*xy*dx*dy+xx*dy*dy)/det).coerceAtLeast(0.0))
    }
}
internal data class OneClassKernel(val points: List<ClusterPoint>, val alpha: List<Double>, val gamma: Double, val rho: Double) {
    fun score(x: Double,y: Double): Double =
        points.indices.sumOf { i ->
            alpha[i]*exp(-gamma*((x-points[i].x).pow(2)+(y-points[i].y).pow(2)))
        }-rho
}

internal object StageTwoAnomalyEngine {
    private fun d(a: ClusterPoint,b: ClusterPoint) = hypot(a.x-b.x,a.y-b.y)
    fun isolationTree(points: List<ClusterPoint>, depth: Int, seed: Int): IsolationNode {
        if (points.size < 2 || depth >= 12) return IsolationNode(0,0.0,points.size)
        val random=Random(seed)
        val axis=random.nextInt(2)
        val values=points.map { if(axis==0) it.x else it.y }
        val min=values.min(); val max=values.max()
        if(max-min<1e-8) return IsolationNode(axis,min,points.size)
        val threshold=random.nextDouble(min,max)
        val left=points.filter { (if(axis==0) it.x else it.y)<=threshold }
        val right=points.filter { (if(axis==0) it.x else it.y)>threshold }
        if(left.isEmpty() || right.isEmpty()) return IsolationNode(axis,threshold,points.size)
        return IsolationNode(axis,threshold,points.size,
            isolationTree(left,depth+1,seed*31+7),isolationTree(right,depth+1,seed*31+13))
    }
    fun path(node: IsolationNode,point: ClusterPoint,depth: Int=0): Int =
        if(node.left==null || node.right==null) depth
        else path(if((if(node.axis==0) point.x else point.y)<=node.threshold) node.left else node.right,
            point,depth+1)
    fun isolationScore(trees: List<IsolationNode>,point: ClusterPoint,sampleSize: Int): Double {
        val harmonic=(1 until sampleSize).sumOf { 1.0/it }
        val expected=(2*harmonic-2.0*(sampleSize-1)/sampleSize).coerceAtLeast(.1)
        return 2.0.pow(-trees.map { path(it,point).toDouble() }.average()/expected)
    }
    fun lof(points: List<ClusterPoint>,k: Int): List<Double> {
        val neighbors=points.indices.map { i -> points.indices.filter { it!=i }
            .sortedBy { d(points[i],points[it]) }.take(k.coerceIn(1,points.size-1)) }
        val kDistance=points.indices.map { i -> d(points[i],points[neighbors[i].last()]) }
        val density=points.indices.map { i ->
            val mean=neighbors[i].map { j -> max(kDistance[j],d(points[i],points[j])) }.average()
            1.0/mean.coerceAtLeast(1e-8)
        }
        return points.indices.map { i ->
            neighbors[i].map { density[it]/density[i] }.average()
        }
    }
    fun robustEllipse(points: List<ClusterPoint>): EllipseFit {
        var retained=points
        repeat(3) {
            val fit=covariance(retained)
            retained=points.sortedBy { fit.distance(it) }.take((points.size*.8).toInt().coerceAtLeast(5))
        }
        return covariance(retained)
    }
    private fun covariance(points: List<ClusterPoint>): EllipseFit {
        val mx=points.map { it.x }.average(); val my=points.map { it.y }.average()
        return EllipseFit(mx,my,
            points.map { (it.x-mx).pow(2) }.average()+.001,
            points.map { (it.x-mx)*(it.y-my) }.average(),
            points.map { (it.y-my).pow(2) }.average()+.001)
    }
    fun oneClassSvm(points: List<ClusterPoint>,nu: Double,gamma: Double): OneClassKernel {
        val n=points.size
        val kernel=Array(n) { i -> DoubleArray(n) { j ->
            exp(-gamma*((points[i].x-points[j].x).pow(2)+(points[i].y-points[j].y).pow(2)))
        } }
        val cap=1.0/(nu.coerceIn(.02,.95)*n)
        var alpha=DoubleArray(n) { 1.0/n }
        repeat(180) {
            val gradient=DoubleArray(n) { i -> (0 until n).sumOf { j -> kernel[i][j]*alpha[j] } }
            val candidate=DoubleArray(n) { (alpha[it]-.055*gradient[it]).coerceIn(0.0,cap) }
            var low=-2.0; var high=2.0
            repeat(36) {
                val shift=(low+high)/2
                val total=candidate.sumOf { (it+shift).coerceIn(0.0,cap) }
                if(total>1.0) high=shift else low=shift
            }
            val shift=(low+high)/2
            alpha=DoubleArray(n) { (candidate[it]+shift).coerceIn(0.0,cap) }
        }
        val raw=DoubleArray(n) { i -> (0 until n).sumOf { j -> alpha[j]*kernel[i][j] } }
        val margin=alpha.indices.filter { alpha[it]>1e-4 && alpha[it]<cap-1e-4 }
        val rho=(if(margin.isEmpty()) raw.sorted()[((1-nu)*n).toInt().coerceIn(0,n-1)]
            else margin.map { raw[it] }.average())
        return OneClassKernel(points,alpha.toList(),gamma,rho)
    }
    fun quartiles(values: List<Double>): Pair<Double,Double> {
        val sorted=values.sorted()
        fun percentile(p: Double): Double {
            val at=p*(sorted.size-1)
            val lo=at.toInt(); val hi=ceil(at).toInt()
            return sorted[lo]*(1-(at-lo))+sorted[hi]*(at-lo)
        }
        return percentile(.25) to percentile(.75)
    }
}
