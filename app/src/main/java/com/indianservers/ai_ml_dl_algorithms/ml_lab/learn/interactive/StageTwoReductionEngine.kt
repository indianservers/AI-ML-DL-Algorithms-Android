package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.*

internal data class EigenSystem(val values: List<Double>, val vectors: List<List<Double>>)
internal data class ReductionState(
    val source: List<ClusterPoint>, val embedded: List<ClusterPoint>,
    val metric: Double, val metricLabel: String, val detail: String,
    val edges: List<Pair<Int,Int>> = emptyList()
)
internal data class SvdState(val original: List<List<Double>>, val reconstruction: List<List<Double>>,
    val singular: List<Double>, val error: Double)

internal object StageTwoReductionEngine {
    fun eigen(matrix: List<List<Double>>): EigenSystem {
        val n=matrix.size
        val a=Array(n) { matrix[it].toDoubleArray() }
        val v=Array(n) { i -> DoubleArray(n) { j -> if(i==j) 1.0 else 0.0 } }
        for (iteration in 0 until n*n*25) {
            var p=0; var q=1; var largest=0.0
            for(i in 0 until n) for(j in i+1 until n)
                if(abs(a[i][j])>largest) { largest=abs(a[i][j]); p=i; q=j }
            if(largest<1e-9) break
            val angle=.5*atan2(2*a[p][q],a[q][q]-a[p][p])
            val c=cos(angle); val s=sin(angle)
            val app=a[p][p]; val aqq=a[q][q]; val apq=a[p][q]
            a[p][p]=c*c*app-2*s*c*apq+s*s*aqq
            a[q][q]=s*s*app+2*s*c*apq+c*c*aqq
            a[p][q]=0.0; a[q][p]=0.0
            for(i in 0 until n) if(i!=p && i!=q) {
                val aip=a[i][p]; val aiq=a[i][q]
                a[i][p]=c*aip-s*aiq; a[p][i]=a[i][p]
                a[i][q]=s*aip+c*aiq; a[q][i]=a[i][q]
            }
            for(i in 0 until n) {
                val vip=v[i][p]; val viq=v[i][q]
                v[i][p]=c*vip-s*viq; v[i][q]=s*vip+c*viq
            }
        }
        val sorted=(0 until n).sortedByDescending { a[it][it] }
        return EigenSystem(sorted.map { a[it][it] },sorted.map { j -> List(n) { i -> v[i][j] } })
    }
    fun kernelPca(points: List<ClusterPoint>,gamma: Double): ReductionState {
        val n=points.size
        val raw=List(n) { i -> List(n) { j ->
            exp(-gamma*((points[i].x-points[j].x).pow(2)+(points[i].y-points[j].y).pow(2)))
        } }
        val row=raw.map { it.average() }; val grand=row.average()
        val centered=List(n) { i -> List(n) { j -> raw[i][j]-row[i]-row[j]+grand } }
        val eig=eigen(centered)
        val x=eig.vectors[0]; val y=eig.vectors[1]
        val sx=sqrt(eig.values[0].coerceAtLeast(0.0))
        val sy=sqrt(eig.values[1].coerceAtLeast(0.0))
        val embedded=normalize(List(n) { i -> ClusterPoint(x[i]*sx,y[i]*sy) })
        return ReductionState(points,embedded,
            eig.values.take(2).sum()/eig.values.filter { it>0 }.sum().coerceAtLeast(1e-9),
            "retained kernel variance","RBF kernel similarities are centered, then eigenvectors provide a nonlinear feature-space projection.")
    }
    fun truncatedSvd(matrix: List<List<Double>>,rank: Int): SvdState {
        val rows=matrix.size; val cols=matrix[0].size
        val ata=List(cols) { i -> List(cols) { j -> (0 until rows).sumOf { r -> matrix[r][i]*matrix[r][j] } } }
        val eig=eigen(ata)
        val singular=eig.values.map { sqrt(it.coerceAtLeast(0.0)) }
        val keep=rank.coerceIn(1,cols)
        val reconstructed=List(rows) { r -> List(cols) { c ->
            (0 until keep).sumOf { component ->
                val v=eig.vectors[component]
                v[c]*(0 until cols).sumOf { j -> matrix[r][j]*v[j] }
            }
        } }
        val error=matrix.indices.sumOf { r -> matrix[r].indices.sumOf { c ->
            (matrix[r][c]-reconstructed[r][c]).pow(2)
        } }/rows/cols
        return SvdState(matrix,reconstructed,singular,error)
    }
    fun mds(points: List<ClusterPoint>,distances: List<List<Double>>?=null): ReductionState {
        val n=points.size
        val squared=List(n) { i -> List(n) { j ->
            (distances?.get(i)?.get(j) ?: hypot(points[i].x-points[j].x,points[i].y-points[j].y)).pow(2)
        } }
        val row=squared.map { it.average() }; val grand=row.average()
        val gram=List(n) { i -> List(n) { j -> -.5*(squared[i][j]-row[i]-row[j]+grand) } }
        val eig=eigen(gram)
        val rawEmbedding=List(n) { i ->
            ClusterPoint(eig.vectors[0][i]*sqrt(eig.values[0].coerceAtLeast(0.0)),
                eig.vectors[1][i]*sqrt(eig.values[1].coerceAtLeast(0.0)))
        }
        val stress=sqrt((0 until n).sumOf { i -> (i+1 until n).sumOf { j ->
            val wanted=sqrt(squared[i][j])
            (wanted-hypot(rawEmbedding[i].x-rawEmbedding[j].x,
                rawEmbedding[i].y-rawEmbedding[j].y)).pow(2)
        } }/(0 until n).sumOf { i -> (i+1 until n).sumOf { j -> squared[i][j] } }.coerceAtLeast(1e-9))
        return ReductionState(points,normalize(rawEmbedding),stress,"normalized stress",
            "Double-center squared distances, retain the top two eigenvectors, and compare reconstructed pairwise distances.")
    }
    fun isomap(points: List<ClusterPoint>,neighbors: Int): ReductionState {
        val n=points.size
        val graph=PhaseThreeEngines.similarityGraph(points,neighbors)
        val distance=Array(n) { i -> DoubleArray(n) { j -> if(i==j) 0.0 else 1e6 } }
        graph.edges.forEach { (a,b) ->
            val d=hypot(points[a].x-points[b].x,points[a].y-points[b].y)
            distance[a][b]=d; distance[b][a]=d
        }
        for(k in 0 until n) for(i in 0 until n) for(j in 0 until n)
            distance[i][j]=min(distance[i][j],distance[i][k]+distance[k][j])
        val max=distance.flatMap { it.toList() }.filter { it<1e5 }.maxOrNull() ?: 1.0
        val result=mds(points,List(n) { i -> List(n) { j ->
            distance[i][j].takeIf { it<1e5 } ?: max*1.5
        } })
        return result.copy(detail="K-neighbor edges define geodesic shortest paths; classical scaling unfolds those distances.",
            edges=graph.edges)
    }
    fun lle(points: List<ClusterPoint>,neighbors: Int): ReductionState {
        val n=points.size; val k=neighbors.coerceIn(2,n-1)
        val graph=PhaseThreeEngines.similarityGraph(points,k)
        val w=Array(n) { DoubleArray(n) }
        for(i in 0 until n) {
            val near=(0 until n).filter { it!=i }.sortedBy {
                (points[i].x-points[it].x).pow(2)+(points[i].y-points[it].y).pow(2)
            }.take(k)
            val gram=List(k) { a -> List(k) { b ->
                (points[i].x-points[near[a]].x)*(points[i].x-points[near[b]].x)+
                (points[i].y-points[near[a]].y)*(points[i].y-points[near[b]].y)+
                if(a==b) .001 else 0.0
            } }
            val weights=solve(gram,List(k) { 1.0 })
            val total=weights.sum().takeIf { abs(it)>1e-9 } ?: 1.0
            near.forEachIndexed { a,j -> w[i][j]=weights[a]/total }
        }
        val m=List(n) { i -> List(n) { j ->
            (0 until n).sumOf { row ->
                (if(row==i) 1.0 else 0.0-w[row][i])*
                    (if(row==j) 1.0 else 0.0-w[row][j])
            }
        } }
        val eig=eigen(m)
        val embedded=normalize(List(n) { i ->
            ClusterPoint(eig.vectors[n-2][i],eig.vectors[n-3][i])
        })
        val loss=(0 until n).sumOf { i ->
            val rx=(0 until n).sumOf { j -> w[i][j]*embedded[j].x }
            val ry=(0 until n).sumOf { j -> w[i][j]*embedded[j].y }
            (embedded[i].x-rx).pow(2)+(embedded[i].y-ry).pow(2)
        }/n
        return ReductionState(points,embedded,loss,"local reconstruction loss",
            "Reconstruction weights from each local neighborhood are preserved after unfolding.",
            graph.edges)
    }
    fun tsne(points: List<ClusterPoint>,perplexity: Double,steps: Int,rate: Double): ReductionState {
        val n=points.size
        val distances=List(n) { i -> List(n) { j ->
            (points[i].x-points[j].x).pow(2)+(points[i].y-points[j].y).pow(2)
        } }
        val high=List(n) { i ->
            var beta=1.0; var lower=0.0; var upper=Double.POSITIVE_INFINITY
            repeat(35) {
                val weights=List(n) { j -> if(i==j) 0.0 else exp(-beta*distances[i][j]) }
                val total=weights.sum().coerceAtLeast(1e-12)
                val entropy=ln(total)+beta*(0 until n).sumOf { j ->
                    distances[i][j]*weights[j]
                }/total
                if(entropy>ln(perplexity.coerceIn(2.0,(n-1).toDouble()))) {
                    lower=beta
                    beta=if(upper.isInfinite()) beta*2 else (beta+upper)/2
                } else {
                    upper=beta; beta=(beta+lower)/2
                }
            }
            List(n) { j -> if(i==j) 0.0 else exp(-beta*distances[i][j]) }
        }
        val conditional=List(n) { i -> val sum=high[i].sum().coerceAtLeast(1e-9)
            List(n) { j -> high[i][j]/sum } }
        val p=List(n) { i -> List(n) { j -> (conditional[i][j]+conditional[j][i])/(2*n) } }
        val embedding=MutableList(n) { i -> ClusterPoint(sin(i*12.989)*.05,cos(i*4.141)*.05) }
        repeat(steps.coerceIn(0,100)) {
            val q=Array(n) { i -> DoubleArray(n) { j ->
                if(i==j) 0.0 else 1.0/(1.0+(embedding[i].x-embedding[j].x).pow(2)+
                    (embedding[i].y-embedding[j].y).pow(2))
            } }
            val sum=q.sumOf { row -> row.sum() }.coerceAtLeast(1e-9)
            val next=embedding.mapIndexed { i,at ->
                var gx=0.0; var gy=0.0
                for(j in 0 until n) if(i!=j) {
                    val coefficient=4*(p[i][j]-q[i][j]/sum)*q[i][j]
                    gx+=coefficient*(at.x-embedding[j].x)
                    gy+=coefficient*(at.y-embedding[j].y)
                }
                ClusterPoint((at.x+rate*gx).coerceIn(-2.0,2.0),
                    (at.y+rate*gy).coerceIn(-2.0,2.0))
            }
            embedding.clear(); embedding.addAll(next)
        }
        val mapped=normalize(embedding)
        val near=PhaseThreeEngines.similarityGraph(points,3).edges
        return ReductionState(points,mapped,
            neighborOverlap(points,mapped,3),"local neighbor preservation",
            "Student-t repulsion and high-dimensional neighbor affinities update the map; global distances are unreliable.",
            near)
    }
    fun umap(points: List<ClusterPoint>,neighbors: Int,minDist: Double,steps: Int): ReductionState {
        val n=points.size
        val graph=PhaseThreeEngines.similarityGraph(points,neighbors)
        val edges=graph.edges.map { (a,b) ->
            Triple(a,b,exp(-hypot(points[a].x-points[b].x,points[a].y-points[b].y)*
                neighbors.coerceAtLeast(1)))
        }
        val embedding=MutableList(n) { i ->
            ClusterPoint(points[i].x*.5+sin(i*2.7)*.2,points[i].y*.5+cos(i*3.1)*.2)
        }
        repeat(steps.coerceIn(0,100)) { iter ->
            val next=embedding.toMutableList()
            val eta=.08*(1.0-iter/110.0)
            edges.forEach { (a,b,w) ->
                val dx=next[b].x-next[a].x; val dy=next[b].y-next[a].y
                val dist=hypot(dx,dy).coerceAtLeast(1e-6)
                val pull=eta*w*(dist-minDist)/dist
                next[a]=next[a].copy(x=next[a].x+pull*dx,y=next[a].y+pull*dy)
                next[b]=next[b].copy(x=next[b].x-pull*dx,y=next[b].y-pull*dy)
            }
            for(i in 0 until n) {
                val j=(i*17+iter*11+7)%n
                if(i==j) continue
                val dx=next[i].x-next[j].x; val dy=next[i].y-next[j].y
                val factor=eta*.002/(dx*dx+dy*dy+.01)
                next[i]=next[i].copy(x=next[i].x+factor*dx,y=next[i].y+factor*dy)
            }
            embedding.clear(); embedding.addAll(next)
        }
        val mapped=normalize(embedding)
        return ReductionState(points,mapped,neighborOverlap(points,mapped,neighbors),
            "graph neighbor preservation",
            "A fuzzy nearest-neighbor graph attracts connected pairs while non-neighbors repel; minDist limits compactness.",
            graph.edges)
    }
    fun linearAutoencoder(points: List<ClusterPoint>,steps: Int): ReductionState {
        val pca=PhaseThreeEngines.pca(points)
        var wx=.7; var wy=.7
        repeat(steps.coerceIn(0,120)) {
            for(p in points) {
                val x=p.x-pca.mean.x; val y=p.y-pca.mean.y
                val z=wx*x+wy*y
                val rx=wx*z; val ry=wy*z
                val gx=2*((rx-x)*z+((rx-x)*wx+(ry-y)*wy)*x)
                val gy=2*((ry-y)*z+((rx-x)*wx+(ry-y)*wy)*y)
                wx-=.008*gx; wy-=.008*gy
            }
        }
        val reconstructed=points.map { p ->
            val z=wx*(p.x-pca.mean.x)+wy*(p.y-pca.mean.y)
            ClusterPoint(pca.mean.x+wx*z,pca.mean.y+wy*z)
        }
        val error=points.indices.sumOf {
            (points[it].x-reconstructed[it].x).pow(2)+
                (points[it].y-reconstructed[it].y).pow(2)
        }/points.size
        return ReductionState(points,reconstructed,error,"reconstruction MSE",
            "A trained tied-weight 2→1→2 linear autoencoder compresses each sample into one latent value.")
    }
    private fun neighborOverlap(a: List<ClusterPoint>,b: List<ClusterPoint>,k: Int): Double =
        a.indices.map { i ->
            fun near(points: List<ClusterPoint>) = points.indices.filter { it!=i }
                .sortedBy { (points[i].x-points[it].x).pow(2)+(points[i].y-points[it].y).pow(2) }
                .take(k).toSet()
            near(a).intersect(near(b)).size.toDouble()/k
        }.average()
    private fun normalize(points: List<ClusterPoint>): List<ClusterPoint> {
        val mx=points.map { it.x }.average(); val my=points.map { it.y }.average()
        val scale=max(points.maxOf { abs(it.x-mx) },points.maxOf { abs(it.y-my) }).coerceAtLeast(1e-9)
        return points.map { ClusterPoint((it.x-mx)/scale*.8,(it.y-my)/scale*.8) }
    }
    private fun solve(matrix: List<List<Double>>,target: List<Double>): List<Double> {
        val n=target.size; val a=Array(n) { i -> DoubleArray(n+1) { j ->
            if(j==n) target[i] else matrix[i][j]
        } }
        for(c in 0 until n) {
            val pivot=(c until n).maxBy { abs(a[it][c]) }
            val tmp=a[c]; a[c]=a[pivot]; a[pivot]=tmp
            val divisor=a[c][c].takeIf { abs(it)>1e-10 } ?: 1e-10
            for(j in c..n) a[c][j]/=divisor
            for(i in 0 until n) if(i!=c) {
                val f=a[i][c]
                for(j in c..n) a[i][j]-=f*a[c][j]
            }
        }
        return List(n) { a[it][n] }
    }
}
