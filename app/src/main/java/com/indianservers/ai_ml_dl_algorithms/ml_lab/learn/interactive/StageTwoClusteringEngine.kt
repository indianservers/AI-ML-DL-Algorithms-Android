package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.*
import kotlin.random.Random

internal data class CentroidFrame(
    val centers: List<ClusterPoint>,
    val assignments: List<Int>,
    val active: Set<Int>,
    val probabilities: List<Double>,
    val explanation: String,
    val inertia: Double
)

internal data class OpticsOrder(val order: List<Int>, val reachability: List<Double>, val core: List<Double>)
internal data class DbscanFrame(
    val labels: List<Int>, val focus: Int, val neighbors: Set<Int>,
    val phase: String, val explanation: String
)
internal data class FuzzyState(val centers: List<ClusterPoint>, val membership: List<List<Double>>)
internal data class DensityHierarchy(val edges: List<Triple<Int, Int, Double>>, val labels: List<Int>, val cutoff: Double)
internal data class CfEntry(val count: Int, val sumX: Double, val sumY: Double, val sumSquares: Double) {
    val center get() = ClusterPoint(sumX / count, sumY / count)
}
internal data class AffinityState(val exemplars: List<Int>, val responsibilities: List<Double>, val availabilities: List<Double>)

internal object StageTwoClusteringEngine {
    private fun d2(a: ClusterPoint, b: ClusterPoint) = (a.x - b.x).pow(2) + (a.y - b.y).pow(2)
    private fun inertia(points: List<ClusterPoint>, centers: List<ClusterPoint>, labels: List<Int>) =
        points.indices.sumOf { d2(points[it], centers[labels[it]]) }
    private fun assign(points: List<ClusterPoint>, centers: List<ClusterPoint>) =
        points.map { p -> centers.indices.minBy { d2(p, centers[it]) } }

    fun dbscanTrace(points: List<ClusterPoint>, eps: Double, minPts: Int): List<DbscanFrame> {
        if (points.isEmpty()) return emptyList()
        val neighborhoods=points.indices.map { i ->
            points.indices.filter { j -> d2(points[i],points[j])<=eps*eps }.toSet()
        }
        val labels=MutableList(points.size) { -99 }
        val visited=BooleanArray(points.size)
        val frames=mutableListOf<DbscanFrame>()
        var cluster=0
        fun emit(focus: Int, phase: String, explanation: String) {
            frames+=DbscanFrame(labels.toList(),focus,neighborhoods[focus],phase,explanation)
        }
        for (seed in points.indices) {
            if (visited[seed]) continue
            emit(seed,"Select","Select an unvisited sample.")
            visited[seed]=true
            emit(seed,"Count","Its epsilon neighborhood contains " +
                neighborhoods[seed].size + " samples; minPts is " + minPts + ".")
            if (neighborhoods[seed].size<minPts) {
                labels[seed]=-1
                emit(seed,"Noise","This sample is provisionally noise; a later core can absorb it as a border point.")
                continue
            }
            labels[seed]=cluster
            emit(seed,"Core","The sample is core. Begin density-connected expansion.")
            val queue=ArrayDeque<Int>()
            queue.addAll(neighborhoods[seed].filter { it!=seed })
            val queued=queue.toMutableSet().apply { add(seed) }
            while (queue.isNotEmpty()) {
                val current=queue.removeFirst()
                if (!visited[current]) {
                    visited[current]=true
                    if (neighborhoods[current].size>=minPts) {
                        neighborhoods[current].forEach { neighbor ->
                            if (neighbor !in queued) {
                                queue.add(neighbor)
                                queued.add(neighbor)
                            }
                        }
                    }
                }
                if (labels[current]<0) labels[current]=cluster
                emit(current,"Expand",if (neighborhoods[current].size>=minPts)
                    "Core sample adds its epsilon neighbors to the expansion queue."
                    else "Border sample joins this cluster but does not expand it.")
            }
            cluster++
        }
        return frames
    }

    fun kMeansTrace(
        points: List<ClusterPoint>, k: Int, seed: Int, plusPlus: Boolean,
        miniBatch: Int?, rounds: Int = 8
    ): List<CentroidFrame> {
        if (points.isEmpty()) return emptyList()
        val count = k.coerceIn(1, points.size)
        val random = Random(seed)
        val centers = mutableListOf<ClusterPoint>()
        centers += points[random.nextInt(points.size)]
        val frames = mutableListOf<CentroidFrame>()
        frames += CentroidFrame(centers.toList(), List(points.size) { 0 }, emptySet(),
            if (plusPlus && count > 1) PhaseThreeEngines.kMeansPlusPlusProbabilities(points, centers) else emptyList(),
            if (plusPlus && count > 1) "Choose the first centroid; orange rings show D² probability for the next seed."
            else "Choose the first initial centroid.", 0.0)
        while (centers.size < count) {
            val probabilities = PhaseThreeEngines.kMeansPlusPlusProbabilities(points, centers)
            val chosen = if (plusPlus) {
                val draw = random.nextDouble()
                var cumulative = 0.0
                probabilities.indices.firstOrNull { i ->
                    cumulative += probabilities[i]
                    cumulative >= draw && points[i] !in centers
                } ?: probabilities.indices.filter { points[it] !in centers }.maxBy { probabilities[it] }
            } else {
                points.indices.filter { points[it] !in centers }.random(random)
            }
            if (plusPlus && centers.size > 1) frames += CentroidFrame(centers.toList(), assign(points, centers), emptySet(),
                probabilities, "D² seeding: farther samples have higher selection probability.", 0.0)
            centers += points[chosen]
            frames += CentroidFrame(centers.toList(), assign(points, centers), setOf(chosen),
                probabilities, if (plusPlus) "Select centroid " + centers.size + " by distance-squared weight."
                    else "Select centroid " + centers.size + " uniformly at random.", 0.0)
        }
        val counts = IntArray(count)
        repeat(rounds) {
            val labels = assign(points, centers)
            val active = if (miniBatch == null) points.indices.toSet() else
                List(miniBatch.coerceIn(1, points.size)) { random.nextInt(points.size) }.toSet()
            frames += CentroidFrame(centers.toList(), labels, active, emptyList(),
                if (miniBatch == null) "Assign each sample to its nearest centroid."
                else "Select a mini-batch of " + active.size + " samples; other points do not update centers.",
                inertia(points, centers, labels))
            val updated = centers.mapIndexed { cluster, center ->
                val members = active.filter { labels[it] == cluster }
                if (members.isEmpty()) center else if (miniBatch == null) {
                    ClusterPoint(members.map { points[it].x }.average(), members.map { points[it].y }.average())
                } else {
                    var x = center.x; var y = center.y
                    members.forEach { i ->
                        counts[cluster]++
                        val rate = 1.0 / counts[cluster]
                        x += rate * (points[i].x - x); y += rate * (points[i].y - y)
                    }
                    ClusterPoint(x, y)
                }
            }
            centers.clear(); centers.addAll(updated)
            frames += CentroidFrame(centers.toList(), assign(points, centers), active, emptyList(),
                if (miniBatch == null) "Recompute each centroid from its assigned samples."
                else "Incrementally update centroids from this mini-batch only.",
                inertia(points, centers, assign(points, centers)))
        }
        return frames
    }

    fun fuzzy(points: List<ClusterPoint>, k: Int, fuzziness: Double, iterations: Int): FuzzyState {
        val count = k.coerceIn(1, points.size.coerceAtLeast(1))
        var centers = List(count) { points[(it * points.size / count).coerceIn(points.indices)] }
        var membership = List(points.size) { List(count) { 1.0 / count } }
        repeat(iterations.coerceIn(0, 25)) {
            membership = points.map { p ->
                val distances = centers.map { d2(p, it).coerceAtLeast(1e-9) }
                if (distances.min() < 1e-8) List(count) { if (distances[it] == distances.min()) 1.0 else 0.0 }
                else List(count) { j ->
                    1.0 / distances.sumOf { other -> (distances[j] / other).pow(1.0 / (fuzziness - 1.0)) }
                }
            }
            centers = centers.indices.map { c ->
                val weights = points.indices.map { membership[it][c].pow(fuzziness) }
                val total = weights.sum().coerceAtLeast(1e-9)
                ClusterPoint(points.indices.sumOf { weights[it] * points[it].x } / total,
                    points.indices.sumOf { weights[it] * points[it].y } / total)
            }
        }
        return FuzzyState(centers, membership)
    }

    fun optics(points: List<ClusterPoint>, eps: Double, minPts: Int): OpticsOrder {
        val n = points.size
        val distances = List(n) { i -> DoubleArray(n) { j -> sqrt(d2(points[i], points[j])) } }
        val core = List(n) { i ->
            distances[i].sorted().getOrElse((minPts - 1).coerceAtLeast(0)) { Double.POSITIVE_INFINITY }
        }
        val processed = BooleanArray(n)
        val reach = DoubleArray(n) { Double.POSITIVE_INFINITY }
        val order = mutableListOf<Int>()
        fun expand(index: Int) {
            processed[index] = true; order += index
            val queue = mutableSetOf<Int>()
            fun update(i: Int) {
                if (core[i] > eps) return
                for (j in 0 until n) if (!processed[j] && distances[i][j] <= eps) {
                    val candidate = max(core[i], distances[i][j])
                    if (candidate < reach[j]) { reach[j] = candidate; queue += j }
                }
            }
            update(index)
            while (queue.isNotEmpty()) {
                val next = queue.minBy { reach[it] }; queue.remove(next)
                if (processed[next]) continue
                processed[next] = true; order += next; update(next)
            }
        }
        for (i in 0 until n) if (!processed[i]) expand(i)
        return OpticsOrder(order, order.map { reach[it] }, order.map { core[it] })
    }

    fun densityHierarchy(points: List<ClusterPoint>, minPts: Int, cutFraction: Double): DensityHierarchy {
        if (points.isEmpty()) return DensityHierarchy(emptyList(), emptyList(), 0.0)
        val n = points.size
        val core = List(n) { i ->
            points.indices.map { sqrt(d2(points[i], points[it])) }.sorted()
                .getOrElse((minPts - 1).coerceAtLeast(0)) { 2.0 }
        }
        val used = BooleanArray(n); used[0] = true
        val edges = mutableListOf<Triple<Int, Int, Double>>()
        repeat(n - 1) {
            var bestA = -1; var bestB = -1; var bestW = Double.POSITIVE_INFINITY
            for (a in 0 until n) if (used[a]) for (b in 0 until n) if (!used[b]) {
                val weight = max(max(core[a], core[b]), sqrt(d2(points[a], points[b])))
                if (weight < bestW) { bestA = a; bestB = b; bestW = weight }
            }
            if (bestB >= 0) { used[bestB] = true; edges += Triple(bestA, bestB, bestW) }
        }
        val sorted = edges.map { it.third }.sorted()
        val threshold = sorted.getOrElse((sorted.size * cutFraction.coerceIn(.05, .95)).toInt()) { 0.0 }
        val parent = IntArray(n) { it }
        fun root(i: Int): Int { var p = i; while (parent[p] != p) p = parent[p]; return p }
        edges.filter { it.third <= threshold }.forEach { (a, b, _) -> parent[root(b)] = root(a) }
        val groups = (0 until n).groupBy { root(it) }.values.filter { it.size >= minPts }
        val labels = List(n) { i -> groups.indexOfFirst { i in it } }
        return DensityHierarchy(edges, labels, threshold)
    }

    fun divisiveTrace(points: List<ClusterPoint>, seed: Int): List<List<Int>> {
        if (points.isEmpty()) return emptyList()
        var groups = listOf(points.indices.toList())
        val trace = mutableListOf(List(points.size) { 0 })
        while (groups.size < minOf(8, points.size)) {
            val candidate = groups.indices.filter { groups[it].size >= 2 }.maxByOrNull { group ->
                val members = groups[group]
                val mx = members.map { points[it].x }.average()
                val my = members.map { points[it].y }.average()
                members.sumOf { (points[it].x - mx).pow(2) + (points[it].y - my).pow(2) }
            } ?: break
            val members = groups[candidate]
            val subset = members.map { points[it] }
            val split = kMeansTrace(subset, 2, seed + groups.size, false, null, 6).last().assignments
            val left = members.filterIndexed { i, _ -> split[i] == 0 }
            val right = members.filterIndexed { i, _ -> split[i] == 1 }
            if (left.isEmpty() || right.isEmpty()) break
            groups = groups.filterIndexed { i, _ -> i != candidate } + listOf(left, right)
            trace += List(points.size) { i -> groups.indexOfFirst { i in it } }
        }
        return trace
    }

    fun hierarchy(points: List<ClusterPoint>, linkage: LinkageMethod): DendrogramState {
        var groups=points.indices.map { setOf(it) }
        val merges=mutableListOf<MergeStep>()
        fun separation(a: Set<Int>,b: Set<Int>): Double {
            val distances=a.flatMap { i -> b.map { j -> sqrt(d2(points[i],points[j])) } }
            return when(linkage) {
                LinkageMethod.Single -> distances.min()
                LinkageMethod.Complete -> distances.max()
                LinkageMethod.Average -> distances.average()
                LinkageMethod.Ward -> {
                    val ax=a.map { points[it].x }.average()
                    val ay=a.map { points[it].y }.average()
                    val bx=b.map { points[it].x }.average()
                    val by=b.map { points[it].y }.average()
                    a.size.toDouble()*b.size/(a.size+b.size)*((ax-bx).pow(2)+(ay-by).pow(2))
                }
            }
        }
        while(groups.size>1) {
            val pair=groups.indices.flatMap { i -> (i+1 until groups.size).map { j -> i to j } }
                .minBy { (i,j) -> separation(groups[i],groups[j]) }
            val left=groups[pair.first]; val right=groups[pair.second]
            merges+=MergeStep(left,right,separation(left,right),left+right)
            groups=groups.filterIndexed { i,_ -> i!=pair.first && i!=pair.second }+
                listOf(left+right)
        }
        val height=merges.getOrNull((merges.size*.65).toInt())?.height ?: 0.0
        return DendrogramState(merges,height,merges.count { it.height>height }+1)
    }
    fun hierarchyLabels(count: Int,merges: List<MergeStep>,step: Int): List<Int> {
        var groups=(0 until count).map { setOf(it) }
        merges.take(step).forEach { merge ->
            groups=groups.filter { it!=merge.left && it!=merge.right }+listOf(merge.merged)
        }
        return List(count) { i -> groups.indexOfFirst { i in it } }
    }

    fun spectralLabels(points: List<ClusterPoint>, neighbors: Int): List<Int> {
        val n = points.size
        if (n < 3) return List(n) { 0 }
        val graph = PhaseThreeEngines.similarityGraph(points, neighbors)
        val weights = Array(n) { DoubleArray(n) }
        val local = points.indices.map { i ->
            points.indices.filter { it != i }.map { sqrt(d2(points[i], points[it])) }.sorted()
                .getOrElse((neighbors - 1).coerceAtLeast(0)) { .2 }.coerceAtLeast(.02)
        }
        graph.edges.forEach { (a, b) ->
            val w = exp(-d2(points[a], points[b]) / (local[a] * local[b]))
            weights[a][b] = w; weights[b][a] = w
        }
        val degree = DoubleArray(n) { i -> weights[i].sum().coerceAtLeast(1e-8) }
        val stationary = DoubleArray(n) { sqrt(degree[it]) }
        val norm = sqrt(stationary.sumOf { it * it })
        for (i in 0 until n) stationary[i] /= norm
        var vector = DoubleArray(n) { i -> sin(i * 1.713 + .1) }
        repeat(80) {
            val next = DoubleArray(n) { i ->
                (0 until n).sumOf { j -> weights[i][j] * vector[j] / sqrt(degree[i] * degree[j]) }
            }
            val projection = (0 until n).sumOf { next[it] * stationary[it] }
            for (i in 0 until n) next[i] -= projection * stationary[i]
            val length = sqrt(next.sumOf { it * it }).coerceAtLeast(1e-9)
            vector = DoubleArray(n) { next[it] / length }
        }
        return vector.map { if (it >= 0.0) 1 else 0 }
    }

    fun cfEntries(points: List<ClusterPoint>, threshold: Double, limit: Int): List<CfEntry> {
        val entries = mutableListOf<CfEntry>()
        for (point in points.take(limit)) {
            val index = entries.indices.minByOrNull { d2(entries[it].center, point) }
            if (index == null || sqrt(d2(entries[index].center, point)) > threshold)
                entries += CfEntry(1, point.x, point.y, point.x * point.x + point.y * point.y)
            else {
                val old = entries[index]
                entries[index] = CfEntry(old.count + 1, old.sumX + point.x, old.sumY + point.y,
                    old.sumSquares + point.x * point.x + point.y * point.y)
            }
        }
        return entries
    }

    fun affinity(points: List<ClusterPoint>, preference: Double, iterations: Int): AffinityState {
        val n = points.size
        if (n == 0) return AffinityState(emptyList(), emptyList(), emptyList())
        val similarity = Array(n) { i -> DoubleArray(n) { j ->
            if (i == j) preference else -d2(points[i], points[j])
        } }
        var responsibility = Array(n) { DoubleArray(n) }
        var availability = Array(n) { DoubleArray(n) }
        repeat(iterations.coerceIn(0, 30)) {
            val nextR = Array(n) { i -> DoubleArray(n) { k ->
                val competitor = (0 until n).filter { it != k }.maxOfOrNull { j ->
                    availability[i][j] + similarity[i][j]
                } ?: 0.0
                .5 * responsibility[i][k] + .5 * (similarity[i][k] - competitor)
            } }
            responsibility = nextR
            availability = Array(n) { i -> DoubleArray(n) { k ->
                val positive = (0 until n).filter { it != k && it != i }
                    .sumOf { j -> max(0.0, responsibility[j][k]) }
                val value = if (i == k) positive
                    else min(0.0, responsibility[k][k] + positive)
                .5 * availability[i][k] + .5 * value
            } }
        }
        val exemplars = (0 until n).filter { responsibility[it][it] + availability[it][it] > 0.0 }
            .ifEmpty { listOf((0 until n).maxBy { responsibility[it][it] + availability[it][it] }) }
        return AffinityState(exemplars, exemplars.map { responsibility[it][it] }, exemplars.map { availability[it][it] })
    }
}
