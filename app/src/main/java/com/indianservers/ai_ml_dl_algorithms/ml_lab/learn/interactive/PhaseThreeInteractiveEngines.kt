package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class PhaseThreeAlgorithmKind(val displayName: String, val category: String) {
    GradientBoostingRegression("Gradient Boosting - Regression", "Boosting / Advanced Ensembles"),
    GradientBoostingClassification("Gradient Boosting - Classification", "Boosting / Advanced Ensembles"),
    XGBoostConcepts("XGBoost Concepts", "Boosting / Advanced Ensembles"),
    HistogramGradientBoosting("Histogram Gradient Boosting Concepts", "Boosting / Advanced Ensembles"),
    LightGbmConcepts("LightGBM Concepts", "Boosting / Advanced Ensembles"),
    CatBoostConcepts("CatBoost Concepts", "Boosting / Advanced Ensembles"),
    KMeans("K-Means", "Clustering"),
    KMeansPlusPlus("K-Means++", "Clustering"),
    MiniBatchKMeans("Mini-Batch K-Means", "Clustering"),
    Hierarchical("Hierarchical Agglomerative Clustering", "Clustering"),
    Dbscan("DBSCAN", "Clustering"),
    Optics("OPTICS", "Clustering"),
    MeanShift("Mean Shift", "Clustering"),
    GaussianMixture("Gaussian Mixture Model", "Clustering"),
    SpectralClustering("Spectral Clustering", "Clustering"),
    Pca("PCA", "Dimensionality Reduction"),
    KernelPca("Kernel PCA", "Dimensionality Reduction"),
    TruncatedSvd("Truncated SVD", "Dimensionality Reduction"),
    Ica("Independent Component Analysis - ICA", "Dimensionality Reduction"),
    Tsne("t-SNE", "Dimensionality Reduction"),
    Umap("UMAP", "Dimensionality Reduction"),
    Isomap("Isomap", "Dimensionality Reduction"),
    Lle("Locally Linear Embedding - LLE", "Dimensionality Reduction")
}

enum class ClusterPreset(val label: String) {
    Blobs("Well-separated blobs"),
    UnequalSizes("Unequal cluster sizes"),
    UnequalVariances("Unequal variances"),
    Elongated("Elongated clusters"),
    Circles("Concentric circles"),
    TwoMoons("Two moons"),
    DenseSparse("Dense + sparse"),
    Noise("Noise points"),
    Outliers("Outliers"),
    Overlap("Overlapping clusters"),
    Single("Single cluster"),
    HighDim("High-dimensional projection")
}

enum class LinkageMethod(val label: String) { Single("Single"), Complete("Complete"), Average("Average"), Ward("Ward") }

data class ClusterPoint(
    val x: Double,
    val y: Double,
    val hiddenLabel: Int = 0,
    val cluster: Int = -1,
    val noise: Boolean = false
)

data class KMeansState(
    val points: List<ClusterPoint>,
    val centers: List<ClusterPoint>,
    val assignments: List<Int>,
    val inertia: Double,
    val history: List<Double>,
    val stepDescription: String
)

data class DbscanState(
    val labels: List<Int>,
    val core: Set<Int>,
    val border: Set<Int>,
    val noise: Set<Int>,
    val averageNeighbors: Double
)

data class MergeStep(val left: Set<Int>, val right: Set<Int>, val height: Double, val merged: Set<Int>)
data class DendrogramState(val merges: List<MergeStep>, val cutHeight: Double, val clusterCount: Int)
data class GmmComponent(val mean: ClusterPoint, val varianceX: Double, val varianceY: Double, val weight: Double)
data class GmmState(val components: List<GmmComponent>, val responsibilities: List<List<Double>>, val logLikelihood: Double)
data class PcaState(val mean: ClusterPoint, val pc1: ClusterPoint, val pc2: ClusterPoint, val variance1: Double, val variance2: Double, val projected: List<Double>, val reconstructionError: Double)
data class MatrixState(val rows: List<String>, val columns: List<String>, val values: List<List<Double>>)
data class GraphState(val nodes: List<ClusterPoint>, val edges: List<Pair<Int, Int>>, val selectedNeighbors: List<Int>)
data class EmbeddingState(val original: List<ClusterPoint>, val embedding: List<ClusterPoint>, val selected: Int, val neighborOverlap: Int)
data class BoostingStage(val stage: Int, val predictionBefore: Double, val residual: Double, val contribution: Double, val learningRate: Double, val predictionAfter: Double)
data class BoostingState(val stages: List<BoostingStage>, val trainError: Double, val testError: Double)
data class SplitGainState(val gainA: Double, val gainB: Double, val lambda: Double, val gamma: Double, val selected: String)
data class SilhouetteState(val overall: Double?, val perSample: List<Double>)

object PhaseThreeTopicMatcher {
    fun kindFor(title: String, section: String, domain: String): PhaseThreeAlgorithmKind? = when {
        title == "Gradient Boosting Regression" -> PhaseThreeAlgorithmKind.GradientBoostingRegression
        title == "Gradient Boosting" && section == "Classification" -> PhaseThreeAlgorithmKind.GradientBoostingClassification
        title == "XGBoost Regression" || title == "XGBoost" -> PhaseThreeAlgorithmKind.XGBoostConcepts
        title == "LightGBM Regression" || title == "LightGBM" -> PhaseThreeAlgorithmKind.LightGbmConcepts
        title == "CatBoost Regression" || title == "CatBoost" -> PhaseThreeAlgorithmKind.CatBoostConcepts
        title == "K-Means" -> PhaseThreeAlgorithmKind.KMeans
        title == "K-Means++" -> PhaseThreeAlgorithmKind.KMeansPlusPlus
        title == "Mini-Batch K-Means" -> PhaseThreeAlgorithmKind.MiniBatchKMeans
        title == "Hierarchical Clustering" || title == "Agglomerative Clustering" -> PhaseThreeAlgorithmKind.Hierarchical
        title == "DBSCAN" -> PhaseThreeAlgorithmKind.Dbscan
        title == "OPTICS" -> PhaseThreeAlgorithmKind.Optics
        title == "Mean Shift" -> PhaseThreeAlgorithmKind.MeanShift
        title == "Gaussian Mixture Models" -> PhaseThreeAlgorithmKind.GaussianMixture
        title == "Spectral Clustering" -> PhaseThreeAlgorithmKind.SpectralClustering
        title == "PCA" -> PhaseThreeAlgorithmKind.Pca
        title == "Kernel PCA" -> PhaseThreeAlgorithmKind.KernelPca
        title == "Truncated SVD" -> PhaseThreeAlgorithmKind.TruncatedSvd
        title == "Independent Component Analysis" -> PhaseThreeAlgorithmKind.Ica
        title == "t-SNE" -> PhaseThreeAlgorithmKind.Tsne
        title == "UMAP" -> PhaseThreeAlgorithmKind.Umap
        title == "Isomap" -> PhaseThreeAlgorithmKind.Isomap
        title == "Locally Linear Embedding" -> PhaseThreeAlgorithmKind.Lle
        domain == "Ensemble Learning" && title == "Gradient Boosting" -> PhaseThreeAlgorithmKind.GradientBoostingClassification
        else -> null
    }
}

object PhaseThreeDatasets {
    fun clusters(preset: ClusterPreset, samples: Int = 120, groups: Int = 3, noise: Double = .08, seed: Int = 31): List<ClusterPoint> {
        val count = samples.coerceIn(20, 500)
        val k = groups.coerceIn(1, 6)
        return List(count) { index ->
            val label = when (preset) {
                ClusterPreset.UnequalSizes -> if (index < count * .55) 0 else if (index < count * .8) 1 else 2 % k
                ClusterPreset.Single -> 0
                ClusterPreset.Circles -> index % 2
                ClusterPreset.TwoMoons -> index % 2
                else -> index % k
            }
            val angle = 2.0 * PI * label / k.coerceAtLeast(2)
            val n1 = centered(seed, index, 1)
            val n2 = centered(seed, index, 2)
            val spread = when (preset) {
                ClusterPreset.UnequalVariances -> .05 + .09 * label
                ClusterPreset.DenseSparse -> if (label == 0) .04 else .22
                ClusterPreset.Overlap -> .28
                ClusterPreset.Single -> .32
                else -> .11
            } + noise
            val point = when (preset) {
                ClusterPreset.Circles -> {
                    val r = if (label == 0) .34 else .72
                    val t = 2.0 * PI * index / count
                    ClusterPoint(r * cos(t) + n1 * noise, r * sin(t) + n2 * noise, label)
                }
                ClusterPreset.TwoMoons -> {
                    val t = PI * (index / 2) / (count / 2.0)
                    val x = if (label == 0) cos(t) * .55 else 1.0 - cos(t) * .55
                    val y = if (label == 0) sin(t) * .38 else .38 - sin(t) * .38
                    ClusterPoint((x - .5) + n1 * noise, (y - .2) + n2 * noise, label)
                }
                ClusterPreset.Elongated -> ClusterPoint(.58 * cos(angle) + n1 * spread * 2.2, .58 * sin(angle) + (n1 * .7 + n2 * .3) * spread, label)
                ClusterPreset.Outliers -> if (index % 23 == 0) ClusterPoint(n1, n2, -1) else ClusterPoint(.62 * cos(angle) + n1 * spread, .62 * sin(angle) + n2 * spread, label)
                ClusterPreset.Noise -> if (index % 7 == 0) ClusterPoint(n1, n2, -1) else ClusterPoint(.62 * cos(angle) + n1 * spread, .62 * sin(angle) + n2 * spread, label)
                ClusterPreset.HighDim -> ClusterPoint(.55 * cos(angle) + n1 * spread + .18 * sin(index.toDouble()), .55 * sin(angle) + n2 * spread + .1 * cos(index * .7), label)
                else -> ClusterPoint(.62 * cos(angle) + n1 * spread, .62 * sin(angle) + n2 * spread, label)
            }
            point.copy(x = point.x.coerceIn(-1.0, 1.0), y = point.y.coerceIn(-1.0, 1.0))
        }
    }

    fun regression(samples: Int = 60, seed: Int = 9): List<LabPoint> = List(samples) { index ->
        val x = -1.0 + 2.0 * index / (samples - 1)
        val y = .55 * x + .45 * sin(3.0 * x) + centered(seed, index, 3) * .06
        LabPoint(x, y, target = y, train = index % 5 != 0)
    }

    private fun centered(seed: Int, index: Int, salt: Int): Double {
        val raw = sin((seed * 109 + index * 43 + salt * 17).toDouble()) * 8719.31
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}

object PhaseThreeEngines {
    fun kMeans(points: List<ClusterPoint>, k: Int, iterations: Int, seed: Int = 3, plusPlus: Boolean = false, miniBatch: Int? = null): KMeansState {
        var centers = if (plusPlus) kMeansPlusPlusCenters(points, k, seed) else List(k.coerceAtLeast(1)) { i -> points[(i * 37 + seed).mod(points.size)] }
        val history = mutableListOf<Double>()
        var assignments = List(points.size) { 0 }
        repeat(iterations.coerceAtLeast(1)) { iter ->
            val active = miniBatch?.let { size -> List(size.coerceIn(1, points.size)) { points[(iter * size + it).mod(points.size)] }.toSet() }
            assignments = points.map { point -> centers.indices.minBy { distance2(point, centers[it]) } }
            history += inertia(points, centers, assignments)
            centers = centers.mapIndexed { cluster, center ->
                val members = points.filterIndexed { index, point -> assignments[index] == cluster && (active == null || point in active) }
                if (members.isEmpty()) center else ClusterPoint(members.map { it.x }.average(), members.map { it.y }.average(), cluster)
            }
        }
        assignments = points.map { point -> centers.indices.minBy { distance2(point, centers[it]) } }
        val finalInertia = inertia(points, centers, assignments)
        return KMeansState(points.mapIndexed { i, p -> p.copy(cluster = assignments[i]) }, centers, assignments, finalInertia, history + finalInertia, "Assign points to nearest centroid, then move each centroid to its assigned mean.")
    }

    fun kMeansPlusPlusProbabilities(points: List<ClusterPoint>, centers: List<ClusterPoint>): List<Double> {
        val d2 = points.map { point -> centers.minOfOrNull { distance2(point, it) } ?: 1.0 }
        val total = d2.sum().coerceAtLeast(1e-9)
        return d2.map { it / total }
    }

    fun dbscan(points: List<ClusterPoint>, eps: Double, minPts: Int): DbscanState {
        val labels = MutableList(points.size) { -99 }
        val core = mutableSetOf<Int>()
        val neighbors = points.indices.map { i -> points.indices.filter { j -> sqrt(distance2(points[i], points[j])) <= eps } }
        neighbors.forEachIndexed { i, ns -> if (ns.size >= minPts) core += i }
        var cluster = 0
        for (i in points.indices) {
            if (labels[i] != -99 || i !in core) continue
            val queue = ArrayDeque<Int>()
            queue.add(i)
            labels[i] = cluster
            while (queue.isNotEmpty()) {
                val p = queue.removeFirst()
                neighbors[p].forEach { n ->
                    if (labels[n] == -99) {
                        labels[n] = cluster
                        if (n in core) queue.add(n)
                    }
                }
            }
            cluster++
        }
        val noise = labels.indices.filter { labels[it] == -99 }.toSet()
        val border = labels.indices.filter { labels[it] >= 0 && it !in core }.toSet()
        val fixed = labels.map { if (it == -99) -1 else it }
        return DbscanState(fixed, core, border, noise, neighbors.map { it.size }.average())
    }

    fun hierarchical(points: List<ClusterPoint>, linkage: LinkageMethod): DendrogramState {
        var clusters = points.indices.map { setOf(it) }
        val merges = mutableListOf<MergeStep>()
        while (clusters.size > 1) {
            val pair = clusters.indices.flatMap { i -> (i + 1 until clusters.size).map { j -> i to j } }.minBy { (i, j) -> clusterDistance(points, clusters[i], clusters[j], linkage) }
            val left = clusters[pair.first]
            val right = clusters[pair.second]
            val height = clusterDistance(points, left, right, linkage)
            merges += MergeStep(left, right, height, left + right)
            clusters = clusters.filterIndexed { index, _ -> index != pair.first && index != pair.second } + listOf(left + right)
        }
        val cut = merges.getOrNull((merges.size * .65).toInt())?.height ?: 0.0
        return DendrogramState(merges, cut, merges.count { it.height > cut } + 1)
    }

    fun meanShift(points: List<ClusterPoint>, bandwidth: Double, iterations: Int = 8): KMeansState {
        var centers = points.take(12)
        repeat(iterations) {
            centers = centers.map { center ->
                val inside = points.filter { sqrt(distance2(it, center)) <= bandwidth }
                if (inside.isEmpty()) center else ClusterPoint(inside.map { it.x }.average(), inside.map { it.y }.average())
            }.distinctBy { "%.1f/%.1f".format(it.x / bandwidth, it.y / bandwidth) }
        }
        val assignments = points.map { point -> centers.indices.minBy { distance2(point, centers[it]) } }
        return KMeansState(points.mapIndexed { i, p -> p.copy(cluster = assignments[i]) }, centers, assignments, inertia(points, centers, assignments), emptyList(), "Move each bandwidth window to the mean of points inside it until modes stabilize.")
    }

    fun gmm(points: List<ClusterPoint>, k: Int, iterations: Int = 5): GmmState {
        var components = kMeans(points, k, 3, plusPlus = true).centers.map { GmmComponent(it, .08, .08, 1.0 / k) }
        var responsibilities = List(points.size) { List(k) { 1.0 / k } }
        repeat(iterations) {
            responsibilities = points.map { point ->
                val raw = components.map { c -> c.weight * gaussian(point.x, c.mean.x, sqrt(c.varianceX)) * gaussian(point.y, c.mean.y, sqrt(c.varianceY)) }
                val total = raw.sum().coerceAtLeast(1e-12)
                raw.map { it / total }
            }
            components = components.indices.map { c ->
                val weightSum = responsibilities.sumOf { it[c] }.coerceAtLeast(1e-9)
                val mx = points.indices.sumOf { responsibilities[it][c] * points[it].x } / weightSum
                val my = points.indices.sumOf { responsibilities[it][c] * points[it].y } / weightSum
                val vx = points.indices.sumOf { responsibilities[it][c] * (points[it].x - mx).pow(2) } / weightSum
                val vy = points.indices.sumOf { responsibilities[it][c] * (points[it].y - my).pow(2) } / weightSum
                GmmComponent(ClusterPoint(mx, my, c), vx.coerceAtLeast(.002), vy.coerceAtLeast(.002), weightSum / points.size)
            }
        }
        val ll = points.sumOf { point -> ln(components.sumOf { it.weight * gaussian(point.x, it.mean.x, sqrt(it.varianceX)) * gaussian(point.y, it.mean.y, sqrt(it.varianceY)) }.coerceAtLeast(1e-12)) }
        return GmmState(components, responsibilities, ll)
    }

    fun opticsReachability(points: List<ClusterPoint>, eps: Double, minPts: Int): List<Double> {
        return points.indices.map { i ->
            val distances = points.indices.filter { it != i }.map { sqrt(distance2(points[i], points[it])) }.sorted()
            distances.getOrElse((minPts - 1).coerceAtLeast(0)) { eps * 1.5 }.coerceAtMost(eps * 2.0)
        }
    }

    fun similarityGraph(points: List<ClusterPoint>, neighbors: Int): GraphState {
        val edges = points.indices.flatMap { i ->
            points.indices.filter { it != i }.sortedBy { distance2(points[i], points[it]) }.take(neighbors.coerceAtLeast(1)).map { j -> minOf(i, j) to maxOf(i, j) }
        }.distinct()
        return GraphState(points, edges, edges.filter { it.first == 0 || it.second == 0 }.flatMap { listOf(it.first, it.second) }.distinct())
    }

    fun pca(points: List<ClusterPoint>, axisAngle: Double? = null): PcaState {
        val meanX = points.map { it.x }.average()
        val meanY = points.map { it.y }.average()
        val xx = points.sumOf { (it.x - meanX).pow(2) } / points.size
        val yy = points.sumOf { (it.y - meanY).pow(2) } / points.size
        val xy = points.sumOf { (it.x - meanX) * (it.y - meanY) } / points.size
        val angle = axisAngle ?: .5 * atan2(2.0 * xy, xx - yy)
        val pc1 = ClusterPoint(cos(angle), sin(angle))
        val pc2 = ClusterPoint(-sin(angle), cos(angle))
        val projected = points.map { (it.x - meanX) * pc1.x + (it.y - meanY) * pc1.y }
        val variance1 = projected.map { it * it }.average()
        val variance2 = points.map { (it.x - meanX) * pc2.x + (it.y - meanY) * pc2.y }.map { it * it }.average()
        val error = points.indices.sumOf { i ->
            val rx = meanX + projected[i] * pc1.x
            val ry = meanY + projected[i] * pc1.y
            (points[i].x - rx).pow(2) + (points[i].y - ry).pow(2)
        } / points.size
        return PcaState(ClusterPoint(meanX, meanY), pc1, pc2, variance1, variance2, projected, error)
    }

    fun truncatedSvd(rank: Int): Pair<MatrixState, Double> {
        val matrix = listOf(
            listOf(4.0, 3.0, 0.0, 0.0),
            listOf(5.0, 4.0, 0.0, 1.0),
            listOf(0.0, 0.0, 4.0, 5.0),
            listOf(0.0, 1.0, 5.0, 4.0)
        )
        val singular = listOf(8.7, 6.4, 1.2, .4)
        val kept = singular.take(rank.coerceIn(1, 4)).sum()
        val error = singular.drop(rank.coerceIn(1, 4)).sum()
        return MatrixState(listOf("d1", "d2", "d3", "d4"), listOf("offer", "money", "project", "report"), matrix) to error / (kept + error)
    }

    fun ica(mixA: Double, mixB: Double, samples: Int = 64): Pair<List<Double>, List<Double>> {
        val sourceA = List(samples) { sin(2.0 * PI * it / samples) }
        val sourceB = List(samples) { if (it % 16 < 8) 1.0 else -1.0 }
        val mixed1 = sourceA.indices.map { mixA * sourceA[it] + (1 - mixA) * sourceB[it] }
        val recovered = mixed1.map { it - mixB * sourceB[mixed1.indexOf(it)] }
        return mixed1 to recovered
    }

    fun embedding(points: List<ClusterPoint>, kind: PhaseThreeAlgorithmKind, seed: Int, neighbors: Int = 8): EmbeddingState {
        val embedded = points.mapIndexed { i, p ->
            val jitter = sin(seed * 13.0 + i) * .04
            when (kind) {
                PhaseThreeAlgorithmKind.KernelPca -> ClusterPoint(p.x, p.x.pow(2) + p.y.pow(2) - .5 + jitter, p.hiddenLabel)
                PhaseThreeAlgorithmKind.Tsne -> ClusterPoint(p.x * .75 + jitter, p.y * .75 + sin(i * .7) * .05, p.hiddenLabel)
                PhaseThreeAlgorithmKind.Umap -> ClusterPoint(p.x * .9 + jitter, p.y * .65 + cos(i * .3) * .05, p.hiddenLabel)
                PhaseThreeAlgorithmKind.Isomap -> ClusterPoint(atan2(p.y, p.x) / PI, sqrt(p.x * p.x + p.y * p.y) - .55, p.hiddenLabel)
                PhaseThreeAlgorithmKind.Lle -> ClusterPoint(p.x + jitter, p.y - jitter, p.hiddenLabel)
                else -> p
            }
        }
        val originalNeighbors = nearest(points, 0, neighbors).toSet()
        val embeddingNeighbors = nearest(embedded, 0, neighbors).toSet()
        return EmbeddingState(points, embedded, 0, originalNeighbors.intersect(embeddingNeighbors).size)
    }

    fun gradientBoosting(points: List<LabPoint>, stages: Int, learningRate: Double): BoostingState {
        val mean = points.filter { it.train }.map { it.target }.average()
        var prediction = mean
        val built = mutableListOf<BoostingStage>()
        repeat(stages.coerceIn(1, 20)) { stage ->
            val sample = points[stage.mod(points.size)]
            val residual = sample.target - prediction
            val contribution = residual * .8
            val after = prediction + learningRate * contribution
            built += BoostingStage(stage + 1, prediction, residual, contribution, learningRate, after)
            prediction = after
        }
        val trainError = points.filter { it.train }.map { (prediction - it.target).pow(2) }.average()
        val testError = points.filter { !it.train }.map { (prediction - it.target).pow(2) }.average()
        return BoostingState(built, trainError, testError)
    }

    fun xgBoostGain(lambda: Double, gamma: Double): SplitGainState {
        fun gain(gLeft: Double, hLeft: Double, gRight: Double, hRight: Double): Double {
            val parentG = gLeft + gRight
            val parentH = hLeft + hRight
            return .5 * (gLeft.pow(2) / (hLeft + lambda) + gRight.pow(2) / (hRight + lambda) - parentG.pow(2) / (parentH + lambda)) - gamma
        }
        val a = gain(-3.2, 5.0, 2.1, 4.0)
        val b = gain(-2.2, 3.0, 1.3, 6.0)
        return SplitGainState(a, b, lambda, gamma, if (a >= b) "A" else "B")
    }

    fun silhouette(points: List<ClusterPoint>, assignments: List<Int>): SilhouetteState {
        val clusters = assignments.filter { it >= 0 }.distinct()
        if (clusters.size < 2) return SilhouetteState(null, emptyList())
        val values = points.indices.map { i ->
            val own = assignments[i]
            val same = points.indices.filter { assignments[it] == own && it != i }
            val a = same.map { sqrt(distance2(points[i], points[it])) }.average().takeIf { !it.isNaN() } ?: 0.0
            val b = clusters.filter { it != own }.minOf { c -> points.indices.filter { assignments[it] == c }.map { sqrt(distance2(points[i], points[it])) }.average() }
            if (a == 0.0 && b == 0.0) 0.0 else (b - a) / maxOf(a, b)
        }
        return SilhouetteState(values.average(), values)
    }

    private fun kMeansPlusPlusCenters(points: List<ClusterPoint>, k: Int, seed: Int): List<ClusterPoint> {
        val centers = mutableListOf(points[seed.mod(points.size)])
        while (centers.size < k) {
            val probs = kMeansPlusPlusProbabilities(points, centers)
            centers += points[probs.indices.maxBy { probs[it] }]
        }
        return centers
    }

    private fun nearest(points: List<ClusterPoint>, index: Int, k: Int): List<Int> =
        points.indices.filter { it != index }.sortedBy { distance2(points[index], points[it]) }.take(k)

    private fun inertia(points: List<ClusterPoint>, centers: List<ClusterPoint>, assignments: List<Int>): Double =
        points.indices.sumOf { distance2(points[it], centers[assignments[it].coerceIn(centers.indices)]) }

    private fun clusterDistance(points: List<ClusterPoint>, a: Set<Int>, b: Set<Int>, linkage: LinkageMethod): Double {
        val distances = a.flatMap { i -> b.map { j -> sqrt(distance2(points[i], points[j])) } }
        return when (linkage) {
            LinkageMethod.Single -> distances.min()
            LinkageMethod.Complete -> distances.max()
            LinkageMethod.Average -> distances.average()
            LinkageMethod.Ward -> distances.average() * (a.size + b.size)
        }
    }

    private fun gaussian(value: Double, mean: Double, sigma: Double): Double {
        val safe = sigma.coerceAtLeast(.001)
        val z = (value - mean) / safe
        return exp(-.5 * z * z) / (safe * sqrt(2.0 * PI))
    }

    private fun distance2(a: ClusterPoint, b: ClusterPoint): Double = (a.x - b.x).pow(2) + (a.y - b.y).pow(2)
}
