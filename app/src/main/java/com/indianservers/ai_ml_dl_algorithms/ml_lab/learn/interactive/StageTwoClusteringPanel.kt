package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageTwoVisualization
import kotlin.math.*

@Composable
internal fun StageTwoClusteringPanel(kind: StageTwoVisualization) {
    var preset by remember(kind) { mutableStateOf(
        when (kind) {
            StageTwoVisualization.Dbscan, StageTwoVisualization.Spectral -> ClusterPreset.TwoMoons
            StageTwoVisualization.Hdbscan, StageTwoVisualization.Optics -> ClusterPreset.DenseSparse
            StageTwoVisualization.Gmm, StageTwoVisualization.FuzzyCMeans -> ClusterPreset.Overlap
            else -> ClusterPreset.Blobs
        }
    ) }
    var seed by remember(kind) { mutableIntStateOf(31) }
    val count = if (kind in setOf(StageTwoVisualization.Hierarchical, StageTwoVisualization.Agglomerative,
            StageTwoVisualization.Divisive)) 18 else if (kind == StageTwoVisualization.Affinity) 35 else 60
    val points = remember(kind, preset, seed) {
        mutableStateListOf<ClusterPoint>().apply {
            addAll(PhaseThreeDatasets.clusters(preset, count, 3, .055, seed))
        }
    }
    var k by remember(kind) { mutableIntStateOf(3) }
    var step by remember(kind, preset, seed) { mutableIntStateOf(0) }
    var eps by remember(kind) { mutableDoubleStateOf(.18) }
    var minPts by remember(kind) { mutableIntStateOf(4) }
    var bandwidth by remember(kind) { mutableDoubleStateOf(.25) }
    var linkage by remember(kind) { mutableStateOf(LinkageMethod.Average) }
    var selected by remember(kind) { mutableIntStateOf(0) }
    var cut by remember(kind) { mutableDoubleStateOf(.62) }
    var compareRandom by remember(kind) { mutableStateOf(false) }
    val snapshot = points.toList()
    val centroid = remember(snapshot, k, seed, kind, bandwidth) {
        if (kind in setOf(StageTwoVisualization.KMeans, StageTwoVisualization.KMeansPlus,
                StageTwoVisualization.MiniBatchKMeans))
            StageTwoClusteringEngine.kMeansTrace(snapshot, k, seed,
                kind == StageTwoVisualization.KMeansPlus,
                if (kind == StageTwoVisualization.MiniBatchKMeans) (bandwidth * 60).toInt().coerceIn(3, 30) else null)
        else emptyList()
    }
    val db = remember(snapshot, eps, minPts) { PhaseThreeEngines.dbscan(snapshot, eps, minPts) }
    val dbTrace = remember(snapshot, eps, minPts, kind) {
        if (kind == StageTwoVisualization.Dbscan)
            StageTwoClusteringEngine.dbscanTrace(snapshot, eps, minPts) else emptyList()
    }
    val hierarchy = remember(snapshot, linkage) {
        if (kind == StageTwoVisualization.Hierarchical || kind == StageTwoVisualization.Agglomerative)
            StageTwoClusteringEngine.hierarchy(snapshot, linkage) else null
    }
    val divisive = remember(snapshot, seed) {
        if (kind == StageTwoVisualization.Divisive) StageTwoClusteringEngine.divisiveTrace(snapshot, seed)
        else emptyList()
    }
    val optics = remember(snapshot, eps, minPts) {
        if (kind == StageTwoVisualization.Optics) StageTwoClusteringEngine.optics(snapshot, eps, minPts)
        else null
    }
    val density = remember(snapshot, minPts, cut) {
        if (kind == StageTwoVisualization.Hdbscan)
            StageTwoClusteringEngine.densityHierarchy(snapshot, minPts, cut) else null
    }
    val gmm = remember(snapshot, k, step) {
        if (kind == StageTwoVisualization.Gmm) PhaseThreeEngines.gmm(snapshot, k, (step / 2 + 1).coerceAtMost(8))
        else null
    }
    val fuzzy = remember(snapshot, k, bandwidth, step) {
        if (kind == StageTwoVisualization.FuzzyCMeans)
            StageTwoClusteringEngine.fuzzy(snapshot, k, 1.2 + bandwidth * 5, (step + 1).coerceAtMost(12))
        else null
    }
    val spectral = remember(snapshot, k) {
        if (kind == StageTwoVisualization.Spectral)
            StageTwoClusteringEngine.spectralLabels(snapshot, k.coerceIn(2, 8)) else emptyList()
    }
    val cf = remember(snapshot, eps, step) {
        if (kind == StageTwoVisualization.Birch)
            StageTwoClusteringEngine.cfEntries(snapshot, eps, (step + 1) * 6) else emptyList()
    }
    val affinity = remember(snapshot, bandwidth, step) {
        if (kind == StageTwoVisualization.Affinity)
            StageTwoClusteringEngine.affinity(snapshot, -bandwidth, (step + 1) * 2) else null
    }
    val maxStep = when (kind) {
        StageTwoVisualization.KMeans, StageTwoVisualization.KMeansPlus, StageTwoVisualization.MiniBatchKMeans ->
            (centroid.size - 1).coerceAtLeast(0)
        StageTwoVisualization.Hierarchical, StageTwoVisualization.Agglomerative ->
            (hierarchy?.merges?.size ?: 0)
        StageTwoVisualization.Divisive -> (divisive.size - 1).coerceAtLeast(0)
        StageTwoVisualization.Dbscan -> (dbTrace.size - 1).coerceAtLeast(0)
        StageTwoVisualization.Optics -> (snapshot.size - 1).coerceAtLeast(0)
        StageTwoVisualization.Birch -> (snapshot.size / 6 - 1).coerceAtLeast(0)
        else -> 12
    }
    val current = step.coerceIn(0, maxStep)
    val frame = centroid.getOrNull(current)
    val dbFrame = dbTrace.getOrNull(current)
    val randomComparison = remember(snapshot,k,seed,kind) {
        if (kind == StageTwoVisualization.KMeansPlus)
            StageTwoClusteringEngine.kMeansTrace(snapshot,k,seed,false,null,1)
        else emptyList()
    }
    val cutStep = hierarchy?.merges?.let { merges ->
        val highest=merges.maxOfOrNull { it.height } ?: 0.0
        merges.count { it.height<=highest*cut }
    } ?: 0
    val labels = when (kind) {
        StageTwoVisualization.KMeans, StageTwoVisualization.KMeansPlus, StageTwoVisualization.MiniBatchKMeans ->
            frame?.assignments.orEmpty()
        StageTwoVisualization.Dbscan -> dbFrame?.labels.orEmpty()
        StageTwoVisualization.Hdbscan -> density?.labels.orEmpty()
        StageTwoVisualization.Divisive -> divisive.getOrNull(current).orEmpty()
        StageTwoVisualization.Hierarchical, StageTwoVisualization.Agglomerative ->
            StageTwoClusteringEngine.hierarchyLabels(snapshot.size,
                hierarchy?.merges.orEmpty(),minOf(current,cutStep))
        StageTwoVisualization.Spectral -> spectral
        StageTwoVisualization.Gmm -> gmm?.responsibilities?.map { row -> row.indices.maxBy { row[it] } }.orEmpty()
        StageTwoVisualization.FuzzyCMeans ->
            fuzzy?.membership?.map { row -> row.indices.maxBy { row[it] } }.orEmpty()
        else -> emptyList()
    }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { StageTwoInfo(kind.title, kind.graphic) }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
                listOf(ClusterPreset.Blobs, ClusterPreset.TwoMoons, ClusterPreset.Circles,
                    ClusterPreset.DenseSparse, ClusterPreset.Noise, ClusterPreset.Overlap)
                    .chunked(3).forEach { row ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                            row.forEach { option ->
                                val short = when(option) {
                                    ClusterPreset.Blobs -> "Blobs"
                                    ClusterPreset.TwoMoons -> "Moons"
                                    ClusterPreset.Circles -> "Circles"
                                    ClusterPreset.DenseSparse -> "Varying"
                                    ClusterPreset.Noise -> "Noise"
                                    else -> "Overlap"
                                }
                                SegmentedOption(short, preset == option, Modifier.weight(1f)) {
                                    preset = option; step = 0
                                }
                            }
                        }
                    }
            }
        }
        if (kind == StageTwoVisualization.KMeansPlus) item {
            Text("Orange rings show D² seed probability; larger means more likely next.",
                color = LabOrange, fontSize = 11.sp)
        }
        item {
            StageTwoPlot(snapshot, labels, selected, onMove = { i, moved ->
                points[i] = moved; selected = i
            }) { scale ->
                when (kind) {
                    StageTwoVisualization.KMeans, StageTwoVisualization.KMeansPlus,
                    StageTwoVisualization.MiniBatchKMeans -> {
                        frame?.let { state ->
                            state.centers.forEachIndexed { cluster, center ->
                                val at = scale.point(center.x, center.y)
                                drawCircle(stageColor(cluster), 12f, at, style = Stroke(3f))
                                drawLine(stageColor(cluster), at + Offset(-8f, 0f), at + Offset(8f, 0f), 3f)
                                drawLine(stageColor(cluster), at + Offset(0f, -8f), at + Offset(0f, 8f), 3f)
                            }
                            if (kind == StageTwoVisualization.KMeansPlus && compareRandom)
                                randomComparison.firstOrNull { it.centers.size == k }?.centers
                                    ?.forEach { center ->
                                        drawCircle(LabMuted,13f,scale.point(center.x,center.y),
                                            style=Stroke(2f))
                                    }
                            val probabilityPeak = state.probabilities.maxOrNull()?.coerceAtLeast(1e-9) ?: 1.0
                            snapshot.indices.forEach { i ->
                                val center = state.centers[state.assignments[i]]
                                drawLine(stageColor(state.assignments[i]).copy(alpha = .22f),
                                    scale.point(snapshot[i].x, snapshot[i].y),
                                    scale.point(center.x, center.y), 1.5f)
                                if (i in state.active && kind == StageTwoVisualization.MiniBatchKMeans)
                                    drawCircle(Color.White, 9f, scale.point(snapshot[i].x, snapshot[i].y),
                                        style = Stroke(2f))
                                if (state.probabilities.isNotEmpty() && kind == StageTwoVisualization.KMeansPlus &&
                                    state.probabilities[i] > 1e-6) {
                                    val relative = state.probabilities[i] / probabilityPeak
                                    drawCircle(LabOrange.copy(alpha = (.2 + .65 * relative).toFloat()),
                                        (7 + 20 * sqrt(relative)).toFloat(),
                                        scale.point(snapshot[i].x, snapshot[i].y), style = Stroke(2.2f))
                                }
                            }
                        }
                    }
                    StageTwoVisualization.Dbscan -> {
                        val focus = snapshot[dbFrame?.focus ?: 0]
                        drawCircle(LabOrange.copy(alpha = .7f),
                            (eps * (scale.right - scale.left) / 2).toFloat(),
                            scale.point(focus.x, focus.y), style = Stroke(2f))
                        db.core.forEach { i ->
                            if (labels.getOrElse(i) { -99 }>=0)
                                drawCircle(stageColor(db.labels[i]), 9f,
                                    scale.point(snapshot[i].x, snapshot[i].y), style = Stroke(2f))
                        }
                        db.border.forEach { i ->
                            if (labels.getOrElse(i) { -99 }>=0) {
                                val at=scale.point(snapshot[i].x,snapshot[i].y)
                                drawRect(stageColor(db.labels[i]),at-Offset(8f,8f),
                                    Size(16f,16f),style=Stroke(2f))
                            }
                        }
                        db.noise.forEach { i ->
                            if (labels.getOrElse(i) { -99 } == -1) {
                                val at = scale.point(snapshot[i].x, snapshot[i].y)
                                drawLine(LabOrange, at + Offset(-6f,-6f), at + Offset(6f,6f), 2f)
                                drawLine(LabOrange, at + Offset(-6f,6f), at + Offset(6f,-6f), 2f)
                            }
                        }
                    }
                    StageTwoVisualization.Hdbscan -> density?.edges?.sortedBy { it.third }
                        ?.take((current+1)*5)?.forEach { (a,b,w) ->
                        if (w <= density.cutoff)
                            drawLine(stageColor(density.labels[a]).copy(alpha = .55f),
                                scale.point(snapshot[a].x,snapshot[a].y),
                                scale.point(snapshot[b].x,snapshot[b].y), 2f)
                    }
                    StageTwoVisualization.Optics -> optics?.order?.take(current + 1)?.zipWithNext()
                        ?.forEach { (a,b) ->
                            drawLine(LabOrange.copy(alpha = .55f), scale.point(snapshot[a].x,snapshot[a].y),
                                scale.point(snapshot[b].x,snapshot[b].y), 2f)
                        }
                    StageTwoVisualization.MeanShift -> {
                        val prior = if (current == 0) snapshot.take(8)
                            else PhaseThreeEngines.meanShift(snapshot, bandwidth, current).centers
                        val next = PhaseThreeEngines.meanShift(snapshot, bandwidth, current + 1).centers
                        prior.take(next.size).forEachIndexed { i, p ->
                            val q = next[i]
                            drawLine(LabOrange, scale.point(p.x,p.y), scale.point(q.x,q.y), 3f)
                            drawCircle(LabOrange, (bandwidth * (scale.right-scale.left)/2).toFloat(),
                                scale.point(q.x,q.y), style = Stroke(1.5f))
                        }
                    }
                    StageTwoVisualization.Gmm -> {
                        if(current%2==0) {
                            val at=scale.point(snapshot[selected].x,snapshot[selected].y)
                            gmm?.responsibilities?.getOrNull(selected)?.forEachIndexed { i,p ->
                                val c=gmm.components[i]
                                drawLine(stageColor(i).copy(alpha=(.25+.7*p).toFloat()),
                                    at,scale.point(c.mean.x,c.mean.y),
                                    (1.5+5*p).toFloat())
                            }
                        }
                        gmm?.components?.forEachIndexed { i, c ->
                            val center = scale.point(c.mean.x,c.mean.y)
                            val discriminant = sqrt((c.varianceX-c.varianceY).pow(2)+
                                4*c.covarianceXY.pow(2))
                            val major = ((c.varianceX+c.varianceY+discriminant)/2).coerceAtLeast(.002)
                            val minor = ((c.varianceX+c.varianceY-discriminant)/2).coerceAtLeast(.002)
                            val radius = (scale.right-scale.left)/2
                            val rx = (sqrt(major)*1.5*radius).toFloat()
                            val ry = (sqrt(minor)*1.5*radius).toFloat()
                            val angle = Math.toDegrees(.5*atan2(2*c.covarianceXY,
                                c.varianceX-c.varianceY)).toFloat()
                            rotate(angle, center) {
                                drawOval(stageColor(i), center-Offset(rx,ry),
                                    Size(rx*2,ry*2), style = Stroke(if(current%2==0) 1.5f else 3.5f))
                            }
                            drawCircle(stageColor(i), 7f, center)
                        }
                    }
                    StageTwoVisualization.Spectral -> {
                        val graph = PhaseThreeEngines.similarityGraph(snapshot, k)
                        graph.edges.take((current+1)*max(1,(graph.edges.size+12)/13)).forEach { (a,b) ->
                            drawLine(if (spectral[a] == spectral[b]) LabCyan.copy(alpha = .38f)
                                else LabPink.copy(alpha = .2f),
                                scale.point(snapshot[a].x,snapshot[a].y),
                                scale.point(snapshot[b].x,snapshot[b].y), 1.5f)
                        }
                    }
                    StageTwoVisualization.Birch -> cf.forEachIndexed { i, entry ->
                        drawCircle(stageColor(i), (7 + sqrt(entry.count.toDouble())*4).toFloat(),
                            scale.point(entry.center.x,entry.center.y), style = Stroke(3f))
                    }
                    StageTwoVisualization.Affinity -> affinity?.exemplars?.forEachIndexed { i, exemplar ->
                        drawCircle(stageColor(i), 14f,
                            scale.point(snapshot[exemplar].x,snapshot[exemplar].y), style = Stroke(3f))
                        snapshot.forEach { p ->
                            if (affinity.exemplars.minBy { e ->
                                (snapshot[e].x-p.x).pow(2)+(snapshot[e].y-p.y).pow(2)
                            } == exemplar)
                                drawLine(stageColor(i).copy(alpha = .3f),scale.point(p.x,p.y),
                                    scale.point(snapshot[exemplar].x,snapshot[exemplar].y),1f)
                        }
                    }
                    StageTwoVisualization.FuzzyCMeans -> {
                        fuzzy?.membership?.forEachIndexed { pointIndex, membership ->
                            val at = scale.point(snapshot[pointIndex].x, snapshot[pointIndex].y)
                            var angle = -90f
                            membership.forEachIndexed { cluster, probability ->
                                val sweep = (360.0 * probability).toFloat()
                                drawArc(stageColor(cluster), angle, sweep, false,
                                    at - Offset(11f, 11f), Size(22f, 22f),
                                    style = Stroke(3.5f))
                                angle += sweep
                            }
                        }
                        fuzzy?.centers?.forEachIndexed { i, center ->
                            drawCircle(stageColor(i), 12f, scale.point(center.x,center.y), style = Stroke(3f))
                        }
                    }
                    else -> Unit
                }
            }
        }
        item { StageTwoStepControls(current, maxStep) { step = it } }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                when (kind) {
                    StageTwoVisualization.KMeans, StageTwoVisualization.KMeansPlus,
                    StageTwoVisualization.MiniBatchKMeans, StageTwoVisualization.Gmm,
                    StageTwoVisualization.FuzzyCMeans -> NativeSlider("Clusters K", k.toDouble(), 2.0, 5.0) {
                        k = it.roundToInt(); step = 0
                    }
                    StageTwoVisualization.Hierarchical, StageTwoVisualization.Agglomerative ->
                        listOf(LinkageMethod.Single,LinkageMethod.Complete,LinkageMethod.Average,
                            LinkageMethod.Ward).chunked(2).forEach { row ->
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                                row.forEach { value ->
                                    SegmentedOption(value.label, linkage == value, Modifier.weight(1f)) {
                                        linkage = value; step = 0
                                    }
                                }
                            }
                        }
                    StageTwoVisualization.Spectral -> NativeSlider(
                        "Graph neighbors",k.toDouble(),2.0,8.0) {
                        k=it.roundToInt(); step=0
                    }
                    else -> Unit
                }
                if (kind in setOf(StageTwoVisualization.Dbscan, StageTwoVisualization.Optics,
                        StageTwoVisualization.Birch)) NativeSlider(
                    if (kind == StageTwoVisualization.Birch) "CF radius threshold" else "Epsilon radius",
                    eps, .04, .65) { eps = it }
                if (kind in setOf(StageTwoVisualization.Dbscan, StageTwoVisualization.Optics,
                        StageTwoVisualization.Hdbscan))
                    NativeSlider("Minimum points", minPts.toDouble(), 2.0, 12.0) {
                        minPts = it.roundToInt()
                    }
                if (kind in setOf(StageTwoVisualization.MeanShift, StageTwoVisualization.Affinity,
                        StageTwoVisualization.FuzzyCMeans, StageTwoVisualization.MiniBatchKMeans))
                    NativeSlider(when(kind) {
                        StageTwoVisualization.MeanShift -> "Kernel bandwidth"
                        StageTwoVisualization.Affinity -> "Exemplar preference magnitude"
                        StageTwoVisualization.FuzzyCMeans -> "Fuzziness"
                        else -> "Mini-batch size"
                    }, bandwidth, .08, .7) { bandwidth = it }
                if (kind == StageTwoVisualization.Hdbscan || kind == StageTwoVisualization.Hierarchical ||
                    kind == StageTwoVisualization.Agglomerative)
                    NativeSlider("Hierarchy cut", cut, .1, .9) { cut = it }
                SegmentedOption("New initialization / dataset", false, Modifier.fillMaxWidth()) {
                    seed++; step = 0
                }
                if (kind == StageTwoVisualization.KMeansPlus)
                    SegmentedOption("Compare random initialization", compareRandom,
                        Modifier.fillMaxWidth()) { compareRandom=!compareRandom }
            }
        }
        item {
            val detail = when (kind) {
                StageTwoVisualization.KMeans, StageTwoVisualization.KMeansPlus,
                StageTwoVisualization.MiniBatchKMeans -> frame?.explanation + " Inertia " +
                    "%.3f".format(frame?.inertia ?: 0.0)
                StageTwoVisualization.Dbscan -> (dbFrame?.phase ?: "Select") + ": " +
                    (dbFrame?.explanation ?: "") + " Core " + db.core.size +
                    " · border " + db.border.size + " · noise " + db.noise.size + "."
                StageTwoVisualization.Hdbscan -> "Mutual-reachability spanning tree: " +
                    minOf((current+1)*5,density?.edges?.size ?: 0) + "/" +
                    (density?.edges?.size ?: 0) + " edges revealed. Size-qualified groups at this cut: " +
                    density?.labels?.filter { it >= 0 }?.distinct()?.size + "."
                StageTwoVisualization.Optics -> "Processing order " + (current + 1) + "/" +
                    snapshot.size + "; reachability " +
                    "%.3f".format(optics?.reachability?.getOrNull(current) ?: 0.0)
                StageTwoVisualization.Gmm -> (if(current%2==0)
                    "E step: weighted links show the selected point's component responsibilities. "
                    else "M step: means and full covariance ellipses update from soft memberships. ") +
                    "Iteration " + (current / 2 + 1) +
                    " · log likelihood " + "%.2f".format(gmm?.logLikelihood ?: 0.0) + "."
                StageTwoVisualization.FuzzyCMeans -> "Membership at selected point: " +
                    fuzzy?.membership?.getOrNull(selected)?.joinToString { "%.2f".format(it) }
                StageTwoVisualization.Birch -> "Streamed " + minOf(snapshot.size,(current+1)*6) +
                    " points into " + cf.size + " CF entries. N, LS and SS are updated on insertion."
                StageTwoVisualization.Affinity -> "Responsibility and availability messages select " +
                    (affinity?.exemplars?.size ?: 0) + " exemplars; no K is supplied."
                StageTwoVisualization.Divisive -> "Start with one cluster and split the group with highest error. Groups: " +
                    (labels.maxOrNull()?.plus(1) ?: 0)
                StageTwoVisualization.Spectral -> "Reveal similarity graph edges, then cut it using its second spectral direction. Stage " +
                    (current+1) + "/" + (maxStep+1) + "."
                StageTwoVisualization.MeanShift -> "Kernel windows move toward local density modes. Bandwidth controls the local mean."
                else -> "Merge nearest groups using " + linkage.label + " linkage. Current merges: " + current
            }
            StageTwoInfo("Current algorithm state", detail)
        }
        if (kind in setOf(StageTwoVisualization.Hierarchical, StageTwoVisualization.Agglomerative))
            item { ClusterDendrogram(hierarchy?.merges.orEmpty(), current, cut) }
        if (kind == StageTwoVisualization.Hdbscan)
            item { MutualReachabilityHierarchy(snapshot.size, density) }
        if (kind == StageTwoVisualization.Optics)
            item { ClusterReachability(optics, current, eps) }
        if (kind == StageTwoVisualization.KMeansPlus && frame?.probabilities?.isNotEmpty() == true)
            item { StageTwoInfo("Distance-squared seeding",
                "The orange rings encode each sample's normalized D² selection probability. Random K-Means does not use this distribution.") }
        if (kind == StageTwoVisualization.KMeansPlus && compareRandom)
            item { StageTwoInfo("Initialization comparison",
                "Grey rings mark uniformly random seeds. D² seeding weights distant samples; both methods then run the same assign/update iterations.") }
        if (kind == StageTwoVisualization.Gmm)
            item { StageTwoInfo("Selected sample responsibilities",
                gmm?.responsibilities?.getOrNull(selected)?.mapIndexed { i, p ->
                    "Component " + (i+1) + ": " + "%.2f".format(p)
                }?.joinToString(" · ") ?: "") }
        item { Spacer(Modifier.height(22.dp)) }
    }
}

@Composable
private fun MutualReachabilityHierarchy(count: Int, hierarchy: DensityHierarchy?) {
    StageTwoInfo("Mutual-reachability hierarchy",
        "Branches merge at mutual-reachability distance. The orange cut selects connected groups; small branches remain noise. This is an educational hierarchy cut, not full HDBSCAN persistence scoring.")
    Canvas(Modifier.fillMaxWidth().height(190.dp)
        .background(Color(0xFF081527))
        .border(1.dp, LabBorder)) {
        if (hierarchy == null || count < 2) return@Canvas
        val edges=hierarchy.edges.sortedBy { it.third }
        val maximum=edges.maxOfOrNull { it.third }?.coerceAtLeast(1e-6) ?: return@Canvas
        val parent=IntArray(count) { it }
        val position=Array(count) { i ->
            Offset(18f+i*(size.width-36f)/(count-1),size.height-17f)
        }
        fun root(index: Int): Int {
            var current=index
            while (parent[current]!=current) current=parent[current]
            return current
        }
        edges.forEach { (a,b,height) ->
            val left=root(a); val right=root(b)
            if (left!=right) {
                val y=size.height-17f-(height/maximum*(size.height-35f)).toFloat()
                val first=position[left]; val second=position[right]
                val color=if (height<=hierarchy.cutoff) LabCyan.copy(alpha=.75f)
                    else LabMuted.copy(alpha=.55f)
                drawLine(color,first,Offset(first.x,y),1.6f)
                drawLine(color,second,Offset(second.x,y),1.6f)
                drawLine(color,Offset(first.x,y),Offset(second.x,y),1.6f)
                parent[right]=left
                position[left]=Offset((first.x+second.x)/2f,y)
            }
        }
        val cutY=size.height-17f-(hierarchy.cutoff/maximum*(size.height-35f)).toFloat()
        drawLine(LabOrange,Offset(8f,cutY),Offset(size.width-8f,cutY),2f)
    }
}

@Composable
private fun ClusterDendrogram(merges: List<MergeStep>, step: Int, cut: Double) {
    StageTwoInfo("Dendrogram", "Each branch is an actual linkage merge. The horizontal line is the current cut.")
    Canvas(Modifier.fillMaxWidth().height(180.dp)) {
        if (merges.isEmpty()) return@Canvas
        val n = merges.size + 1
        val maxHeight = merges.maxOf { it.height }.coerceAtLeast(1e-6)
        val leafX = (0 until n).associateWith { i -> 18f + i*(size.width-36f)/(n-1) }
        val positions = mutableMapOf<Set<Int>,Pair<Float,Float>>()
        for (i in 0 until n) positions[setOf(i)] = leafX.getValue(i) to size.height-12f
        merges.take(step).forEach { merge ->
            val left = positions[merge.left] ?: return@forEach
            val right = positions[merge.right] ?: return@forEach
            val y = size.height-16f-(merge.height/maxHeight*(size.height-32f)).toFloat()
            drawLine(LabCyan, Offset(left.first,left.second), Offset(left.first,y),2f)
            drawLine(LabPink, Offset(right.first,right.second), Offset(right.first,y),2f)
            drawLine(LabPurple, Offset(left.first,y), Offset(right.first,y),2f)
            positions[merge.merged] = ((left.first+right.first)/2) to y
        }
        val yCut = size.height-16f-(cut*(size.height-32f)).toFloat()
        drawLine(LabOrange.copy(alpha=.8f),Offset(8f,yCut),Offset(size.width-8f,yCut),2f)
    }
}

@Composable
private fun ClusterReachability(order: OpticsOrder?, step: Int, eps: Double) {
    StageTwoInfo("Reachability order", "Bars are ordered by OPTICS processing. Valleys indicate dense cluster structure.")
    Canvas(Modifier.fillMaxWidth().height(150.dp)) {
        val values = order?.reachability ?: return@Canvas
        values.forEachIndexed { i, value ->
            val height = ((value.takeIf { it.isFinite() } ?: eps*2).coerceAtMost(eps*2)/(eps*2)*size.height).toFloat()
            val width = size.width/values.size
            drawRect(if (i <= step) LabCyan else LabMuted.copy(alpha=.35f),
                Offset(i*width,size.height-height), Size((width-1).coerceAtLeast(1f),height))
        }
    }
}
