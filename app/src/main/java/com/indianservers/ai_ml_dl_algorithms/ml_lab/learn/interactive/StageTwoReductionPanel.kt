package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageTwoVisualization
import kotlin.math.*

private val documentMatrix = listOf(
    listOf(4.0,3.0,0.0,0.0), listOf(5.0,4.0,0.0,1.0),
    listOf(0.0,0.0,4.0,5.0), listOf(0.0,1.0,5.0,4.0),
    listOf(3.0,2.0,0.0,0.0), listOf(0.0,0.0,3.0,4.0)
)

@Composable
internal fun StageTwoReductionPanel(kind: StageTwoVisualization) {
    var preset by remember(kind) { mutableStateOf(when(kind) {
        StageTwoVisualization.KernelPca -> ClusterPreset.Circles
        StageTwoVisualization.Tsne,StageTwoVisualization.Umap -> ClusterPreset.TwoMoons
        else -> ClusterPreset.Elongated
    }) }
    var seed by remember(kind) { mutableIntStateOf(73) }
    val points=remember(kind,preset,seed) {
        mutableStateListOf<ClusterPoint>().apply {
            addAll(PhaseThreeDatasets.clusters(preset,28,2,.07,seed))
        }
    }
    var step by remember(kind,preset,seed) { mutableIntStateOf(0) }
    var parameter by remember(kind) {
        mutableDoubleStateOf(if(kind==StageTwoVisualization.Ica) .55 else 3.0)
    }
    var secondary by remember(kind) { mutableDoubleStateOf(.2) }
    var selected by remember(kind) { mutableIntStateOf(0) }
    val snapshot=points.toList()
    val active=if(kind==StageTwoVisualization.IncrementalPca)
        snapshot.take(((step+1)*4).coerceAtMost(snapshot.size)) else snapshot
    val pca=remember(active) { PhaseThreeEngines.pca(active) }
    val svd=remember(parameter) {
        StageTwoReductionEngine.truncatedSvd(documentMatrix,parameter.roundToInt().coerceIn(1,4))
    }
    val reduction=remember(snapshot,kind,parameter,secondary,step) {
        when(kind) {
            StageTwoVisualization.KernelPca -> StageTwoReductionEngine.kernelPca(snapshot,parameter)
            StageTwoVisualization.Tsne ->
                StageTwoReductionEngine.tsne(snapshot,parameter,(step+1)*5,secondary)
            StageTwoVisualization.Umap ->
                StageTwoReductionEngine.umap(snapshot,parameter.roundToInt().coerceIn(2,10),
                    secondary,(step+1)*6)
            StageTwoVisualization.Isomap ->
                StageTwoReductionEngine.isomap(snapshot,parameter.roundToInt().coerceIn(2,9))
            StageTwoVisualization.Lle ->
                StageTwoReductionEngine.lle(snapshot,parameter.roundToInt().coerceIn(2,9))
            StageTwoVisualization.Mds -> StageTwoReductionEngine.mds(snapshot)
            StageTwoVisualization.AutoencoderReduction ->
                StageTwoReductionEngine.linearAutoencoder(snapshot,(step+1)*8)
            else -> null
        }
    }
    val isEmbedding=kind in setOf(StageTwoVisualization.KernelPca,StageTwoVisualization.Tsne,
        StageTwoVisualization.Umap,StageTwoVisualization.Isomap,StageTwoVisualization.Lle,
        StageTwoVisualization.Mds,StageTwoVisualization.AutoencoderReduction)
    LazyColumn(Modifier.fillMaxSize().padding(horizontal=14.dp),verticalArrangement=Arrangement.spacedBy(10.dp)) {
        item { StageTwoInfo(kind.title,kind.graphic) }
        if(kind!=StageTwoVisualization.TruncatedSvd && kind!=StageTwoVisualization.Ica)
            item {
                Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(5.dp)) {
                    listOf(ClusterPreset.Elongated,ClusterPreset.Circles,ClusterPreset.TwoMoons)
                        .forEach { option ->
                            val short = when (option) {
                                ClusterPreset.Elongated -> "Elongated"
                                ClusterPreset.Circles -> "Circles"
                                else -> "Moons"
                            }
                            SegmentedOption(short,preset==option,Modifier.weight(1f)) {
                                preset=option; step=0
                            }
                        }
                }
            }
        if(kind in setOf(StageTwoVisualization.IncrementalPca,StageTwoVisualization.Tsne,
                StageTwoVisualization.Umap,StageTwoVisualization.AutoencoderReduction))
            item { StageTwoStepControls(step,11) { step=it } }
        if(kind in setOf(StageTwoVisualization.KernelPca,StageTwoVisualization.Tsne,
                StageTwoVisualization.Umap,StageTwoVisualization.Isomap,StageTwoVisualization.Lle))
            item { NativeSlider(when(kind) {
                StageTwoVisualization.KernelPca -> "RBF gamma"
                StageTwoVisualization.Tsne -> "Perplexity"
                else -> "Neighbors"
            },parameter,if(kind==StageTwoVisualization.KernelPca) .1 else 2.0,
                if(kind==StageTwoVisualization.KernelPca) 12.0 else 10.0) {
                parameter=it
            } }
        if(kind==StageTwoVisualization.Tsne)
            item { NativeSlider("Learning rate",secondary,.05,2.0) { secondary=it } }
        if(kind==StageTwoVisualization.Umap)
            item { NativeSlider("Minimum distance",secondary,.02,.8) { secondary=it } }
        if(kind==StageTwoVisualization.SparsePca)
            item { NativeSlider("Sparsity penalty",secondary,0.0,.8) { secondary=it } }
        if(kind==StageTwoVisualization.TruncatedSvd) {
            item { MatrixHeatmap(documentMatrix,"Original document-term matrix") }
            item { MatrixHeatmap(svd.reconstruction,"Rank-" + parameter.roundToInt() + " reconstruction") }
            item { StageTwoInfo("Singular spectrum",
                svd.singular.mapIndexed { i,value -> "σ" + (i+1) + "=" + "%.2f".format(value) }
                    .joinToString("  ") + "\nReconstruction MSE " + "%.3f".format(svd.error)) }
            item { NativeSlider("Retained rank",parameter,1.0,4.0) { parameter=it } }
        } else if(kind==StageTwoVisualization.Ica) {
            item { NativeSlider("Mixing coefficient",parameter,.1,.9) { parameter=it } }
            item { NativeSlider("Second-channel mix",secondary,.1,.9) { secondary=it } }
            item { IcaSignals(parameter,secondary) }
            item {
                StageTwoInfo("Independent source recovery",
                    "Two independent source signals are mixed into observed channels. The recovered traces use the inverse mixing matrix; this is source separation, not variance maximization.")
            }
        } else {
            item {
                StageTwoInfo("Input observations",
                    when(kind) {
                        StageTwoVisualization.Pca -> "Drag samples; covariance and principal axes refit."
                        StageTwoVisualization.IncrementalPca -> "Only the first " + active.size +
                            " samples have arrived in mini-batches."
                        StageTwoVisualization.KernelPca -> "Circles cannot be separated by one straight projection; an RBF kernel changes the geometry."
                        StageTwoVisualization.SparsePca -> "Four engineered features expose which loadings shrink to zero."
                        StageTwoVisualization.FactorAnalysis -> "Correlated observed features are explained by shared latent factors plus unique noise."
                        StageTwoVisualization.Tsne -> "Input local neighbors become high-dimensional affinity probabilities."
                        StageTwoVisualization.Umap -> "The input forms a fuzzy nearest-neighbor graph."
                        StageTwoVisualization.Isomap -> "K-neighbor edges approximate manifold geodesics."
                        StageTwoVisualization.Lle -> "Each point is reconstructed from its local neighbors."
                        StageTwoVisualization.Mds -> "All pairwise distances drive the low-dimensional layout."
                        else -> "A two-dimensional input is compressed through a one-dimensional bottleneck."
                    })
            }
            item {
                StageTwoPlot(snapshot,selected=selected,onMove={ i,moved ->
                    points[i]=moved; selected=i
                }) { scale ->
                    when(kind) {
                        StageTwoVisualization.Pca,StageTwoVisualization.IncrementalPca,
                        StageTwoVisualization.SparsePca,StageTwoVisualization.FactorAnalysis -> {
                            val mean=scale.point(pca.mean.x,pca.mean.y)
                            listOf(pca.pc1 to LabOrange,pca.pc2 to LabPurple).forEach { (axis,color) ->
                                drawLine(color,scale.point(pca.mean.x-axis.x*.9,pca.mean.y-axis.y*.9),
                                    scale.point(pca.mean.x+axis.x*.9,pca.mean.y+axis.y*.9),3f)
                            }
                            drawCircle(Color.White,7f,mean)
                            snapshot.forEach { p ->
                                val value=(p.x-pca.mean.x)*pca.pc1.x+(p.y-pca.mean.y)*pca.pc1.y
                                drawLine(LabCyan.copy(alpha=.5f),scale.point(p.x,p.y),
                                    scale.point(pca.mean.x+value*pca.pc1.x,
                                        pca.mean.y+value*pca.pc1.y),1.5f)
                            }
                        }
                        StageTwoVisualization.Tsne,StageTwoVisualization.Umap,
                        StageTwoVisualization.Isomap,StageTwoVisualization.Lle ->
                            reduction?.edges?.forEach { (a,b) ->
                                drawLine(LabOrange.copy(alpha=.35f),
                                    scale.point(snapshot[a].x,snapshot[a].y),
                                    scale.point(snapshot[b].x,snapshot[b].y),1.5f)
                            }
                        else -> Unit
                    }
                }
            }
            if(isEmbedding) {
                item {
                    StageTwoInfo(if(kind==StageTwoVisualization.AutoencoderReduction)
                        "Reconstruction from bottleneck" else "Derived embedding",
                        reduction?.detail ?: "")
                }
                item {
                    val embedded=reduction?.embedded.orEmpty()
                    StageTwoPlot(embedded,labels=List(embedded.size) { snapshot[it].hiddenLabel })
                }
                item { StageTwoInfo(reduction?.metricLabel ?: "Metric",
                    "%.4f".format(reduction?.metric ?: 0.0)) }
            }
            when(kind) {
                StageTwoVisualization.Pca,StageTwoVisualization.IncrementalPca -> {
                    item {
                        StageTwoInfo("Principal components",
                            "λ₁ " + "%.3f".format(pca.variance1) + " · λ₂ " +
                                "%.3f".format(pca.variance2) + " · PC1 explained variance " +
                                "%.1f".format(pca.variance1/
                                    (pca.variance1+pca.variance2).coerceAtLeast(1e-9)*100) +
                                "% · 1D reconstruction MSE " + "%.3f".format(pca.reconstructionError))
                    }
                    item { ProjectionStrip(pca.projected) }
                }
                StageTwoVisualization.SparsePca -> {
                    item {
                        val load=listOf(pca.pc1.x,pca.pc1.y,
                            .65*pca.pc1.x+.35*pca.pc1.y,.15*pca.pc1.x-.4*pca.pc1.y)
                        val sparse=load.map { sign(it)*max(0.0,abs(it)-secondary) }
                        LoadingBars(sparse)
                    }
                }
                StageTwoVisualization.FactorAnalysis -> {
                    item {
                        val loading=listOf(pca.pc1.x*sqrt(pca.variance1),
                            pca.pc1.y*sqrt(pca.variance1))
                        StageTwoInfo("Latent factor → observed features",
                            "Factor 1 → X₁: " + "%.2f".format(loading[0]) +
                                " · X₂: " + "%.2f".format(loading[1]) +
                                "\nUnique noise X₁: " +
                                "%.3f".format((snapshot.map { (it.x-pca.mean.x).pow(2) }
                                    .average()-loading[0].pow(2)).coerceAtLeast(0.0)) +
                                " · X₂: " +
                                "%.3f".format((snapshot.map { (it.y-pca.mean.y).pow(2) }
                                    .average()-loading[1].pow(2)).coerceAtLeast(0.0)))
                    }
                    item { LoadingBars(listOf(pca.pc1.x,pca.pc1.y)) }
                }
                else -> Unit
            }
            if(kind==StageTwoVisualization.Tsne)
                item { StageTwoInfo("Interpretation warning",
                    "t-SNE preserves local neighborhoods. The distance between separated islands is not a reliable global distance.") }
            item { SegmentedOption("New dataset",false,Modifier.fillMaxWidth()) { seed++; step=0 } }
        }
        item { Spacer(Modifier.height(22.dp)) }
    }
}

@Composable
private fun MatrixHeatmap(matrix: List<List<Double>>,title: String) {
    StageTwoInfo(title,"Each cell encodes a measured matrix value.")
    Canvas(Modifier.fillMaxWidth().height(160.dp)) {
        val rows=matrix.size; val cols=matrix[0].size
        val scale=matrix.flatten().maxOf { abs(it) }.coerceAtLeast(1e-9)
        matrix.forEachIndexed { i,row ->
            row.forEachIndexed { j,value ->
                drawRect(LabCyan.copy(alpha=(.08+.85*abs(value)/scale).toFloat()),
                    Offset(j*size.width/cols,i*size.height/rows),
                    Size(size.width/cols-3,size.height/rows-3))
            }
        }
    }
}

@Composable
private fun ProjectionStrip(values: List<Double>) {
    StageTwoInfo("Projection onto PC1","Lines in the input plot show each point's one-dimensional reconstruction.")
    Canvas(Modifier.fillMaxWidth().height(65.dp)) {
        drawLine(LabMuted,Offset(12f,size.height/2),Offset(size.width-12f,size.height/2),2f)
        val max=values.maxOf { abs(it) }.coerceAtLeast(1e-6)
        values.forEach { value ->
            drawCircle(LabCyan,4.5f,Offset(
                size.width/2+(value/max*size.width*.44).toFloat(),size.height/2))
        }
    }
}

@Composable
private fun LoadingBars(loadings: List<Double>) {
    StageTwoInfo("Feature loadings",
        loadings.mapIndexed { i,v -> "Feature " + (i+1) + ": " + "%.2f".format(v) }
            .joinToString(" · ") + "\nZero bars are excluded features.")
    Canvas(Modifier.fillMaxWidth().height(110.dp)) {
        val width=size.width/loadings.size
        loadings.forEachIndexed { i,v ->
            val height=(abs(v).coerceIn(0.0,1.0)*size.height*.85).toFloat()
            drawRect(if(v==0.0) LabMuted else stageColor(i),
                Offset(i*width+width*.2f,size.height-height),
                Size(width*.6f,height.coerceAtLeast(2f)))
        }
    }
}

@Composable
private fun IcaSignals(mix: Double,second: Double) {
    val n=64
    val first=List(n) { sin(2*PI*it/n) }
    val other=List(n) { if(it%16<8) 1.0 else -1.0 }
    val observedA=List(n) { first[it]+mix*other[it] }
    val observedB=List(n) { second*first[it]+other[it] }
    val determinant=(1-mix*second).coerceAtLeast(.02)
    val recovered=List(n) { (observedA[it]-mix*observedB[it])/determinant }
    listOf("Source A","Mixed observation A","Recovered source A").zip(
        listOf(first,observedA,recovered)).forEach { (title,values) ->
        StageTwoInfo(title,when(title) {
            "Source A" -> "Independent sine source"
            "Mixed observation A" -> "Sine plus square-wave source"
            else -> "Recovered by unmixing the two observed channels"
        })
        Canvas(Modifier.fillMaxWidth().height(72.dp)) {
            val path=Path()
            values.forEachIndexed { i,value ->
                val at=Offset(i*size.width/(n-1),
                    size.height/2-(value*size.height*.32).toFloat())
                if(i==0) path.moveTo(at.x,at.y) else path.lineTo(at.x,at.y)
            }
            drawPath(path,LabCyan,style=Stroke(2.5f))
        }
    }
}
