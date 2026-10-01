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
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageTwoVisualization
import kotlin.math.*

@Composable
internal fun StageTwoAnomalyPanel(kind: StageTwoVisualization) {
    var preset by remember(kind) { mutableStateOf(ClusterPreset.Outliers) }
    var seed by remember(kind) { mutableIntStateOf(62) }
    val points=remember(kind,preset,seed) {
        mutableStateListOf<ClusterPoint>().apply {
            addAll(PhaseThreeDatasets.clusters(preset,50,2,.045,seed))
        }
    }
    var selected by remember(kind) { mutableIntStateOf(0) }
    var k by remember(kind) { mutableIntStateOf(5) }
    var step by remember(kind) { mutableIntStateOf(0) }
    var threshold by remember(kind) { mutableDoubleStateOf(.6) }
    var gamma by remember(kind) { mutableDoubleStateOf(2.0) }
    var method by remember(kind) { mutableStateOf("Z-score") }
    val snapshot=points.toList()
    val sample=snapshot[selected.coerceIn(snapshot.indices)]
    val trees=remember(snapshot,k,seed) {
        if(kind==StageTwoVisualization.IsolationForest)
            List(k.coerceIn(1,20)) { i ->
                val random=kotlin.random.Random(seed+i*71)
                val subset=List(32) { snapshot[random.nextInt(snapshot.size)] }
                StageTwoAnomalyEngine.isolationTree(subset,0,seed+i*47)
            } else emptyList()
    }
    val lof=remember(snapshot,k) {
        if(kind==StageTwoVisualization.Lof) StageTwoAnomalyEngine.lof(snapshot,k) else emptyList()
    }
    val ellipse=remember(snapshot) { StageTwoAnomalyEngine.robustEllipse(snapshot) }
    val svm=remember(snapshot,threshold,gamma) {
        if(kind==StageTwoVisualization.OneClassSvm)
            StageTwoAnomalyEngine.oneClassSvm(snapshot.filter { it.hiddenLabel>=0 },
                threshold.coerceIn(.05,.9),gamma) else null
    }
    val pca=remember(snapshot) { PhaseThreeEngines.pca(snapshot.filter { it.hiddenLabel>=0 }) }
    val reconstructed=ClusterPoint(
        pca.mean.x+((sample.x-pca.mean.x)*pca.pc1.x+(sample.y-pca.mean.y)*pca.pc1.y)*pca.pc1.x,
        pca.mean.y+((sample.x-pca.mean.x)*pca.pc1.x+(sample.y-pca.mean.y)*pca.pc1.y)*pca.pc1.y
    )
    val reconstructionError=(sample.x-reconstructed.x).pow(2)+(sample.y-reconstructed.y).pow(2)
    val values=snapshot.map { it.x }
    val mean=values.average()
    val sd=sqrt(values.map { (it-mean).pow(2) }.average()).coerceAtLeast(1e-8)
    val quartiles=StageTwoAnomalyEngine.quartiles(values)
    val iqr=quartiles.second-quartiles.first
    val score=when(kind) {
        StageTwoVisualization.IsolationForest ->
            StageTwoAnomalyEngine.isolationScore(trees,sample,32)
        StageTwoVisualization.Lof -> lof[selected.coerceIn(lof.indices)]
        StageTwoVisualization.OneClassSvm -> svm?.score(sample.x,sample.y) ?: 0.0
        StageTwoVisualization.EllipticEnvelope -> ellipse.distance(sample)
        StageTwoVisualization.AutoencoderAnomaly -> reconstructionError
        else -> if(method=="Z-score") abs(sample.x-mean)/sd else
            maxOf(quartiles.first-sample.x,sample.x-quartiles.second,0.0)/iqr.coerceAtLeast(1e-8)
    }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal=14.dp),verticalArrangement=Arrangement.spacedBy(10.dp)) {
        item { StageTwoInfo(kind.title,kind.graphic) }
        item {
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(6.dp)) {
                listOf(ClusterPreset.Outliers,ClusterPreset.DenseSparse,ClusterPreset.Elongated)
                    .forEach { option ->
                        val short = when (option) {
                            ClusterPreset.Outliers -> "Outliers"
                            ClusterPreset.DenseSparse -> "Density"
                            else -> "Elongated"
                        }
                        SegmentedOption(short,preset==option,Modifier.weight(1f)) {
                            preset=option; selected=0; step=0
                        }
                    }
            }
        }
        item {
            StageTwoPlot(snapshot,selected=selected,onMove={ i,moved ->
                points[i]=moved; selected=i
            }) { scale ->
                when(kind) {
                    StageTwoVisualization.IsolationForest -> {
                        val tree=trees.firstOrNull()
                        fun drawPartition(node: IsolationNode?,x0: Double,x1: Double,y0: Double,y1: Double,depth: Int) {
                            if(node==null || depth>step.coerceAtLeast(1) || node.left==null) return
                            if(node.axis==0) {
                                drawLine(LabOrange.copy(alpha=.7f),scale.point(node.threshold,y0),
                                    scale.point(node.threshold,y1),2f)
                                drawPartition(node.left,x0,node.threshold,y0,y1,depth+1)
                                drawPartition(node.right,node.threshold,x1,y0,y1,depth+1)
                            } else {
                                drawLine(LabOrange.copy(alpha=.7f),scale.point(x0,node.threshold),
                                    scale.point(x1,node.threshold),2f)
                                drawPartition(node.left,x0,x1,y0,node.threshold,depth+1)
                                drawPartition(node.right,x0,x1,node.threshold,y1,depth+1)
                            }
                        }
                        drawPartition(tree,-1.0,1.0,-1.0,1.0,0)
                    }
                    StageTwoVisualization.Lof -> {
                        val near=snapshot.indices.filter { it!=selected }
                            .sortedBy { hypot(snapshot[it].x-sample.x,snapshot[it].y-sample.y) }.take(k)
                        near.forEach { i ->
                            drawLine(LabCyan.copy(alpha=.7f),scale.point(sample.x,sample.y),
                                scale.point(snapshot[i].x,snapshot[i].y),2f)
                        }
                        val radius=near.maxOfOrNull {
                            hypot(snapshot[it].x-sample.x,snapshot[it].y-sample.y)
                        } ?: 0.0
                        drawCircle(LabCyan.copy(alpha=.7f),
                            (radius*(scale.right-scale.left)/2).toFloat(),
                            scale.point(sample.x,sample.y),style=Stroke(2f))
                    }
                    StageTwoVisualization.OneClassSvm -> {
                        val model=svm
                        if(model!=null) for(ix in 0 until 34) for(iy in 0 until 34) {
                            val x=-1.0+2.0*ix/34; val y=-1.0+2.0*iy/34
                            val s=model.score(x,y)
                            if(s>=0) drawRect(LabCyan.copy(alpha=.07f),
                                scale.point(x,y+2.0/34),Size(
                                    (scale.right-scale.left)/34,(scale.bottom-scale.top)/34))
                        }
                        model?.alpha?.forEachIndexed { i,weight ->
                            if(weight>1e-3) drawCircle(LabOrange,9f,
                                scale.point(model.points[i].x,model.points[i].y),style=Stroke(2f))
                        }
                    }
                    StageTwoVisualization.EllipticEnvelope -> {
                        val n=90
                        for(i in 0 until n) {
                            val a=2*PI*i/n; val b=2*PI*(i+1)/n
                            fun point(t: Double): Offset {
                                val rx=cos(t); val ry=sin(t)
                                val x=ellipse.centerX+threshold*2.5*sqrt(ellipse.xx)*rx
                                val y=ellipse.centerY+threshold*2.5*(
                                    ellipse.xy/sqrt(ellipse.xx)*rx+
                                        sqrt((ellipse.yy-ellipse.xy.pow(2)/ellipse.xx).coerceAtLeast(0.0))*ry)
                                return scale.point(x,y)
                            }
                            drawLine(LabOrange,point(a),point(b),2f)
                        }
                        drawCircle(LabGreen,7f,scale.point(ellipse.centerX,ellipse.centerY))
                    }
                    StageTwoVisualization.AutoencoderAnomaly -> {
                        drawLine(LabOrange,scale.point(sample.x,sample.y),
                            scale.point(reconstructed.x,reconstructed.y),3f)
                        drawCircle(LabGreen,8f,scale.point(reconstructed.x,reconstructed.y))
                    }
                    StageTwoVisualization.StatisticalOutlier -> {
                        val lower=if(method=="Z-score") mean-threshold*3*sd
                            else quartiles.first-threshold*2*iqr
                        val upper=if(method=="Z-score") mean+threshold*3*sd
                            else quartiles.second+threshold*2*iqr
                        listOf(lower,upper).forEach { x ->
                            drawLine(LabOrange,scale.point(x,-1.0),scale.point(x,1.0),2f)
                        }
                    }
                    else -> Unit
                }
            }
        }
        item {
            Column(verticalArrangement=Arrangement.spacedBy(6.dp)) {
                NativeSlider("Selected sample",selected.toDouble(),0.0,(snapshot.size-1).toDouble()) {
                    selected=it.roundToInt().coerceIn(snapshot.indices)
                }
                when(kind) {
                    StageTwoVisualization.IsolationForest -> NativeSlider("Trees",k.toDouble(),2.0,20.0) {
                        k=it.roundToInt()
                    }
                    StageTwoVisualization.Lof -> NativeSlider("Neighbors K",k.toDouble(),2.0,15.0) {
                        k=it.roundToInt()
                    }
                    StageTwoVisualization.OneClassSvm -> {
                        NativeSlider("Nu",threshold,.05,.8) { threshold=it }
                        NativeSlider("Gamma",gamma,.2,8.0) { gamma=it }
                    }
                    StageTwoVisualization.EllipticEnvelope,
                    StageTwoVisualization.AutoencoderAnomaly,
                    StageTwoVisualization.StatisticalOutlier ->
                        NativeSlider("Threshold",threshold,.1,1.0) { threshold=it }
                    else -> Unit
                }
                if(kind==StageTwoVisualization.StatisticalOutlier)
                    Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(7.dp)) {
                        listOf("Z-score","IQR").forEach { option ->
                            SegmentedOption(option,method==option,Modifier.weight(1f)) { method=option }
                        }
                    }
            }
        }
        if(kind==StageTwoVisualization.IsolationForest)
            item { StageTwoStepControls(step,11) { step=it } }
        item {
            StageTwoInfo("Selected sample · " + selected,
                when(kind) {
                    StageTwoVisualization.IsolationForest ->
                        "Average isolation path " + "%.2f".format(trees.map {
                            StageTwoAnomalyEngine.path(it,sample).toDouble()
                        }.average()) + " · anomaly score " + "%.3f".format(score) +
                            ". Shorter paths produce higher scores."
                    StageTwoVisualization.Lof ->
                        "LOF " + "%.3f".format(score) +
                            " compares this sample's local reachability density with its neighbors."
                    StageTwoVisualization.OneClassSvm ->
                        "Kernel decision score " + "%.3f".format(score) +
                            ". Negative values lie outside the learned normal boundary."
                    StageTwoVisualization.EllipticEnvelope ->
                        "Robust Mahalanobis distance " + "%.3f".format(score) +
                            "; the ellipse comes from trimmed covariance."
                    StageTwoVisualization.AutoencoderAnomaly ->
                        "Linear bottleneck reconstruction error " + "%.4f".format(score) +
                            ". The green point is the reconstructed sample."
                    else -> method + " deviation " + "%.3f".format(score) +
                        " from the fitted reference distribution."
                })
        }
        if(kind==StageTwoVisualization.StatisticalOutlier)
            item { AnomalyHistogram(values,mean,quartiles.first,quartiles.second) }
        item { Spacer(Modifier.height(22.dp)) }
    }
}

@Composable
private fun AnomalyHistogram(values: List<Double>,mean: Double,q1: Double,q3: Double) {
    StageTwoInfo("Feature distribution","Center " + "%.2f".format(mean) +
        " · Q1 " + "%.2f".format(q1) + " · Q3 " + "%.2f".format(q3))
    Canvas(Modifier.fillMaxWidth().height(140.dp)) {
        val bins=IntArray(18)
        values.forEach { bins[((it+1)/2*18).toInt().coerceIn(0,17)]++ }
        val max=bins.max().coerceAtLeast(1)
        bins.forEachIndexed { i,count ->
            val w=size.width/18
            val h=size.height*count/max
            drawRect(LabCyan,Offset(i*w,size.height-h),Size(w-2,h))
        }
    }
}
