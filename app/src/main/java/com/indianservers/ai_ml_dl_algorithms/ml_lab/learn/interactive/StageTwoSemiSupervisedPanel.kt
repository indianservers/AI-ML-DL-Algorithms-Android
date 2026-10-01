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
internal fun StageTwoSemiSupervisedPanel(kind: StageTwoVisualization) {
    var seed by remember(kind) { mutableIntStateOf(91) }
    var preset by remember(kind) { mutableStateOf(ClusterPreset.TwoMoons) }
    val points=remember(kind,seed,preset) {
        mutableStateListOf<ClusterPoint>().apply {
            addAll(PhaseThreeDatasets.clusters(preset,44,2,.06,seed))
        }
    }
    var step by remember(kind,seed,preset) { mutableIntStateOf(0) }
    var threshold by remember(kind) { mutableDoubleStateOf(.78) }
    var secondary by remember(kind) { mutableDoubleStateOf(.8) }
    var selected by remember(kind) { mutableIntStateOf(8) }
    val snapshot=points.toList()
    val graphKind=kind in setOf(StageTwoVisualization.LabelPropagation,
        StageTwoVisualization.LabelSpreading)
    val selfKind=kind in setOf(StageTwoVisualization.SelfTraining,
        StageTwoVisualization.PseudoLabelling)
    val state=remember(snapshot,kind,step,threshold,secondary) {
        when {
            selfKind -> StageTwoSemiSupervisedEngine.selfTraining(snapshot,step,threshold,
                kind==StageTwoVisualization.PseudoLabelling)
            graphKind -> StageTwoSemiSupervisedEngine.graphPropagation(snapshot,step,
                kind==StageTwoVisualization.LabelSpreading,secondary,5)
            else -> null
        }
    }
    val co=remember(snapshot,step,threshold) {
        if(kind==StageTwoVisualization.CoTraining)
            StageTwoSemiSupervisedEngine.coTraining(snapshot,step,threshold) else null
    }
    val consistency=remember(snapshot,selected,secondary,threshold,kind) {
        StageTwoSemiSupervisedEngine.consistency(snapshot,selected.coerceIn(snapshot.indices),
            secondary,threshold,kind==StageTwoVisualization.FixMatch)
    }
    val teacher=remember(step,secondary) {
        StageTwoSemiSupervisedEngine.teacherTrace(step,secondary)
    }
    val mix=remember(snapshot,selected,secondary,threshold) {
        StageTwoSemiSupervisedEngine.mixMatch(snapshot,selected.coerceIn(snapshot.indices),
            secondary.coerceAtLeast(.1),threshold)
    }
    val labeled=state?.labeled ?: co?.first?.labeled ?: setOf(0,1,2,3)
    val probabilities=state?.probabilities ?: co?.first?.probabilities ?:
        List(snapshot.size) { .5 }
    val labels=List(snapshot.size) { i ->
        if(i !in labeled) -1 else if(probabilities[i]>=.5) 1 else 0
    }
    LazyColumn(Modifier.fillMaxSize().padding(horizontal=14.dp),verticalArrangement=Arrangement.spacedBy(10.dp)) {
        item { StageTwoInfo(kind.title,kind.graphic) }
        item {
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(6.dp)) {
                listOf(ClusterPreset.TwoMoons,ClusterPreset.Blobs,ClusterPreset.Overlap)
                    .forEach { option ->
                        val short = when (option) {
                            ClusterPreset.TwoMoons -> "Moons"
                            ClusterPreset.Blobs -> "Blobs"
                            else -> "Overlap"
                        }
                        SegmentedOption(short,preset==option,Modifier.weight(1f)) {
                            preset=option; selected=8; step=0
                        }
                    }
            }
        }
        item {
            StageTwoPlot(snapshot,labels,selected,onMove={ i,moved ->
                points[i]=moved; selected=i
            }) { scale ->
                if(graphKind) {
                    PhaseThreeEngines.similarityGraph(snapshot,5).edges.forEach { (a,b) ->
                        drawLine(LabPurple.copy(alpha=.3f),
                            scale.point(snapshot[a].x,snapshot[a].y),
                            scale.point(snapshot[b].x,snapshot[b].y),1.5f)
                    }
                    snapshot.forEachIndexed { i,p ->
                        if(i !in labeled) drawCircle(if(probabilities[i]>=.5) LabPink else LabCyan,
                            (5+abs(probabilities[i]-.5)*10).toFloat(),
                            scale.point(p.x,p.y),style=Stroke(1.5f))
                    }
                }
                if(selfKind) state?.newlyLabeled?.forEach { i ->
                    drawCircle(LabOrange,12f,scale.point(snapshot[i].x,snapshot[i].y),
                        style=Stroke(2.5f))
                }
                if(kind in setOf(StageTwoVisualization.Consistency,StageTwoVisualization.FixMatch,
                        StageTwoVisualization.MixMatch)) {
                    val focus=snapshot[selected.coerceIn(snapshot.indices)]
                    drawLine(LabOrange,scale.point(focus.x,focus.y),
                        scale.point(consistency.weak.x,consistency.weak.y),2f)
                    drawLine(LabPurple,scale.point(focus.x,focus.y),
                        scale.point(consistency.strong.x,consistency.strong.y),2f)
                    drawCircle(LabOrange,8f,scale.point(consistency.weak.x,consistency.weak.y))
                    drawCircle(LabPurple,8f,scale.point(consistency.strong.x,consistency.strong.y))
                }
                if(kind==StageTwoVisualization.CoTraining) {
                    val focus=snapshot[selected.coerceIn(snapshot.indices)]
                    drawLine(LabCyan.copy(alpha=.6f),scale.point(focus.x,-1.0),
                        scale.point(focus.x,1.0),2f)
                    drawLine(LabPink.copy(alpha=.6f),scale.point(-1.0,focus.y),
                        scale.point(1.0,focus.y),2f)
                }
            }
        }
        item { StageTwoStepControls(step,12) { step=it } }
        item {
            Column(verticalArrangement=Arrangement.spacedBy(6.dp)) {
                if(kind in setOf(StageTwoVisualization.SelfTraining,StageTwoVisualization.PseudoLabelling,
                        StageTwoVisualization.CoTraining,StageTwoVisualization.FixMatch))
                    NativeSlider("Confidence threshold",threshold,.5,.98) { threshold=it }
                if(kind in setOf(StageTwoVisualization.LabelSpreading,StageTwoVisualization.MeanTeacher,
                        StageTwoVisualization.Consistency,StageTwoVisualization.FixMatch,
                        StageTwoVisualization.MixMatch))
                    NativeSlider(when(kind) {
                        StageTwoVisualization.LabelSpreading -> "Soft-clamping alpha"
                        StageTwoVisualization.MeanTeacher -> "Teacher EMA decay"
                        StageTwoVisualization.MixMatch -> "Sharpen temperature"
                        else -> "Augmentation strength"
                    },secondary,.1,.98) { secondary=it }
                NativeSlider("Selected unlabeled sample",selected.toDouble(),4.0,
                    (snapshot.size-1).toDouble()) { selected=it.roundToInt().coerceIn(4,snapshot.lastIndex) }
                SegmentedOption("New dataset",false,Modifier.fillMaxWidth()) { seed++; step=0 }
            }
        }
        when(kind) {
            StageTwoVisualization.SelfTraining,StageTwoVisualization.PseudoLabelling -> {
                item { StageTwoInfo("Pseudo-label admission",
                    (state?.detail ?: "") + "\nLabeled samples " + (state?.labeled?.size ?: 0) +
                        " · admitted this step " + (state?.newlyLabeled?.size ?: 0) +
                        " · mean confidence " + "%.2f".format(state?.confidence ?: 0.0)) }
                item { SemiProbabilityBars(probabilities.take(16),labeled) }
            }
            StageTwoVisualization.LabelPropagation,StageTwoVisualization.LabelSpreading -> {
                item { StageTwoInfo(if(graphKind) "Graph diffusion" else "Diffusion",
                    (state?.detail ?: "") + "\nSeed labels 4 · iteration " + step +
                        " · selected P(class 1) " +
                        "%.2f".format(probabilities[selected.coerceIn(probabilities.indices)])) }
                item { SemiProbabilityBars(probabilities.take(16),labeled) }
            }
            StageTwoVisualization.CoTraining -> {
                item { StageTwoInfo("View A · feature X1",
                    "P(class 1) for selected sample " +
                        "%.2f".format(co?.first?.probabilities?.get(selected) ?: 0.0) +
                        " · labeled " + (co?.first?.labeled?.size ?: 0) +
                        "\n" + (co?.first?.detail ?: "")) }
                item { StageTwoInfo("View B · feature X2",
                    "P(class 1) for selected sample " +
                        "%.2f".format(co?.second?.probabilities?.get(selected) ?: 0.0) +
                        " · labeled " + (co?.second?.labeled?.size ?: 0) +
                        "\n" + (co?.second?.detail ?: "")) }
            }
            StageTwoVisualization.Consistency -> {
                item { StageTwoInfo("Consistency loss",
                    "Weak augmentation P(class 1) " + "%.2f".format(consistency.weakProbability) +
                        " · second perturbation " + "%.2f".format(consistency.strongProbability) +
                        " · squared disagreement " + "%.4f".format(consistency.loss)) }
            }
            StageTwoVisualization.MeanTeacher -> {
                item { StageTwoInfo("Student → EMA teacher",
                    "Student " + "%.3f".format(teacher.last().first) +
                        " · teacher " + "%.3f".format(teacher.last().second) +
                        " · consistency loss " +
                        "%.4f".format((teacher.last().first-teacher.last().second).pow(2))) }
                item { TeacherTrace(teacher) }
            }
            StageTwoVisualization.FixMatch -> {
                item { StageTwoInfo("Weak → gate → strong",
                    "Weak prediction " + "%.2f".format(consistency.weakProbability) +
                        " · confidence gate " + "%.2f".format(threshold) +
                        " · " + (if(consistency.accepted) "pseudo-label accepted" else "sample skipped") +
                        "\nStrong prediction " + "%.2f".format(consistency.strongProbability) +
                        " · gated consistency loss " + "%.4f".format(consistency.loss)) }
            }
            StageTwoVisualization.MixMatch -> {
                item { StageTwoInfo("Guess → sharpen → MixUp",
                    "Augmentation predictions " + "%.2f".format(mix[0]) + " and " +
                        "%.2f".format(mix[1]) + " → mean " + "%.2f".format(mix[2]) +
                        " → sharpened " + "%.2f".format(mix[3]) +
                        " → mixed target " + "%.2f".format(mix[4])) }
                item { SemiProbabilityBars(mix,emptySet()) }
            }
            else -> Unit
        }
        item { Spacer(Modifier.height(22.dp)) }
    }
}

@Composable
private fun SemiProbabilityBars(values: List<Double>,labeled: Set<Int>) {
    StageTwoInfo("Class probabilities", "Bar length encodes P(class 1). Outlined samples are currently labeled.")
    Canvas(Modifier.fillMaxWidth().height(130.dp)) {
        val width=size.width/values.size
        values.forEachIndexed { i,value ->
            val h=(value*size.height*.85).toFloat()
            drawRect(if(i in labeled) LabPink else LabCyan.copy(alpha=.55f),
                Offset(i*width,size.height-h),Size(width-2,h))
            if(i in labeled) drawRect(Color.White,Offset(i*width,size.height-h),
                Size(width-2,h),style=Stroke(1f))
        }
    }
}

@Composable
private fun TeacherTrace(values: List<Pair<Double,Double>>) {
    Canvas(Modifier.fillMaxWidth().height(140.dp)) {
        if(values.size<2) return@Canvas
        values.zipWithNext().forEachIndexed { i,(a,b) ->
            val x1=i*size.width/(values.size-1); val x2=(i+1)*size.width/(values.size-1)
            drawLine(LabCyan,Offset(x1,size.height*(1-a.first).toFloat()),
                Offset(x2,size.height*(1-b.first).toFloat()),3f)
            drawLine(LabOrange,Offset(x1,size.height*(1-a.second).toFloat()),
                Offset(x2,size.height*(1-b.second).toFloat()),3f)
        }
    }
}
