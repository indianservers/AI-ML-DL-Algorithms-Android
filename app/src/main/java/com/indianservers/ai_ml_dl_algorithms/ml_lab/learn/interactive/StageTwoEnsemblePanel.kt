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
internal fun StageTwoEnsemblePanel(kind: StageTwoVisualization) {
    var preset by remember(kind) { mutableStateOf(ClusterPreset.TwoMoons) }
    var seed by remember(kind) { mutableIntStateOf(105) }
    val points=remember(kind,preset,seed) {
        mutableStateListOf<ClusterPoint>().apply {
            addAll(PhaseThreeDatasets.clusters(preset,42,2,.08,seed))
        }
    }
    var query by remember(kind) { mutableStateOf(LabPoint(.1,.05)) }
    var learnerCount by remember(kind) { mutableIntStateOf(5) }
    var step by remember(kind) { mutableIntStateOf(0) }
    var fraction by remember(kind) { mutableDoubleStateOf(.25) }
    val snapshot=points.toList()
    val training=snapshot.map { LabPoint(it.x,it.y,label=it.hiddenLabel.coerceIn(0,1)) }
    val bag=remember(training,learnerCount,seed,query) {
        if(kind==StageTwoVisualization.Bagging)
            StageTwoEnsembleEngine.bootstrap(training,learnerCount,seed,query) else emptyList()
    }
    val combined=remember(training,kind,query,fraction) {
        when(kind) {
            StageTwoVisualization.Stacking -> StageTwoEnsembleEngine.stacking(training,query,
                (fraction*10).roundToInt().coerceIn(2,5))
            StageTwoVisualization.Blending ->
                StageTwoEnsembleEngine.blending(training,query,fraction)
            else -> StageTwoEnsembleEngine.votes(training,query)
        }
    }
    val current=step.coerceIn(0,if(kind==StageTwoVisualization.Bagging) learnerCount else 3)
    LazyColumn(Modifier.fillMaxSize().padding(horizontal=14.dp),
        verticalArrangement=Arrangement.spacedBy(10.dp)) {
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
                            preset=option; step=0
                        }
                    }
            }
        }
        item {
            StageTwoPlot(snapshot,labels=snapshot.map { it.hiddenLabel },
                onMove={ i,moved -> points[i]=moved },
                onTap={ at -> query=query.copy(x=at.x,y=at.y) }) { scale ->
                val at=scale.point(query.x,query.y)
                drawCircle(Color.White,13f,at,style=Stroke(2.5f))
                drawCircle(LabOrange,6f,at)
                if(kind==StageTwoVisualization.Bagging) {
                    val selected=bag.getOrNull((current-1).coerceAtLeast(0))
                    selected?.indices?.groupingBy { it }?.eachCount()?.forEach { (i,count) ->
                        if(count>1) drawCircle(LabOrange,(8+count*2).toFloat(),
                            scale.point(snapshot[i].x,snapshot[i].y),style=Stroke(2f))
                    }
                }
            }
        }
        item { StageTwoStepControls(current,
            if(kind==StageTwoVisualization.Bagging) learnerCount else 3) { step=it } }
        if(kind==StageTwoVisualization.Bagging) {
            item {
                NativeSlider("Independent learners",learnerCount.toDouble(),2.0,15.0) {
                    learnerCount=it.roundToInt(); step=0
                }
            }
            item {
                val selected=bag.getOrNull((current-1).coerceAtLeast(0))
                val duplicates=selected?.indices?.groupingBy { it }?.eachCount()
                    ?.count { it.value>1 } ?: 0
                StageTwoInfo("Bootstrap sample → tree " + current,
                    if(selected==null) "Each tree independently samples the original data with replacement."
                    else "Sampled " + selected.indices.size + " rows; " + duplicates +
                        " original rows were drawn more than once. Tree vote: class " +
                        selected.probability.roundToInt())
            }
            item {
                val used=bag.take(current)
                StageTwoInfo("Parallel aggregation",
                    used.mapIndexed { i,learner -> "Tree " + (i+1) + " → class " +
                        learner.probability.roundToInt() }.joinToString("\n") +
                    "\nMajority vote → class " +
                    if(used.isEmpty()) "—" else
                        (if(used.sumOf { it.probability }*2>=used.size) "1" else "0"))
            }
        } else {
            item {
                StageTwoInfo("Three independent base models",
                    listOf("Logistic regression","Distance-weighted KNN","Decision tree")
                        .mapIndexed { i,name ->
                            name + " → P(class 1) " + "%.2f".format(combined.probabilities[i]) +
                                " → class " + combined.hardVotes[i]
                        }.joinToString("\n"))
            }
            item {
                when(kind) {
                    StageTwoVisualization.SoftVoting ->
                        StageTwoInfo("Average class probabilities",
                            "P(class 1) = (" + combined.probabilities.joinToString(" + ") {
                                "%.2f".format(it)
                            } + ") / 3 = " + "%.3f".format(combined.softVote) +
                                "\nFinal class " + if(combined.softVote>=.5) "1" else "0")
                    StageTwoVisualization.HardVoting ->
                        StageTwoInfo("Discrete majority",
                            "Votes: " + combined.hardVotes.joinToString(" + ") +
                                " · class 1 votes " + combined.hardVotes.sum() +
                                " of 3 · final class " + combined.hardVote)
                    StageTwoVisualization.Voting ->
                        StageTwoInfo("Vote comparison",
                            "Hard majority → class " + combined.hardVote +
                                "\nSoft probability average → " +
                                "%.3f".format(combined.softVote) +
                                ". Change the query to compare them.")
                    StageTwoVisualization.Stacking ->
                        StageTwoInfo("Out-of-fold meta-features",
                            "Each training row's base predictions come from models that did not train on that row. " +
                                combined.metaRows.take(5).mapIndexed { i,row ->
                                    "Row " + (i+1) + " [" + row.joinToString { "%.2f".format(it) } +
                                        "] → " + combined.metaLabels[i]
                                }.joinToString("\n"))
                    StageTwoVisualization.Blending ->
                        StageTwoInfo("Holdout blend matrix",
                            "Base learners train on " + (training.size-combined.metaRows.size) +
                                " rows. A separate holdout of " + combined.metaRows.size +
                                " rows trains the meta-model.\n" +
                                combined.metaRows.take(5).mapIndexed { i,row ->
                                    "[" + row.joinToString { "%.2f".format(it) } +
                                        "] → " + combined.metaLabels[i]
                                }.joinToString("\n"))
                    else -> Unit
                }
            }
            item {
                if(kind==StageTwoVisualization.Stacking || kind==StageTwoVisualization.Blending)
                    StageTwoInfo("Learned meta-model",
                        "Weights " + combined.metaWeights.joinToString { "%.2f".format(it) } +
                            "\nMeta-model P(class 1) " +
                            "%.3f".format(combined.metaPrediction) +
                            " · final class " + if(combined.metaPrediction>=.5) "1" else "0")
                else EnsembleProbabilityBars(combined.probabilities,
                    kind!=StageTwoVisualization.HardVoting)
            }
            if(kind==StageTwoVisualization.Stacking || kind==StageTwoVisualization.Blending)
                item { NativeSlider(if(kind==StageTwoVisualization.Stacking) "Folds" else "Holdout fraction",
                    fraction,.2,.45) { fraction=it } }
        }
        item { SegmentedOption("New dataset",false,Modifier.fillMaxWidth()) { seed++; step=0 } }
        item { Spacer(Modifier.height(22.dp)) }
    }
}

@Composable
private fun EnsembleProbabilityBars(probabilities: List<Double>,soft: Boolean) {
    StageTwoInfo(if(soft) "Base probabilities" else "Class votes",
        if(soft) "Average these calibrated outputs to get soft voting."
        else "Threshold each output to a class before majority voting.")
    Canvas(Modifier.fillMaxWidth().height(130.dp)) {
        val width=size.width/probabilities.size
        probabilities.forEachIndexed { i,p ->
            val value=if(soft) p else if(p>=.5) 1.0 else 0.0
            val h=(value*size.height*.9).toFloat()
            drawRect(stageColor(i),Offset(i*width+width*.15f,size.height-h),
                Size(width*.7f,h.coerceAtLeast(2f)))
        }
    }
}
