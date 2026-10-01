package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import kotlin.math.exp
import kotlin.math.sqrt

internal object StageThreeGraphEngine {
    val points = listOf(.15 to .5, .38 to .15, .42 to .48, .36 to .83, .7 to .27, .78 to .66, .9 to .44)
    val edges = listOf(0 to 1, 0 to 2, 0 to 3, 1 to 2, 1 to 4, 2 to 3, 2 to 4, 2 to 5, 3 to 5, 4 to 5, 4 to 6, 5 to 6)
    val features = listOf(.2, .8, .4, .6, .9, .3, .7)
    fun neighbors(node: Int): List<Int> = edges.mapNotNull { (a,b) -> when (node) { a -> b; b -> a; else -> null } }
    fun meanAggregate(node: Int): Double = neighbors(node).map { features[it] }.average()
    fun gcn(node: Int): Double {
        val withSelf = neighbors(node) + node
        return withSelf.sumOf { other ->
            features[other] / sqrt((neighbors(node).size+1.0)*(neighbors(other).size+1.0))
        }
    }
    fun attention(node: Int): List<Pair<Int, Double>> {
        val adjacent = neighbors(node)
        val scores = adjacent.map { exp(features[node] * features[it] + features[it]) }
        val total = scores.sum()
        return adjacent.mapIndexed { i, neighbor -> neighbor to scores[i]/total }
    }
    fun sampled(node: Int, count: Int): List<Int> = neighbors(node).sortedBy { (it*13+node*7)%11 }.take(count)
    fun embedding(node: Int): List<Double> = listOf(features[node], meanAggregate(node), gcn(node))
    fun linkProbability(a: Int, b: Int): Double {
        val x=embedding(a); val y=embedding(b)
        val dot=x.zip(y).sumOf { it.first*it.second }
        return 1.0/(1.0+exp(-dot))
    }
}

@Composable
internal fun StageThreeGraphScreen(topic: LearnTopic, kind: DeepVisualization) {
    var node by remember(topic.id) { mutableIntStateOf(2) }
    var sampleSize by remember(topic.id) { mutableIntStateOf(2) }
    var target by remember(topic.id) { mutableIntStateOf(6) }
    val neighbors = StageThreeGraphEngine.neighbors(node)
    val selectedNeighbors = if (kind == DeepVisualization.GraphSage) StageThreeGraphEngine.sampled(node,sampleSize) else neighbors
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, "Select a node and inspect its incoming messages") }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Node $node receives from ${selectedNeighbors.joinToString()}", color = LabText, fontWeight = FontWeight.Bold)
                GraphCanvas(node, selectedNeighbors, kind) { node=it }
                Text("Node features are the colors and numeric values below; edges carry information between neighbors.", color = LabMuted, fontSize = 11.sp)
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                Text(when (kind) {
                    DeepVisualization.Gnn -> "Message → mean aggregate → update"
                    DeepVisualization.Gcn -> "Degree-normalized graph convolution"
                    DeepVisualization.Gat -> "Feature-dependent attention weights on incoming edges"
                    DeepVisualization.GraphSage -> "Sample neighbors → aggregate → combine self"
                    else -> "GNN encoder → embeddings → link decoder"
                }, color = LabText, fontWeight = FontWeight.Bold)
                when (kind) {
                    DeepVisualization.Gnn -> {
                        neighbors.forEach { GraphValue("Message from node $it", StageThreeGraphEngine.features[it]) }
                        GraphValue("Mean aggregate", StageThreeGraphEngine.meanAggregate(node))
                        GraphValue("Updated state (self + aggregate)/2", (StageThreeGraphEngine.features[node]+StageThreeGraphEngine.meanAggregate(node))/2)
                    }
                    DeepVisualization.Gcn -> {
                        neighbors.forEach { it ->
                            val weight = 1/sqrt((neighbors.size+1.0)*(StageThreeGraphEngine.neighbors(it).size+1.0))
                            GraphValue("Node $it normalized weight", weight)
                        }
                        GraphValue("Normalized aggregate (incl. self)", StageThreeGraphEngine.gcn(node))
                        Text("D⁻½(A+I)D⁻½XW", color = LabCyan, fontSize = 12.sp)
                    }
                    DeepVisualization.Gat -> {
                        StageThreeGraphEngine.attention(node).forEach { (other, weight) -> GraphValue("α from node $other",weight) }
                        GraphValue("Attention-weighted update", StageThreeGraphEngine.attention(node).sumOf { (other,w)->w*StageThreeGraphEngine.features[other] })
                    }
                    DeepVisualization.GraphSage -> {
                        selectedNeighbors.forEach { GraphValue("Sampled node $it",StageThreeGraphEngine.features[it]) }
                        GraphValue("Sample mean",selectedNeighbors.map { StageThreeGraphEngine.features[it] }.average())
                        Text("Fixed-size sampling avoids reading every neighbor in a large graph.", color = LabMuted, fontSize = 12.sp)
                        Slider(sampleSize.toFloat(),{ sampleSize=it.toInt().coerceIn(1,neighbors.size) },valueRange=1f..neighbors.size.toFloat())
                    }
                    else -> {
                        StageThreeGraphEngine.embedding(node).forEachIndexed { i,v -> GraphValue("Embedding dimension ${i+1}",v) }
                        GraphValue("Link probability to node $target",StageThreeGraphEngine.linkProbability(node,target))
                        Text("Decoder applies sigmoid(dot(z_node, z_target)).",color=LabCyan,fontSize=12.sp)
                        Slider(target.toFloat(),{target=it.toInt().coerceIn(0,6)},valueRange=0f..6f)
                    }
                }
            }
        } }
    }
}

@Composable private fun GraphValue(label:String,value:Double) {
    Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween) {
        Text(label,color=LabMuted,fontSize=12.sp)
        Text("%.3f".format(value),color=LabCyan,fontSize=12.sp)
    }
}

@Composable private fun GraphCanvas(selected:Int,neighbors:List<Int>,kind:DeepVisualization,onSelect:(Int)->Unit) {
    Canvas(Modifier.fillMaxWidth().height(240.dp).background(Color(0xFF071B37)).pointerInput(selected) {
        detectTapGestures { p ->
            val best=StageThreeGraphEngine.points.indices.minBy { i ->
                val v=StageThreeGraphEngine.points[i]
                val dx=p.x/size.width-v.first; val dy=p.y/size.height-v.second
                dx*dx+dy*dy
            }; onSelect(best)
        }
    }) {
        val positions=StageThreeGraphEngine.points.map { Offset((it.first*size.width).toFloat(),(it.second*size.height).toFloat()) }
        StageThreeGraphEngine.edges.forEach { (a,b) ->
            val active=(a==selected&&b in neighbors)||(b==selected&&a in neighbors)
            val width=if (active&&kind==DeepVisualization.Gat) {
                val other=if(a==selected)b else a
                (StageThreeGraphEngine.attention(selected).firstOrNull{it.first==other}?.second?:0.0).toFloat()*15+2
            } else if (active) 4f else 1.5f
            drawLine(if(active) LabOrange else Color(0xFF345277),positions[a],positions[b],width)
        }
        positions.forEachIndexed { i,p -> drawCircle(if(i==selected) LabCyan else if(i in neighbors) LabPurple else Color(0xFF43638A),
            if(i==selected) 16f else 11f,p) }
    }
}
