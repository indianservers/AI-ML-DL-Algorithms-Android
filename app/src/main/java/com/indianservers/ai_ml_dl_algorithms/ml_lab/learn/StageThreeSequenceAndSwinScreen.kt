package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
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
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenEngines

@Composable
internal fun StageThreeSequenceScreen(topic: LearnTopic, kind: DeepVisualization) {
    var step by remember(topic.id) { mutableIntStateOf(2) }
    val words=listOf("the","small","cat","sat","down")
    val values=words.map { it.length/6.0 }
    val forward=PhaseSevenEngines.rnnForward(values).steps
    val backward=PhaseSevenEngines.rnnForward(values.reversed()).steps.reversed()
    val at=step.coerceIn(words.indices)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, "Watch source information become context and output") }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                Text("Source tokens",color=LabText,fontWeight=FontWeight.Bold)
                Row(horizontalArrangement=Arrangement.spacedBy(5.dp)) { words.forEachIndexed { i,w ->
                    SegmentedOption(w,i==at,Modifier.weight(1f)){step=i}
                } }
                if(kind==DeepVisualization.Bidirectional) {
                    Text("Forward pass →",color=LabCyan)
                    Text(forward.joinToString("   ") { "%.2f".format(it.hidden) },color=LabMuted,fontSize=12.sp)
                    Text("← Backward pass",color=LabOrange)
                    Text(backward.joinToString("   ") { "%.2f".format(it.hidden) },color=LabMuted,fontSize=12.sp)
                    Text("${words[at]}: concatenate [${"%.3f".format(forward[at].hidden)}, ${"%.3f".format(backward[at].hidden)}]",color=LabCyan,fontSize=12.sp)
                } else {
                    Text("Encoder context = %.3f".format(forward.last().hidden),color=LabCyan)
                    if(kind==DeepVisualization.Seq2Seq) {
                        val target=listOf("le","petit","chat","assis")
                        val decoder = PhaseSevenEngines.teacherForcedDecoder(
                            forward.last().hidden,
                            listOf(0.0) + target.dropLast(1).map { it.length / 6.0 }
                        )
                        val targetStep = at.coerceAtMost(target.lastIndex)
                        Text("Example target (teacher forced): ${target.take(targetStep+1).joinToString(" ")}",color=LabOrange)
                        Text("Decoder state at step ${targetStep+1} = %.3f".format(decoder[targetStep]),color=LabCyan,fontSize=12.sp)
                        Text("The decoder state uses encoder context and the prior target token. The French example is supplied for teaching; no translation model is trained here.",color=LabMuted,fontSize=12.sp)
                    } else {
                        Text("Bottleneck representation → decoder reconstruction",color=LabOrange)
                        Text("Encoder-decoder is an architectural pattern; a decoder can reconstruct or transform the input, not only translate text.",
                            color=LabMuted,fontSize=12.sp)
                    }
                }
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column {
                Text("Time step ${at+1}",color=LabText)
                Slider(at.toFloat(),{step=it.toInt().coerceIn(words.indices)},valueRange=0f..words.lastIndex.toFloat())
            }
        } }
    }
}

@Composable
internal fun StageThreeSwinScreen(topic: LearnTopic) {
    var shifted by remember(topic.id) { mutableIntStateOf(0) }
    var selected by remember(topic.id) { mutableIntStateOf(10) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Swin Transformer", "Local window attention followed by shifted windows") }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                Text(if(shifted==0) "Stage 1: regular 4×4 windows" else "Stage 2: windows shifted by 2 patches",color=LabText,fontWeight=FontWeight.Bold)
                SwinGrid(shifted==1,selected)
                Text("Patch $selected attends to ${swinWindow(selected,shifted==1).size} patches in its current window.",color=LabCyan,fontSize=12.sp)
                Text("Shifting lets patches exchange information across previous window boundaries.",color=LabMuted,fontSize=12.sp)
                SegmentedOption(if(shifted==0) "Shift windows →" else "← Regular windows",false,Modifier.fillMaxWidth()){shifted=1-shifted}
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column {
                Text("Selected patch: $selected",color=LabText)
                Slider(selected.toFloat(),{selected=it.toInt().coerceIn(0,63)},valueRange=0f..63f)
                Text("Visible patch indices: ${swinWindow(selected,shifted==1).joinToString()}",color=LabMuted,fontSize=11.sp)
            }
        } }
    }
}

internal fun swinWindow(index:Int,shifted:Boolean):List<Int> {
    val row=index/8; val col=index%8; val offset=if(shifted)2 else 0
    val groupRow=(row+offset)/4; val groupCol=(col+offset)/4
    return (0..63).filter { i -> (i/8+offset)/4==groupRow && (i%8+offset)/4==groupCol }
}

@Composable private fun SwinGrid(shifted:Boolean,selected:Int) {
    val visible=swinWindow(selected,shifted).toSet()
    Canvas(Modifier.fillMaxWidth().height(250.dp).background(Color(0xFF071B37))) {
        val width=size.width/8; val height=size.height/8
        for(i in 0..63) {
            val r=i/8; val c=i%8
            drawRect(if(i==selected) LabOrange else if(i in visible) LabCyan.copy(alpha=.55f) else LabPurple.copy(alpha=.18f),
                Offset(c*width+1,r*height+1),Size(width-2,height-2))
        }
        val offset=if(shifted)2 else 0
        for(i in 0..2) {
            val x=(i*4-offset)*width; val y=(i*4-offset)*height
            if(x>=0&&x<=size.width)drawLine(LabOrange,Offset(x,0f),Offset(x,size.height),2f)
            if(y>=0&&y<=size.height)drawLine(LabOrange,Offset(0f,y),Offset(size.width,y),2f)
        }
    }
}
