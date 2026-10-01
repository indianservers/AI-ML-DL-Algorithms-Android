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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import kotlin.math.abs
import kotlin.math.ln
import kotlin.math.max

internal object StageThreeOptimizationEngine {
    val samples=listOf(-1.2,-.7,-.3,.1,.4,.8,1.1,1.6)
    fun gradient(position:Double,indices:List<Int>):Double = indices.map { i -> 2*(position-samples[i]) }.average()
    fun path(kind:DeepVisualization,rate:Double,batchSize:Int,steps:Int):List<Double> {
        val result=mutableListOf(2.5)
        repeat(steps) { t ->
            val indexes=when(kind) {
                DeepVisualization.GradientDescent -> samples.indices.toList()
                DeepVisualization.Sgd -> listOf((t*5+1)%samples.size)
                else -> (0 until batchSize.coerceIn(1,samples.size)).map { (t*batchSize+it)%samples.size }
            }
            result += result.last()-rate*gradient(result.last(),indexes)
        }
        return result
    }
    fun loss(kind:Int,prediction:Double,target:Double=1.0):Double {
        val error=prediction-target
        return when(kind) {
            0 -> error*error
            1 -> abs(error)
            2 -> if(abs(error)<=.5) .5*error*error else .5*(abs(error)-.25)
            3 -> -(target*ln(prediction.coerceIn(1e-6,1.0))+(1-target)*ln((1-prediction).coerceIn(1e-6,1.0)))
            4 -> -ln(prediction.coerceIn(1e-6,1.0))
            else -> max(0.0,1-target*prediction)
        }
    }
}

@Composable
internal fun StageThreeOptimizationScreen(topic:LearnTopic,kind:DeepVisualization) {
    var control by remember(topic.id){mutableIntStateOf(4)}
    var step by remember(topic.id){mutableIntStateOf(5)}
    var lossKind by remember(topic.id){mutableIntStateOf(0)}
    val isLoss=kind==DeepVisualization.Loss
    val rate=control/20.0
    val batchSize=control.coerceIn(2,7)
    val path=if(!isLoss) StageThreeOptimizationEngine.path(kind,rate,batchSize,12) else emptyList()
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title,if(isLoss)"Move a prediction and watch the objective curve" else "Follow actual gradient updates on a small quadratic loss") }
        if(isLoss) {
            item { GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                    Text("Loss geometry",color=LabText,fontWeight=FontWeight.Bold)
                    val names=listOf("MSE","MAE","Huber","Binary CE","Categorical CE","Hinge")
                    names.chunked(3).forEach { group -> Row(horizontalArrangement=Arrangement.spacedBy(5.dp)) {
                        group.forEach { label -> val i=names.indexOf(label); SegmentedOption(label,i==lossKind,Modifier.weight(1f)){lossKind=i} }
                    } }
                    LossCurve(lossKind,control/10.0)
                    Text("True value 1.0; prediction %.1f; loss %.3f".format(control/10.0,StageThreeOptimizationEngine.loss(lossKind,control/10.0)),color=LabCyan)
                    Slider(control.toFloat(),{control=it.toInt().coerceIn(1,9)},valueRange=1f..9f)
                }
            } }
        } else {
            item { GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                    Text("Loss landscape and parameter path",color=LabText,fontWeight=FontWeight.Bold)
                    OptimizationCurve(path,step)
                    Text(when(kind) {
                        DeepVisualization.GradientDescent -> "Full batch: all 8 samples contribute to each gradient."
                        DeepVisualization.Sgd -> "SGD: exactly one selected sample produces a noisy update."
                        else -> "Mini-batch: a changing subset produces each gradient."
                    },color=LabMuted,fontSize=12.sp)
                    val indexes=when(kind) {
                        DeepVisualization.GradientDescent -> StageThreeOptimizationEngine.samples.indices.toList()
                        DeepVisualization.Sgd -> listOf(((step-1)*5+1)%StageThreeOptimizationEngine.samples.size)
                        else -> (0 until batchSize).map { ((step-1)*batchSize+it)%StageThreeOptimizationEngine.samples.size }
                    }
                    Text("Selected sample indices: ${indexes.joinToString()}",color=LabCyan,fontSize=12.sp)
                    Text("θ = %.3f; gradient = %.3f; next θ = %.3f".format(path[step-1],StageThreeOptimizationEngine.gradient(path[step-1],indexes),path[step]),color=LabOrange,fontSize=12.sp)
                    Text("θₜ₊₁ = θₜ − learning rate × gradient",color=LabMuted,fontSize=12.sp)
                    Text("Learning rate %.2f${if(kind==DeepVisualization.MiniBatch) "; batch size $batchSize" else ""}".format(rate),color=LabText,fontSize=12.sp)
                    Slider(control.toFloat(),{control=it.toInt().coerceIn(2,8)},valueRange=2f..8f)
                    Text("Update step $step",color=LabText,fontSize=12.sp)
                    Slider(step.toFloat(),{step=it.toInt().coerceIn(1,12)},valueRange=1f..12f)
                }
            } }
        }
    }
}

@Composable private fun OptimizationCurve(path:List<Double>,selected:Int) {
    Canvas(Modifier.fillMaxWidth().height(205.dp).background(Color(0xFF071B37))) {
        val mean=StageThreeOptimizationEngine.samples.average()
        fun p(theta:Double)=Offset((((theta+2)/5)*size.width.toDouble()).toFloat(),
            (size.height-20-(theta-mean)*(theta-mean)*17).toFloat())
        val curve=Path()
        for(i in 0..100) { val q=p(-2+i*.05);if(i==0)curve.moveTo(q.x,q.y) else curve.lineTo(q.x,q.y) }
        drawPath(curve,LabCyan,style=Stroke(3f))
        for(i in 1..selected)drawLine(LabOrange,p(path[i-1]),p(path[i]),3f)
        drawCircle(LabOrange,8f,p(path[selected]))
    }
}

@Composable private fun LossCurve(kind:Int,prediction:Double) {
    Canvas(Modifier.fillMaxWidth().height(205.dp).background(Color(0xFF071B37))) {
        val curve=Path()
        for(i in 0..100) {
            val x=i/100.0
            val loss=StageThreeOptimizationEngine.loss(kind,x).coerceAtMost(5.0)
            val y=(size.height-15-loss/5*(size.height-30)).toFloat()
            val px=i*size.width/100
            if(i==0)curve.moveTo(px,y) else curve.lineTo(px,y)
        }
        drawPath(curve,LabCyan,style=Stroke(3f))
        val value=StageThreeOptimizationEngine.loss(kind,prediction).coerceAtMost(5.0)
        drawCircle(LabOrange,8f,Offset((prediction*size.width).toFloat(),(size.height-15-value/5*(size.height-30)).toFloat()))
    }
}
