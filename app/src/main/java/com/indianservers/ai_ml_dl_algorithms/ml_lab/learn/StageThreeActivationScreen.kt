package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
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
import kotlin.math.PI
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.tanh

internal enum class ActivationKind(val label:String) {
    Sigmoid("Sigmoid"), Tanh("Tanh"), Relu("ReLU"), Leaky("Leaky ReLU"),
    Elu("ELU"), Gelu("GELU"), Softplus("Softplus"), Softmax("Softmax")
}

internal object StageThreeActivationEngine {
    fun output(kind:ActivationKind,x:Double):Double=when(kind) {
        ActivationKind.Sigmoid -> 1/(1+exp(-x))
        ActivationKind.Tanh -> tanh(x)
        ActivationKind.Relu -> max(0.0,x)
        ActivationKind.Leaky -> if(x>=0)x else .01*x
        ActivationKind.Elu -> if(x>=0)x else exp(x)-1
        ActivationKind.Gelu -> .5*x*(1+tanh(kotlin.math.sqrt(2/PI)*(x+.044715*x.pow(3))))
        ActivationKind.Softplus -> ln(1+exp(x))
        ActivationKind.Softmax -> softmax(listOf(x,1.0,-.5)).first()
    }
    fun derivative(kind:ActivationKind,x:Double):Double {
        val epsilon=1e-4
        return (output(kind,x+epsilon)-output(kind,x-epsilon))/(2*epsilon)
    }
    fun softmax(logits:List<Double>):List<Double> {
        val maximum=logits.max()
        val values=logits.map { exp(it-maximum) }
        return values.map { it/values.sum() }
    }
}

@Composable
internal fun StageThreeActivationScreen(topic:LearnTopic) {
    var selected by remember(topic.id){mutableIntStateOf(0)}
    var input by remember(topic.id){mutableIntStateOf(50)}
    val kind=ActivationKind.entries[selected]
    val x=(input-50)/10.0
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) {
        item { SectionTitle("Activation Functions","Change x to see output, derivative, and saturation") }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement=Arrangement.spacedBy(8.dp)) {
                Row(Modifier.horizontalScroll(rememberScrollState()),horizontalArrangement=Arrangement.spacedBy(5.dp)) {
                    ActivationKind.entries.forEachIndexed { i,k -> SegmentedOption(k.label,i==selected){selected=i} }
                }
                ActivationCurve(kind,x)
                Text("Cyan: f(x)     Orange: derivative",color=LabMuted,fontSize=12.sp)
                Text("x = %.2f   f(x) = %.3f   f′(x) = %.3f".format(x,StageThreeActivationEngine.output(kind,x),
                    StageThreeActivationEngine.derivative(kind,x)),color=LabCyan,fontSize=12.sp)
                Text("Input x = %.1f".format(x),color=LabText,fontSize=12.sp)
                Slider(input.toFloat(),{input=it.toInt().coerceIn(0,100)},valueRange=0f..100f)
            }
        } }
        item { GlassPanel(Modifier.fillMaxWidth()) {
            Column(verticalArrangement=Arrangement.spacedBy(6.dp)) {
                Text(if(kind==ActivationKind.Softmax)"Vector normalization" else "What the curve means",color=LabText,fontWeight=FontWeight.Bold)
                Text(when(kind) {
                    ActivationKind.Sigmoid -> "Large positive or negative inputs saturate; derivatives approach zero."
                    ActivationKind.Tanh -> "Outputs range from −1 to 1; both tails saturate."
                    ActivationKind.Relu -> "Negative inputs become zero, with zero gradient there."
                    ActivationKind.Leaky -> "Negative inputs retain a small nonzero gradient."
                    ActivationKind.Elu -> "Negative values approach −1 smoothly."
                    ActivationKind.Gelu -> "Smoothly gates inputs with a magnitude-dependent curve."
                    ActivationKind.Softplus -> "Smooth approximation of ReLU; derivative is sigmoid."
                    ActivationKind.Softmax -> "One logit changes its probability relative to the other logits."
                },color=LabMuted,fontSize=12.sp)
                if(kind==ActivationKind.Softmax) {
                    val probabilities=StageThreeActivationEngine.softmax(listOf(x,1.0,-.5))
                    Text("Logits [${"%.1f".format(x)}, 1.0, −0.5]",color=LabCyan,fontSize=12.sp)
                    Text("Probabilities ${probabilities.joinToString { "%.3f".format(it) }}; sum = %.3f".format(probabilities.sum()),
                        color=LabOrange,fontSize=12.sp)
                }
            }
        } }
    }
}

@Composable private fun ActivationCurve(kind:ActivationKind,input:Double) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF071B37))) {
        fun point(x:Double,y:Double)=Offset((((x+5)/10)*size.width).toFloat(),
            (size.height*.55f-(y.coerceIn(-2.0,3.0)*size.height*.14)).toFloat())
        drawLine(Color(0xFF385573),Offset(0f,size.height*.55f),Offset(size.width,size.height*.55f),1f)
        drawLine(Color(0xFF385573),Offset(size.width/2,0f),Offset(size.width/2,size.height),1f)
        val outputPath=Path();val derivativePath=Path()
        for(i in 0..100) {
            val x=-5+i*.1
            val p=point(x,StageThreeActivationEngine.output(kind,x))
            val d=point(x,StageThreeActivationEngine.derivative(kind,x))
            if(i==0){outputPath.moveTo(p.x,p.y);derivativePath.moveTo(d.x,d.y)}
            else {outputPath.lineTo(p.x,p.y);derivativePath.lineTo(d.x,d.y)}
        }
        drawPath(outputPath,LabCyan,style=Stroke(3f))
        drawPath(derivativePath,LabOrange,style=Stroke(2f))
        drawCircle(LabCyan,7f,point(input,StageThreeActivationEngine.output(kind,input)))
    }
}
