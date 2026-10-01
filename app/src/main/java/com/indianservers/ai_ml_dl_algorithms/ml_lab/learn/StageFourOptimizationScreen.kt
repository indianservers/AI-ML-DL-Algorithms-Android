package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption

internal data class OptSpec(val subtitle:String,val flow:List<String>,val equation:String,val difference:String)

internal object StageFourOptimizationSpecs {
    fun forTitle(name:String):OptSpec=when(name){
        "Batch Gradient Descent"->OptSpec("Use all samples for one update",listOf("Full dataset","Average gradient","One parameter update"),"θ ← θ − η∇Lfull","Every sample contributes to each gradient.")
        "Stochastic Gradient Descent"->OptSpec("Use one sample per update",listOf("One example","Noisy gradient","Immediate update"),"θ ← θ − η∇Lᵢ","The path zig-zags because one sample differs from the full mean.")
        "Mini-Batch Gradient Descent"->OptSpec("Average a small subset",listOf("Sample batch","Mean batch gradient","Update"),"θ ← θ − η meanᵢ∈B ∇Lᵢ","A batch smooths the single-sample noise without using all data.")
        "Momentum"->OptSpec("Accumulate velocity through gradients",listOf("Current gradient","Velocity βv+g","Move"),"vₜ=βvₜ₋₁+gₜ; θ←θ−ηvₜ","Velocity persists across steps and damps zig-zag.")
        "Nesterov Momentum"->OptSpec("Measure gradient at the lookahead point",listOf("Look ahead θ−ηβv","Gradient there","Correct velocity"),"vₜ=βvₜ₋₁+∇L(θ−ηβvₜ₋₁)","The gradient is evaluated ahead of the current position.")
        "AdaGrad"->OptSpec("Accumulate squared gradients per coordinate",listOf("g² accumulated","Per-axis denominator","Adaptive step"),"Gₜ=Σg²; θ←θ−ηg/√Gₜ","Frequently updated coordinates receive smaller later steps.")
        "RMSProp"->OptSpec("Decay old squared-gradient history",listOf("EMA of g²","Per-axis denominator","Adaptive step"),"vₜ=βvₜ₋₁+(1−β)g²","The exponential average avoids AdaGrad's permanently growing denominator.")
        "Adam"->OptSpec("Combine first and second moments",listOf("m=EMA(g)","v=EMA(g²)","Bias correction","Update"),"θ←θ−ηm̂/(√v̂+ε)","Both momentum and adaptive scale affect the step.")
        "AdamW"->OptSpec("Apply weight decay separately",listOf("Adam gradient step","Decoupled −ηλθ","New weights"),"θ←θ−ηm̂/(√v̂+ε)−ηλθ","Decay is not folded into the gradient moments.")
        "Nadam"->OptSpec("Add Nesterov-style moment correction",listOf("Adam moments","Lookahead m̂","Adaptive update"),"θ←θ−η[βm̂+(1−β)ĝ]/(√v̂+ε)","The first-moment contribution looks ahead.")
        "Learning Rate Scheduling"->OptSpec("Watch learning rate change with epoch",listOf("Choose schedule","Compute ηₜ","Update path"),"θₜ₊₁=θₜ−ηₜ∇L","Step, exponential, cosine, warmup, and one-cycle have different timing.")
        "Coordinate Descent"->OptSpec("Optimize one coordinate at a time",listOf("Update x","Hold y","Update y","Repeat"),"x←argminₓL(x,y); y←argminᵧL(x,y)","Only one axis moves on each step.")
        "Newton's Method"->OptSpec("Use true local curvature",listOf("Gradient","Hessian","Solve HΔ=g","Curvature step"),"θ←θ−H⁻¹g","The Hessian can make one large correction on this quadratic bowl.")
        "Quasi-Newton / BFGS"->OptSpec("Learn inverse curvature from secant history",listOf("Gradient step","s=Δθ; y=Δg","BFGS inverse update"),"Hₖ₊₁≈ inverse Hessian from (sₖ,yₖ)","Curvature is estimated from observed position and gradient changes.")
        "L-BFGS"->OptSpec("Keep only recent curvature pairs",listOf("Recent (s,y) queue","Two-loop recursion","Direction"),"dₖ=−Hₖgₖ via limited-memory pairs","Only three recent secant pairs are retained in this educational run.")
        else->error("Unregistered optimizer: $name")
    }
}

@Composable
internal fun StageFourOptimizationScreen(topic:LearnTopic){
    val name=topic.title;val spec=StageFourOptimizationSpecs.forTitle(name)
    var step by remember(topic.id){mutableIntStateOf(4)}
    var control by remember(topic.id){mutableIntStateOf(4)}
    var schedule by remember(topic.id){mutableIntStateOf(0)}
    val rate=.025+control*.02
    val path=StageFourOptimizationEngine.path(name,rate,14,schedule)
    val current=path[step.coerceIn(path.indices)]
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)){
        item{SectionTitle(name,spec.subtitle)}
        item{FourPanel("Optimization landscape"){
            FourLandscape(path.map{it.x to it.y},step)
            FourMetric("Objective L(x,y)","%.4f".format(current.loss),LabGreen)
            FourMetric("Parameters x, y","%.3f, %.3f".format(current.x,current.y))
        }}
        item{FourPanel("Algorithm-specific update"){
            FourFlow(spec.flow,step%spec.flow.size)
            FourText(spec.equation,LabCyan)
            FourText(spec.difference)
            FourMetric("Gradient gx, gy","%.3f, %.3f".format(current.gx,current.gy))
            when(name){
                "Stochastic Gradient Descent"->FourMetric("Selected sample", "${((step*5+1)%8)+1} of 8",LabOrange)
                "Mini-Batch Gradient Descent"->FourMetric("Selected batch", "${(step%4)*2+1}–${(step%4)*2+2} of 8",LabOrange)
                "Momentum","Nesterov Momentum"->FourMetric("Velocity / lookahead", "%.3f / %.3f".format(current.firstMoment,current.x-.8*rate*current.firstMoment))
                "AdaGrad","RMSProp","Adam","AdamW","Nadam"->{
                    FourMetric("First moment", "%.3f".format(current.firstMoment))
                    FourMetric("Squared-gradient accumulator", "%.3f".format(current.secondMoment))
                    if(name=="AdamW")FourMetric("Separate decay ηλx", "%.4f".format(rate*.03*current.x),LabOrange)
                }
                "Coordinate Descent"->FourMetric("Active coordinate",if(step%2==0)"x; y held fixed" else "y; x held fixed")
                "Newton's Method"->{
                    val h=StageFourOptimizationEngine.hessian()
                    FourMetric("True Hessian","[${h[0][0]}, ${h[0][1]}; ${h[1][0]}, ${h[1][1]}]")
                }
                "Quasi-Newton / BFGS"->FourText("An inverse-Hessian approximation is revised after each observed gradient change.")
                "L-BFGS"->FourMetric("Curvature pairs retained","${current.memory} of 3",LabOrange)
                else->Unit
            }
        }}
        item{FourPanel("Loss and learning-rate trace"){
            FourLine(path.map{it.loss},if(name=="Learning Rate Scheduling")path.map{it.rate*20}else emptyList())
            if(name=="Learning Rate Scheduling"){
                FourMetric("Current rate","%.4f".format(current.rate),LabOrange)
                Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(3.dp)){
                    listOf("Step","Exp","Cos","Warm","Cycle").forEachIndexed{i,label->
                        SegmentedOption(label,schedule==i,Modifier.weight(1f)){schedule=i;step=0}
                    }
                }
            }
        }}
        item{FourPanel("Explore"){
            FourSteps(step,14){step=it}
            FourText("Learning rate %.3f".format(rate))
            Slider(control.toFloat(),{control=it.toInt().coerceIn(0,7);step=0},valueRange=0f..7f)
        }}
    }
}
