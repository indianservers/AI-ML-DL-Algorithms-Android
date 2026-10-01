package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Slider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

internal object StageFourExplanationEngine {
    val data=listOf(listOf(.2,.7,.4),listOf(.8,.2,.6),listOf(.5,.4,.2),listOf(.9,.7,.8),listOf(.3,.3,.7),listOf(.7,.5,.4))
    val coefficients=listOf(1.2,-1.0,.3)
    fun score(x:List<Double>):Double=1/(1+exp(-(coefficients.zip(x).sumOf{it.first*it.second}-.2)))
    fun nonlinear(x:List<Double>):Double=1/(1+exp(-(1.4*x[0]-1.1*x[1]+.5*x[0]*x[2]-.2)))
    fun baseline():Double=data.map(::score).average()
    fun exactLinearContributions(x:List<Double>):List<Double> {
        val means=(0..2).map{j->data.map{it[j]}.average()}
        return x.indices.map{i->coefficients[i]*(x[i]-means[i])}
    }
    fun permutationDrop(feature:Int):Double {
        val truth=data.map(::score)
        val permuted=data.indices.map{i->data[i].toMutableList().apply{this[feature]=data[(i+2)%data.size][feature]}.toList()}
        return data.indices.map{i->(truth[i]-score(permuted[i])).let{it*it}}.average()
    }
    fun pdp(feature:Int):List<Double> = (0..20).map{k->
        data.map{x->score(x.toMutableList().apply{this[feature]=k/20.0})}.average()
    }
    fun limeSamples(x:List<Double>):List<List<Double>> = (0..24).map{i->listOf(
            (x[0]+((i*7)%11-5)*.055).coerceIn(0.0,1.0),
            (x[1]+((i*13)%11-5)*.055).coerceIn(0.0,1.0),x[2])}
    fun lime(x:List<Double>):List<Double> {
        val samples=limeSamples(x)
        val weighted=Array(3){DoubleArray(4)}
        samples.forEach{sample->
            val distance=sqrt((sample[0]-x[0])*(sample[0]-x[0])+(sample[1]-x[1])*(sample[1]-x[1]))
            val weight=exp(-distance*distance/.09)
            val row=listOf(1.0,sample[0]-x[0],sample[1]-x[1])
            val output=nonlinear(sample)
            repeat(3){i->repeat(3){j->weighted[i][j]+=weight*row[i]*row[j]};weighted[i][3]+=weight*row[i]*output}
        }
        repeat(3){i->weighted[i][i]+=1e-5}
        repeat(3){pivot->
            val scale=weighted[pivot][pivot].coerceAtLeast(1e-9)
            for(j in pivot..3)weighted[pivot][j]/=scale
            repeat(3){row->if(row!=pivot){val factor=weighted[row][pivot];for(j in pivot..3)weighted[row][j]-=factor*weighted[pivot][j]}}
        }
        return (0..2).map{weighted[it][3]}
    }
    val pixels=(0..4).map{r->(0..4).map{c->if(abs(r-2)+abs(c-2)<3).8 else .12}}
    val imageWeights=(0..4).map{r->(0..4).map{c->if(r==2||c==2).9 else .08}}
    fun imageScore(image:List<List<Double>>):Double=image.indices.sumOf{r->
        image[r].indices.sumOf{c->image[r][c]*imageWeights[r][c]}}
    val saliency=imageWeights.map{row->row.map{abs(it)}}
    fun gradCam():List<List<Double>> {
        val edge=(0..4).map{r->(0..4).map{c->abs(pixels[r][(c+1).coerceAtMost(4)]-pixels[r][(c-1).coerceAtLeast(0)])}}
        val center=(0..4).map{r->(0..4).map{c->pixels[r][c]}}
        return (0..4).map{r->(0..4).map{c->max(0.0,.7*edge[r][c]+.3*center[r][c])}}
    }
    fun attention(head:Int):List<List<Double>> {
        val vectors=listOf(listOf(1.0,.1),listOf(.6,.8),listOf(.2,.9),listOf(.8,.2))
        return vectors.indices.map{i->
            val logits=vectors.indices.map{j->(vectors[i][0]*vectors[j][0]+vectors[i][1]*vectors[j][1])*(if(head==0)1.0 else (-1.0+head*.7))}
            val exps=logits.map{exp(it-logits.max())};exps.map{it/exps.sum()}
        }
    }
    fun counterfactual(x:List<Double>):Pair<List<Double>,Double> {
        val original=score(x)>=.5
        val candidates=(0..20).flatMap{i->(0..20).map{j->listOf(i/20.0,j/20.0,x[2])}}
        val found=candidates.filter{(score(it)>=.5)!=original}.minBy{abs(it[0]-x[0])+abs(it[1]-x[1])}
        return found to (abs(found[0]-x[0])+abs(found[1]-x[1]))
    }
}

@Composable
private fun LimeNeighborhood(x:List<Double>,surrogate:List<Double>){
    val samples=StageFourExplanationEngine.limeSamples(x)
    Canvas(Modifier.fillMaxWidth().height(185.dp).background(Color(0xFF071C36),RoundedCornerShape(8.dp)).padding(10.dp)){
        fun point(a:Double,b:Double)=Offset((a*size.width).toFloat(),((1-b)*size.height).toFloat())
        samples.forEach{sample->
            drawCircle(if(StageFourExplanationEngine.nonlinear(sample)>=.5)LabGreen else LabOrange,5f,
                point(sample[0],sample[1]))
        }
        if(abs(surrogate[2])>1e-5){
            fun boundary(a:Double)=x[1]+(.5-surrogate[0]-surrogate[1]*(a-x[0]))/surrogate[2]
            drawLine(LabCyan,point(0.0,boundary(0.0)),point(1.0,boundary(1.0)),3f)
        }
        drawCircle(LabText,9f,point(x[0],x[1]),style=Stroke(3f))
    }
}

@Composable
internal fun StageFourExplanationScreen(topic:LearnTopic){
    val name=topic.title
    var sample by remember(topic.id){mutableIntStateOf(0)}
    var feature by remember(topic.id){mutableIntStateOf(0)}
    var control by remember(topic.id){mutableIntStateOf(5)}
    val x=StageFourExplanationEngine.data[sample]
    val labels=listOf("Income","Debt","Age")
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)){
        item{SectionTitle(name,when(name){
            "Feature Importance"->"Compare global coefficient magnitudes of the fixed model"
            "Permutation Importance"->"Shuffle one feature and measure the loss increase"
            "Partial Dependence Plot"->"Average predictions as one feature value changes"
            "SHAP"->"Add signed feature effects to a baseline score"
            "LIME"->"Fit a local surrogate around one prediction"
            "Saliency Maps"->"Inspect input-pixel gradients"
            "Grad-CAM"->"Weight CNN feature maps by class gradients"
            "Attention Visualization"->"Inspect token-to-token attention weights"
            "Counterfactual Explanations"->"Find a small feature change that flips the decision"
            else->"Inspect how model behavior changes with inputs"
        })}
        item{FourPanel("Explanation visual"){
            when(name){
                "Feature Importance"->{
                    FourBars(labels.mapIndexed{i,label->label to abs(StageFourExplanationEngine.coefficients[i])},feature)
                    FourText("Global coefficient magnitude from a fixed logistic model; coefficient size depends on feature scaling.")
                }
                "Permutation Importance"->{
                    FourFlow(listOf("Baseline predictions","Shuffle ${labels[feature]}","Re-score","MSE increase"),control%4)
                    FourBars(labels.mapIndexed{i,label->label to StageFourExplanationEngine.permutationDrop(i)},feature,LabOrange)
                    FourMetric("Selected performance drop","%.4f".format(StageFourExplanationEngine.permutationDrop(feature)),LabGreen)
                }
                "Partial Dependence Plot"->{
                    FourLine(StageFourExplanationEngine.pdp(feature))
                    FourText("Each feature value is substituted for every row; predictions are then averaged. Other feature rows stay as observed.")
                }
                "SHAP"->{
                    val contributions=StageFourExplanationEngine.exactLinearContributions(x)
                    val base=StageFourExplanationEngine.data.map{row->StageFourExplanationEngine.coefficients.zip(row).sumOf{it.first*it.second}-.2}.average()
                    val cumulative=contributions.runningFold(base){a,b->a+b}
                    FourWaterfall(base,contributions,labels)
                    FourBars(labels.mapIndexed{i,label->label to contributions[i]},feature)
                    FourMetric("Baseline logit","%.3f".format(base))
                    FourMetric("Final logit","%.3f".format(cumulative.last()),LabGreen)
                    FourText("For this additive linear score, each contribution is exact relative to the feature-mean baseline. Positive and negative effects sum to the final logit.")
                }
                "LIME"->{
                    val surrogate=StageFourExplanationEngine.lime(x)
                    FourFlow(listOf("Selected sample","25 nearby perturbations","Black-box scores","Distance weights","Local linear fit"),control%5)
                    LimeNeighborhood(x,surrogate)
                    FourBars(listOf("local intercept" to surrogate[0],"income slope" to surrogate[1],"debt slope" to surrogate[2]),feature+1)
                    FourMetric("Black-box probability","%.3f".format(StageFourExplanationEngine.nonlinear(x)))
                    FourMetric("Surrogate at sample","%.3f".format(surrogate[0]),LabGreen)
                }
                "Saliency Maps"->{
                    FourGrid(StageFourExplanationEngine.pixels,false)
                    FourText("Input pixels ↑   |   magnitude of ∂class score/∂input pixel ↓")
                    FourGrid(StageFourExplanationEngine.saliency.map{row->row.map{it*control/10.0}},false,normalize=false)
                    FourMetric("Selected class score","%.3f".format(StageFourExplanationEngine.imageScore(StageFourExplanationEngine.pixels)))
                    FourMetric("Overlay opacity","%.1f".format(control/10.0))
                }
                "Grad-CAM"->{
                    FourGrid(StageFourExplanationEngine.pixels,false)
                    FourFlow(listOf("CNN feature maps","Class gradients αₖ","Σ αₖAₖ","ReLU heatmap"),control%4)
                    FourGrid(StageFourExplanationEngine.gradCam(),false)
                    FourText("Class-weighted convolution activations are coarser than direct input gradients.")
                }
                "Attention Visualization"->{
                    FourGrid(StageFourExplanationEngine.attention(control%3))
                    FourText("Rows are query tokens; columns are key tokens. Each row is normalized by softmax.")
                    FourMetric("Selected head","${control%3+1} of 3")
                }
                "Counterfactual Explanations"->{
                    val (changed,distance)=StageFourExplanationEngine.counterfactual(x)
                    FourFlow(listOf("Original sample","Search feature edits","Class flips"),control%3)
                    FourBars(labels.mapIndexed{i,label->label to (changed[i]-x[i])},feature,LabOrange)
                    FourMetric("Original → alternate probability","%.2f → %.2f".format(StageFourExplanationEngine.score(x),StageFourExplanationEngine.score(changed)),LabGreen)
                    FourMetric("Smallest grid L1 change","%.2f".format(distance))
                    FourText("This changes the model prediction. It does not establish a causal intervention.")
                }
                else->error("Unregistered explainability topic $name")
            }
        }}
        item{FourPanel("Choose input"){
            FourMetric("Selected sample","${sample+1} of ${StageFourExplanationEngine.data.size}")
            FourMetric("Prediction","%.3f".format(StageFourExplanationEngine.score(x)),LabGreen)
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(4.dp)){
                labels.forEachIndexed{i,label->SegmentedOption(label,feature==i,Modifier.weight(1f)){feature=i}}
            }
            Slider(sample.toFloat(),{sample=it.toInt().coerceIn(0,StageFourExplanationEngine.data.lastIndex)},valueRange=0f..StageFourExplanationEngine.data.lastIndex.toFloat())
            if(name in setOf("Saliency Maps","Grad-CAM","Attention Visualization","LIME")){
                FourText("View parameter $control")
                Slider(control.toFloat(),{control=it.toInt().coerceIn(1,9)},valueRange=1f..9f)
            }
        }}
    }
}
