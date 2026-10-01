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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.SupervisedVisualizationScreen
import kotlin.math.exp
import kotlin.math.pow

internal object StageFourProbabilityEngine {
    fun likelihood(heads:Int,tails:Int,theta:Double):Double =
        theta.coerceIn(1e-6,1.0).pow(heads)*(1-theta).coerceIn(1e-6,1.0).pow(tails)
    fun betaDensity(alpha:Double,beta:Double,theta:Double):Double {
        fun factorial(n:Int)=(2..n).fold(1.0){value,i->value*i}
        val coefficient=factorial((alpha+beta-1).toInt())/
            (factorial((alpha-1).toInt())*factorial((beta-1).toInt()))
        return coefficient*theta.coerceIn(1e-6,1.0).pow(alpha-1.0)*
            (1-theta).coerceIn(1e-6,1.0).pow(beta-1.0)
    }
    fun wetPosterior(rainEvidence:Boolean?, wetWithRain:Double=.90):Double {
        val rain=.30; val sprinkler=.35
        fun wet(r:Boolean,s:Boolean)=when {r&&s->.99;r->wetWithRain;s->.80;else->.02}
        val joint=listOf(false,true).sumOf { r ->
            if(rainEvidence!=null&&r!=rainEvidence) 0.0 else (if(r)rain else 1-rain)*
                listOf(false,true).sumOf { s->(if(s)sprinkler else 1-sprinkler)*wet(r,s) }
        }
        return joint/(when(rainEvidence){true->rain;false->1-rain;null->1.0})
    }
    fun rainGivenWet(wetWithRain:Double=.90):Double =
        .30*wetPosterior(true,wetWithRain)/wetPosterior(null,wetWithRain)
    fun acyclic(edges:List<Pair<String,String>>):Boolean {
        val visiting=mutableSetOf<String>();val visited=mutableSetOf<String>()
        fun visit(node:String):Boolean {
            if(node in visiting)return false
            if(node in visited)return true
            visiting+=node
            for(child in edges.filter{it.first==node}.map{it.second})if(!visit(child))return false
            visiting-=node;visited+=node;return true
        }
        return edges.flatMap{listOf(it.first,it.second)}.distinct().all(::visit)
    }
}

@Composable
internal fun StageFourProbabilityScreen(topic:LearnTopic) {
    val name=topic.title
    if(name=="Bayesian Linear Regression"){SupervisedVisualizationScreen(topic,SupervisedVisualization.BayesianLinear,null);return}
    if(name=="Gaussian Mixture Models"){StageTwoVisualizationScreen(topic,StageTwoVisualization.Gmm);return}
    if(name=="Gaussian Processes"){SupervisedVisualizationScreen(topic,SupervisedVisualization.GaussianProcessRegression,null);return}
    if(name=="Hidden Markov Models"||name.startsWith("Hidden Markov Model")){StageThreeLanguageScreen(topic,LanguageVisualization.Hmm);return}
    var step by remember(topic.id){mutableIntStateOf(0)}
    var parameter by remember(topic.id){mutableIntStateOf(3)}
    var evidence by remember(topic.id){mutableIntStateOf(0)}
    var edgeChoice by remember(topic.id){mutableIntStateOf(0)}
    var query by remember(topic.id){mutableIntStateOf(0)}
    val prior=.05+parameter*.05
    val bayes=PhaseFourEngines.bayes(prior,.85,.90)
    val beta=PhaseFourEngines.betaBernoulli(2.0,2.0,step.coerceAtMost(8),(step/2).coerceAtMost(6))
    val heads=2+step.coerceAtMost(8);val tails=3+(step/2).coerceAtMost(6)
    val mle=PhaseFourEngines.mleTheta(heads,tails)
    val map=PhaseFourEngines.mapTheta(heads,tails,2.0+parameter,2.0)
    val rainEvidence=when(evidence){1->true;2->false;else->null}
    val wetWithRain=.70+parameter*.035
    val baseEdges=listOf("Rain" to "Wet Grass","Sprinkler" to "Wet Grass")
    val proposedEdge=when(edgeChoice){1->"Season" to "Rain";2->"Wet Grass" to "Rain";else->null}
    val graphEdges=baseEdges+(if(proposedEdge!=null&&StageFourProbabilityEngine.acyclic(baseEdges+proposedEdge))listOf(proposedEdge)else emptyList())
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(name,when(name){
            "Bayes Theorem"->"Multiply a prior by evidence likelihoods"
            "Bayesian Inference"->"Observe data and update a distribution"
            "Maximum Likelihood Estimation"->"Find the coin bias that best explains observations"
            "Maximum A Posteriori Estimation"->"Combine likelihood with a prior"
            "Bayesian Networks","Construction of Bayesian Network","Inference from Bayesian Network"->"Inspect a conditional probability graph"
            "Markov Chain Monte Carlo"->"Follow a chain toward the target density"
            "Gibbs Sampling"->"Alternate axis-aligned conditional samples"
            "Metropolis-Hastings"->"Propose, compare densities, accept or reject"
            else->"Fit a simple approximation to a posterior"
        }) }
        item { FourPanel("Distribution or graph") {
            when(name) {
                "Bayes Theorem"->{
                    FourFlow(listOf("Prior","Likelihood","Joint evidence","Posterior"),step%4)
                    FourBars(listOf("Prior" to prior,"True +" to bayes.truePositive,"False +" to bayes.falsePositive,"Posterior" to bayes.posterior),3)
                    FourMetric("P(condition | positive)","%.3f".format(bayes.posterior),LabGreen)
                }
                "Bayesian Inference"->{
                    FourLine((1..49).map{x->StageFourProbabilityEngine.betaDensity(beta.alpha,beta.beta,x/50.0)},
                        (1..49).map{x->StageFourProbabilityEngine.betaDensity(2.0,2.0,x/50.0)})
                    FourMetric("Posterior Beta(α,β)","Beta(%.0f, %.0f)".format(beta.alpha,beta.beta))
                    FourMetric("Posterior mean","%.3f".format(beta.mean),LabGreen)
                }
                "Maximum Likelihood Estimation","Maximum A Posteriori Estimation"->{
                    FourLine((1..49).map{x->StageFourProbabilityEngine.likelihood(heads,tails,x/50.0)},
                        if(name=="Maximum A Posteriori Estimation")(1..49).map{x->StageFourProbabilityEngine.likelihood(heads+parameter+1,tails+1,x/50.0)}else emptyList())
                    FourMetric("MLE θ","%.3f".format(mle))
                    if(name=="Maximum A Posteriori Estimation")FourMetric("MAP θ with Beta prior","%.3f".format(map),LabOrange)
                }
                "Bayesian Networks","Construction of Bayesian Network","Inference from Bayesian Network"->{
                    FourFlow(listOf("Rain","Wet Grass","Observation"),if(evidence==0)0 else 2)
                    FourText("Directed edges: "+graphEdges.joinToString{"${it.first} → ${it.second}"})
                    FourText("P(Rain)·P(Sprinkler)·P(Wet Grass | Rain, Sprinkler)",LabCyan)
                    FourMetric(if(query==0)"P(Wet Grass | evidence)" else "P(Rain | Wet Grass)",
                        "%.3f".format(if(query==0)StageFourProbabilityEngine.wetPosterior(rainEvidence,wetWithRain) else StageFourProbabilityEngine.rainGivenWet(wetWithRain)),LabGreen)
                    FourMetric("Graph is acyclic",StageFourProbabilityEngine.acyclic(graphEdges).toString())
                    if(edgeChoice==2)FourText("Wet Grass → Rain is rejected because it would create a directed cycle.",LabOrange)
                    FourText("CPT P(Wet | Rain,Sprinkler): 0.99 / %.2f / 0.80 / 0.02".format(wetWithRain))
                }
                "Markov Chain Monte Carlo","Metropolis-Hastings"->{
                    val chain=mutableListOf(0.0)
                    repeat(step+1){i->val move=PhaseFourEngines.metropolis(chain.last(),.45,i+1);chain+=if(move.accepted)move.proposal else chain.last()}
                    FourLine(chain)
                    val move=PhaseFourEngines.metropolis(chain[chain.lastIndex-1],.45,step+1)
                    FourMetric("Proposal / acceptance ratio","%.2f / %.2f".format(move.proposal,move.ratio))
                    FourMetric("Decision",if(move.accepted)"Accepted" else "Rejected",if(move.accepted)LabGreen else LabOrange)
                    if(name=="Markov Chain Monte Carlo")FourBars(chain.groupBy{(it*2).toInt()}.entries.take(8).map{"bin ${it.key}" to it.value.size.toDouble()})
                }
                "Gibbs Sampling"->{
                    val state=PhaseFourEngines.gibbs(step+1)
                    FourLine(state.path.map{it.first},state.path.map{it.second})
                    FourMetric("Conditional update",state.lastMove,LabGreen)
                    FourText("Each step changes only x given y, or only y given x.")
                }
                "Variational Inference"->{
                    val state=PhaseFourEngines.variational(step*3)
                    FourLine((-20..20).map{x->exp(-.5*((x/10.0-state.targetMean)/.35).pow(2))},
                        (-20..20).map{x->exp(-.5*((x/10.0-state.approximateMean)/state.approximateVariance).pow(2))})
                    FourMetric("Target p mean / q mean","%.2f / %.2f".format(state.targetMean,state.approximateMean))
                    FourMetric("ELBO proxy","%.3f".format(state.elboProxy),LabGreen)
                }
                else->error("Unregistered probability topic $name")
            }
        } }
        item { FourPanel("Inspect the computation") {
            FourText(when(name){
                "Bayes Theorem"->"Posterior = (sensitivity × prior) / [(sensitivity × prior) + ((1−specificity) × (1−prior))]"
                "Bayesian Inference"->"Beta prior plus observed heads and tails yields a Beta posterior; the mean changes as evidence arrives."
                "Maximum Likelihood Estimation"->"Likelihood θ^$heads(1−θ)^$tails; maximizing yields heads / observations = %.3f.".format(mle)
                "Maximum A Posteriori Estimation"->"MAP maximizes log likelihood + log prior; MLE ignores the prior."
                "Bayesian Networks","Construction of Bayesian Network","Inference from Bayesian Network"->"Enumeration sums unobserved parent assignments using CPTs. Reverse edge Wet Grass → Rain would create a cycle and is rejected: ${!StageFourProbabilityEngine.acyclic(graphEdges+listOf("Wet Grass" to "Rain"))}."
                "Markov Chain Monte Carlo"->"The trace records accepted states. The histogram approaches the target as samples accumulate; this short demonstration is not a convergence claim."
                "Gibbs Sampling"->"Conditional samples alternate x|y and y|x, unlike a free 2D proposal."
                "Metropolis-Hastings"->"Acceptance α = min(1,p(proposal)/p(current)) for this symmetric proposal. A uniform draw decides whether the chain moves."
                else->"q(z) is updated toward p(z|x) by improving a simplified ELBO. The plotted curves are an educational Gaussian approximation."
            })
        } }
        item { FourPanel("Explore") {
            FourSteps(step,12){step=it}
            if(name in setOf("Bayesian Networks","Construction of Bayesian Network","Inference from Bayesian Network")){
                Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(5.dp)){
                    listOf("No evidence","Rain=true","Rain=false").forEachIndexed{i,label->SegmentedOption(label,evidence==i,Modifier.weight(1f)){evidence=i}}
                }
                Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(5.dp)){
                    listOf("Query wet","Query rain | wet").forEachIndexed{i,label->SegmentedOption(label,query==i,Modifier.weight(1f)){query=i}}
                }
                Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(5.dp)){
                    listOf("Base graph","Add Season→Rain","Try Wet→Rain").forEachIndexed{i,label->SegmentedOption(label,edgeChoice==i,Modifier.weight(1f)){edgeChoice=i}}
                }
                FourText("P(Wet | Rain, no sprinkler) = %.2f".format(wetWithRain))
                Slider(parameter.toFloat(),{parameter=it.toInt().coerceIn(0,7)},valueRange=0f..7f)
            } else {
                FourText((if(name=="Bayes Theorem")"Prior probability" else "Scenario parameter")+": $parameter")
                Slider(parameter.toFloat(),{parameter=it.toInt().coerceIn(0,7)},valueRange=0f..7f)
            }
        } }
    }
}
