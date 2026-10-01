package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import kotlin.math.abs
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

internal data class EvoFrame(val flow:List<String>,val equation:String,val distinction:String,
                             val bars:List<Pair<String,Double>>,val points:List<Pair<Double,Double>>,
                             val metrics:List<Pair<String,String>>)

internal data class GpTree(val symbol:String,val left:GpTree?=null,val right:GpTree?=null) {
    fun evaluate(x:Double):Double=when(symbol){
        "x"->x
        "1"->1.0
        "+"->left!!.evaluate(x)+right!!.evaluate(x)
        "×"->left!!.evaluate(x)*right!!.evaluate(x)
        else->error("Unknown GP symbol $symbol")
    }
    fun expression():String=if(left==null)symbol else "(${left.expression()} $symbol ${right!!.expression()})"
    fun diagram():String {
        fun children(node:GpTree,prefix:String):List<String> {
            val leftChild=node.left ?: return emptyList()
            val rightChild=node.right ?: return emptyList()
            return listOf("${prefix}├─ ${leftChild.symbol}")+children(leftChild,"$prefix│  ")+
                listOf("${prefix}└─ ${rightChild.symbol}")+children(rightChild,"$prefix   ")
        }
        return (listOf(symbol)+children(this,"")).joinToString("\n")
    }
}
internal data class EsState(val parent:Pair<Double,Double>,val offspring:List<Pair<Double,Double>>,val sigma:Double)
internal data class SaState(val current:Double,val candidate:Double,val next:Double,val temperature:Double,
                            val acceptance:Double,val accepted:Boolean)
internal data class BeeState(val sources:List<Pair<Double,Double>>,val role:String,val selected:Int)

internal object StageFourEvolutionEngine {
    private val gpX=GpTree("x")
    private val gpOne=GpTree("1")
    private fun gpAdd(a:GpTree,b:GpTree)=GpTree("+",a,b)
    private fun gpTimes(a:GpTree,b:GpTree)=GpTree("×",a,b)
    fun gpFitness(tree:GpTree):Double {
        val error=listOf(-2.0,-1.0,0.0,1.0,2.0).sumOf{x->
            val delta=tree.evaluate(x)-(x*x+x)
            delta*delta
        }
        return 1.0/(1.0+error)
    }
    fun gpCrossover(a:GpTree,b:GpTree):GpTree =
        GpTree(a.symbol,a.left,b.right ?: b)
    fun gpMutation(tree:GpTree):GpTree =
        if(tree.left==null)gpAdd(tree,gpOne) else GpTree(tree.symbol,tree.left,gpAdd(gpX,gpOne))
    fun gpPopulation(generation:Int):List<GpTree> {
        var population=listOf(gpAdd(gpX,gpOne),gpTimes(gpX,gpX),gpTimes(gpAdd(gpX,gpOne),gpOne),gpX)
        repeat(generation.coerceIn(0,12)){
            val parents=population.sortedByDescending(::gpFitness).take(2)
            val crossed=gpCrossover(gpAdd(parents[0],gpX),gpTimes(gpX,gpX))
            val mutated=gpMutation(parents[1])
            population=(parents+crossed+mutated).sortedByDescending(::gpFitness).take(4)
        }
        return population
    }
    fun objective(x:Double,y:Double)=-(x-1.0).pow(2)-(y+.4).pow(2)+2.0
    fun crossover(a:String,b:String,cut:Int)=a.take(cut)+b.drop(cut)
    fun mutate(bits:String,index:Int)=bits.mapIndexed{i,c->if(i==index)if(c=='0')'1' else '0' else c}.joinToString("")
    fun deMutant(a:Pair<Double,Double>,b:Pair<Double,Double>,c:Pair<Double,Double>,factor:Double)=
        a.first+factor*(b.first-c.first) to a.second+factor*(b.second-c.second)
    fun psoVelocity(position:Pair<Double,Double>,velocity:Pair<Double,Double>,personal:Pair<Double,Double>,global:Pair<Double,Double>):Pair<Double,Double> =
        .7*velocity.first+.9*(personal.first-position.first)+1.1*(global.first-position.first) to
            .7*velocity.second+.9*(personal.second-position.second)+1.1*(global.second-position.second)
    fun annealProbability(old:Double,candidate:Double,temp:Double)=
        if(candidate>=old)1.0 else exp((candidate-old)/temp.coerceAtLeast(1e-6))
    private fun normal(seed:Int):Double {
        val u1=((seed*73)%997+1)/998.0
        val u2=((seed*191)%991+1)/992.0
        return sqrt(-2*ln(u1))*cos(2*PI*u2)
    }
    fun esState(generation:Int,control:Int):EsState {
        var parent=-1.4 to 1.1
        var sigma=.45+control*.09
        fun children(t:Int)=List(4){i->
            parent.first+sigma*normal(t*17+i*2+1) to parent.second+sigma*normal(t*17+i*2+2)
        }
        repeat(generation.coerceIn(0,12)){t->
            val winner=(listOf(parent)+children(t)).maxBy{objective(it.first,it.second)}
            sigma=(sigma*if(winner!=parent).92 else 1.04).coerceIn(.08,1.4)
            parent=winner
        }
        return EsState(parent,children(generation.coerceIn(0,12)),sigma)
    }
    fun saState(step:Int,control:Int):SaState {
        var current=-1.4
        var last=SaState(current,current,current,1.8,1.0,true)
        repeat(step.coerceIn(0,12)+1){t->
            val temperature=(1.8*.82.pow(t)).coerceAtLeast(.05)
            val candidate=current+sin((t+1)*2.7)*(.15+control*.1)
            val acceptance=annealProbability(objective(current,0.0),objective(candidate,0.0),temperature)
            val accepted=abs(sin((t+1)*7.1))<acceptance
            last=SaState(current,candidate,if(accepted)candidate else current,temperature,acceptance,accepted)
            current=last.next
        }
        return last
    }
    fun acoPheromone(step:Int):List<Double> {
        val lengths=listOf(6.0,3.0,4.0,5.0)
        val pheromone=DoubleArray(4){.3}
        repeat(step.coerceIn(0,12)+1){generation->
            val chosen=if(generation<4)generation else pheromone.indices.maxBy{pheromone[it]/lengths[it]}
            repeat(4){i->pheromone[i]*=.85}
            pheromone[chosen]+=1.0/lengths[chosen]
        }
        return pheromone.toList()
    }
    fun abcState(step:Int):BeeState {
        val sources=mutableListOf(-1.5 to .8,-.7 to .4,.1 to .1,.9 to -.2)
        var selected=0
        repeat(step.coerceIn(0,12)+1){generation->
            when(generation%3){
                0->{
                    val index=(generation/3)%sources.size
                    val old=sources[index]
                    val candidate=old.first+.22*sin(generation+1.0) to old.second+.22*cos(generation+1.0)
                    if(objective(candidate.first,candidate.second)>objective(old.first,old.second))sources[index]=candidate
                    selected=index
                }
                1->{
                    val index=sources.indices.maxBy{objective(sources[it].first,sources[it].second)}
                    val old=sources[index]
                    val candidate=old.first+.16 to old.second-.12
                    if(objective(candidate.first,candidate.second)>objective(old.first,old.second))sources[index]=candidate
                    selected=index
                }
                else->{
                    val index=sources.indices.minBy{objective(sources[it].first,sources[it].second)}
                    sources[index]=-.8+generation*.08 to .7-generation*.06
                    selected=index
                }
            }
        }
        return BeeState(sources,(listOf("Employed","Onlooker","Scout"))[step%3],selected)
    }
    fun population(generation:Int):List<String>{
        var current=listOf("010101","111000","101110","000111","110011","001101")
        repeat(generation.coerceIn(0,20)){t->
            val ranked=current.sortedByDescending {bits->bits.count{it=='1'}}
            val children=(0..3).map{i->
                val child=crossover(ranked[i%2],ranked[(i+1)%2],2+(t+i)%3)
                if((t+i)%3==0)mutate(child,(t*3+i)%6)else child
            }
            current=ranked.take(2)+children
        }
        return current
    }
    fun pso(step:Int):List<Pair<Double,Double>>{
        var positions=listOf(-1.7 to 1.2, -.8 to -.9,1.8 to 1.0,1.3 to -1.8)
        var velocities=List(4){0.0 to 0.0}
        var personal=positions
        repeat(step.coerceIn(0,20)){
            val global=personal.maxBy { objective(it.first,it.second) }
            velocities=positions.indices.map{i->psoVelocity(positions[i],velocities[i],personal[i],global)}
            positions=positions.indices.map{i->positions[i].first+velocities[i].first*.35 to positions[i].second+velocities[i].second*.35}
            personal=personal.indices.map{i->if(objective(positions[i].first,positions[i].second)>objective(personal[i].first,personal[i].second))positions[i]else personal[i]}
        }
        return positions
    }
    fun frame(name:String,step:Int,control:Int):EvoFrame {
        val rate=.15+control*.1
        return when(name){
            "Genetic Algorithm"->{
                val pop=population(step)
                val score=pop.map{it.count{b->b=='1'}.toDouble()}
                val a=pop.sortedByDescending{it.count{b->b=='1'}}
                EvoFrame(listOf("Fitness","Select parents","Crossover","Mutation","Next generation"),
                    "child = crossover(parent₁,parent₂); occasional bit flip",
                    "Chromosomes reproduce; particles and velocities are absent.",pop.mapIndexed{i,p->p to score[i]},emptyList(),
                    listOf("Best chromosome" to a.first(),"Mean fitness" to "%.2f".format(score.average())))
            }
            "Genetic Programming"->{
                val programs=gpPopulation(step)
                val best=programs.maxBy(::gpFitness)
                EvoFrame(listOf("Tree programs","Evaluate on x","Subtree crossover","Subtree mutation"),
                    "fitness = 1 / (1 + Σx [tree(x) − (x²+x)]²)","Whole expression subtrees are exchanged rather than bit positions.",
                    programs.mapIndexed{i,p->"Tree ${i+1}" to gpFitness(p)},emptyList(),
                    listOf("Selected tree" to best.expression(),"Selected fitness" to "%.3f".format(gpFitness(best))))
            }
            "Evolution Strategies"->{
                val state=esState(step,control)
                val sigma=state.sigma
                val parent=state.parent
                val offspring=state.offspring
                EvoFrame(listOf("Parent vector","Gaussian mutation σ","Offspring","Select best"),
                    "xchild = xparent + σ ε; ε ~ N(0,I)","Mutation strength controls the search radius.",
                    offspring.mapIndexed{i,p->"child $i" to objective(p.first,p.second)},listOf(parent)+offspring,
                    listOf("Mutation strength σ" to "%.3f".format(sigma),"Best offspring" to "%.3f".format(offspring.maxOf{objective(it.first,it.second)})))
            }
            "Differential Evolution"->{
                val a=-1.1 to .8;val b=1.5 to .9;val c=-.4 to -1.4
                val mutant=deMutant(a,b,c,rate)
                val target=.3 to .6
                val trial=mutant.first to target.second
                EvoFrame(listOf("Choose a,b,c","a + F(b−c)","Coordinate crossover","Select fitter"),
                    "mutant = a + %.2f(b − c)".format(rate),"Vector differences, not chromosome crossover, create the mutant.",
                    listOf("Target" to objective(target.first,target.second),"Trial" to objective(trial.first,trial.second)),
                    listOf(a,b,c,target,mutant,trial),listOf("Mutant x,y" to "%.2f, %.2f".format(mutant.first,mutant.second),
                        "Selected" to if(objective(trial.first,trial.second)>objective(target.first,target.second))"Trial" else "Target"))
            }
            "Particle Swarm Optimization"->{
                val particles=pso(step)
                val best=particles.maxBy{objective(it.first,it.second)}
                val velocity=psoVelocity(particles[0],.1 to -.1,particles[0],best)
                EvoFrame(listOf("Position + velocity","Personal best","Global best","Updated velocity"),
                    "v ← ωv + c₁r₁(pbest−x) + c₂r₂(gbest−x)","Particles move through velocity and social attraction; no selection or crossover.",
                    particles.mapIndexed{i,p->"P$i" to objective(p.first,p.second)},particles,
                    listOf("Global best" to "%.2f, %.2f".format(best.first,best.second),"Particle 1 velocity" to "%.2f, %.2f".format(velocity.first,velocity.second)))
            }
            "Ant Colony Optimization"->{
                val pheromone=acoPheromone(step)
                val best=pheromone.indices.maxBy{pheromone[it]/listOf(6.0,3.0,4.0,5.0)[it]}
                EvoFrame(listOf("Ant routes","Route length","Evaporate pheromone","Deposit on good path"),
                    "τedge ← (1−ρ)τedge + Σant Q / routeLength","Pheromone strengthens good graph edges; no particles move in a numeric landscape.",
                    pheromone.mapIndexed{i,p->"route ${i+1}" to p},emptyList(),listOf("Evaporation ρ" to "0.15","Best route" to "${best+1} (length ${listOf(6,3,4,5)[best]})"))
            }
            "Artificial Bee Colony"->{
                val state=abcState(step)
                val sources=state.sources
                val nectar=sources.map{max(.01,objective(it.first,it.second))}
                EvoFrame(listOf("Employed bees","Onlookers select nectar","Scout replaces stale source"),
                    "P(source i) = fitnessᵢ / Σ fitness","Bee roles differ: employed explore, onlookers choose, scouts restart.",
                    nectar.mapIndexed{i,v->"food $i" to v},sources,
                    listOf("Current role" to state.role,"Touched food source" to "${state.selected+1}",
                        "Best nectar" to "%.3f".format(nectar.max())))
            }
            "Simulated Annealing"->{
                val state=saState(step,control)
                val old=objective(state.current,0.0);val proposed=objective(state.candidate,0.0)
                EvoFrame(listOf("Current solution","Neighbor proposal","Temperature","Accept/reject"),
                    "P(accept worse) = exp((fitnessnew−fitnessold)/T)","High temperature permits some worse moves; cooling reduces their chance.",
                    listOf("Current" to old,"Neighbor" to proposed),listOf(state.current to 0.0,state.candidate to 0.0,state.next to 0.0),
                    listOf("Temperature" to "%.3f".format(state.temperature),"Acceptance probability" to "%.3f".format(state.acceptance),
                        "Decision" to if(state.accepted)"Accept" else "Reject","Next solution" to "%.3f".format(state.next)))
            }
            else->error("Unregistered evolution topic: $name")
        }
    }
}

@Composable
private fun GpTreeDiagram(tree:GpTree) {
    Canvas(Modifier.fillMaxWidth().height(170.dp)) {
        val radius=17.dp.toPx()
        val top=27.dp.toPx()
        val gap=54.dp.toPx()
        val labelPaint=Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color=android.graphics.Color.WHITE
            textAlign=Paint.Align.CENTER
            textSize=17.dp.toPx()
            typeface=android.graphics.Typeface.DEFAULT_BOLD
        }
        fun node(current:GpTree,x:Float,y:Float,spread:Float) {
            val childY=y+gap
            current.left?.let { child ->
                val childX=x-spread
                drawLine(LabCyan.copy(alpha=.7f),Offset(x,y+radius),Offset(childX,childY-radius),2.dp.toPx())
                node(child,childX,childY,spread*.55f)
            }
            current.right?.let { child ->
                val childX=x+spread
                drawLine(LabCyan.copy(alpha=.7f),Offset(x,y+radius),Offset(childX,childY-radius),2.dp.toPx())
                node(child,childX,childY,spread*.55f)
            }
            drawCircle(if(current.left==null)LabCyan else Color(0xFF8B5CF6),radius,Offset(x,y))
            drawContext.canvas.nativeCanvas.drawText(current.symbol,x,y+6.dp.toPx(),labelPaint)
        }
        node(tree,size.width/2,top,size.width*.24f)
    }
}

@Composable
internal fun StageFourEvolutionScreen(topic:LearnTopic){
    val name=topic.title
    var step by remember(topic.id){mutableIntStateOf(0)}
    var control by remember(topic.id){mutableIntStateOf(4)}
    val frame=StageFourEvolutionEngine.frame(name,step,control)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)){
        item{SectionTitle(name,when(name){
            "Genetic Algorithm"->"Select, cross, and mutate chromosome candidates"
            "Genetic Programming"->"Exchange and mutate expression-tree subtrees"
            "Evolution Strategies"->"Sample Gaussian offspring and select the fittest"
            "Differential Evolution"->"Form mutants from vector differences, then select"
            "Particle Swarm Optimization"->"Move particles toward personal and global best positions"
            "Ant Colony Optimization"->"Evaporate and deposit pheromone on candidate routes"
            "Artificial Bee Colony"->"Compare employed, onlooker, and scout bee updates"
            "Simulated Annealing"->"Accept or reject neighbors as the temperature falls"
            else->frame.distinction
        })}
        item{FourPanel("Mechanism"){
            FourFlow(frame.flow,step%frame.flow.size)
            FourText(frame.equation,LabCyan)
            FourText(frame.distinction)
        }}
        item{FourPanel(when(name){"Ant Colony Optimization"->"Pheromone on graph routes";"Genetic Algorithm"->"Chromosome population";"Genetic Programming"->"Expression-tree population";else->"Candidate population and fitness"}){
            if(name=="Genetic Programming"){
                val selected=StageFourEvolutionEngine.gpPopulation(step).maxBy(StageFourEvolutionEngine::gpFitness)
                GpTreeDiagram(selected)
            }
            if(frame.points.isNotEmpty())FourLandscape(frame.points,connect=name=="Simulated Annealing")
            val selected=if(name=="Simulated Annealing"){
                if(StageFourEvolutionEngine.saState(step,control).accepted)1 else 0
            }else frame.bars.indices.maxBy{frame.bars[it].second}
            FourBars(frame.bars,selected)
            frame.metrics.forEach{(label,value)->FourMetric(label,value,if(label.contains("Best"))LabGreen else LabOrange)}
        }}
        item{FourPanel("Explore"){
            FourSteps(step,12){step=it}
            FourText(when(name){"Evolution Strategies"->"Mutation strength";"Differential Evolution"->"Difference factor F";"Simulated Annealing"->"Neighbor span";else->"Scenario variation"}+": $control")
            Slider(control.toFloat(),{control=it.toInt().coerceIn(0,7);step=0},valueRange=0f..7f)
        }}
    }
}
