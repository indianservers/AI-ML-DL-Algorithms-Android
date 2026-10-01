package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxSize
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

internal data class SearchSpec(val subtitle:String,val flow:List<String>,val rule:String,val difference:String)
internal object StageFourSearchSpecs {
    fun forTitle(name:String):SearchSpec=when(name){
        "Breadth First Search"->SearchSpec("Visit graph nodes level by level",listOf("FIFO queue","Dequeue oldest","Enqueue children"),"priority = depth","Queue order finds the fewest-edge path.")
        "AI Depth First Search"->SearchSpec("Follow one branch before backtracking",listOf("LIFO stack","Pop newest","Push children"),"priority = stack top","Stack order can find a deep solution before a shallow one.")
        "Uniform Cost Search"->SearchSpec("Expand lowest accumulated path cost",listOf("Priority queue","Read g(n)","Expand lowest g"),"priority = g(n)","Edge costs determine order; no heuristic is used.")
        "Greedy Best First Search"->SearchSpec("Expand lowest estimated distance",listOf("Priority queue","Read h(n)","Expand lowest h"),"priority = h(n)","It ignores the cost already spent.")
        "A Star Search"->SearchSpec("Combine path cost and estimate",listOf("Priority queue","g(n)","h(n)","lowest f"),"f(n) = g(n) + h(n)","Both paid cost and estimated remaining cost rank the frontier.")
        "Simplified Memory-Bounded A Star Search"->SearchSpec("Prune expensive leaves under a memory limit",listOf("A* frontier","Memory full","Forget worst leaf","Back up score"),"f = g+h; keep limited frontier","The memory cap visibly discards a frontier leaf and records its score.")
        "Bidirectional Search"->SearchSpec("Meet frontiers grown from both ends",listOf("Start frontier →","← Goal frontier","Intersection"),"Fstart ∩ Fgoal ≠ ∅","Two separate visited sets expand toward a meeting node.")
        "Beam Search"->SearchSpec("Prune low-ranked candidates at each level",listOf("Generate level","Rank by h","Keep top k","Discard rest"),"frontier = top-k(level, h)","A fixed beam width drops candidates even if they might lead to a solution.")
        "Iterative Deepening Depth First Search"->SearchSpec("Restart DFS with increasing depth limits",listOf("Limit 0","Restart limit 1","Restart limit 2","Continue"),"DFS(depth ≤ limit)","The frontier restarts at S whenever the limit grows.")
        "Monte Carlo Tree Search"->SearchSpec("Balance search and simulation",listOf("Selection","Expansion","Simulation","Backpropagation"),"UCT = Q/N + c√(ln Nparent/N)","Visit counts and rollout rewards decide which branch is explored next.")
        "Minimax Search"->SearchSpec("Back up game utilities through MAX/MIN",listOf("Leaf utilities","MIN child values","MAX root value"),"root = max(min(left leaves),min(right leaves))","Every relevant leaf is evaluated in a full minimax tree.")
        "Alpha-Beta Pruning"->SearchSpec("Skip game branches that cannot change minimax",listOf("Evaluate left","Set α","Evaluate right","Prune if β≤α"),"prune when β ≤ α","The minimax result stays the same while a branch is not evaluated.")
        "Hill Climbing Search"->SearchSpec("Move to a better local neighbor",listOf("Current state","Evaluate neighbors","Choose best improving","Stop at local peak"),"x′ = argmaxneighbor f(x′)","One current state can become stuck at a local optimum.")
        "Local Beam Search"->SearchSpec("Evolve several local states together",listOf("k current states","All neighbors","Keep global top k"),"states′ = top-k(∪ neighbors(states))","Multiple current states share candidate information, unlike tree-level beam search.")
        else->error("Unregistered AI search topic $name")
    }
}

@Composable
internal fun StageFourSearchScreen(topic:LearnTopic){
    val name=topic.title;val spec=StageFourSearchSpecs.forTitle(name)
    var step by remember(topic.id){mutableIntStateOf(0)}
    var width by remember(topic.id){mutableIntStateOf(3)}
    val game=name in setOf("Minimax Search","Alpha-Beta Pruning","Monte Carlo Tree Search")
    val local=name in setOf("Hill Climbing Search","Local Beam Search")
    val frames=when{
        game||local->emptyList()
        name=="Bidirectional Search"->StageFourSearchEngine.bidirectional()
        else->StageFourSearchEngine.trace(name,width.coerceIn(2,5),4)
    }
    val maxStep=if(game||local)12 else (frames.size-1).coerceAtLeast(0)
    val s=step.coerceIn(0,maxStep)
    val frame=frames.getOrNull(s)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)){
        item{SectionTitle(name,spec.subtitle)}
        item{FourPanel("Algorithm rule"){
            FourFlow(spec.flow,s%spec.flow.size)
            FourText(spec.rule,LabCyan)
            FourText(spec.difference)
        }}
        item{FourPanel(when{game->"Game/search tree";local->"Local objective landscape";else->"Search graph and frontier"}){
            when(name){
                "Minimax Search","Alpha-Beta Pruning"->{
                    val leaves=listOf(3,5,2,9)
                    val (result,pruned)=StageFourSearchEngine.alphaBeta(leaves)
                    FourGameTree(leaves,(s-1).coerceIn(0,3),if(name=="Alpha-Beta Pruning"&&s>=3)pruned else emptySet())
                    FourMetric("Root minimax utility",StageFourSearchEngine.minimax(leaves).toString(),LabGreen)
                    if(name=="Alpha-Beta Pruning"){
                        FourMetric("α, β after right-left leaf","3, 2")
                        FourMetric("Pruned leaves",if(s>=3)pruned.joinToString() else "none yet",LabOrange)
                        FourText("Alpha-beta result = $result; minimax result = ${StageFourSearchEngine.minimax(leaves)}.")
                    }
                }
                "Monte Carlo Tree Search"->{
                    val visits=listOf(2+s,1+s/2,1+s/3,1)
                    val values=listOf(1.5+s*.25,.8+s*.16,.4+s*.1,.1)
                    val uct=visits.indices.map{i->StageFourSearchEngine.uct(values[i],visits[i],visits.sum())}
                    FourGameTree(listOf(3,5,2,9),s%4,emptySet())
                    FourBars(uct.mapIndexed{i,v->"child $i" to v},uct.indices.maxBy{uct[it]})
                    FourMetric("Selected UCT child",uct.indices.maxBy{uct[it]}.toString())
                    FourText("Simulation reward is backed up into Q and visit counts before the next selection.")
                }
                "Hill Climbing Search"->{
                    val path=StageFourSearchEngine.hillPath(s)
                    val landscape=(0..80).map{StageFourSearchEngine.hillFitness(-2.5+it/16.0)}
                    FourLine(landscape,markers=path.map{(x,y)->((x+2.5)/5.0) to y},connectMarkers=true)
                    FourMetric("Current position x","%.2f".format(path.last().first))
                    FourMetric("Current local score","%.3f".format(path.last().second))
                    FourText("Each new point is the better of the left and right neighbors. The trace stops improving at a local peak.")
                }
                "Local Beam Search"->{
                    val states=StageFourSearchEngine.localBeam(s,width).sortedByDescending(StageFourSearchEngine::localFitness)
                    val landscape=(0..80).map{StageFourSearchEngine.localFitness(-2.5+it/16.0)}
                    FourLine(landscape,markers=states.reversed().map{x->((x+2.5)/5.0) to StageFourSearchEngine.localFitness(x)})
                    states.forEachIndexed{i,x->FourMetric("${if(i==0) "Best" else "Candidate ${i+1}"} x=%.2f".format(x),"%.3f".format(StageFourSearchEngine.localFitness(x)),if(i==0)LabOrange else LabCyan)}
                    FourMetric("Current states",states.size.toString())
                    FourText("All current states generate neighboring positions; the best $width scores survive together.")
                }
                else->{
                    FourSearchGraph(frame?.current,frame?.visited?:emptySet(),frame?.frontier?.map{it.node}?.toSet()?:emptySet(),frame?.scores?:emptyMap(),frame?.discarded?:emptySet())
                    FourMetric("Current node",frame?.current?:"Ready")
                    FourText("Frontier order: "+(frame?.frontier?.joinToString{"${it.node}(g=${it.cost.toInt()},h=${StageFourSearchEngine.heuristic[it.node]?.toInt()})"}?:"empty"),LabCyan)
                    FourMetric("Visited",frame?.visited?.joinToString()?:"none")
                    if(name=="Beam Search"||name=="Simplified Memory-Bounded A Star Search")FourMetric("Discarded candidates",frame?.discarded?.joinToString()?:"none",LabOrange)
                    if(name=="Simplified Memory-Bounded A Star Search")FourText("Forgotten-leaf scores backed up to parents: "+(frame?.scores?.filterKeys{it in setOf("S","A","B","C","D")}?.entries?.joinToString{"${it.key}=%.1f".format(it.value)}?:"none"))
                    if(name=="Bidirectional Search")FourText(frame?.note?:"Two frontiers are ready.")
                    if(name=="Iterative Deepening Depth First Search")FourText("Depth restarts are part of the trace; nodes can be revisited at a larger limit.")
                    if(frame?.path?.lastOrNull()=="G")FourMetric("Found path",frame.path.joinToString(" → "),LabGreen)
                    if(frame?.note?.isNotBlank()==true&&name!="Bidirectional Search")FourText(frame.note)
                }
            }
        }}
        item{FourPanel("Explore search"){
            FourSteps(s,maxStep){step=it}
            if(name in setOf("Beam Search","Local Beam Search","Simplified Memory-Bounded A Star Search")){
                FourText(if(name=="Simplified Memory-Bounded A Star Search")"Frontier memory limit $width" else "Beam width $width")
                Slider(width.toFloat(),{width=it.toInt().coerceIn(2,5);step=0},valueRange=2f..5f)
            }
        }}
    }
}
