package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

internal data class SearchCandidate(val node:String,val path:List<String>,val cost:Double) {
    val depth:Int get()=path.size-1
}
internal data class SearchFrame(val current:String?,val visited:Set<String>,val frontier:List<SearchCandidate>,
                                val path:List<String>,val scores:Map<String,Double> = emptyMap(),
                                val discarded:Set<String> = emptySet(),val note:String="")

internal object StageFourSearchEngine {
    val graph=mapOf(
        "S" to listOf("A" to 2.0,"B" to 5.0),
        "A" to listOf("C" to 2.0,"D" to 5.0),
        "B" to listOf("D" to 1.0,"G" to 10.0),
        "C" to listOf("G" to 6.0),
        "D" to listOf("G" to 2.0),"G" to emptyList())
    val heuristic=mapOf("S" to 6.0,"A" to 5.0,"B" to 3.0,"C" to 4.0,"D" to 2.0,"G" to 0.0)
    fun priority(kind:String,c:SearchCandidate):Double=when(kind){
        "Uniform Cost Search"->c.cost
        "Greedy Best First Search","Beam Search"->heuristic.getValue(c.node)
        "A Star Search","Simplified Memory-Bounded A Star Search"->c.cost+heuristic.getValue(c.node)
        else->c.depth.toDouble()
    }
    fun trace(kind:String,width:Int=2,limit:Int=3):List<SearchFrame> {
        require(kind in setOf("Breadth First Search","AI Depth First Search","Uniform Cost Search",
            "Greedy Best First Search","A Star Search","Beam Search","Simplified Memory-Bounded A Star Search",
            "Iterative Deepening Depth First Search"))
        if(kind=="Beam Search")return beamTrace(width)
        val result=mutableListOf<SearchFrame>()
        val limits=if(kind=="Iterative Deepening Depth First Search")0..limit.coerceAtLeast(0) else 20..20
        for(depthLimit in limits){
            val frontier=mutableListOf(SearchCandidate("S",listOf("S"),0.0))
            val visited=linkedSetOf<String>()
            val discarded=mutableSetOf<String>()
            val backups=mutableMapOf<String,Double>()
            result+=SearchFrame(null,visited.toSet(),frontier.toList(),emptyList(),note="Depth limit $depthLimit")
            var expanded=0
            while(frontier.isNotEmpty()&&expanded++<28){
                val index=when(kind){
                    "AI Depth First Search","Iterative Deepening Depth First Search"->frontier.lastIndex
                    "Breadth First Search"->0
                    else->frontier.indices.minBy{priority(kind,frontier[it])}
                }
                val current=frontier.removeAt(index)
                if(current.node in visited&&kind!="Iterative Deepening Depth First Search")continue
                visited+=current.node
                val scores=(frontier+current).associate{it.node to priority(kind,it)}+backups
                result+=SearchFrame(current.node,visited.toSet(),frontier.toList(),current.path,scores,discarded.toSet(),
                    "Expanded ${current.node}; g=%.1f h=%.1f f=%.1f".format(current.cost,heuristic.getValue(current.node),current.cost+heuristic.getValue(current.node)))
                if(current.node=="G")return result
                if(current.depth>=depthLimit)continue
                val children=graph.getValue(current.node).filter{it.first !in current.path}.map{(node,cost)->
                    SearchCandidate(node,current.path+node,current.cost+cost)
                }
                frontier+=if(kind=="AI Depth First Search"||kind=="Iterative Deepening Depth First Search")children.reversed()else children
                if(kind=="Simplified Memory-Bounded A Star Search"&&frontier.size>width.coerceAtLeast(2)){
                    val worst=frontier.maxBy{priority(kind,it)}
                    discarded+=worst.node
                    val parent=worst.path.getOrNull(worst.path.lastIndex-1)
                    if(parent!=null)backups[parent]=minOf(backups[parent]?:Double.POSITIVE_INFINITY,priority(kind,worst))
                    frontier.remove(worst)
                }
            }
        }
        return result
    }
    fun beamTrace(width:Int):List<SearchFrame>{
        val frames=mutableListOf<SearchFrame>()
        var level=listOf(SearchCandidate("S",listOf("S"),0.0))
        val visited=mutableSetOf<String>()
        val discarded=mutableSetOf<String>()
        repeat(5){depth->
            if(level.isEmpty())return frames
            frames+=SearchFrame(null,visited.toSet(),level,emptyList(),level.associate{it.node to heuristic.getValue(it.node)},discarded.toSet(),"Level $depth: retain top $width by h")
            val next=mutableListOf<SearchCandidate>()
            level.forEach{candidate->
                visited+=candidate.node
                frames+=SearchFrame(candidate.node,visited.toSet(),level, candidate.path,
                    level.associate{it.node to heuristic.getValue(it.node)},discarded.toSet(),"Expanded level $depth")
                if(candidate.node=="G")return frames
                next+=graph.getValue(candidate.node).filter{it.first !in candidate.path}.map{(node,cost)->
                    SearchCandidate(node,candidate.path+node,candidate.cost+cost)
                }
            }
            val ranked=next.distinctBy{it.node}.sortedBy{heuristic.getValue(it.node)}
            discarded+=ranked.drop(width.coerceAtLeast(1)).map{it.node}
            level=ranked.take(width.coerceAtLeast(1))
        }
        return frames
    }
    fun bidirectional():List<SearchFrame> {
        val forward=ArrayDeque(listOf("S"));val backward=ArrayDeque(listOf("G"))
        val seenF=mutableSetOf("S");val seenB=mutableSetOf("G")
        val parentF=mutableMapOf("S" to "")
        val parentB=mutableMapOf("G" to "")
        val frames=mutableListOf<SearchFrame>()
        repeat(5){step->
            val fromStart=step%2==0
            val queue=if(fromStart)forward else backward
            if(queue.isEmpty())return@repeat
            val node=queue.removeFirst()
            val neighbors=if(fromStart)graph.getValue(node).map{it.first}else graph.filterValues{edges->edges.any{it.first==node}}.keys.toList()
            neighbors.filter{it !in if(fromStart)seenF else seenB}.forEach{neighbor->
                queue.addLast(neighbor);if(fromStart)seenF+=neighbor else seenB+=neighbor
                if(fromStart)parentF[neighbor]=node else parentB[neighbor]=node
            }
            val meeting=seenF.intersect(seenB).firstOrNull()
            fun startPath(end:String):List<String>{
                val path=mutableListOf<String>();var cursor=end
                while(cursor.isNotEmpty()){path+=cursor;cursor=parentF[cursor]?:""}
                return path.reversed()
            }
            fun endPath(start:String):List<String>{
                val path=mutableListOf<String>();var cursor=parentB[start]?:""
                while(cursor.isNotEmpty()){path+=cursor;cursor=parentB[cursor]?:""}
                return path
            }
            frames+=SearchFrame(node,seenF+seenB,(forward+backward).map{SearchCandidate(it,listOf(it),0.0)},
                if(meeting!=null)startPath(meeting)+endPath(meeting) else emptyList(),
                note="Start frontier ${seenF.joinToString()} | goal frontier ${seenB.joinToString()}")
            if(meeting!=null)return frames
        }
        return frames
    }
    fun minimax(leaves:List<Int>):Int=maxOf(minOf(leaves[0],leaves[1]),minOf(leaves[2],leaves[3]))
    fun alphaBeta(leaves:List<Int>):Pair<Int,Set<Int>> {
        var alpha=Int.MIN_VALUE
        var pruned=emptySet<Int>()
        for(child in 0..1){
            var beta=Int.MAX_VALUE
            for(leaf in child*2..child*2+1){
                beta=minOf(beta,leaves[leaf])
                if(beta<=alpha){pruned=((leaf+1)..(child*2+1)).toSet();break}
            }
            alpha=maxOf(alpha,beta)
        }
        return alpha to pruned
    }
    fun uct(value:Double,visits:Int,parentVisits:Int,exploration:Double=1.4)=
        value/visits.coerceAtLeast(1)+exploration*sqrt(ln(parentVisits.coerceAtLeast(2).toDouble())/visits.coerceAtLeast(1))
    fun hillFitness(x:Double)=1.6-(x+.5)*(x+.5)+.25*x
    fun localFitness(x:Double)=2-(x-.8)*(x-.8)+.3*kotlin.math.sin(5*x)
    fun hillPath(steps:Int):List<Pair<Double,Double>> {
        var x=-1.7
        val path=mutableListOf(x to hillFitness(x))
        repeat(steps){
            val candidates=listOf(x-.15,x+.15)
            val best=candidates.maxBy(::hillFitness)
            if(hillFitness(best)>path.last().second)x=best
            path+=x to hillFitness(x)
        }
        return path
    }
    fun localBeam(steps:Int,width:Int):List<Double>{
        var states=listOf(-1.8,-.4,1.4).take(width.coerceIn(1,3))
        repeat(steps){states=states.flatMap{listOf(it-.18,it+.18)}.sortedByDescending(::localFitness).take(width.coerceIn(1,3))}
        return states
    }
}
