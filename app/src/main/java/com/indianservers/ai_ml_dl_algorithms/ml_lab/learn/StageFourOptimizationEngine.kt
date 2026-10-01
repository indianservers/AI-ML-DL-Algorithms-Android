package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

internal data class OptPoint(val x:Double,val y:Double,val loss:Double,val gx:Double,val gy:Double,val rate:Double,
                             val firstMoment:Double=0.0,val secondMoment:Double=0.0,val memory:Int=0)

internal object StageFourOptimizationEngine {
    fun loss(x:Double,y:Double):Double=(x-1).pow(2)+3*(y+.4*x).pow(2)
    fun gradient(x:Double,y:Double):Pair<Double,Double> =
        (2*(x-1)+2.4*(y+.4*x)) to (6*(y+.4*x))
    fun hessian():List<List<Double>> = listOf(listOf(2.96,2.4),listOf(2.4,6.0))
    fun schedule(kind:Int,step:Int,base:Double):Double = when(kind) {
        0->base*(.5).pow(step/4)
        1->base*exp(-step*.12)
        2->base*.5*(1+cos(PI*step/14))
        3->base*((step+1)/4.0).coerceAtMost(1.0)
        else->base*(if(step<7)(step+1)/8.0 else (15-step)/8.0).coerceAtLeast(.08)
    }
    private fun dot(a:Pair<Double,Double>,b:Pair<Double,Double>)=a.first*b.first+a.second*b.second
    private fun add(a:Pair<Double,Double>,b:Pair<Double,Double>)=a.first+b.first to a.second+b.second
    private fun scale(a:Pair<Double,Double>,s:Double)=a.first*s to a.second*s
    private fun sub(a:Pair<Double,Double>,b:Pair<Double,Double>)=a.first-b.first to a.second-b.second

    fun path(name:String,rate:Double,steps:Int,schedule:Int=0):List<OptPoint> {
        var x=-1.6;var y=1.6
        var vx=0.0;var vy=0.0;var ax=0.0;var ay=0.0;var mx=0.0;var my=0.0
        var h00=1.0;var h01=0.0;var h10=0.0;var h11=1.0
        val memory=ArrayDeque<Pair<Pair<Double,Double>,Pair<Double,Double>>>()
        val result=mutableListOf<OptPoint>()
        repeat(steps.coerceIn(0,50)+1){t->
            val grad=gradient(x,y)
            val lr=if(name=="Learning Rate Scheduling")schedule(schedule,t,rate)else rate
            result+=OptPoint(x,y,loss(x,y),grad.first,grad.second,lr,
                if(name=="Momentum"||name=="Nesterov Momentum")vx else mx,ax,memory.size)
            if(t==steps)return@repeat
            val sample=if(name=="Stochastic Gradient Descent") (t*5+1)%8 else t%8
            val noise=listOf(-.36,.12,.27,-.18,.33,-.27,.09,0.0)[sample]
            val g=when(name){
                "Stochastic Gradient Descent"->grad.first+noise to grad.second-noise*.6
                "Mini-Batch Gradient Descent"->grad.first+noise*.25 to grad.second-noise*.15
                else->grad
            }
            val previous=x to y
            val direction:Pair<Double,Double> = when(name) {
                "Momentum"->{vx=.8*vx+g.first;vy=.8*vy+g.second;vx to vy}
                "Nesterov Momentum"->{val look=gradient(x-.8*rate*vx,y-.8*rate*vy);vx=.8*vx+look.first;vy=.8*vy+look.second;vx to vy}
                "AdaGrad"->{ax+=g.first*g.first;ay+=g.second*g.second;g.first/sqrt(ax+1e-8) to g.second/sqrt(ay+1e-8)}
                "RMSProp"->{ax=.9*ax+.1*g.first*g.first;ay=.9*ay+.1*g.second*g.second;g.first/sqrt(ax+1e-8) to g.second/sqrt(ay+1e-8)}
                "Adam","AdamW","Nadam"->{
                    mx=.9*mx+.1*g.first;my=.9*my+.1*g.second
                    ax=.999*ax+.001*g.first*g.first;ay=.999*ay+.001*g.second*g.second
                    val mhatX=mx/(1-.9.pow(t+1));val mhatY=my/(1-.9.pow(t+1))
                    val vhatX=ax/(1-.999.pow(t+1));val vhatY=ay/(1-.999.pow(t+1))
                    if(name=="Nadam"){
                        val nx=.9*mhatX+(.1*g.first)/(1-.9.pow(t+1))
                        val ny=.9*mhatY+(.1*g.second)/(1-.9.pow(t+1))
                        nx/(sqrt(vhatX)+1e-8) to ny/(sqrt(vhatY)+1e-8)
                    } else mhatX/(sqrt(vhatX)+1e-8) to mhatY/(sqrt(vhatY)+1e-8)
                }
                "Coordinate Descent"->if(t%2==0)g.first to 0.0 else 0.0 to g.second
                "Newton's Method"->{
                    val determinant=2.96*6-2.4*2.4
                    ((6*g.first-2.4*g.second)/determinant) to ((-2.4*g.first+2.96*g.second)/determinant)
                }
                "Quasi-Newton / BFGS"->(h00*g.first+h01*g.second) to (h10*g.first+h11*g.second)
                "L-BFGS"->{
                    var q=g
                    val alphas=mutableListOf<Double>()
                    memory.toList().asReversed().forEach{(s,yy)->
                        val rho=1/dot(s,yy).coerceAtLeast(1e-9)
                        val a=rho*dot(s,q);alphas+=a;q=sub(q,scale(yy,a))
                    }
                    val last=memory.lastOrNull()
                    val initial=if(last==null)1.0 else dot(last.first,last.second)/dot(last.second,last.second).coerceAtLeast(1e-9)
                    var r=scale(q,initial)
                    memory.toList().forEachIndexed{i,(s,yy)->
                        val rho=1/dot(s,yy).coerceAtLeast(1e-9)
                        val a=alphas[alphas.lastIndex-i]
                        r=add(r,scale(s,a-rho*dot(yy,r)))
                    }
                    r
                }
                else->g
            }
            val stepRate=if(name=="Newton's Method")1.0 else lr
            x-=stepRate*direction.first;y-=stepRate*direction.second
            if(name=="AdamW"){x-=lr*.03*previous.first;y-=lr*.03*previous.second}
            if(name=="Quasi-Newton / BFGS"||name=="L-BFGS"){
                val s=sub(x to y,previous)
                val yy=sub(gradient(x,y),grad)
                val curvature=dot(s,yy)
                if(curvature>1e-8){
                    if(name=="L-BFGS"){
                        memory.addLast(s to yy)
                        if(memory.size>3)memory.removeFirst()
                    } else {
                        val rho=1/curvature
                        val hs0=h00*yy.first+h01*yy.second
                        val hs1=h10*yy.first+h11*yy.second
                        val yHy=yy.first*hs0+yy.second*hs1
                        val factor=(1+yHy*rho)*rho
                        val n00=h00+factor*s.first*s.first-rho*(s.first*hs0+hs0*s.first)
                        val n01=h01+factor*s.first*s.second-rho*(s.first*hs1+hs0*s.second)
                        val n10=h10+factor*s.second*s.first-rho*(s.second*hs0+hs1*s.first)
                        val n11=h11+factor*s.second*s.second-rho*(s.second*hs1+hs1*s.second)
                        h00=n00;h01=n01;h10=n10;h11=n11
                    }
                }
            }
        }
        return result
    }
}
