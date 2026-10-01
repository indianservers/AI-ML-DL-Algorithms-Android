package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
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
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourData
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFourEngines
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.RatingData
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt
import kotlin.math.tanh

internal data class RecFactors(val users:List<List<Double>>,val items:List<List<Double>>,val loss:Double) {
    fun predict(user:Int,item:Int)=users[user].zip(items[item]).sumOf{(a,b)->a*b}
}
internal data class RecSvd(val singularValues:List<Double>,val reconstruction:List<List<Double>>)

internal object StageFourRecommendationEngine {
    val data:RatingData=PhaseFourData.ratings
    val features=listOf(listOf(1.0,.1,.2),listOf(.7,.3,.3),listOf(.2,1.0,.4),listOf(.1,.3,1.0),listOf(.2,.8,.8))
    fun popularity():List<Pair<String,Double>> = data.items.indices.map{i->data.items[i] to data.ratings.count{it[i]!=null}.toDouble()}
    fun profile(user:Int):List<Double> {
        val observed=data.items.indices.filter{data.ratings[user][it]!=null}
        val weights=observed.map{data.ratings[user][it]!!}
        return (0..2).map{f->observed.indices.sumOf{k->features[observed[k]][f]*weights[k]}/weights.sum().coerceAtLeast(1e-9)}
    }
    fun content(user:Int):List<Pair<String,Double>> = data.items.indices.map{i->data.items[i] to PhaseFourEngines.cosine(profile(user),features[i])}
    fun userNeighbors(user:Int,item:Int):List<Pair<String,Double>> = data.users.indices.filter{it!=user&&data.ratings[it][item]!=null}.map{other->
        data.users[other] to PhaseFourEngines.cosine(data.ratings[user].map{it?:0.0},data.ratings[other].map{it?:0.0})
    }.sortedByDescending{it.second}
    fun itemNeighbors(user:Int,item:Int):List<Pair<String,Double>> = data.items.indices.filter{it!=item&&data.ratings[user][it]!=null}.map{other->
        data.items[other] to PhaseFourEngines.cosine(data.ratings.map{it[item]?:0.0},data.ratings.map{it[other]?:0.0})
    }.sortedByDescending{it.second}
    fun factorize(epochs:Int,rank:Int=2):RecFactors {
        val u=Array(data.users.size){i->DoubleArray(rank){f->.4+.12*i+.07*f}}
        val v=Array(data.items.size){i->DoubleArray(rank){f->.5+.08*i-.04*f}}
        repeat(epochs.coerceIn(0,100)){
            data.users.indices.forEach{i->data.items.indices.forEach{j->
                val actual=data.ratings[i][j]?:return@forEach
                val estimate=(0 until rank).sumOf{u[i][it]*v[j][it]}
                val error=actual-estimate
                repeat(rank){f->val before=u[i][f];u[i][f]+=.025*(error*v[j][f]-.04*u[i][f]);v[j][f]+=.025*(error*before-.04*v[j][f])}
            }}
        }
        val loss=data.users.indices.sumOf{i->data.items.indices.sumOf{j->
            val target=data.ratings[i][j]?:return@sumOf 0.0
            (target-(0 until rank).sumOf{u[i][it]*v[j][it]}).let{it*it}
        }}
        return RecFactors(u.map{it.toList()},v.map{it.toList()},loss)
    }
    fun als(iterations:Int):RecFactors {
        val rank=2
        val u=Array(data.users.size){i->doubleArrayOf(.6+.1*i,.3+.04*i)}
        val v=Array(data.items.size){j->doubleArrayOf(.4+.08*j,.7-.05*j)}
        fun solve(vectors:List<DoubleArray>,targets:List<Double>):DoubleArray {
            var a=.2;var b=0.0;var c=.2;var d=0.0;var e=0.0
            vectors.indices.forEach{k->val x=vectors[k][0];val y=vectors[k][1];a+=x*x;b+=x*y;c+=y*y;d+=x*targets[k];e+=y*targets[k]}
            val det=(a*c-b*b).coerceAtLeast(1e-9)
            return doubleArrayOf((d*c-b*e)/det,(a*e-b*d)/det)
        }
        repeat(iterations.coerceIn(0,30)){
            data.users.indices.forEach{i->val js=data.items.indices.filter{data.ratings[i][it]!=null};u[i]=solve(js.map{v[it]},js.map{data.ratings[i][it]!!})}
            data.items.indices.forEach{j->val users=data.users.indices.filter{data.ratings[it][j]!=null};v[j]=solve(users.map{u[it]},users.map{data.ratings[it][j]!!})}
        }
        val loss=data.users.indices.sumOf{i->data.items.indices.sumOf{j->val actual=data.ratings[i][j]?:return@sumOf 0.0;(actual-u[i].zip(v[j]).sumOf{it.first*it.second}).let{it*it}}}
        return RecFactors(u.map{it.toList()},v.map{it.toList()},loss)
    }
    fun svd():RecSvd {
        val residual=Array(5){i->DoubleArray(5){j->data.ratings[i][j]?:0.0}}
        val reconstruction=Array(5){DoubleArray(5)}
        val values=mutableListOf<Double>()
        repeat(5){component->
            var right=DoubleArray(5){j->1.0/(j+component+1)}
            repeat(25){
                val left=DoubleArray(5){i->(0..4).sumOf{j->residual[i][j]*right[j]}}
                val updated=DoubleArray(5){j->(0..4).sumOf{i->residual[i][j]*left[i]}}
                val norm=sqrt(updated.sumOf{it*it}).coerceAtLeast(1e-9)
                right=DoubleArray(5){updated[it]/norm}
            }
            val left=DoubleArray(5){i->(0..4).sumOf{j->residual[i][j]*right[j]}}
            val sigma=sqrt(left.sumOf{it*it})
            values+=sigma
            if(sigma>1e-9)repeat(5){i->repeat(5){j->
                residual[i][j]-=left[i]*right[j]
                if(component<2)reconstruction[i][j]+=left[i]*right[j]
            }}
        }
        return RecSvd(values,reconstruction.map{it.toList()})
    }
    fun singularValues():List<Double> = svd().singularValues
    fun neuralScore(factors:RecFactors,user:Int,item:Int):Double {
        val a=factors.users[user];val b=factors.items[item]
        val hidden1=tanh(.8*a[0]*b[0]+.3*a[1]+.2*b[1])
        val hidden2=tanh(.4*a[1]*b[1]+.2*a[0]-.3*b[0])
        return 2.5+1.5*hidden1+hidden2
    }
    fun historyMatch(user:Int,item:Int):Double {
        val liked=data.items.indices.filter{(data.ratings[user][it]?:0.0)>=3.0}
        return liked.maxOfOrNull{PhaseFourEngines.cosine(features[it],features[item])}?:0.0
    }
    fun contextSignal(user:Int,item:Int):Double=if((user+item)%3==0) .15 else -.05
    fun deepScore(factors:RecFactors,user:Int,item:Int):Double =
        neuralScore(factors,user,item)+.3*PhaseFourEngines.cosine(profile(user),features[item])+
            .15*historyMatch(user,item)+contextSignal(user,item)
}

@Composable
internal fun StageFourRecommendationScreen(topic:LearnTopic){
    val name=topic.title
    var user by remember(topic.id){mutableIntStateOf(0)}
    var item by remember(topic.id){mutableIntStateOf(2)}
    var step by remember(topic.id){mutableIntStateOf(0)}
    val data=StageFourRecommendationEngine.data
    val factor=remember(step){StageFourRecommendationEngine.factorize(step*3,2)}
    val als=remember(step){StageFourRecommendationEngine.als(step+1)}
    val svd=remember{StageFourRecommendationEngine.svd()}
    val ranked:List<Pair<String,Double>> = when(name){
        "Popularity-Based Recommendation"->StageFourRecommendationEngine.popularity()
        "Content-Based Filtering"->StageFourRecommendationEngine.content(user)
        "User-Based Collaborative Filtering"->StageFourRecommendationEngine.userNeighbors(user,item)
        "Item-Based Collaborative Filtering"->StageFourRecommendationEngine.itemNeighbors(user,item)
        "Matrix Factorization"->data.items.indices.map{data.items[it] to factor.predict(user,it)}
        "SVD"->svd.singularValues.mapIndexed{i,v->"σ${i+1}" to v}
        "Alternating Least Squares"->data.items.indices.map{data.items[it] to als.predict(user,it)}
        "Neural Collaborative Filtering"->data.items.indices.map{data.items[it] to StageFourRecommendationEngine.neuralScore(factor,user,it)}
        "Deep Recommendation Systems"->data.items.indices.map{data.items[it] to StageFourRecommendationEngine.deepScore(factor,user,it)}
        else->error("Unregistered recommender $name")
    }
    val displayedRanked=ranked.sortedByDescending{it.second}.take(5)
    val flow=when(name){
        "Popularity-Based Recommendation"->listOf("Interaction counts","Sort","Same rank for everyone")
        "Content-Based Filtering"->listOf("Liked item features","User profile","Cosine with items")
        "User-Based Collaborative Filtering"->listOf("Target user","Similar users","Weighted ratings")
        "Item-Based Collaborative Filtering"->listOf("Liked items","Similar items","Weighted ratings")
        "Matrix Factorization"->listOf("Sparse rating R","User factors U","Item factors V","U·Vᵀ")
        "SVD"->listOf("Filled rating matrix","U","Singular values Σ","Vᵀ","Rank-k reconstruction")
        "Alternating Least Squares"->listOf("Fix items → solve users","Fix users → solve items","Repeat")
        "Neural Collaborative Filtering"->listOf("User/item embeddings","Nonlinear hidden units","Score")
        else->listOf("Candidate items","User + item embeddings","History + context","Nonlinear ranking","Top items")
    }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp),verticalArrangement=Arrangement.spacedBy(12.dp)){
        item{SectionTitle(name,when(name){
            "Popularity-Based Recommendation"->"Rank items by aggregate interaction counts"
            "Content-Based Filtering"->"Compare item features with a user's liked-item profile"
            "User-Based Collaborative Filtering"->"Weight ratings from users with similar histories"
            "Item-Based Collaborative Filtering"->"Weight similar items the user has already rated"
            "Matrix Factorization"->"Fit user and item factors to observed ratings"
            "SVD"->"Inspect singular values and a low-rank rating reconstruction"
            "Alternating Least Squares"->"Alternate solving user factors and item factors"
            "Neural Collaborative Filtering"->"Pass user and item embeddings through nonlinear units"
            "Deep Recommendation Systems"->"Combine embeddings, history, and context to rank items"
            else->"Inspect the recommendation score"
        })}
        item{FourPanel("Scoring mechanism"){
            FourFlow(flow,step%flow.size)
            FourBars(displayedRanked,displayedRanked.indexOfFirst{it.first==data.items[item]})
            when(name){
                "Popularity-Based Recommendation"->FourText("Counts are shared by all users; no user profile enters the score.")
                "Content-Based Filtering"->{FourText("User feature profile: "+StageFourRecommendationEngine.profile(user).joinToString{ "%.2f".format(it) });FourMetric("Selected cosine","%.3f".format(StageFourRecommendationEngine.content(user)[item].second))}
                "User-Based Collaborative Filtering"->{FourMetric("Weighted neighbor prediction","%.3f".format(PhaseFourEngines.userCf(data,user,item).score));FourText("Nearest users are highlighted above; similarity weights their ratings.")}
                "Item-Based Collaborative Filtering"->{FourMetric("Weighted item prediction","%.3f".format(PhaseFourEngines.itemCf(data,user,item).score));FourText("Only items already rated by this user provide the neighborhood.")}
                "Matrix Factorization","Alternating Least Squares"->{val f=if(name=="Matrix Factorization")factor else als;FourMetric("Missing-rating prediction","%.3f".format(f.predict(user,item)),LabGreen);FourMetric("Known-rating squared error","%.3f".format(f.loss));FourText(if(name=="Alternating Least Squares")"User and item factors are solved in alternating least-squares passes." else "Observed ratings update latent user and item vectors by gradient descent.")}
                "SVD"->{
                    FourText("R ≈ U Σ Vᵀ; the bars are singular values of the zero-filled teaching matrix.")
                    FourGrid(svd.reconstruction)
                    FourMetric("Rank-2 predicted rating","%.3f".format(svd.reconstruction[user][item]),LabGreen)
                    FourMetric("Retained energy (rank 2)","%.1f%%".format(100*svd.singularValues.take(2).sumOf{it*it}/svd.singularValues.sumOf{it*it}))
                }
                "Neural Collaborative Filtering"->FourText("An embedding interaction passes through tanh hidden units; score is not a dot product alone.")
                else->{
                    FourMetric("History similarity","%.3f".format(StageFourRecommendationEngine.historyMatch(user,item)))
                    FourMetric("Session context (illustrative)","%+.2f".format(StageFourRecommendationEngine.contextSignal(user,item)))
                    FourMetric("Ranked candidate score","%.3f".format(StageFourRecommendationEngine.deepScore(factor,user,item)),LabGreen)
                    FourText("Embeddings, content, observed history and a synthetic session signal contribute to candidate ranking.")
                }
            }
        }}
        item{FourPanel("User–item rating matrix"){
            FourGrid(data.ratings.map{row->row.map{it?:0.0}})
            FourText("0 marks a missing rating. Select a user and target item below.")
        }}
        item{FourPanel("Choose evidence and step"){
            FourSteps(step,12){step=it}
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(4.dp)){
                data.users.forEachIndexed{i,label->SegmentedOption(label,user==i,Modifier.weight(1f)){user=i}}
            }
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(4.dp)){
                data.items.forEachIndexed{i,label->SegmentedOption(label.takeLast(1),item==i,Modifier.weight(1f)){item=i}}
            }
        }}
    }
}
