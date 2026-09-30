package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/** Visual vocabulary used by the Learn catalog. Every topic resolves to a semantic icon. */
internal enum class AlgorithmIconKind {
    Regression, Classification, Neighbours, Probability, Tree, Forest, Boosting,
    Clustering, Density, Projection, Association, Anomaly, SemiSupervised,
    NeuralNetwork, Convolution, Sequence, Attention, Autoencoder, Generative, Graph,
    Reinforcement, Language, Vision, TimeSeries, Bayesian, Optimization, Evolution,
    Recommendation, Explanation
}

internal fun algorithmIconKind(topic: LearnTopic): AlgorithmIconKind =
    algorithmIconKind(topic.title, topic.section, topic.domain)

private fun algorithmIconKind(title: String, section: String, domain: String): AlgorithmIconKind {
    val name = title.lowercase()
    return when {
        "nearest neighbor" in name || "nearest neighbour" in name || "knn" in name -> AlgorithmIconKind.Neighbours
        "naive bayes" in name || "gaussian process" in name || "mixture" in name -> AlgorithmIconKind.Probability
        "random forest" in name || "extra trees" in name -> AlgorithmIconKind.Forest
        "tree" in name || name in listOf("birch", "isolation forest") -> AlgorithmIconKind.Tree
        "boost" in name || name in listOf("adaboost", "xgboost", "lightgbm", "catboost") -> AlgorithmIconKind.Boosting
        name in listOf("dbscan", "hdbscan", "optics", "mean shift", "local outlier factor") -> AlgorithmIconKind.Density
        section == "Clustering" -> AlgorithmIconKind.Clustering
        section == "Dimensionality Reduction" || "embedding" in name || name in listOf("svd", "pca", "umap", "t-sne") -> AlgorithmIconKind.Projection
        section == "Association Learning" -> AlgorithmIconKind.Association
        section == "Anomaly Detection" || "outlier" in name || "one-class" in name -> AlgorithmIconKind.Anomaly
        domain == "Semi-Supervised Learning" -> AlgorithmIconKind.SemiSupervised
        "convolution" in name || name in listOf("cnn", "lenet", "alexnet", "vgg", "resnet", "densenet", "mobilenet", "efficientnet", "convnext") -> AlgorithmIconKind.Convolution
        "attention" in name || "transformer" in name || name in listOf("bert", "gpt", "t5", "swin transformer") -> AlgorithmIconKind.Attention
        "rnn" in name || "lstm" in name || "gru" in name || "sequence" in name || "encoder-decoder" in name -> AlgorithmIconKind.Sequence
        "autoencoder" in name -> AlgorithmIconKind.Autoencoder
        "gan" in name || "diffusion" in name || "generative" in name -> AlgorithmIconKind.Generative
        "graph" in name -> AlgorithmIconKind.Graph
        domain == "Deep Learning" -> AlgorithmIconKind.NeuralNetwork
        domain == "Reinforcement Learning" -> AlgorithmIconKind.Reinforcement
        domain == "Natural Language Processing" -> AlgorithmIconKind.Language
        domain == "Computer Vision" -> AlgorithmIconKind.Vision
        domain == "Time-Series Algorithms" -> AlgorithmIconKind.TimeSeries
        domain == "Probabilistic & Bayesian Learning" -> AlgorithmIconKind.Bayesian
        domain == "Optimization Algorithms" -> AlgorithmIconKind.Optimization
        domain == "Evolutionary Algorithms" -> AlgorithmIconKind.Evolution
        domain == "Recommendation Algorithms" -> AlgorithmIconKind.Recommendation
        domain == "Explainable AI" -> AlgorithmIconKind.Explanation
        section == "Regression" || "regression" in name -> AlgorithmIconKind.Regression
        section == "Classification" -> AlgorithmIconKind.Classification
        else -> AlgorithmIconKind.Classification
    }
}

@Composable
internal fun AlgorithmIcon(
    topic: LearnTopic,
    modifier: Modifier = Modifier,
    size: Dp = 40.dp
) {
    val color = Color(topic.accent)
    Box(
        modifier.size(size).background(color.copy(alpha = .15f), RoundedCornerShape(size * .22f)),
        contentAlignment = Alignment.Center
    ) {
        Canvas(Modifier.size(size * .64f)) { drawAlgorithmIcon(algorithmIconKind(topic), color) }
    }
}

@Composable
internal fun AlgorithmIcon(
    title: String,
    section: String,
    domain: String,
    accent: Long,
    modifier: Modifier = Modifier,
    size: Dp = 40.dp
) {
    val color = Color(accent)
    Box(
        modifier.size(size).background(color.copy(alpha = .15f), RoundedCornerShape(size * .22f)),
        contentAlignment = Alignment.Center
    ) {
        Canvas(Modifier.size(size * .64f)) { drawAlgorithmIcon(algorithmIconKind(title, section, domain), color) }
    }
}

private fun DrawScope.drawAlgorithmIcon(kind: AlgorithmIconKind, color: Color) {
    val w = size.width
    val h = size.height
    val stroke = (w * .09f).coerceAtLeast(1.6f)
    fun line(a: Offset, b: Offset, width: Float = stroke) =
        drawLine(color, a, b, width, StrokeCap.Round)
    fun node(x: Float, y: Float, r: Float = w * .09f, filled: Boolean = true) =
        if (filled) drawCircle(color, r, Offset(w * x, h * y))
        else drawCircle(color, r, Offset(w * x, h * y), style = Stroke(stroke * .7f))

    when (kind) {
        AlgorithmIconKind.Regression -> {
            line(Offset(w * .12f, h * .82f), Offset(w * .88f, h * .2f))
            listOf(.2f to .69f, .38f to .64f, .57f to .38f, .78f to .3f).forEach { (x, y) -> node(x, y, w * .055f) }
        }
        AlgorithmIconKind.Classification -> {
            line(Offset(w * .48f, h * .12f), Offset(w * .55f, h * .88f))
            listOf(.2f to .28f, .3f to .52f, .23f to .74f).forEach { (x, y) -> node(x, y, w * .065f) }
            listOf(.73f to .25f, .78f to .5f, .72f to .75f).forEach { (x, y) -> node(x, y, w * .075f, false) }
        }
        AlgorithmIconKind.Neighbours -> {
            node(.5f, .5f, w * .11f, false)
            listOf(.18f to .28f, .75f to .2f, .82f to .63f, .28f to .78f).forEach { (x, y) ->
                line(Offset(w * .5f, h * .5f), Offset(w * x, h * y), stroke * .55f); node(x, y, w * .06f)
            }
        }
        AlgorithmIconKind.Probability, AlgorithmIconKind.Bayesian -> {
            val path = Path().apply { moveTo(w * .1f, h * .75f); cubicTo(w * .3f, h * .75f, w * .3f, h * .2f, w * .5f, h * .2f); cubicTo(w * .7f, h * .2f, w * .7f, h * .75f, w * .9f, h * .75f) }
            drawPath(path, color, style = Stroke(stroke, cap = StrokeCap.Round)); line(Offset(w * .1f, h * .78f), Offset(w * .9f, h * .78f), stroke * .65f)
        }
        AlgorithmIconKind.Tree, AlgorithmIconKind.Forest, AlgorithmIconKind.Boosting -> {
            val roots = if (kind == AlgorithmIconKind.Forest) listOf(.28f, .5f, .72f) else listOf(.5f)
            roots.forEach { root ->
                line(Offset(w * root, h * .2f), Offset(w * root, h * .48f), stroke * .7f)
                line(Offset(w * root, h * .48f), Offset(w * (root - .14f), h * .75f), stroke * .7f)
                line(Offset(w * root, h * .48f), Offset(w * (root + .14f), h * .75f), stroke * .7f)
                node(root, .18f, w * .06f); node(root - .14f, .78f, w * .055f); node(root + .14f, .78f, w * .055f)
            }
            if (kind == AlgorithmIconKind.Boosting) line(Offset(w * .18f, h * .88f), Offset(w * .85f, h * .16f), stroke * .65f)
        }
        AlgorithmIconKind.Clustering, AlgorithmIconKind.Density -> {
            val centers = listOf(.3f to .32f, .7f to .67f)
            centers.forEach { (cx, cy) ->
                if (kind == AlgorithmIconKind.Density) drawCircle(color.copy(alpha = .35f), w * .22f, Offset(w * cx, h * cy), style = Stroke(stroke * .55f))
                listOf(-.1f to -.08f, .08f to -.05f, -.04f to .11f).forEach { (dx, dy) -> node(cx + dx, cy + dy, w * .055f) }
            }
        }
        AlgorithmIconKind.Projection -> {
            line(Offset(w * .12f, h * .8f), Offset(w * .86f, h * .25f))
            listOf(.24f to .3f, .4f to .55f, .64f to .43f, .78f to .72f).forEach { (x, y) ->
                node(x, y, w * .05f); line(Offset(w * x, h * y), Offset(w * (x + .08f), h * (y + .11f)), stroke * .45f)
            }
        }
        AlgorithmIconKind.Association -> {
            listOf(.2f to .3f, .5f to .2f, .78f to .42f, .35f to .75f, .72f to .78f).forEach { (x, y) -> node(x, y, w * .07f, false) }
            listOf(.2f to .3f to (.5f to .2f), .5f to .2f to (.78f to .42f), .2f to .3f to (.35f to .75f), .35f to .75f to (.72f to .78f)).forEach { edge ->
                val a = edge.first; val b = edge.second; line(Offset(w * a.first, h * a.second), Offset(w * b.first, h * b.second), stroke * .55f)
            }
        }
        AlgorithmIconKind.Anomaly -> {
            drawCircle(color, w * .36f, Offset(w * .5f, h * .5f), style = Stroke(stroke * .65f)); drawCircle(color, w * .2f, Offset(w * .5f, h * .5f), style = Stroke(stroke * .65f)); node(.5f, .5f, w * .06f); node(.88f, .14f, w * .08f)
        }
        AlgorithmIconKind.SemiSupervised -> {
            listOf(.18f to .3f, .38f to .68f, .68f to .25f, .82f to .7f).forEachIndexed { i, (x, y) -> node(x, y, w * .08f, i < 2) }
            line(Offset(w * .2f, h * .5f), Offset(w * .8f, h * .5f), stroke * .6f)
        }
        AlgorithmIconKind.NeuralNetwork, AlgorithmIconKind.Graph -> {
            val left = listOf(.2f to .25f, .2f to .75f); val middle = listOf(.5f to .18f, .5f to .5f, .5f to .82f); val right = listOf(.82f to .5f)
            (left.flatMap { a -> middle.map { a to it } } + middle.flatMap { a -> right.map { a to it } }).forEach { (a, b) -> line(Offset(w*a.first,h*a.second), Offset(w*b.first,h*b.second), stroke*.4f) }
            (left + middle + right).forEach { (x, y) -> node(x, y, w * .06f) }
        }
        AlgorithmIconKind.Convolution, AlgorithmIconKind.Vision -> {
            drawRect(color, Offset(w*.12f,h*.18f), Size(w*.45f,h*.45f), style=Stroke(stroke*.65f)); drawRect(color, Offset(w*.42f,h*.4f), Size(w*.45f,h*.45f), style=Stroke(stroke*.65f));
            line(Offset(w*.57f,h*.3f), Offset(w*.85f,h*.45f), stroke*.45f); line(Offset(w*.57f,h*.62f), Offset(w*.85f,h*.78f), stroke*.45f)
        }
        AlgorithmIconKind.Sequence, AlgorithmIconKind.TimeSeries -> {
            val points = listOf(.1f to .62f, .28f to .36f, .46f to .7f, .65f to .25f, .9f to .45f)
            points.zipWithNext().forEach { (a,b) -> line(Offset(w*a.first,h*a.second),Offset(w*b.first,h*b.second),stroke*.75f) }; points.forEach { (x,y)->node(x,y,w*.045f) }
        }
        AlgorithmIconKind.Attention -> {
            val left = listOf(.17f to .2f, .17f to .5f, .17f to .8f); val right = listOf(.83f to .25f, .83f to .75f)
            left.flatMap { a -> right.map { a to it } }.forEachIndexed { i,(a,b)-> line(Offset(w*a.first,h*a.second),Offset(w*b.first,h*b.second),stroke*(if(i==2) .9f else .35f)) }
            (left+right).forEach { (x,y)->node(x,y,w*.06f) }
        }
        AlgorithmIconKind.Autoencoder -> {
            val xs = listOf(.12f,.36f,.5f,.64f,.88f); val counts = listOf(3,2,1,2,3)
            xs.forEachIndexed { i,x -> repeat(counts[i]) { j -> node(x,(j+1f)/(counts[i]+1f),w*.045f) } }
            line(Offset(w*.15f,h*.18f),Offset(w*.5f,h*.5f),stroke*.45f); line(Offset(w*.5f,h*.5f),Offset(w*.85f,h*.18f),stroke*.45f)
        }
        AlgorithmIconKind.Generative -> {
            drawCircle(color, w*.27f, Offset(w*.5f,h*.5f), style=Stroke(stroke*.6f)); drawCircle(color,w*.1f,Offset(w*.5f,h*.5f));
            repeat(6) { i -> val a=i*Math.PI.toFloat()/3f; line(Offset(w*.5f,h*.5f),Offset(w*(.5f+.4f*kotlin.math.cos(a)),h*(.5f+.4f*kotlin.math.sin(a))),stroke*.45f) }
        }
        AlgorithmIconKind.Reinforcement -> {
            drawArc(color,-55f,280f,false,Offset(w*.12f,h*.12f),Size(w*.76f,h*.76f),style=Stroke(stroke,cap=StrokeCap.Round));
            val p=Path().apply{moveTo(w*.82f,h*.12f);lineTo(w*.9f,h*.34f);lineTo(w*.67f,h*.29f);close()};drawPath(p,color); node(.5f,.5f,w*.1f,false)
        }
        AlgorithmIconKind.Language -> {
            drawRoundRect(color,Offset(w*.1f,h*.18f),Size(w*.8f,h*.58f),cornerRadius=androidx.compose.ui.geometry.CornerRadius(w*.1f),style=Stroke(stroke*.7f));
            val tail=Path().apply{moveTo(w*.3f,h*.74f);lineTo(w*.24f,h*.9f);lineTo(w*.47f,h*.75f)};drawPath(tail,color,style=Stroke(stroke*.7f,cap=StrokeCap.Round)); line(Offset(w*.27f,h*.38f),Offset(w*.73f,h*.38f),stroke*.55f);line(Offset(w*.27f,h*.56f),Offset(w*.6f,h*.56f),stroke*.55f)
        }
        AlgorithmIconKind.Optimization -> {
            val path=Path().apply{moveTo(w*.1f,h*.25f);cubicTo(w*.28f,h*.3f,w*.28f,h*.82f,w*.5f,h*.82f);cubicTo(w*.72f,h*.82f,w*.72f,h*.3f,w*.9f,h*.25f)};drawPath(path,color,style=Stroke(stroke,cap=StrokeCap.Round));node(.5f,.82f,w*.065f)
        }
        AlgorithmIconKind.Evolution -> {
            val path=Path().apply{moveTo(w*.2f,h*.85f);cubicTo(w*.25f,h*.5f,w*.7f,h*.62f,w*.78f,h*.18f);moveTo(w*.25f,h*.62f);cubicTo(w*.52f,h*.55f,w*.42f,h*.26f,w*.68f,h*.18f)};drawPath(path,color,style=Stroke(stroke*.75f,cap=StrokeCap.Round));node(.2f,.85f,w*.06f);node(.78f,.18f,w*.06f);node(.68f,.18f,w*.06f)
        }
        AlgorithmIconKind.Recommendation -> {
            val star=Path(); repeat(10){i->val a=(-Math.PI/2+i*Math.PI/5).toFloat();val r=if(i%2==0)w*.38f else w*.16f;val p=Offset(w*.5f+r*kotlin.math.cos(a),h*.5f+r*kotlin.math.sin(a));if(i==0)star.moveTo(p.x,p.y)else star.lineTo(p.x,p.y)};star.close();drawPath(star,color,style=Stroke(stroke*.7f,cap=StrokeCap.Round))
        }
        AlgorithmIconKind.Explanation -> {
            drawCircle(color,w*.32f,Offset(w*.45f,h*.43f),style=Stroke(stroke*.7f));line(Offset(w*.68f,h*.68f),Offset(w*.9f,h*.9f),stroke);node(.45f,.43f,w*.055f);line(Offset(w*.45f,h*.2f),Offset(w*.45f,h*.31f),stroke*.65f)
        }
    }
}
