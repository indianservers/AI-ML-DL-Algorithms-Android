package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class PhaseFourAlgorithmKind(val displayName: String, val family: String) {
    IsolationForest("Isolation Forest", "Anomaly Detection"),
    Lof("Local Outlier Factor - LOF", "Anomaly Detection"),
    OneClassSvm("One-Class SVM", "Anomaly Detection"),
    EllipticEnvelope("Elliptic Envelope / Robust Covariance", "Anomaly Detection"),
    ZScore("Statistical Z-Score Outlier Detection", "Anomaly Detection"),
    Iqr("IQR-Based Outlier Detection", "Anomaly Detection"),
    Apriori("Apriori", "Association Rule Mining"),
    FpGrowth("FP-Growth", "Association Rule Mining"),
    Eclat("ECLAT", "Association Rule Mining"),
    AssociationRules("Association Rule Generation", "Association Rule Mining"),
    Popularity("Popularity-Based Recommendation", "Recommendation Systems"),
    ContentBased("Content-Based Filtering", "Recommendation Systems"),
    UserCf("User-Based Collaborative Filtering", "Recommendation Systems"),
    ItemCf("Item-Based Collaborative Filtering", "Recommendation Systems"),
    MatrixFactorization("Matrix Factorization", "Recommendation Systems"),
    SvdRecommendation("SVD-Based Recommendation", "Recommendation Systems"),
    Als("Alternating Least Squares - ALS", "Recommendation Systems"),
    NeuralCf("Neural Collaborative Filtering Concepts", "Recommendation Systems"),
    BayesTheorem("Bayes Theorem", "Probabilistic / Bayesian Learning"),
    BayesianInference("Bayesian Inference", "Probabilistic / Bayesian Learning"),
    Mle("Maximum Likelihood Estimation - MLE", "Probabilistic / Bayesian Learning"),
    Map("Maximum A Posteriori - MAP", "Probabilistic / Bayesian Learning"),
    BayesianLinearRegression("Bayesian Linear Regression", "Probabilistic / Bayesian Learning"),
    BayesianNetworks("Bayesian Networks", "Probabilistic / Bayesian Learning"),
    Hmm("Hidden Markov Models", "Probabilistic / Bayesian Learning"),
    GaussianProcesses("Gaussian Processes", "Probabilistic / Bayesian Learning"),
    Mcmc("Markov Chain Monte Carlo Concepts", "Probabilistic / Bayesian Learning"),
    MetropolisHastings("Metropolis-Hastings", "Probabilistic / Bayesian Learning"),
    GibbsSampling("Gibbs Sampling", "Probabilistic / Bayesian Learning"),
    VariationalInference("Variational Inference Concepts", "Probabilistic / Bayesian Learning")
}

enum class AnomalyPreset(val label: String) {
    SingleOutlier("Single obvious outlier"),
    MultipleOutliers("Multiple isolated outliers"),
    DenseDistant("Dense cluster + distant point"),
    TwoClustersLocal("Two clusters + local anomaly"),
    GlobalLocal("Global vs local anomaly"),
    Elliptical("Elliptical Gaussian data"),
    Curved("Non-Gaussian curved data"),
    HighDim("High-dimensional synthetic anomaly"),
    Clustered("Clustered anomalies")
}

data class AnomalyPoint(val x: Double, val y: Double, val hiddenAnomaly: Boolean = false, val score: Double = 0.0)
data class IsolationPath(val tree: Int, val pathLength: Int, val splits: List<String>)
data class IsolationState(val scores: List<Double>, val selectedPaths: List<IsolationPath>, val threshold: Double)
data class LofState(val neighbors: List<Int>, val kDistance: Double, val localReachabilityDensity: Double, val neighborDensity: Double, val lof: Double)
data class EnvelopeState(val centerX: Double, val centerY: Double, val varX: Double, val varY: Double, val threshold: Double, val distances: List<Double>)
data class Basket(val id: String, val items: Set<String>)
data class ItemsetLevel(val level: Int, val candidates: List<Set<String>>, val frequent: List<Set<String>>)
data class RuleMetric(val antecedent: Set<String>, val consequent: Set<String>, val support: Double, val confidence: Double, val lift: Double)
data class FpNode(val item: String, val count: Int, val children: List<FpNode>)
data class RatingData(val users: List<String>, val items: List<String>, val ratings: List<List<Double?>>)
data class Recommendation(val item: String, val score: Double, val explanation: String)
data class FactorState(val userFactors: List<List<Double>>, val itemFactors: List<List<Double>>, val prediction: Double, val error: Double)
data class BayesState(val prior: Double, val sensitivity: Double, val specificity: Double, val posterior: Double, val truePositive: Double, val falsePositive: Double)
data class BetaState(val alpha: Double, val beta: Double, val mean: Double)
data class HmmState(val forward: List<Map<String, Double>>, val viterbi: List<String>, val probability: Double)
data class GpState(val xs: List<Double>, val mean: List<Double>, val variance: List<Double>)
data class McmcStep(val current: Double, val proposal: Double, val currentDensity: Double, val proposalDensity: Double, val ratio: Double, val u: Double, val accepted: Boolean)
data class GibbsState(val path: List<Pair<Double, Double>>, val lastMove: String)
data class VariationalState(val targetMean: Double, val approximateMean: Double, val approximateVariance: Double, val elboProxy: Double)

object PhaseFourTopicMatcher {
    fun kindFor(title: String): PhaseFourAlgorithmKind? = when (title) {
        "Isolation Forest" -> PhaseFourAlgorithmKind.IsolationForest
        "Local Outlier Factor" -> PhaseFourAlgorithmKind.Lof
        "One-Class SVM" -> PhaseFourAlgorithmKind.OneClassSvm
        "Elliptic Envelope" -> PhaseFourAlgorithmKind.EllipticEnvelope
        "Statistical Outlier Detection" -> PhaseFourAlgorithmKind.ZScore
        "Apriori" -> PhaseFourAlgorithmKind.Apriori
        "FP-Growth" -> PhaseFourAlgorithmKind.FpGrowth
        "ECLAT" -> PhaseFourAlgorithmKind.Eclat
        "Association Rule Mining" -> PhaseFourAlgorithmKind.AssociationRules
        "Popularity-Based Recommendation" -> PhaseFourAlgorithmKind.Popularity
        "Content-Based Filtering" -> PhaseFourAlgorithmKind.ContentBased
        "User-Based Collaborative Filtering" -> PhaseFourAlgorithmKind.UserCf
        "Item-Based Collaborative Filtering" -> PhaseFourAlgorithmKind.ItemCf
        "Matrix Factorization" -> PhaseFourAlgorithmKind.MatrixFactorization
        "SVD" -> PhaseFourAlgorithmKind.SvdRecommendation
        "Alternating Least Squares" -> PhaseFourAlgorithmKind.Als
        "Neural Collaborative Filtering" -> PhaseFourAlgorithmKind.NeuralCf
        "Bayes Theorem" -> PhaseFourAlgorithmKind.BayesTheorem
        "Bayesian Inference" -> PhaseFourAlgorithmKind.BayesianInference
        "Maximum Likelihood Estimation" -> PhaseFourAlgorithmKind.Mle
        "Maximum A Posteriori Estimation" -> PhaseFourAlgorithmKind.Map
        "Bayesian Linear Regression" -> PhaseFourAlgorithmKind.BayesianLinearRegression
        "Bayesian Networks" -> PhaseFourAlgorithmKind.BayesianNetworks
        "Hidden Markov Models" -> PhaseFourAlgorithmKind.Hmm
        "Gaussian Processes" -> PhaseFourAlgorithmKind.GaussianProcesses
        "Markov Chain Monte Carlo" -> PhaseFourAlgorithmKind.Mcmc
        "Metropolis-Hastings" -> PhaseFourAlgorithmKind.MetropolisHastings
        "Gibbs Sampling" -> PhaseFourAlgorithmKind.GibbsSampling
        "Variational Inference" -> PhaseFourAlgorithmKind.VariationalInference
        else -> null
    }
}

object PhaseFourData {
    val marketItems = listOf("Bread", "Milk", "Eggs", "Butter", "Coffee", "Tea", "Rice", "Pasta", "Cheese", "Apples")
    val baskets = listOf(
        Basket("T1", setOf("Bread", "Milk", "Butter")),
        Basket("T2", setOf("Bread", "Milk", "Eggs")),
        Basket("T3", setOf("Coffee", "Tea", "Apples")),
        Basket("T4", setOf("Bread", "Milk", "Cheese")),
        Basket("T5", setOf("Rice", "Pasta", "Cheese")),
        Basket("T6", setOf("Bread", "Butter", "Eggs")),
        Basket("T7", setOf("Coffee", "Tea")),
        Basket("T8", setOf("Milk", "Bread", "Eggs", "Butter"))
    )
    val ratings = RatingData(
        users = listOf("Alice", "Bob", "Cara", "Dev", "Eva"),
        items = listOf("Movie A", "Movie B", "Movie C", "Movie D", "Movie E"),
        ratings = listOf(
            listOf(5.0, 4.0, null, 2.0, null),
            listOf(4.0, null, 5.0, 1.0, null),
            listOf(null, 5.0, 4.0, null, 2.0),
            listOf(1.0, 2.0, null, 5.0, 4.0),
            listOf(null, 1.0, 2.0, 4.0, 5.0)
        )
    )

    fun anomalies(preset: AnomalyPreset, samples: Int = 80, contamination: Double = .08, noise: Double = .08, seed: Int = 41): List<AnomalyPoint> {
        val count = samples.coerceIn(20, 250)
        val anomalyCount = (count * contamination).toInt().coerceIn(1, count / 3)
        return List(count) { i ->
            val isAnomaly = i >= count - anomalyCount
            val n1 = centered(seed, i, 1)
            val n2 = centered(seed, i, 2)
            when {
                isAnomaly && preset == AnomalyPreset.Clustered -> AnomalyPoint(.72 + n1 * .06, -.72 + n2 * .06, true)
                isAnomaly -> AnomalyPoint((.75 * if (i % 2 == 0) 1 else -1) + n1 * .18, (.75 * if (i % 3 == 0) 1 else -1) + n2 * .18, true)
                preset == AnomalyPreset.Curved -> {
                    val t = 2.0 * PI * i / (count - anomalyCount)
                    AnomalyPoint(.55 * cos(t) + n1 * noise, .28 * sin(t) + n2 * noise, false)
                }
                preset == AnomalyPreset.TwoClustersLocal -> {
                    val side = if (i % 2 == 0) -1 else 1
                    AnomalyPoint(side * .45 + n1 * (.09 + noise), side * .18 + n2 * (.09 + noise), false)
                }
                preset == AnomalyPreset.Elliptical -> AnomalyPoint(n1 * (.35 + noise), n1 * .18 + n2 * (.12 + noise), false)
                else -> AnomalyPoint(n1 * (.26 + noise), n2 * (.26 + noise), false)
            }
        }
    }

    private fun centered(seed: Int, index: Int, salt: Int): Double {
        val raw = sin((seed * 127 + index * 47 + salt * 29).toDouble()) * 5913.17
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}

object PhaseFourEngines {
    fun isolationForest(points: List<AnomalyPoint>, selected: Int, trees: Int): IsolationState {
        val centerX = points.map { it.x }.average()
        val centerY = points.map { it.y }.average()
        val scores = points.map { p ->
            val distance = sqrt((p.x - centerX).pow(2) + (p.y - centerY).pow(2))
            (distance / 1.3).coerceIn(0.0, 1.0)
        }
        val paths = List(trees.coerceIn(1, 8)) { tree ->
            val score = scores[selected]
            val length = (8 - score * 6 + (tree % 3)).toInt().coerceAtLeast(1)
            IsolationPath(tree + 1, length, List(length) { step -> if ((step + tree) % 2 == 0) "x < %.2f".format(points[selected].x) else "y < %.2f".format(points[selected].y) })
        }
        return IsolationState(scores, paths, percentile(scores, .9))
    }

    fun lof(points: List<AnomalyPoint>, selected: Int, k: Int): LofState {
        val neighbors = points.indices.filter { it != selected }.sortedBy { distance(points[selected], points[it]) }.take(k.coerceAtLeast(1))
        val kDistance = neighbors.maxOf { distance(points[selected], points[it]) }
        fun density(i: Int): Double {
            val ns = points.indices.filter { it != i }.sortedBy { distance(points[i], points[it]) }.take(k.coerceAtLeast(1))
            return 1.0 / ns.map { max(distance(points[i], points[it]), kDistance) }.average().coerceAtLeast(1e-6)
        }
        val lrd = density(selected)
        val neighborDensity = neighbors.map { density(it) }.average()
        return LofState(neighbors, kDistance, lrd, neighborDensity, neighborDensity / lrd.coerceAtLeast(1e-9))
    }

    fun envelope(points: List<AnomalyPoint>): EnvelopeState {
        val cx = points.map { it.x }.average()
        val cy = points.map { it.y }.average()
        val vx = points.map { (it.x - cx).pow(2) }.average().coerceAtLeast(.001)
        val vy = points.map { (it.y - cy).pow(2) }.average().coerceAtLeast(.001)
        val distances = points.map { (it.x - cx).pow(2) / vx + (it.y - cy).pow(2) / vy }
        return EnvelopeState(cx, cy, vx, vy, percentile(distances, .9), distances)
    }

    fun zScores(values: List<Double>, threshold: Double): List<Boolean> {
        val mean = values.average()
        val sd = sqrt(values.map { (it - mean).pow(2) }.average()).coerceAtLeast(1e-9)
        return values.map { abs((it - mean) / sd) > threshold }
    }

    fun iqrOutliers(values: List<Double>, multiplier: Double = 1.5): Pair<Pair<Double, Double>, List<Boolean>> {
        val sorted = values.sorted()
        val q1 = sorted[(sorted.size * .25).toInt()]
        val q3 = sorted[(sorted.size * .75).toInt()]
        val iqr = q3 - q1
        val fences = q1 - multiplier * iqr to q3 + multiplier * iqr
        return fences to values.map { it < fences.first || it > fences.second }
    }

    fun support(baskets: List<Basket>, itemset: Set<String>): Double = baskets.count { it.items.containsAll(itemset) } / baskets.size.toDouble()

    fun associationRule(baskets: List<Basket>, antecedent: Set<String>, consequent: Set<String>): RuleMetric {
        val both = antecedent + consequent
        val supBoth = support(baskets, both)
        val supA = support(baskets, antecedent).coerceAtLeast(1e-9)
        val supB = support(baskets, consequent).coerceAtLeast(1e-9)
        val confidence = supBoth / supA
        return RuleMetric(antecedent, consequent, supBoth, confidence, confidence / supB)
    }

    fun apriori(baskets: List<Basket>, minSupport: Double): List<ItemsetLevel> {
        val items = baskets.flatMap { it.items }.distinct().sorted()
        val levels = mutableListOf<ItemsetLevel>()
        var candidates = items.map { setOf(it) }
        var level = 1
        while (candidates.isNotEmpty() && level <= 3) {
            val frequent = candidates.filter { support(baskets, it) >= minSupport }
            levels += ItemsetLevel(level, candidates, frequent)
            candidates = frequent.flatMap { a -> frequent.map { b -> a + b } }.filter { it.size == level + 1 }.distinct()
            level++
        }
        return levels
    }

    fun fpTree(baskets: List<Basket>): FpNode {
        val ordered = baskets.flatMap { it.items }.groupingBy { it }.eachCount().entries.sortedByDescending { it.value }.map { it.key }
        val children = ordered.take(5).map { item -> FpNode(item, baskets.count { item in it.items }, emptyList()) }
        return FpNode("root", baskets.size, children)
    }

    fun eclatTidsets(baskets: List<Basket>): Map<String, Set<String>> =
        PhaseFourData.marketItems.associateWith { item -> baskets.filter { item in it.items }.map { it.id }.toSet() }.filterValues { it.isNotEmpty() }

    fun cosine(a: List<Double>, b: List<Double>): Double {
        val dot = a.indices.sumOf { a[it] * b[it] }
        val na = sqrt(a.sumOf { it * it })
        val nb = sqrt(b.sumOf { it * it })
        return dot / (na * nb).coerceAtLeast(1e-9)
    }

    fun popularity(data: RatingData): List<Recommendation> = data.items.indices.map { item ->
        val ratings = data.ratings.mapNotNull { it[item] }
        Recommendation(data.items[item], ratings.average(), "Average rating %.2f from %d known ratings.".format(ratings.average(), ratings.size))
    }.sortedByDescending { it.score }

    fun userCf(data: RatingData, user: Int, item: Int): Recommendation {
        val sims = data.users.indices.filter { it != user && data.ratings[it][item] != null }.map { other ->
            other to cosine(filled(data.ratings[user]), filled(data.ratings[other]))
        }.sortedByDescending { it.second }.take(3)
        val score = sims.sumOf { (other, sim) -> sim * (data.ratings[other][item] ?: 0.0) } / sims.sumOf { abs(it.second) }.coerceAtLeast(1e-9)
        return Recommendation(data.items[item], score, sims.joinToString { "${data.users[it.first]} sim %.2f".format(it.second) })
    }

    fun itemCf(data: RatingData, user: Int, item: Int): Recommendation {
        val target = data.ratings.map { it[item] ?: 0.0 }
        val sims = data.items.indices.filter { it != item && data.ratings[user][it] != null }.map { other ->
            other to cosine(target, data.ratings.map { it[other] ?: 0.0 })
        }
        val score = sims.sumOf { (other, sim) -> sim * (data.ratings[user][other] ?: 0.0) } / sims.sumOf { abs(it.second) }.coerceAtLeast(1e-9)
        return Recommendation(data.items[item], score, "Similar rated items: ${sims.joinToString { data.items[it.first] }}")
    }

    fun factorState(data: RatingData, user: Int, item: Int, factors: Int): FactorState {
        val uf = data.users.indices.map { u -> List(factors) { f -> .25 + .13 * (u + 1) * (f + 1) } }
        val itf = data.items.indices.map { i -> List(factors) { f -> .22 + .11 * (i + 1) / (f + 1) } }
        val prediction = uf[user].indices.sumOf { uf[user][it] * itf[item][it] }
        val known = data.ratings[user][item] ?: prediction
        return FactorState(uf, itf, prediction, abs(known - prediction))
    }

    fun bayes(prior: Double, sensitivity: Double, specificity: Double): BayesState {
        val tp = sensitivity * prior
        val fp = (1.0 - specificity) * (1.0 - prior)
        return BayesState(prior, sensitivity, specificity, tp / (tp + fp).coerceAtLeast(1e-12), tp, fp)
    }

    fun betaBernoulli(alpha: Double, beta: Double, heads: Int, tails: Int): BetaState {
        val a = alpha + heads
        val b = beta + tails
        return BetaState(a, b, a / (a + b))
    }

    fun mleTheta(heads: Int, tails: Int): Double = heads / (heads + tails).coerceAtLeast(1).toDouble()
    fun mapTheta(heads: Int, tails: Int, alpha: Double, beta: Double): Double = (heads + alpha - 1.0) / (heads + tails + alpha + beta - 2.0).coerceAtLeast(1e-9)

    fun hmm(observations: List<String>): HmmState {
        val states = listOf("Sunny", "Rainy")
        val init = mapOf("Sunny" to .6, "Rainy" to .4)
        val trans = mapOf("Sunny" to mapOf("Sunny" to .7, "Rainy" to .3), "Rainy" to mapOf("Sunny" to .4, "Rainy" to .6))
        val emit = mapOf("Sunny" to mapOf("Walk" to .6, "Shop" to .3, "Clean" to .1), "Rainy" to mapOf("Walk" to .1, "Shop" to .4, "Clean" to .5))
        val forward = mutableListOf<Map<String, Double>>()
        observations.forEachIndexed { t, obs ->
            val raw = states.associateWith { s ->
                val prev = if (t == 0) init.getValue(s) else states.sumOf { p -> forward[t - 1].getValue(p) * trans.getValue(p).getValue(s) }
                prev * emit.getValue(s).getValue(obs)
            }
            val total = raw.values.sum().coerceAtLeast(1e-12)
            forward += raw.mapValues { it.value / total }
        }
        return HmmState(forward, forward.map { it.maxBy { e -> e.value }.key }, forward.last().values.max())
    }

    fun gaussianProcess(observations: List<Pair<Double, Double>>, lengthScale: Double, noise: Double): GpState {
        val xs = List(50) { -1.0 + 2.0 * it / 49.0 }
        val means = xs.map { x ->
            if (observations.isEmpty()) 0.0 else observations.sumOf { (ox, oy) -> rbf(x, ox, lengthScale) * oy } / observations.sumOf { (ox, _) -> rbf(x, ox, lengthScale) }.coerceAtLeast(1e-9)
        }
        val vars = xs.map { x ->
            val closeness = observations.maxOfOrNull { (ox, _) -> rbf(x, ox, lengthScale) } ?: 0.0
            (1.0 - closeness + noise).coerceIn(0.02, 1.2)
        }
        return GpState(xs, means, vars)
    }

    fun metropolis(current: Double, proposalStd: Double, step: Int): McmcStep {
        val proposal = current + sin(step * 12.989) * proposalStd
        val cd = targetDensity(current)
        val pd = targetDensity(proposal)
        val ratio = min(1.0, pd / cd.coerceAtLeast(1e-12))
        val u = abs(sin(step * 78.23)).mod(1.0)
        return McmcStep(current, proposal, cd, pd, ratio, u, u < ratio)
    }

    fun gibbs(steps: Int): GibbsState {
        val path = mutableListOf(.6 to -.4)
        repeat(steps.coerceAtLeast(1)) { i ->
            val (x, y) = path.last()
            path += if (i % 2 == 0) (.75 * y + sin(i.toDouble()) * .15) to y else x to (.75 * x + cos(i.toDouble()) * .15)
        }
        return GibbsState(path, if (steps % 2 == 0) "sample x | y" else "sample y | x")
    }

    fun variational(step: Int): VariationalState {
        val mean = 1.5 * (1.0 - exp(-step / 12.0))
        val variance = .25 + .75 * exp(-step / 10.0)
        val elbo = -abs(1.5 - mean) - abs(.35 - variance)
        return VariationalState(1.5, mean, variance, elbo)
    }

    private fun filled(row: List<Double?>): List<Double> = row.map { it ?: 0.0 }
    private fun distance(a: AnomalyPoint, b: AnomalyPoint): Double = sqrt((a.x - b.x).pow(2) + (a.y - b.y).pow(2))
    private fun percentile(values: List<Double>, q: Double): Double = values.sorted()[(values.lastIndex * q).toInt().coerceIn(values.indices)]
    private fun rbf(a: Double, b: Double, length: Double): Double = exp(-((a - b).pow(2)) / (2.0 * length.coerceAtLeast(.01).pow(2)))
    private fun targetDensity(x: Double): Double = .55 * exp(-.5 * ((x + 1.0) / .45).pow(2)) + .45 * exp(-.5 * ((x - .9) / .25).pow(2))
}
