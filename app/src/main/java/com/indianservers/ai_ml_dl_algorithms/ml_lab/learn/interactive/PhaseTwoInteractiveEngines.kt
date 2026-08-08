package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class PhaseTwoAlgorithmKind(val displayName: String, val category: String, val difficulty: String) {
    GaussianNaiveBayes("Gaussian Naive Bayes", "Classification", "Intermediate"),
    MultinomialNaiveBayes("Multinomial Naive Bayes", "Classification", "Intermediate"),
    BernoulliNaiveBayes("Bernoulli Naive Bayes", "Classification", "Intermediate"),
    Lda("Linear Discriminant Analysis", "Classification", "Intermediate"),
    Qda("Quadratic Discriminant Analysis", "Classification", "Intermediate"),
    LinearSvm("Support Vector Machine - Linear SVM", "SVM", "Intermediate"),
    KernelSvm("Support Vector Machine - Kernel SVM", "SVM", "Advanced"),
    SoftMarginSvm("Support Vector Classification with Soft Margin", "SVM", "Advanced"),
    SgdClassifier("SGD Classifier", "Classification", "Intermediate"),
    BaggingClassifier("Bagging Classifier", "Ensemble Learning", "Intermediate"),
    RandomForestClassifier("Random Forest Classifier", "Ensemble Learning", "Intermediate"),
    ExtraTreesClassifier("Extra Trees Classifier", "Ensemble Learning", "Intermediate"),
    AdaBoostClassifier("AdaBoost Classifier", "Ensemble Learning", "Advanced"),
    VotingClassifier("Voting Classifier", "Ensemble Learning", "Intermediate"),
    StackingClassifier("Stacking Classifier", "Ensemble Learning", "Advanced")
}

enum class PhaseTwoDatasetPreset(val label: String) {
    WellSeparatedGaussian("Well separated Gaussian"),
    DifferentVariances("Different variances"),
    CorrelatedFeatures("Correlated features"),
    ImbalancedPriors("Imbalanced priors"),
    PerfectlySeparable("Perfectly separable"),
    OneOutlier("One outlier"),
    OverlappingClasses("Overlapping classes"),
    CircularData("Circular data"),
    XorLike("XOR-like"),
    NoisyEnsemble("Noisy ensemble"),
    LabelNoise("Label noise")
}

enum class KernelType(val label: String) { Linear("Linear"), Polynomial("Polynomial"), Rbf("RBF") }
enum class VotingMode(val label: String) { Hard("Hard voting"), Soft("Soft voting") }

data class ClassSummary(
    val label: Int,
    val prior: Double,
    val meanX: Double,
    val meanY: Double,
    val varianceX: Double,
    val varianceY: Double,
    val covariance: Double = 0.0
)

data class PosteriorBreakdown(
    val label: Int,
    val prior: Double,
    val likelihoodX: Double,
    val likelihoodY: Double,
    val score: Double,
    val posterior: Double
)

data class MultiClassMetrics(
    val classes: List<Int>,
    val confusion: List<List<Int>>,
    val accuracy: Double,
    val macroPrecision: Double,
    val macroRecall: Double,
    val macroF1: Double,
    val perClass: List<PerClassMetric>
)

data class PerClassMetric(val label: Int, val precision: Double, val recall: Double, val f1: Double)

data class SvmState(
    val weightX: Double,
    val weightY: Double,
    val bias: Double,
    val marginWidth: Double,
    val supportVectorIndices: List<Int>,
    val violations: List<Int>,
    val hingeLoss: Double
)

data class KernelState(
    val kernel: KernelType,
    val gamma: Double,
    val degree: Int,
    val transformedSeparation: Double,
    val warning: String?
)

data class BootstrapState(
    val memberId: Int,
    val frequencies: List<Int>,
    val outOfBag: List<Int>,
    val featureSubset: List<String>,
    val splitFeature: String,
    val threshold: Double,
    val polarity: Int,
    val weight: Double = 1.0
)

data class EnsembleMemberState(
    val id: String,
    val prediction: Int,
    val weight: Double?,
    val isSelected: Boolean,
    val metadata: Map<String, String>
)

data class EnsembleState(
    val members: List<EnsembleMemberState>,
    val voteDistribution: Map<Int, Double>,
    val prediction: Int,
    val bootstrapStates: List<BootstrapState>
)

data class AdaBoostRound(
    val round: Int,
    val stump: BootstrapState,
    val weightedError: Double,
    val learnerWeight: Double,
    val sampleWeights: List<Double>,
    val misclassified: List<Int>
)

data class TextNaiveBayesState(
    val vocabulary: List<String>,
    val counts: Map<String, Int>,
    val classScores: Map<Int, Double>,
    val prediction: Int
)

object PhaseTwoTopicMatcher {
    fun kindFor(title: String, section: String, domain: String): PhaseTwoAlgorithmKind? = when {
        title == "Gaussian Naive Bayes" -> PhaseTwoAlgorithmKind.GaussianNaiveBayes
        title == "Multinomial Naive Bayes" -> PhaseTwoAlgorithmKind.MultinomialNaiveBayes
        title == "Bernoulli Naive Bayes" -> PhaseTwoAlgorithmKind.BernoulliNaiveBayes
        title == "Linear Discriminant Analysis" -> PhaseTwoAlgorithmKind.Lda
        title == "Quadratic Discriminant Analysis" -> PhaseTwoAlgorithmKind.Qda
        title == "Support Vector Machine" -> PhaseTwoAlgorithmKind.LinearSvm
        title == "SGD Classifier" -> PhaseTwoAlgorithmKind.SgdClassifier
        title == "Bagging" -> PhaseTwoAlgorithmKind.BaggingClassifier
        title == "Random Forest" -> PhaseTwoAlgorithmKind.RandomForestClassifier
        title == "Extra Trees" && domain == "Ensemble Learning" -> PhaseTwoAlgorithmKind.ExtraTreesClassifier
        title == "AdaBoost" -> PhaseTwoAlgorithmKind.AdaBoostClassifier
        title == "Voting" || title == "Soft Voting" || title == "Hard Voting" -> PhaseTwoAlgorithmKind.VotingClassifier
        title == "Stacking" -> PhaseTwoAlgorithmKind.StackingClassifier
        title == "Decision Tree" && section == "Classification" -> null
        else -> null
    }
}

object PhaseTwoDatasets {
    fun generate(
        preset: PhaseTwoDatasetPreset,
        samples: Int = 60,
        classes: Int = 3,
        noise: Double = 0.12,
        seed: Int = 17
    ): List<LabPoint> {
        val classCount = classes.coerceIn(2, 4)
        val count = samples.coerceIn(12, 200)
        return List(count) { index ->
            val label = labelFor(index, count, classCount, preset)
            val angle = 2.0 * PI * label / classCount
            val baseRadius = if (preset == PhaseTwoDatasetPreset.CircularData) {
                if (label == 0) 0.24 else 0.72
            } else 0.58
            val spreadX = when (preset) {
                PhaseTwoDatasetPreset.DifferentVariances -> 0.08 + label * 0.08
                PhaseTwoDatasetPreset.OverlappingClasses -> 0.28
                PhaseTwoDatasetPreset.CorrelatedFeatures -> 0.22
                else -> 0.14
            } + noise
            val spreadY = when (preset) {
                PhaseTwoDatasetPreset.DifferentVariances -> 0.24 - label.coerceAtMost(2) * 0.04
                PhaseTwoDatasetPreset.OverlappingClasses -> 0.28
                else -> 0.14
            } + noise
            val n1 = centered(seed, index, 1)
            val n2 = centered(seed, index, 2)
            val correlated = if (preset == PhaseTwoDatasetPreset.CorrelatedFeatures) n1 * 0.72 + n2 * 0.28 else n2
            val centerX = baseRadius * cos(angle)
            val centerY = baseRadius * sin(angle)
            val rawX = when (preset) {
                PhaseTwoDatasetPreset.XorLike -> if (index % 4 < 2) -0.55 else 0.55
                PhaseTwoDatasetPreset.PerfectlySeparable -> centerX * 1.18
                else -> centerX
            }
            val rawY = when (preset) {
                PhaseTwoDatasetPreset.XorLike -> if (index % 2 == 0) -0.55 else 0.55
                PhaseTwoDatasetPreset.PerfectlySeparable -> centerY * 1.18
                else -> centerY
            }
            val noisyLabel = if (preset == PhaseTwoDatasetPreset.LabelNoise && index % 13 == 0) (label + 1) % classCount else label
            val outlier = preset == PhaseTwoDatasetPreset.OneOutlier && index == count - 1
            LabPoint(
                x = (if (outlier) -rawX else rawX + n1 * spreadX).coerceIn(-1.0, 1.0),
                y = (if (outlier) -rawY else rawY + correlated * spreadY).coerceIn(-1.0, 1.0),
                label = noisyLabel,
                target = noisyLabel.toDouble(),
                train = index % 5 != 0
            )
        }
    }

    private fun labelFor(index: Int, count: Int, classes: Int, preset: PhaseTwoDatasetPreset): Int = when (preset) {
        PhaseTwoDatasetPreset.ImbalancedPriors -> when {
            index < count * 0.62 -> 0
            index < count * 0.86 -> 1
            else -> (2).coerceAtMost(classes - 1)
        }
        PhaseTwoDatasetPreset.CircularData -> if (index % 3 == 0) 0 else 1
        PhaseTwoDatasetPreset.XorLike -> if ((index % 4 == 0) || (index % 4 == 3)) 1 else 0
        else -> index % classes
    }

    private fun centered(seed: Int, index: Int, salt: Int): Double {
        val raw = sin((seed * 101 + index * 37 + salt * 19).toDouble()) * 12973.123
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}

object PhaseTwoEngines {
    val textVocabulary = listOf("offer", "money", "meeting", "project", "free", "report")

    fun classSummaries(points: List<LabPoint>): List<ClassSummary> = points.groupBy { it.label }.toSortedMap().map { (label, group) ->
        val meanX = group.map { it.x }.average()
        val meanY = group.map { it.y }.average()
        val varianceX = group.sumOf { (it.x - meanX).pow(2) } / group.size.coerceAtLeast(1)
        val varianceY = group.sumOf { (it.y - meanY).pow(2) } / group.size.coerceAtLeast(1)
        val covariance = group.sumOf { (it.x - meanX) * (it.y - meanY) } / group.size.coerceAtLeast(1)
        ClassSummary(label, group.size / points.size.toDouble(), meanX, meanY, varianceX.coerceAtLeast(0.0025), varianceY.coerceAtLeast(0.0025), covariance)
    }

    fun gaussianNaiveBayes(points: List<LabPoint>, sample: LabPoint): Pair<Int, List<PosteriorBreakdown>> {
        val raw = classSummaries(points).map { summary ->
            val lx = gaussianDensity(sample.x, summary.meanX, sqrt(summary.varianceX))
            val ly = gaussianDensity(sample.y, summary.meanY, sqrt(summary.varianceY))
            val score = summary.prior * lx * ly
            PosteriorBreakdown(summary.label, summary.prior, lx, ly, score, 0.0)
        }
        val total = raw.sumOf { it.score }.coerceAtLeast(1e-12)
        val posterior = raw.map { it.copy(posterior = it.score / total) }
        return posterior.maxBy { it.posterior }.label to posterior
    }

    fun multinomialNaiveBayes(counts: Map<String, Int>): TextNaiveBayesState {
        val classWordCounts = mapOf(
            0 to mapOf("offer" to 8, "money" to 7, "free" to 9, "meeting" to 1, "project" to 1, "report" to 2),
            1 to mapOf("offer" to 1, "money" to 1, "free" to 1, "meeting" to 8, "project" to 7, "report" to 8)
        )
        val scores = classWordCounts.mapValues { (_, freqs) ->
            val total = freqs.values.sum() + textVocabulary.size
            ln(0.5) + textVocabulary.sumOf { word ->
                val probability = (freqs.getValue(word) + 1.0) / total
                counts.getOrDefault(word, 0) * ln(probability)
            }
        }
        return TextNaiveBayesState(textVocabulary, counts, scores, scores.maxBy { it.value }.key)
    }

    fun bernoulliNaiveBayes(present: Map<String, Boolean>): TextNaiveBayesState {
        val pFeatureGivenClass = mapOf(
            0 to mapOf("offer" to .82, "money" to .78, "free" to .85, "meeting" to .12, "project" to .18, "report" to .2),
            1 to mapOf("offer" to .14, "money" to .16, "free" to .2, "meeting" to .82, "project" to .76, "report" to .8)
        )
        val scores = pFeatureGivenClass.mapValues { (_, probs) ->
            ln(0.5) + textVocabulary.sumOf { word ->
                val p = probs.getValue(word)
                if (present[word] == true) ln(p) else ln(1.0 - p)
            }
        }
        return TextNaiveBayesState(textVocabulary, present.mapValues { if (it.value) 1 else 0 }, scores, scores.maxBy { it.value }.key)
    }

    fun ldaPredict(points: List<LabPoint>, sample: LabPoint): Pair<Int, Double> {
        val summaries = classSummaries(points)
        val globalMeanX = points.map { it.x }.average()
        val globalMeanY = points.map { it.y }.average()
        val directionX = summaries.sumOf { it.prior * (it.meanX - globalMeanX) }
        val directionY = summaries.sumOf { it.prior * (it.meanY - globalMeanY) }
        val scores = summaries.associate { summary ->
            val meanProjection = summary.meanX * directionX + summary.meanY * directionY
            val sampleProjection = sample.x * directionX + sample.y * directionY
            summary.label to -abs(sampleProjection - meanProjection) + ln(summary.prior)
        }
        val within = summaries.sumOf { it.varianceX + it.varianceY }.coerceAtLeast(1e-9)
        val between = summaries.sumOf { (it.meanX - globalMeanX).pow(2) + (it.meanY - globalMeanY).pow(2) }
        return scores.maxBy { it.value }.key to between / within
    }

    fun qdaPredict(points: List<LabPoint>, sample: LabPoint): Pair<Int, List<PosteriorBreakdown>> {
        val raw = classSummaries(points).map { summary ->
            val lx = gaussianDensity(sample.x, summary.meanX, sqrt(summary.varianceX))
            val ly = gaussianDensity(sample.y, summary.meanY, sqrt(summary.varianceY))
            val score = summary.prior * lx * ly / sqrt((summary.varianceX * summary.varianceY).coerceAtLeast(1e-9))
            PosteriorBreakdown(summary.label, summary.prior, lx, ly, score, 0.0)
        }
        val total = raw.sumOf { it.score }.coerceAtLeast(1e-12)
        val posterior = raw.map { it.copy(posterior = it.score / total) }
        return posterior.maxBy { it.posterior }.label to posterior
    }

    fun svmState(points: List<LabPoint>, c: Double = 1.0, angle: Double? = null, offset: Double = 0.0): SvmState {
        val binary = points.filter { it.label <= 1 }
        val class0 = binary.filter { it.label == 0 }
        val class1 = binary.filter { it.label == 1 }
        val mean0x = class0.map { it.x }.average()
        val mean0y = class0.map { it.y }.average()
        val mean1x = class1.map { it.x }.average()
        val mean1y = class1.map { it.y }.average()
        val wx = angle?.let { cos(it) } ?: mean1x - mean0x
        val wy = angle?.let { sin(it) } ?: mean1y - mean0y
        val norm = sqrt(wx * wx + wy * wy).coerceAtLeast(1e-9)
        val normalizedX = wx / norm
        val normalizedY = wy / norm
        val bias = if (angle == null) -0.5 * (normalizedX * (mean0x + mean1x) + normalizedY * (mean0y + mean1y)) + offset else offset
        val margins = binary.map { point ->
            val y = if (point.label == 1) 1.0 else -1.0
            y * (normalizedX * point.x + normalizedY * point.y + bias)
        }
        val support = margins.indices.filter { margins[it] <= 1.12 }.take(12)
        val violations = margins.indices.filter { margins[it] < 1.0 }
        val loss = margins.sumOf { max(0.0, 1.0 - it) } / margins.size.coerceAtLeast(1) + 0.5 / c.coerceAtLeast(0.001) * (normalizedX * normalizedX + normalizedY * normalizedY)
        return SvmState(normalizedX, normalizedY, bias, 2.0 / sqrt(normalizedX * normalizedX + normalizedY * normalizedY), support, violations, loss)
    }

    fun hingeLoss(label: Int, score: Double): Double {
        val y = if (label == 1) 1.0 else -1.0
        return max(0.0, 1.0 - y * score)
    }

    fun kernel(a: LabPoint, b: LabPoint, type: KernelType, gamma: Double = 1.0, degree: Int = 3): Double = when (type) {
        KernelType.Linear -> a.x * b.x + a.y * b.y
        KernelType.Polynomial -> (gamma * (a.x * b.x + a.y * b.y) + 1.0).pow(degree)
        KernelType.Rbf -> exp(-gamma * ((a.x - b.x).pow(2) + (a.y - b.y).pow(2)))
    }

    fun kernelState(type: KernelType, gamma: Double, degree: Int): KernelState {
        val separation = when (type) {
            KernelType.Linear -> 0.35
            KernelType.Polynomial -> 0.55 + degree * 0.04
            KernelType.Rbf -> 0.45 + gamma.coerceAtMost(8.0) * 0.06
        }
        val warning = if (type == KernelType.Rbf && gamma > 6.0) "Very high gamma creates local influence regions and can overfit." else null
        return KernelState(type, gamma, degree, separation, warning)
    }

    fun sgdClassifierStep(points: List<LabPoint>, iteration: Int, rate: Double, weightX: Double, weightY: Double, bias: Double): TrainingStep {
        val index = iteration % points.size
        val sample = points[index]
        val y = if (sample.label == 1) 1.0 else -1.0
        val score = weightX * sample.x + weightY * sample.y + bias
        val loss = hingeLoss(sample.label, score)
        val update = if (loss > 0.0) y else 0.0
        return TrainingStep(
            iteration + 1,
            "SGD classifier reads one sample, computes hinge loss, and updates the linear boundary if the margin is violated.",
            loss,
            mapOf("w1" to (weightX + rate * update * sample.x), "w2" to (weightY + rate * update * sample.y), "bias" to (bias + rate * update), "score" to score),
            listOf(index),
            if (loss == 0.0) "The sample is outside the margin, so no corrective update is needed." else "The sample is inside the margin or misclassified, so the boundary moves toward a larger margin."
        )
    }

    fun bootstrap(points: List<LabPoint>, seed: Int, memberId: Int, featurePool: List<String> = listOf("x1", "x2"), randomThreshold: Boolean = false): BootstrapState {
        val frequencies = MutableList(points.size) { 0 }
        repeat(points.size) { draw ->
            val index = deterministicIndex(seed + memberId * 97, draw, points.size)
            frequencies[index] += 1
        }
        val subset = featurePool.filterIndexed { index, _ -> (index + memberId + seed) % 2 == 0 }.ifEmpty { listOf(featurePool[memberId % featurePool.size]) }
        val splitFeature = subset.first()
        val values = points.map { if (splitFeature == "x1") it.x else it.y }.sorted()
        val threshold = if (randomThreshold) {
            -0.75 + 1.5 * ((seed + memberId * 17) % 100) / 99.0
        } else {
            values[values.size / 2]
        }
        val polarity = if ((seed + memberId) % 2 == 0) 1 else -1
        return BootstrapState(memberId, frequencies, frequencies.indices.filter { frequencies[it] == 0 }, subset, splitFeature, threshold, polarity)
    }

    fun ensemble(points: List<LabPoint>, query: LabPoint, trees: Int, kind: PhaseTwoAlgorithmKind, seed: Int): EnsembleState {
        val states = List(trees.coerceIn(1, 50)) { member ->
            bootstrap(points, seed, member, randomThreshold = kind == PhaseTwoAlgorithmKind.ExtraTreesClassifier)
        }
        val members = states.map { state ->
            val vote = stumpPredict(query, state)
            EnsembleMemberState("Tree ${state.memberId + 1}", vote, state.weight, state.memberId == 0, mapOf("feature" to state.splitFeature, "threshold" to "%.2f".format(state.threshold), "oob" to state.outOfBag.size.toString()))
        }
        val distribution = members.groupingBy { it.prediction }.fold(0.0) { total, member -> total + (member.weight ?: 1.0) }
        val normalized = distribution.mapValues { it.value / distribution.values.sum().coerceAtLeast(1e-9) }
        return EnsembleState(members, normalized, normalized.maxBy { it.value }.key, states)
    }

    fun adaBoost(points: List<LabPoint>, rounds: Int, seed: Int = 5): List<AdaBoostRound> {
        var weights = List(points.size) { 1.0 / points.size }
        return List(rounds.coerceIn(1, 12)) { round ->
            val stump = bootstrap(points, seed, round, randomThreshold = true)
            val predictions = points.map { stumpPredict(it, stump) }
            val missed = predictions.indices.filter { predictions[it] != points[it].label.coerceAtMost(1) }
            val error = missed.sumOf { weights[it] }.coerceIn(1e-6, 0.499)
            val learnerWeight = 0.5 * ln((1.0 - error) / error)
            weights = weights.mapIndexed { index, weight ->
                if (index in missed) weight * exp(learnerWeight) else weight * exp(-learnerWeight)
            }.let { updated ->
                val total = updated.sum().coerceAtLeast(1e-9)
                updated.map { it / total }
            }
            AdaBoostRound(round + 1, stump.copy(weight = learnerWeight), error, learnerWeight, weights, missed)
        }
    }

    fun hardVoting(predictions: List<Int>): Int =
        predictions.groupingBy { it }.eachCount().maxWith(compareBy<Map.Entry<Int, Int>> { it.value }.thenBy { -it.key }).key

    fun softVoting(probabilities: List<Map<Int, Double>>): Pair<Int, Map<Int, Double>> {
        val classes = probabilities.flatMap { it.keys }.toSet()
        val averaged = classes.associateWith { label -> probabilities.sumOf { it[label] ?: 0.0 } / probabilities.size.coerceAtLeast(1) }
        return averaged.maxBy { it.value }.key to averaged
    }

    fun stackingMetaFeatures(baseProbabilities: List<Double>): List<Double> = baseProbabilities
    fun stackingMetaProbability(features: List<Double>): Double = sigmoid(-1.2 + features.sum() / features.size.coerceAtLeast(1) * 2.4)

    fun multiclassMetrics(actual: List<Int>, predicted: List<Int>): MultiClassMetrics {
        val classes = (actual + predicted).distinct().sorted()
        val confusion = classes.map { a -> classes.map { p -> actual.indices.count { actual[it] == a && predicted[it] == p } } }
        val total = actual.size.coerceAtLeast(1)
        val perClass = classes.mapIndexed { index, label ->
            val tp = confusion[index][index]
            val fp = confusion.indices.sumOf { row -> if (row == index) 0 else confusion[row][index] }
            val fn = confusion[index].sum() - tp
            val precision = tp / (tp + fp).coerceAtLeast(1).toDouble()
            val recall = tp / (tp + fn).coerceAtLeast(1).toDouble()
            val f1 = if (precision + recall == 0.0) 0.0 else 2.0 * precision * recall / (precision + recall)
            PerClassMetric(label, precision, recall, f1)
        }
        return MultiClassMetrics(
            classes,
            confusion,
            actual.indices.count { actual[it] == predicted[it] } / total.toDouble(),
            perClass.map { it.precision }.average(),
            perClass.map { it.recall }.average(),
            perClass.map { it.f1 }.average(),
            perClass
        )
    }

    fun stumpPredict(point: LabPoint, state: BootstrapState): Int {
        val value = if (state.splitFeature == "x1") point.x else point.y
        val positive = if (state.polarity > 0) value > state.threshold else value <= state.threshold
        return if (positive) 1 else 0
    }

    private fun gaussianDensity(value: Double, mean: Double, sigma: Double): Double {
        val safeSigma = sigma.coerceAtLeast(0.001)
        val z = (value - mean) / safeSigma
        return exp(-0.5 * z * z) / (safeSigma * sqrt(2.0 * PI))
    }

    private fun deterministicIndex(seed: Int, draw: Int, size: Int): Int {
        val raw = sin((seed * 53 + draw * 41).toDouble()) * 10000.0
        return (((raw - kotlin.math.floor(raw)) * size).toInt()).coerceIn(0, size - 1)
    }

    private fun sigmoid(value: Double): Double = 1.0 / (1.0 + exp(-value))
}
