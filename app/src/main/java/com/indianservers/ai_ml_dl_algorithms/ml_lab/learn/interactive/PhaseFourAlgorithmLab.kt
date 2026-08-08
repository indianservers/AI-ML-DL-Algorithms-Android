package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableDoubleStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GradientButton
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBorder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

private enum class PhaseFourSection(val label: String) { Learn("Learn"), Visualize("Visualize"), Step("Step"), Experiment("Experiment"), Compare("Compare") }

@Composable
fun PhaseFourAlgorithmLab(
    topic: LearnTopic,
    kind: PhaseFourAlgorithmKind,
    depth: LearningDepth,
    completed: Boolean,
    onBack: () -> Unit,
    onComplete: () -> Unit
) {
    var section by remember(topic.id) { mutableStateOf(PhaseFourSection.Learn) }
    Column(Modifier.fillMaxSize()) {
        Column(Modifier.padding(start = 16.dp, end = 16.dp, top = 10.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp), verticalAlignment = Alignment.CenterVertically) {
                SegmentedOption("<", false, Modifier.size(42.dp), onBack)
                Column(Modifier.weight(1f)) {
                    Text(topic.title, color = LabText, fontSize = 20.sp, fontWeight = FontWeight.Bold, maxLines = 2)
                    Text("${kind.family} - ${depth.title}", color = Color(topic.accent), fontSize = 11.sp)
                }
                Text(if (completed) "Completed" else "Phase 4", color = if (completed) LabGreen else LabMuted, fontSize = 11.sp)
            }
            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                PhaseFourSection.entries.forEach { SegmentedOption(it.label, section == it) { section = it } }
            }
        }
        when (section) {
            PhaseFourSection.Learn -> Learn4(kind)
            PhaseFourSection.Visualize -> Visualize4(kind)
            PhaseFourSection.Step -> Step4(kind)
            PhaseFourSection.Experiment -> Experiment4(kind, onComplete)
            PhaseFourSection.Compare -> Compare4(kind)
        }
    }
}

@Composable
private fun Learn4(kind: PhaseFourAlgorithmKind) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { Info4("Core question", coreQuestion4(kind)) }
        item { Equation4(equation4(kind)) }
        item { Info4("What becomes visible", mechanism4(kind)) }
    }
}

@Composable
private fun Visualize4(kind: PhaseFourAlgorithmKind) {
    var preset by remember(kind) { mutableStateOf(AnomalyPreset.SingleOutlier) }
    var selected by remember(kind) { mutableIntStateOf(0) }
    var k by remember(kind) { mutableIntStateOf(5) }
    var threshold by remember(kind) { mutableDoubleStateOf(2.5) }
    var prior by remember(kind) { mutableDoubleStateOf(.01) }
    var sensitivity by remember(kind) { mutableDoubleStateOf(.9) }
    var specificity by remember(kind) { mutableDoubleStateOf(.95) }
    val anomalies = PhaseFourData.anomalies(preset, 90, .09, .08, 7)
    val iso = PhaseFourEngines.isolationForest(anomalies, selected.coerceIn(anomalies.indices), 6)
    val lof = PhaseFourEngines.lof(anomalies, selected.coerceIn(anomalies.indices), k)
    val envelope = PhaseFourEngines.envelope(anomalies)
    val bayes = PhaseFourEngines.bayes(prior, sensitivity, specificity)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Visualize", "Why this result happened") }
        item {
            when (family4(kind)) {
                "anomaly" -> AnomalyCanvas4(anomalies, iso.scores, envelope, selected)
                "association" -> AssociationPanel4(kind)
                "recommender" -> RecommenderPanel4(kind)
                else -> ProbabilityPanel4(kind, bayes)
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    when (family4(kind)) {
                        "anomaly" -> {
                            Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                AnomalyPreset.entries.take(6).forEach { SegmentedOption(it.label, preset == it) { preset = it } }
                            }
                            Slider4("Selected point", selected.toDouble(), 0.0, (anomalies.lastIndex).toDouble()) { selected = it.toInt() }
                            Slider4("Neighbors / threshold", k.toDouble(), 2.0, 20.0) { k = it.toInt() }
                            if (kind == PhaseFourAlgorithmKind.ZScore) Slider4("Z threshold", threshold, 1.0, 4.0) { threshold = it }
                        }
                        "bayes" -> {
                            Slider4("Prior", prior, .001, .5) { prior = it }
                            Slider4("Sensitivity", sensitivity, .5, .99) { sensitivity = it }
                            Slider4("Specificity", specificity, .5, .99) { specificity = it }
                        }
                        else -> Text("Tap through Step and Compare to inspect calculations on the fixed offline dataset.", color = LabMuted, fontSize = 12.sp)
                    }
                }
            }
        }
        item { StateSummary4(kind, iso, lof, envelope, bayes) }
    }
}

@Composable
private fun Step4(kind: PhaseFourAlgorithmKind) {
    var step by remember(kind) { mutableIntStateOf(4) }
    val beta = PhaseFourEngines.betaBernoulli(2.0, 2.0, step, 2)
    val hmm = PhaseFourEngines.hmm(listOf("Walk", "Shop", "Clean", "Walk").take((step % 4) + 1))
    val gp = PhaseFourEngines.gaussianProcess(listOf(-.6 to .4, .2 to -.2, .7 to .5).take((step % 3) + 1), .28, .05)
    val mh = PhaseFourEngines.metropolis(.2, .45, step)
    val gibbs = PhaseFourEngines.gibbs(step)
    val vi = PhaseFourEngines.variational(step)
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Step Mode", "Current calculation with actual values") }
        item {
            when (kind) {
                PhaseFourAlgorithmKind.BayesianInference, PhaseFourAlgorithmKind.Mle, PhaseFourAlgorithmKind.Map -> BayesianStepPanel4(beta, step)
                PhaseFourAlgorithmKind.Hmm -> HmmPanel4(hmm)
                PhaseFourAlgorithmKind.GaussianProcesses, PhaseFourAlgorithmKind.BayesianLinearRegression -> GpCanvas4(gp)
                PhaseFourAlgorithmKind.Mcmc, PhaseFourAlgorithmKind.MetropolisHastings -> McmcPanel4(mh)
                PhaseFourAlgorithmKind.GibbsSampling -> GibbsCanvas4(gibbs)
                PhaseFourAlgorithmKind.VariationalInference -> ViPanel4(vi)
                else -> GenericStep4(kind)
            }
        }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                SegmentedOption("Next", true, Modifier.weight(1f)) { step += 1 }
                SegmentedOption("Reset", false, Modifier.weight(1f)) { step = 1 }
            }
        }
    }
}

@Composable
private fun Experiment4(kind: PhaseFourAlgorithmKind, onComplete: () -> Unit) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Experiment", "Break-it scenario and recovery hint") }
        item { Info4("Break It", break4(kind)) }
        item { Info4("How can we fix this?", fix4(kind)) }
        item { GradientButton("Mark lesson complete", Modifier.fillMaxWidth(), onComplete) }
    }
}

@Composable
private fun Compare4(kind: PhaseFourAlgorithmKind) {
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { SectionTitle("Compare", "Same data, different assumptions") }
        item { Info4("Curated comparison", compare4(kind)) }
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Assumption", family4(kind), LabCyan, Modifier.weight(1f))
                MetricPill("Offline", "Yes", LabGreen, Modifier.weight(1f))
                MetricPill("Explainable", "State", LabOrange, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun AnomalyCanvas4(points: List<AnomalyPoint>, scores: List<Double>, envelope: EnvelopeState, selected: Int) {
    Canvas(Modifier.fillMaxWidth().height(300.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(10.dp)) {
        fun sx(x: Double) = (size.width * (x + 1.1) / 2.2).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.1) / 2.2)).toFloat()
        val rx = (sqrt(envelope.varX * envelope.threshold) * size.width / 2.2).toFloat()
        val ry = (sqrt(envelope.varY * envelope.threshold) * size.height / 2.2).toFloat()
        drawOval(LabPurple.copy(alpha = .25f), Offset(sx(envelope.centerX) - rx, sy(envelope.centerY) - ry), Size(rx * 2, ry * 2), style = Stroke(3f))
        points.forEachIndexed { i, p ->
            val anomaly = scores[i] >= scores.sorted()[(scores.size * .9).toInt()]
            val color = if (anomaly) LabPink else LabCyan
            if (p.hiddenAnomaly) drawCircle(Color.White.copy(alpha = .4f), 10f, Offset(sx(p.x), sy(p.y)), style = Stroke(2f))
            drawCircle(if (i == selected) LabOrange else color, if (i == selected) 9f else 6f, Offset(sx(p.x), sy(p.y)))
        }
    }
}

@Composable
private fun AssociationPanel4(kind: PhaseFourAlgorithmKind) {
    val levels = PhaseFourEngines.apriori(PhaseFourData.baskets, .3)
    val rule = PhaseFourEngines.associationRule(PhaseFourData.baskets, setOf("Bread"), setOf("Milk"))
    val tree = PhaseFourEngines.fpTree(PhaseFourData.baskets)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Market Basket", color = LabText, fontWeight = FontWeight.Bold)
            Text("Rule Bread -> Milk: support %.2f, confidence %.2f, lift %.2f".format(rule.support, rule.confidence, rule.lift), color = LabGreen, fontSize = 12.sp)
            Text("Apriori levels: ${levels.joinToString { "L${it.level}: ${it.frequent.size}/${it.candidates.size}" }}", color = LabMuted, fontSize = 12.sp)
            Text("FP-tree root ${tree.count}, children ${tree.children.joinToString { "${it.item}(${it.count})" }}", color = LabMuted, fontSize = 12.sp)
            if (kind == PhaseFourAlgorithmKind.Eclat) Text("ECLAT Bread TIDs: ${PhaseFourEngines.eclatTidsets(PhaseFourData.baskets)["Bread"]}", color = LabCyan, fontSize = 12.sp)
        }
    }
}

@Composable
private fun RecommenderPanel4(kind: PhaseFourAlgorithmKind) {
    val data = PhaseFourData.ratings
    val pop = PhaseFourEngines.popularity(data).first()
    val user = PhaseFourEngines.userCf(data, 0, 2)
    val item = PhaseFourEngines.itemCf(data, 0, 4)
    val factor = PhaseFourEngines.factorState(data, 0, 2, 2)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("User x Item Matrix", color = LabText, fontWeight = FontWeight.Bold)
            data.ratings.forEachIndexed { i, row -> Text("${data.users[i]}: ${row.joinToString { it?.toInt()?.toString() ?: "?" }}", color = LabMuted, fontSize = 12.sp) }
            Text(when (kind) {
                PhaseFourAlgorithmKind.Popularity -> "Top item: ${pop.item}, ${pop.explanation}"
                PhaseFourAlgorithmKind.UserCf -> "User-CF predicts ${user.item}=%.2f because ${user.explanation}".format(user.score)
                PhaseFourAlgorithmKind.ItemCf -> "Item-CF predicts ${item.item}=%.2f. ${item.explanation}".format(item.score)
                else -> "Latent factor prediction Movie C for Alice = %.2f, error %.2f".format(factor.prediction, factor.error)
            }, color = LabGreen, fontSize = 12.sp)
        }
    }
}

@Composable
private fun ProbabilityPanel4(kind: PhaseFourAlgorithmKind, bayes: BayesState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Prior x Likelihood -> Posterior", color = LabText, fontWeight = FontWeight.Bold)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                MetricPill("Prior", "%.3f".format(bayes.prior), LabCyan, Modifier.weight(1f))
                MetricPill("TP count", "%.3f".format(bayes.truePositive), LabGreen, Modifier.weight(1f))
                MetricPill("FP count", "%.3f".format(bayes.falsePositive), LabPink, Modifier.weight(1f))
                MetricPill("Posterior", "%.3f".format(bayes.posterior), LabOrange, Modifier.weight(1f))
            }
            Text("This is a generic educational probability example, not domain advice.", color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun StateSummary4(kind: PhaseFourAlgorithmKind, iso: IsolationState, lof: LofState, envelope: EnvelopeState, bayes: BayesState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("Selected calculation", color = LabText, fontWeight = FontWeight.Bold)
            when (kind) {
                PhaseFourAlgorithmKind.IsolationForest -> Text("Average path length %.1f, score threshold %.2f. ${iso.selectedPaths.first().splits.take(3)}".format(iso.selectedPaths.map { it.pathLength }.average(), iso.threshold), color = LabMuted, fontSize = 12.sp)
                PhaseFourAlgorithmKind.Lof -> Text("k-distance %.2f, LRD %.2f, neighbor density %.2f, LOF %.2f".format(lof.kDistance, lof.localReachabilityDensity, lof.neighborDensity, lof.lof), color = LabMuted, fontSize = 12.sp)
                PhaseFourAlgorithmKind.EllipticEnvelope -> Text("Mahalanobis threshold %.2f from covariance ellipse centered at %.2f, %.2f".format(envelope.threshold, envelope.centerX, envelope.centerY), color = LabMuted, fontSize = 12.sp)
                else -> Text("Posterior = %.4f from %.4f / (%.4f + %.4f)".format(bayes.posterior, bayes.truePositive, bayes.truePositive, bayes.falsePositive), color = LabMuted, fontSize = 12.sp)
            }
        }
    }
}

@Composable
private fun BayesianStepPanel4(beta: BetaState, step: Int) {
    Info4("Beta-Bernoulli update", "After $step heads and 2 tails: Beta(alpha=%.1f, beta=%.1f), posterior mean %.3f".format(beta.alpha, beta.beta, beta.mean))
}

@Composable
private fun HmmPanel4(state: HmmState) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text("HMM Forward / Viterbi", color = LabText, fontWeight = FontWeight.Bold)
            state.forward.forEachIndexed { i, probs -> Text("t${i + 1}: ${probs.mapValues { "%.2f".format(it.value) }} best=${state.viterbi[i]}", color = LabMuted, fontSize = 12.sp) }
        }
    }
}

@Composable
private fun GpCanvas4(state: GpState) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        val path = Path()
        state.xs.forEachIndexed { i, x ->
            val px = size.width * i / state.xs.lastIndex
            val py = size.height * (.5f - state.mean[i].toFloat() * .25f)
            val band = (state.variance[i] * size.height * .12).toFloat()
            drawLine(LabPurple.copy(alpha = .18f), Offset(px, py - band), Offset(px, py + band), 3f)
            if (i == 0) path.moveTo(px, py) else path.lineTo(px, py)
        }
        drawPath(path, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
    }
}

@Composable
private fun McmcPanel4(step: McmcStep) {
    Info4("Metropolis-Hastings step", "Current %.2f density %.3f -> proposal %.2f density %.3f. ratio %.2f, u %.2f, %s".format(step.current, step.currentDensity, step.proposal, step.proposalDensity, step.ratio, step.u, if (step.accepted) "accepted" else "rejected"))
}

@Composable
private fun GibbsCanvas4(state: GibbsState) {
    Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(8.dp)).border(1.dp, LabBorder, RoundedCornerShape(8.dp)).padding(8.dp)) {
        fun sx(x: Double) = (size.width * (x + 1.2) / 2.4).toFloat()
        fun sy(y: Double) = (size.height * (1.0 - (y + 1.2) / 2.4)).toFloat()
        state.path.zipWithNext().forEach { (a, b) -> drawLine(LabCyan, Offset(sx(a.first), sy(a.second)), Offset(sx(b.first), sy(b.second)), 3f) }
        state.path.forEach { drawCircle(LabOrange, 4f, Offset(sx(it.first), sy(it.second))) }
    }
}

@Composable
private fun ViPanel4(state: VariationalState) {
    Info4("Variational approximation", "Target mean %.2f, q mean %.2f, q variance %.2f, ELBO proxy %.3f".format(state.targetMean, state.approximateMean, state.approximateVariance, state.elboProxy))
}

@Composable
private fun GenericStep4(kind: PhaseFourAlgorithmKind) {
    Info4("Step calculation", mechanism4(kind))
}

@Composable
private fun Slider4(label: String, value: Double, min: Double, max: Double, onChange: (Double) -> Unit) {
    Column {
        Text("$label: %.2f".format(value), color = LabText, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        Slider(value.toFloat(), { onChange(it.toDouble().coerceIn(min, max)) }, valueRange = min.toFloat()..max.toFloat())
    }
}

@Composable
private fun Equation4(text: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Box(Modifier.fillMaxWidth().background(Color(0xFF081126), RoundedCornerShape(7.dp)).border(1.dp, LabBorder, RoundedCornerShape(7.dp)).padding(12.dp)) {
            Text(text, color = LabCyan, fontSize = 14.sp, fontWeight = FontWeight.Bold)
        }
    }
}

@Composable
private fun Info4(title: String, body: String) {
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold)
            Text(body, color = LabMuted, fontSize = 13.sp)
        }
    }
}

private fun family4(kind: PhaseFourAlgorithmKind) = when (kind.family) {
    "Anomaly Detection" -> "anomaly"
    "Association Rule Mining" -> "association"
    "Recommendation Systems" -> "recommender"
    else -> "bayes"
}

private fun coreQuestion4(kind: PhaseFourAlgorithmKind) = when (family4(kind)) {
    "anomaly" -> "Why is this point considered unusual?"
    "association" -> "Which item combinations occur often enough to become useful rules?"
    "recommender" -> "Why was this item recommended to this user?"
    else -> "How did prior belief, likelihood, or transition structure become posterior/inferred state?"
}

private fun mechanism4(kind: PhaseFourAlgorithmKind) = when (kind) {
    PhaseFourAlgorithmKind.IsolationForest -> "Random splits isolate unusual samples quickly; shorter path length raises anomaly score."
    PhaseFourAlgorithmKind.Lof -> "LOF compares selected-point density against neighbor density."
    PhaseFourAlgorithmKind.EllipticEnvelope -> "Mahalanobis contours mark points far from a robust Gaussian-like center."
    PhaseFourAlgorithmKind.Apriori -> "Frequent itemsets are grown level by level; infrequent candidates are pruned."
    PhaseFourAlgorithmKind.FpGrowth -> "Transactions are compressed into an FP-tree rather than repeatedly generating candidates."
    PhaseFourAlgorithmKind.Eclat -> "Item transaction-ID sets are intersected to compute support."
    PhaseFourAlgorithmKind.MatrixFactorization, PhaseFourAlgorithmKind.Als, PhaseFourAlgorithmKind.SvdRecommendation -> "A sparse user-item matrix is approximated by low-dimensional user and item factors."
    PhaseFourAlgorithmKind.Hmm -> "Forward probabilities track hidden-state belief; Viterbi keeps the best path."
    PhaseFourAlgorithmKind.GaussianProcesses -> "Kernel covariance shrinks uncertainty near observations and leaves it wider far away."
    PhaseFourAlgorithmKind.MetropolisHastings, PhaseFourAlgorithmKind.Mcmc -> "A proposal is accepted or rejected according to the target-density ratio."
    PhaseFourAlgorithmKind.GibbsSampling -> "Coordinates update from conditional distributions in alternating directions."
    PhaseFourAlgorithmKind.VariationalInference -> "A simpler distribution is optimized to approximate a complex posterior."
    else -> "The lab exposes the current calculation and its explanation from the offline toy data."
}

private fun equation4(kind: PhaseFourAlgorithmKind) = when (kind) {
    PhaseFourAlgorithmKind.BayesTheorem -> "P(A|B) = P(B|A)P(A) / P(B)"
    PhaseFourAlgorithmKind.BayesianInference -> "Beta(alpha,beta) + H/T observations -> Beta(alpha+H,beta+T)"
    PhaseFourAlgorithmKind.Mle -> "theta_MLE = argmax P(data | theta)"
    PhaseFourAlgorithmKind.Map -> "theta_MAP = argmax P(data | theta)P(theta)"
    PhaseFourAlgorithmKind.Hmm -> "alpha_t(s) = emission(o_t|s) * sum alpha_(t-1)(s') transition(s'->s)"
    PhaseFourAlgorithmKind.GaussianProcesses -> "posterior mean/variance derive from kernel covariance K(x,x')"
    PhaseFourAlgorithmKind.Apriori, PhaseFourAlgorithmKind.AssociationRules -> "confidence(A->B) = support(A union B) / support(A)"
    PhaseFourAlgorithmKind.Lof -> "LOF_k(p) = average neighbor LRD / LRD(p)"
    PhaseFourAlgorithmKind.Iqr -> "IQR = Q3 - Q1; fences = Q1 - 1.5 IQR, Q3 + 1.5 IQR"
    PhaseFourAlgorithmKind.ZScore -> "z = (x - mean) / sigma"
    else -> "score/state = model(data, parameters)"
}

private fun break4(kind: PhaseFourAlgorithmKind) = when (family4(kind)) {
    "anomaly" -> "Try local anomalies, non-elliptical data, or clustered anomalies; global and local methods disagree for good reasons."
    "association" -> "Very low support can cause candidate explosion; high confidence can still be uninteresting if lift is near 1."
    "recommender" -> "New users/items and extreme sparsity expose cold-start and missing-signal failures."
    else -> "A strong wrong prior, ambiguous emissions, poor kernel, or bad proposal scale can make inference misleading or slow."
}

private fun fix4(kind: PhaseFourAlgorithmKind) = when (family4(kind)) {
    "anomaly" -> "Compare Isolation Forest, LOF, and covariance assumptions; tune threshold/k/contamination on validation labels if available."
    "association" -> "Raise support, inspect lift, and compare Apriori candidate counts with FP-tree compression."
    "recommender" -> "Blend popularity/content for cold start, and use factor models only when enough interactions exist."
    else -> "Show the current calculation, test sensitivity to priors/parameters, and compare posterior, sampling, and approximation views."
}

private fun compare4(kind: PhaseFourAlgorithmKind) = when (family4(kind)) {
    "anomaly" -> "Isolation Forest asks how quickly a point is isolated; LOF asks whether local density is low; Elliptic Envelope assumes Gaussian-like contours."
    "association" -> "Apriori prunes candidates, FP-Growth compresses transactions, and ECLAT intersects transaction-ID sets."
    "recommender" -> "Popularity is non-personalized; content uses features; collaborative filtering uses neighbors; matrix factorization uses latent compatibility."
    else -> "MLE uses likelihood only, MAP adds a prior, Bayesian posterior keeps uncertainty, MCMC samples it, and VI approximates it parametrically."
}
