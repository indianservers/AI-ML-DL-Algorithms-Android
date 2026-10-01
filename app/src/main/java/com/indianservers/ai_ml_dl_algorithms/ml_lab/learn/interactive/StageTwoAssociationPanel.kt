package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageTwoVisualization
import kotlin.math.roundToInt

@Composable
internal fun StageTwoAssociationPanel(kind: StageTwoVisualization) {
    var basketChoice by remember(kind) { mutableStateOf("Grocery") }
    var minSupport by remember(kind) { mutableDoubleStateOf(.2) }
    var minConfidence by remember(kind) { mutableDoubleStateOf(.55) }
    var step by remember(kind) { mutableIntStateOf(0) }
    var selectedRule by remember(kind) { mutableIntStateOf(0) }
    var showTransactions by remember(kind) { mutableStateOf(false) }
    val data = when(basketChoice) {
        "Shopping" -> StageTwoAssociationEngine.shoppingClicks
        "Courses" -> StageTwoAssociationEngine.courseBundles
        else -> StageTwoAssociationEngine.baskets
    }
    val frequent = remember(data,minSupport) { StageTwoAssociationEngine.frequent(data,minSupport) }
    val vertical = remember(data,minSupport) { StageTwoAssociationEngine.eclat(data,minSupport) }
    val rules = remember(data,minSupport,minConfidence) {
        StageTwoAssociationEngine.rules(data,minSupport,minConfidence)
    }
    val last = when(kind) {
        StageTwoVisualization.FpGrowth -> data.size
        StageTwoVisualization.Eclat -> vertical.size.coerceAtLeast(1)
        StageTwoVisualization.AssociationRules -> rules.size.coerceAtLeast(1)
        else -> frequent.maxOfOrNull { it.items.size } ?: 1
    }
    val current = step.coerceIn(0,last)
    LazyColumn(Modifier.fillMaxSize().padding(horizontal = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item { StageTwoInfo(kind.title,kind.graphic) }
        item {
            Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.spacedBy(6.dp)) {
                listOf("Grocery","Shopping","Courses").forEach { option ->
                    SegmentedOption(option,basketChoice==option,Modifier.weight(1f)) {
                        basketChoice=option; step=0; selectedRule=0
                    }
                }
            }
        }
        item {
            SegmentedOption(if (showTransactions) "Hide ${data.size} transactions" else "Show ${data.size} transactions",
                showTransactions, Modifier.fillMaxWidth()) { showTransactions = !showTransactions }
        }
        if (showTransactions) item {
            StageTwoInfo("Market baskets · " + data.size + " transactions",
                data.mapIndexed { i, basket -> "T" + (i+1) + ": " + basket.joinToString(", ") }
                    .joinToString("\n"))
        }
        item {
            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                NativeSlider("Minimum support",minSupport,.1,.6) { minSupport = it; step = 0 }
                if (kind == StageTwoVisualization.Apriori ||
                    kind == StageTwoVisualization.AssociationRules)
                    NativeSlider("Minimum confidence",minConfidence,.2,.95) {
                        minConfidence = it
                    }
            }
        }
        item { StageTwoStepControls(current,last) { step = it } }
        when(kind) {
            StageTwoVisualization.Apriori -> {
                item {
                    val candidate = data.flatten().distinct().sorted()
                    val level = current.coerceAtLeast(1)
                    StageTwoInfo("Candidate level " + level,
                        if (current == 0) "Start with singleton items, count support, then join frequent sets."
                        else frequent.filter { it.items.size == level }.joinToString("\n") {
                            it.items.sorted().joinToString(" + ") + " → " + it.count + "/" + data.size
                        }.ifEmpty { "No frequent candidates at this level; pruning stops." })
                }
                item { CandidateLattice(frequent,current) }
                item { StageTwoInfo("Apriori pruning",
                    "Candidates whose subset failed support are never extended. " +
                    frequent.size + " itemsets pass the current threshold.") }
            }
            StageTwoVisualization.FpGrowth -> {
                val tree = StageTwoAssociationEngine.fpTree(data,minSupport,current)
                item { StageTwoInfo("FP-tree after " + current + " insertions",
                    "Frequent items are sorted by corpus frequency before each transaction follows an existing prefix or creates a child.") }
                item { FpTreeRows(tree) }
                item { StageTwoInfo("Conditional prefix paths",
                    tree.children.values.flatMap { child ->
                        child.children.values.map { grandchild ->
                            child.item + " → " + grandchild.item + " : " + grandchild.count
                        }
                    }.joinToString("\n").ifEmpty { "Insert a transaction to build prefix paths." }) }
            }
            StageTwoVisualization.Eclat -> {
                item {
                    StageTwoInfo("Vertical transaction ID sets",
                        data.flatten().distinct().sorted().joinToString("\n") { item ->
                            item + " → " + StageTwoAssociationEngine.tidset(data,setOf(item))
                                .sorted().joinToString(prefix="{",postfix="}") { "T" + (it+1) }
                        })
                }
                item {
                    val itemset = vertical.getOrNull((current-1).coerceAtLeast(0))
                    StageTwoInfo("TID-set intersection",
                        if (itemset == null) "Select Next to intersect candidate transaction-ID sets."
                        else itemset.items.sorted().joinToString(" ∩ ") + " → " +
                            itemset.tids.sorted().joinToString(prefix="{",postfix="}") { "T" + (it+1) } +
                            " · support " + "%.2f".format(itemset.count.toDouble()/data.size))
                }
            }
            StageTwoVisualization.AssociationRules -> {
                item {
                    val shown = rules.getOrNull(selectedRule.coerceIn(0,(rules.size-1).coerceAtLeast(0)))
                    StageTwoInfo("Selected rule",
                        if (shown == null) "No rule passes these thresholds."
                        else shown.premise.joinToString(" + ") + " → " + shown.conclusion.joinToString() +
                            "\nSupport " + "%.2f".format(shown.support) +
                            " · confidence " + "%.2f".format(shown.confidence) +
                            " · lift " + "%.2f".format(shown.lift) +
                            "\nLift above 1 means the conclusion is more common after the premise than overall.")
                }
                items(rules.size) { index ->
                    val rule = rules[index]
                    SegmentedOption(rule.premise.joinToString("+") + " → " +
                        rule.conclusion.joinToString() + " · lift " + "%.2f".format(rule.lift),
                        selectedRule == index,Modifier.fillMaxWidth()) { selectedRule = index; step = index+1 }
                }
            }
            else -> Unit
        }
        item { Spacer(Modifier.height(22.dp)) }
    }
}

@Composable
private fun CandidateLattice(sets: List<Itemset>, step: Int) {
    Canvas(Modifier.fillMaxWidth().height(155.dp)) {
        val maxSize = sets.maxOfOrNull { it.items.size } ?: return@Canvas
        val counts = (1..maxSize).map { level -> sets.count { it.items.size == level } }
        val barWidth = size.width/(maxSize*2f)
        counts.forEachIndexed { i, count ->
            val x = (i+.5f)*size.width/maxSize
            val h = count.toFloat()/counts.max().coerceAtLeast(1)*size.height*.82f
            drawRect(if (i+1 <= step) LabCyan else LabMuted,
                Offset(x-barWidth/2,size.height-h), Size(barWidth,h))
        }
    }
}

@Composable
private fun FpTreeRows(root: BasketTreeNode) {
    val rows = mutableListOf<Pair<Int,BasketTreeNode>>()
    fun visit(node: BasketTreeNode,depth: Int) {
        rows += depth to node
        node.children.values.forEach { visit(it,depth+1) }
    }
    visit(root,0)
    GlassPanel(Modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Frequency prefix tree",color=LabText,fontSize=15.sp)
            rows.forEach { (depth,node) ->
                Text("  ".repeat(depth) + "↳ " + node.item + "  ×" + node.count,
                    color=if(depth==0) LabOrange else stageColor(depth),fontSize=12.sp)
            }
        }
    }
}
