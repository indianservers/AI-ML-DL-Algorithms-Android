package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.awaitTouchSlopOrCancellation
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import kotlin.math.pow
import kotlin.math.roundToInt

internal data class PlotScale(val width: Float, val height: Float) {
    val left = 29f
    val right = width - 12f
    val top = 13f
    val bottom = height - 27f
    fun x(value: Double) = left + ((value + 1.0) / 2.0 * (right - left)).toFloat()
    fun y(value: Double) = bottom - ((value + 1.0) / 2.0 * (bottom - top)).toFloat()
    fun point(x: Double, y: Double) = Offset(x(x), y(y))
    fun from(offset: Offset): LabPoint {
        val x = ((offset.x - left) / (right - left) * 2.0 - 1.0).coerceIn(-1.0, 1.0)
        val y = ((bottom - offset.y) / (bottom - top) * 2.0 - 1.0).coerceIn(-1.0, 1.0)
        return LabPoint(x, y, target = y)
    }
}

@Composable
internal fun NativeSupervisedPlot(
    points: MutableList<LabPoint>,
    classification: Boolean,
    xLabel: String = "Feature 1",
    yLabel: String = if (classification) "Feature 2" else "Target",
    selectedIndices: Set<Int> = emptySet(),
    query: LabPoint? = null,
    onQuery: ((LabPoint) -> Unit)? = null,
    onPointAdded: ((LabPoint) -> Unit)? = null,
    onPointChanged: ((Int, LabPoint) -> Unit)? = null,
    onPointRemoved: ((Int) -> Unit)? = null,
    onBackgroundTap: ((LabPoint) -> Unit)? = null,
    overlay: DrawScope.(PlotScale) -> Unit
) {
    Column(Modifier.fillMaxWidth()) {
        Canvas(
            Modifier.fillMaxWidth().height(310.dp)
                .background(Color(0xFF081527), RoundedCornerShape(15.dp))
                .border(1.dp, LabBorder, RoundedCornerShape(15.dp))
                .pointerInput(points.size, query, onQuery, onPointAdded, onBackgroundTap) {
                    val scale = PlotScale(size.width.toFloat(), size.height.toFloat())
                    detectTapGestures(
                        onTap = { tap ->
                            val data = scale.from(tap)
                            val nearest = points.indices.minByOrNull { i ->
                                (points[i].x - data.x).pow(2) +
                                    ((if (classification) points[i].y else points[i].target) - data.y).pow(2)
                            }
                            val close = nearest != null &&
                                (points[nearest].x - data.x).pow(2) +
                                ((if (classification) points[nearest].y else points[nearest].target) - data.y).pow(2) < 0.012
                            when {
                                close -> Unit
                                onBackgroundTap != null -> onBackgroundTap(data)
                                onQuery != null -> onQuery(data)
                                else -> onPointAdded?.invoke(data)
                            }
                        },
                        onLongPress = { tap ->
                            val data = scale.from(tap)
                            val nearest = points.indices.minByOrNull { i ->
                                (points[i].x - data.x).pow(2) +
                                    ((if (classification) points[i].y else points[i].target) - data.y).pow(2)
                            }
                            if (nearest != null &&
                                (points[nearest].x - data.x).pow(2) +
                                ((if (classification) points[nearest].y else points[nearest].target) - data.y).pow(2) < .012)
                                onPointRemoved?.invoke(nearest)
                        }
                    )
                }
                .pointerInput(points.size, classification, onPointChanged) {
                    val scale = PlotScale(size.width.toFloat(), size.height.toFloat())
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        val nearest = points.indices.minByOrNull { i ->
                            val sample = points[i]
                            (scale.point(sample.x, if (classification) sample.y else sample.target) - down.position).getDistanceSquared()
                        } ?: -1
                        if (onPointChanged != null && nearest >= 0 &&
                            (scale.point(points[nearest].x, if (classification) points[nearest].y else points[nearest].target) -
                                down.position).getDistanceSquared() < 28f * 28f) {
                            val start = awaitTouchSlopOrCancellation(down.id) { change, _ -> change.consume() }
                            if (start != null) {
                                drag(start.id) { change ->
                                    val moved = scale.from(change.position)
                                    val old = points[nearest]
                                    onPointChanged(nearest, if (classification)
                                        old.copy(x = moved.x, y = moved.y)
                                    else old.copy(x = moved.x, y = moved.y, target = moved.y))
                                    change.consume()
                                }
                            }
                        }
                    }
                }
                .padding(1.dp)
        ) {
            val plot = PlotScale(size.width, size.height)
            for (tick in -1..1) {
                drawLine(Color.White.copy(alpha = .09f), plot.point(tick.toDouble(), -1.0), plot.point(tick.toDouble(), 1.0))
                drawLine(Color.White.copy(alpha = .09f), plot.point(-1.0, tick.toDouble()), plot.point(1.0, tick.toDouble()))
            }
            drawLine(LabMuted, Offset(plot.left, plot.bottom), Offset(plot.right, plot.bottom), 2f)
            drawLine(LabMuted, Offset(plot.left, plot.top), Offset(plot.left, plot.bottom), 2f)
            clipRect(plot.left, plot.top, plot.right, plot.bottom) {
                overlay(plot)
            }
            points.forEachIndexed { index, sample ->
                val location = plot.point(sample.x, if (classification) sample.y else sample.target)
                val color = if (classification) {
                    if (sample.label == 0) LabCyan else LabPink
                } else LabCyan
                if (index in selectedIndices) drawCircle(Color.White, 11f, location, style = Stroke(2.5f))
                drawCircle(color.copy(alpha = .22f), 10f, location)
                drawCircle(color, 5.5f, location)
            }
            query?.let {
                val location = plot.point(it.x, it.y)
                drawCircle(Color.White, 11f, location, style = Stroke(3f))
                drawCircle(LabGreen, 6f, location)
            }
        }
        Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(yLabel, color = LabMuted, fontSize = 11.sp)
            Text(xLabel, color = LabMuted, fontSize = 11.sp)
        }
    }
}

internal fun DrawScope.drawFunction(
    plot: PlotScale, function: (Double) -> Double,
    color: Color, stroke: Float = 3f
) {
    val path = Path()
    repeat(101) { i ->
        val x = -1.0 + 2.0 * i / 100.0
        val location = plot.point(x, function(x).coerceIn(-2.0, 2.0))
        if (i == 0) path.moveTo(location.x, location.y) else path.lineTo(location.x, location.y)
    }
    drawPath(path, color, style = Stroke(stroke))
}

internal fun DrawScope.drawStepFunction(
    plot: PlotScale, function: (Double) -> Double,
    color: Color, stroke: Float = 3f
) {
    val path = Path()
    val first = plot.point(-1.0, function(-1.0))
    path.moveTo(first.x, first.y)
    var previous = first
    repeat(240) { index ->
        val x = -1.0 + 2.0 * (index + 1) / 240.0
        val next = plot.point(x, function(x))
        path.lineTo(next.x, previous.y)
        path.lineTo(next.x, next.y)
        previous = next
    }
    drawPath(path, color, style = Stroke(stroke))
}

internal fun DrawScope.drawUncertainty(
    plot: PlotScale, predictions: List<UncertainPrediction>, multiplier: Double = 1.96
) {
    if (predictions.isEmpty()) return
    val fill = Path()
    predictions.forEachIndexed { i, value ->
        val point = plot.point(value.x, value.mean + multiplier * value.standardDeviation)
        if (i == 0) fill.moveTo(point.x, point.y) else fill.lineTo(point.x, point.y)
    }
    predictions.asReversed().forEach { value ->
        val point = plot.point(value.x, value.mean - multiplier * value.standardDeviation)
        fill.lineTo(point.x, point.y)
    }
    fill.close()
    drawPath(fill, LabPurple.copy(alpha = .22f))
    val mean = Path()
    predictions.forEachIndexed { i, value ->
        val point = plot.point(value.x, value.mean)
        if (i == 0) mean.moveTo(point.x, point.y) else mean.lineTo(point.x, point.y)
    }
    drawPath(mean, LabGreen, style = Stroke(3.5f))
}

internal fun DrawScope.drawTreePartitions(
    plot: PlotScale, tree: SupervisedTreeNode, selectedPath: String = ""
) {
    fun visit(node: SupervisedTreeNode, x0: Double, x1: Double, y0: Double, y1: Double, path: String) {
        if (node.isLeaf) {
            val color = if (node.prediction.roundToIntCompat() == 0) LabCyan else LabPink
            drawRect(color.copy(alpha = if (selectedPath == path) .36f else .12f),
                plot.point(x0, y1), Size(plot.x(x1) - plot.x(x0), plot.y(y0) - plot.y(y1)))
            return
        }
        val threshold = node.threshold ?: return
        if (node.feature == 0) {
            drawLine(LabPurple.copy(alpha = .9f), plot.point(threshold, y0), plot.point(threshold, y1), 2.3f)
            node.left?.let { visit(it, x0, threshold, y0, y1, path + "L") }
            node.right?.let { visit(it, threshold, x1, y0, y1, path + "R") }
        } else {
            drawLine(LabPurple.copy(alpha = .9f), plot.point(x0, threshold), plot.point(x1, threshold), 2.3f)
            node.left?.let { visit(it, x0, x1, y0, threshold, path + "L") }
            node.right?.let { visit(it, x0, x1, threshold, y1, path + "R") }
        }
    }
    visit(tree, -1.0, 1.0, -1.0, 1.0, "")
}

private fun Double.roundToIntCompat() = this.roundToInt()

internal fun SupervisedTreeNode.pathFor(x: Double, y: Double): String {
    var node = this
    val path = StringBuilder()
    while (!node.isLeaf) {
        val coordinate = if (node.feature == 0) x else y
        val left = coordinate <= (node.threshold ?: 0.0)
        path.append(if (left) "L" else "R")
        node = (if (left) node.left else node.right) ?: break
    }
    return path.toString()
}

@Composable
internal fun NativeTreeDiagram(
    root: SupervisedTreeNode,
    regression: Boolean,
    selectedPath: String,
    onSelected: (String) -> Unit
) {
    val visible = root.nodes().size.coerceAtMost(31)
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Text("Decision tree · $visible nodes", color = LabText, fontWeight = FontWeight.Bold, fontSize = 13.sp)
        fun collect(node: SupervisedTreeNode?, path: String, output: MutableList<Pair<String, SupervisedTreeNode>>) {
            if (node == null || output.size >= 31) return
            output += path to node
            collect(node.left, path + "L", output)
            collect(node.right, path + "R", output)
        }
        val rows = remember(root) { mutableListOf<Pair<String, SupervisedTreeNode>>().also { collect(root, "", it) } }
        rows.forEach { (path, node) ->
            val title = if (node.isLeaf) {
                if (regression) "Leaf → ${"%.2f".format(node.prediction)}"
                else "Leaf → Class ${node.prediction.toInt()}"
            } else "${if (node.feature == 0) "X1" else "X2"} ≤ ${"%.2f".format(node.threshold)}"
            val detail = "n=${node.indices.size} · impurity ${"%.3f".format(node.impurity)}" +
                if (regression) "" else " · ${node.counts.toSortedMap().entries.joinToString { "C${it.key}:${it.value}" }}"
            Column(
                Modifier.fillMaxWidth().padding(start = (path.length * 13).dp)
                    .background(if (selectedPath == path) LabPurple.copy(alpha = .35f) else LabPanelSoft, RoundedCornerShape(9.dp))
                    .border(1.dp, if (selectedPath == path) LabCyan else LabBorder, RoundedCornerShape(9.dp))
                    .clickable { onSelected(path) }.padding(9.dp)
            ) {
                Text(title, color = LabText, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
                Text(detail, color = LabMuted, fontSize = 10.sp)
            }
        }
    }
}
