package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.*
import kotlinx.coroutines.delay

internal val stageColors = listOf(LabCyan, LabPink, LabGreen, LabOrange, LabPurple)

internal fun stageColor(label: Int): Color =
    if (label < 0) LabMuted else stageColors[label % stageColors.size]

@Composable
internal fun StageTwoPlot(
    points: List<ClusterPoint>,
    labels: List<Int> = emptyList(),
    selected: Int = -1,
    modifier: Modifier = Modifier,
    onMove: ((Int, ClusterPoint) -> Unit)? = null,
    onTap: ((ClusterPoint) -> Unit)? = null,
    drawOverlay: DrawScope.(PlotScale) -> Unit = {}
) {
    Canvas(
        modifier.fillMaxWidth().height(300.dp)
            .background(Color(0xFF081527), RoundedCornerShape(15.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(15.dp))
            .pointerInput(onTap) {
                val scale = PlotScale(size.width.toFloat(), size.height.toFloat())
                detectTapGestures { at ->
                    val point = scale.from(at)
                    onTap?.invoke(ClusterPoint(point.x,point.y))
                }
            }
            .pointerInput(points.size, onMove) {
                val scale = PlotScale(size.width.toFloat(), size.height.toFloat())
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    val nearest = points.indices.minByOrNull {
                        (scale.point(points[it].x, points[it].y) - down.position).getDistanceSquared()
                    } ?: -1
                    if (onMove != null && nearest >= 0 &&
                        (scale.point(points[nearest].x, points[nearest].y) - down.position).getDistanceSquared() < 26f * 26f) {
                        val start = awaitTouchSlopOrCancellation(down.id) { change, _ -> change.consume() }
                        if (start != null) {
                            drag(start.id) { change ->
                                val at = scale.from(change.position)
                                onMove(nearest, points[nearest].copy(x = at.x, y = at.y))
                                change.consume()
                            }
                        }
                    }
                }
            }
    ) {
        val scale = PlotScale(size.width, size.height)
        for (tick in -1..1) {
            drawLine(Color.White.copy(alpha = .08f), scale.point(tick.toDouble(), -1.0), scale.point(tick.toDouble(), 1.0))
            drawLine(Color.White.copy(alpha = .08f), scale.point(-1.0, tick.toDouble()), scale.point(1.0, tick.toDouble()))
        }
        drawLine(LabMuted, Offset(scale.left, scale.bottom), Offset(scale.right, scale.bottom), 2f)
        drawLine(LabMuted, Offset(scale.left, scale.top), Offset(scale.left, scale.bottom), 2f)
        drawOverlay(scale)
        points.forEachIndexed { i, point ->
            val at = scale.point(point.x, point.y)
            val color = stageColor(labels.getOrElse(i) { point.cluster })
            if (i == selected) drawCircle(Color.White, 11f, at, style = Stroke(2.5f))
            drawCircle(color.copy(alpha = .2f), 10f, at)
            drawCircle(color, 5.5f, at)
        }
    }
}

@Composable
internal fun StageTwoInfo(title: String, detail: String, modifier: Modifier = Modifier) {
    GlassPanel(modifier.fillMaxWidth()) {
        Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text(title, color = LabText, fontWeight = FontWeight.Bold, fontSize = 15.sp)
            Text(detail, color = LabMuted, fontSize = 12.sp, lineHeight = 18.sp)
        }
    }
}

@Composable
internal fun StageTwoStepControls(step: Int, last: Int, onStep: (Int) -> Unit) {
    var auto by remember { mutableStateOf(false) }
    LaunchedEffect(auto, last, step) {
        while (auto) {
            delay(750)
            if (step >= last) auto = false else onStep(step + 1)
        }
    }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) {
        SegmentedOption("Reset", false, Modifier.weight(1f)) { auto = false; onStep(0) }
        SegmentedOption("Previous", false, Modifier.weight(1f)) { auto = false; onStep((step - 1).coerceAtLeast(0)) }
        SegmentedOption("Next", false, Modifier.weight(1f)) { auto = false; onStep((step + 1).coerceAtMost(last)) }
        SegmentedOption(if (auto) "Pause" else "Auto", auto, Modifier.weight(1f)) { auto = !auto }
    }
}
