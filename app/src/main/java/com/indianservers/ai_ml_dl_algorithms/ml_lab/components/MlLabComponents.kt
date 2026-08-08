package com.indianservers.ai_ml_dl_algorithms.ml_lab.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.Point2D
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.TrainingSnapshot
import kotlin.math.max

val LabBackground = Color(0xFF050B18)
val LabPanel = Color(0xFF111A31)
val LabPanelSoft = Color(0xFF17213B)
val LabBorder = Color(0xFF2A365A)
val LabText = Color(0xFFF5F7FF)
val LabMuted = Color(0xFFA9B3C8)
val LabBlue = Color(0xFF2F7BFF)
val LabPurple = Color(0xFF9B3DFF)
val LabCyan = Color(0xFF20D9E8)
val LabGreen = Color(0xFF35E58F)
val LabOrange = Color(0xFFFFA52E)
val LabPink = Color(0xFFFF48BE)

@Composable
fun LabGradientBackground(content: @Composable () -> Unit) {
    Box(
        Modifier.background(
            Brush.verticalGradient(
                listOf(Color(0xFF020612), Color(0xFF07122B), Color(0xFF040814))
            )
        )
    ) {
        Canvas(Modifier.matchParentSize()) {
            drawCircle(LabBlue.copy(alpha = 0.18f), radius = size.width * 0.55f, center = Offset(size.width * 0.12f, -size.height * 0.1f))
            drawCircle(LabPurple.copy(alpha = 0.15f), radius = size.width * 0.45f, center = Offset(size.width * 0.98f, size.height * 0.18f))
            for (i in 0..36) {
                val x = size.width * i / 36f
                val y = size.height * (0.12f + 0.03f * kotlin.math.sin(i * 0.8f))
                drawCircle(LabCyan.copy(alpha = 0.18f), 1.6f, Offset(x, y))
            }
        }
        content()
    }
}

@Composable
fun GlassPanel(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Box(
        modifier
            .background(LabPanel.copy(alpha = 0.82f), RoundedCornerShape(8.dp))
            .border(1.dp, LabBorder.copy(alpha = 0.85f), RoundedCornerShape(8.dp))
            .padding(14.dp)
    ) {
        content()
    }
}

@Composable
fun GradientButton(text: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Box(
        modifier
            .height(48.dp)
            .background(Brush.horizontalGradient(listOf(LabBlue, LabPurple)), RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(text, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
    }
}

@Composable
fun SegmentedOption(text: String, selected: Boolean, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Box(
        modifier
            .background(if (selected) LabPurple else LabPanelSoft, RoundedCornerShape(8.dp))
            .border(1.dp, if (selected) Color.White.copy(alpha = 0.16f) else LabBorder, RoundedCornerShape(8.dp))
            .clickable(onClick = onClick)
            .padding(horizontal = 12.dp, vertical = 9.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(text, color = if (selected) Color.White else LabMuted, fontSize = 12.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

@Composable
fun MetricPill(label: String, value: String, accent: Color, modifier: Modifier = Modifier) {
    Column(
        modifier
            .background(accent.copy(alpha = 0.14f), RoundedCornerShape(8.dp))
            .border(1.dp, accent.copy(alpha = 0.32f), RoundedCornerShape(8.dp))
            .padding(10.dp)
    ) {
        Text(label, color = LabMuted, fontSize = 11.sp)
        Text(value, color = LabText, fontWeight = FontWeight.Bold, fontSize = 16.sp, maxLines = 1)
    }
}

@Composable
fun SectionTitle(title: String, subtitle: String? = null) {
    Column {
        Text(title, color = LabText, fontSize = 18.sp, fontWeight = FontWeight.Bold)
        if (subtitle != null) {
            Text(subtitle, color = LabMuted, fontSize = 12.sp)
        }
    }
}

@Composable
fun DatasetGraph(
    points: List<Point2D>,
    modifier: Modifier = Modifier,
    line: TrainingSnapshot? = null,
    selectedNeighbours: List<Point2D> = emptyList(),
    pcaDirection: Point2D? = null,
    onPointAdded: ((Point2D) -> Unit)? = null
) {
    Canvas(
        modifier
            .height(230.dp)
            .fillMaxWidth()
            .background(Color(0xFF081126), RoundedCornerShape(8.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(8.dp))
            .clickable(enabled = onPointAdded != null) {
                onPointAdded?.invoke(Point2D(0.35f, 0.2f, points.size % 2))
            }
            .padding(8.dp)
    ) {
        fun sx(x: Float) = size.width * (x + 1f) / 2f
        fun sy(y: Float) = size.height * (1f - (y + 1f) / 2f)
        val grid = Color.White.copy(alpha = 0.08f)
        for (i in 0..4) {
            val x = size.width * i / 4f
            val y = size.height * i / 4f
            drawLine(grid, Offset(x, 0f), Offset(x, size.height), 1f)
            drawLine(grid, Offset(0f, y), Offset(size.width, y), 1f)
        }
        drawLine(Color.White.copy(alpha = 0.18f), Offset(0f, sy(0f)), Offset(size.width, sy(0f)), 1.4f)
        drawLine(Color.White.copy(alpha = 0.18f), Offset(sx(0f), 0f), Offset(sx(0f), size.height), 1.4f)

        line?.let {
            val y1 = it.weight * -1f + it.bias
            val y2 = it.weight * 1f + it.bias
            drawLine(LabGreen, Offset(sx(-1f), sy(y1)), Offset(sx(1f), sy(y2)), 4f, cap = StrokeCap.Round)
            points.forEach { point ->
                val predicted = it.weight * point.x + it.bias
                drawLine(LabPink.copy(alpha = 0.38f), Offset(sx(point.x), sy(point.y)), Offset(sx(point.x), sy(predicted)), 1.5f)
            }
        }

        pcaDirection?.let {
            val center = Offset(sx(0f), sy(0f))
            val end = Offset(sx(it.x * 0.92f), sy(it.y * 0.92f))
            val start = Offset(sx(-it.x * 0.92f), sy(-it.y * 0.92f))
            drawLine(LabOrange, start, end, 5f, cap = StrokeCap.Round)
            drawCircle(LabOrange, 6f, center)
        }

        points.forEach { point ->
            val color = if (point.label == 0) LabCyan else if (point.label == 1) LabPink else LabOrange
            val selected = selectedNeighbours.any { it.x == point.x && it.y == point.y && it.label == point.label }
            drawCircle(if (selected) Color.White else color.copy(alpha = 0.22f), if (selected) 12f else 10f, Offset(sx(point.x), sy(point.y)))
            drawCircle(color, if (selected) 7f else 5.5f, Offset(sx(point.x), sy(point.y)))
        }
    }
}

@Composable
fun LossChart(snapshots: List<TrainingSnapshot>, selectedEpoch: Int, modifier: Modifier = Modifier) {
    Canvas(
        modifier
            .height(132.dp)
            .fillMaxWidth()
            .background(Color(0xFF081126), RoundedCornerShape(8.dp))
            .border(1.dp, LabBorder, RoundedCornerShape(8.dp))
            .padding(8.dp)
    ) {
        if (snapshots.size < 2) return@Canvas
        val maxLoss = max(0.001f, snapshots.maxOf { it.loss })
        val path = Path()
        snapshots.forEachIndexed { index, snapshot ->
            val x = size.width * index / (snapshots.lastIndex.toFloat())
            val y = size.height * (1f - snapshot.loss / maxLoss)
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, LabCyan, style = Stroke(4f, cap = StrokeCap.Round))
        val selected = snapshots[selectedEpoch.coerceIn(snapshots.indices)]
        val x = size.width * selected.epoch / snapshots.lastIndex.toFloat()
        val y = size.height * (1f - selected.loss / maxLoss)
        drawCircle(LabPurple, 9f, Offset(x, y))
        drawCircle(Color.White, 3.5f, Offset(x, y))
    }
}

@Composable
fun HeroPipeline(modifier: Modifier = Modifier) {
    GlassPanel(modifier) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            SectionTitle("ML & Deep Learning", "Learn - Train - Visualize - Infer")
            Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
                PipelineNode("Dataset", LabCyan)
                PipelineNode("Model", LabPurple)
                PipelineNode("Boundary", LabGreen)
                PipelineNode("Prediction", LabOrange)
            }
            Canvas(Modifier.fillMaxWidth().height(96.dp)) {
                val mid = size.height / 2f
                drawLine(LabCyan.copy(alpha = 0.5f), Offset(28f, mid), Offset(size.width - 28f, mid), 3f, cap = StrokeCap.Round)
                for (i in 0..8) {
                    val x = 26f + (size.width - 52f) * i / 8f
                    val y = mid + kotlin.math.sin(i.toFloat()) * 28f
                    drawCircle(if (i % 2 == 0) LabCyan else LabPink, 5f, Offset(x, y))
                }
                drawRect(LabPurple.copy(alpha = 0.28f), topLeft = Offset(size.width * 0.43f, 18f), size = Size(size.width * 0.14f, 60f))
            }
        }
    }
}

@Composable
private fun PipelineNode(title: String, color: Color) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            Modifier
                .size(42.dp)
                .background(color.copy(alpha = 0.16f), RoundedCornerShape(8.dp))
                .border(1.dp, color.copy(alpha = 0.45f), RoundedCornerShape(8.dp)),
            contentAlignment = Alignment.Center
        ) {
            Text(title.first().toString(), color = color, fontWeight = FontWeight.Bold)
        }
        Spacer(Modifier.height(6.dp))
        Text(title, color = LabMuted, fontSize = 11.sp, style = MaterialTheme.typography.bodySmall)
    }
}
