package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.toMutableStateList
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import kotlin.math.max

@Composable
internal fun StageThreeForecastScreen(topic: LearnTopic, kind: ForecastVisualization) {
    var seed by remember(topic.id) { mutableIntStateOf(0) }
    var step by remember(topic.id) { mutableIntStateOf(40) }
    var control by remember(topic.id) { mutableIntStateOf(4) }
    var differenceOrder by remember(topic.id) { mutableIntStateOf(1) }
    var errorOrder by remember(topic.id) { mutableIntStateOf(1) }
    val seasonal = kind in setOf(ForecastVisualization.HoltWinters, ForecastVisualization.Sarima, ForecastVisualization.Prophet)
    val data = remember(seed, seasonal) { StageThreeForecastEngine.sample(seasonal, seed = seed).toMutableStateList() }
    val series = data.toList()
    val firstStep = if (kind == ForecastVisualization.Sarima) 25 else 12
    val parameter = when (kind) {
        ForecastVisualization.ExponentialSmoothing, ForecastVisualization.Holt, ForecastVisualization.HoltWinters -> control / 10.0
        ForecastVisualization.Kalman, ForecastVisualization.StateSpace -> control / 2.0
        else -> control.toDouble()
    }
    val frame = remember(kind, series, step, parameter, differenceOrder, errorOrder) {
        StageThreeForecastEngine.evaluate(kind, series, step, parameter, differenceOrder, errorOrder)
    }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, when (kind) {
            ForecastVisualization.MovingAverage -> "Average a sliding window of recent observations"
            ForecastVisualization.ExponentialSmoothing -> "Give recent observations exponentially more weight"
            ForecastVisualization.Holt -> "Update level and trend before forecasting forward"
            ForecastVisualization.HoltWinters -> "Combine level, trend, and repeating seasonality"
            ForecastVisualization.Ar -> "Predict from previous observed values"
            ForecastVisualization.Ma -> "Predict from previous forecast errors"
            ForecastVisualization.Arma -> "Combine observed lags and previous errors"
            ForecastVisualization.Arima -> "Difference the series, model residuals, then integrate"
            ForecastVisualization.Sarima -> "Remove seasonal structure before ARIMA forecasting"
            ForecastVisualization.Var -> "Predict each series from lags of multiple series"
            ForecastVisualization.StateSpace -> "Track a hidden state through transition and observation"
            ForecastVisualization.Kalman -> "Balance predicted state against a noisy measurement"
            ForecastVisualization.Prophet -> "Add trend, seasonality, and change points"
            ForecastVisualization.Lstm -> "Carry gated memory through successive time steps"
            ForecastVisualization.Gru -> "Blend prior hidden state with a gated candidate"
            ForecastVisualization.TemporalCnn -> "Use temporal filters over a fixed receptive field"
            ForecastVisualization.Transformer -> "Weight past positions with time-aware attention"
        }) }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    Text("Time series", color = LabText, fontWeight = FontWeight.Bold)
                    ForecastChart(frame)
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("Observed", color = LabCyan, fontSize = 12.sp)
                        Text("Model estimate", color = LabOrange, fontSize = 12.sp)
                        Text("Next: %.2f".format(frame.forecast.first()), color = LabText, fontSize = 12.sp)
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text("Explore the calculation", color = LabText, fontWeight = FontWeight.Bold)
                    Text(frame.explanation, color = LabMuted, fontSize = 13.sp)
                    Text(frame.equation, color = LabCyan, fontSize = 13.sp)
                    frame.components.forEach { (name, value) ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(name, color = LabMuted, fontSize = 12.sp)
                            Text("%.3f".format(value), color = LabText, fontSize = 12.sp)
                        }
                    }
                    if (kind == ForecastVisualization.Kalman) {
                        Text("Predict  →  Measure  →  Update", color = LabOrange, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                        Text("Uncertainty after update: ±%.2f".format(frame.uncertainty), color = LabMuted, fontSize = 12.sp)
                    }
                }
            }
        }
        if (frame.auxiliary.isNotEmpty()) item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
                    Text(frame.auxiliaryLabel, color = LabText, fontWeight = FontWeight.Bold)
                    ForecastBarChart(frame.auxiliary.takeLast(16))
                    Text(when (kind) {
                        ForecastVisualization.Arima, ForecastVisualization.Sarima -> "These are differences; the final forecast is restored to the original scale."
                        ForecastVisualization.Ma, ForecastVisualization.Arma -> "These residuals are prediction errors, not lagged observations."
                        ForecastVisualization.Transformer -> "Bar height is the normalized attention weight on one historical position."
                        else -> "Values come from the current series and update when the time step changes."
                    }, color = LabMuted, fontSize = 11.sp)
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("Time step ${step + 1} of ${data.size}", color = LabText, fontSize = 13.sp)
                    Slider(value = step.toFloat().coerceAtLeast(firstStep.toFloat()), onValueChange = { step = it.toInt().coerceIn(firstStep, data.lastIndex) }, valueRange = firstStep.toFloat()..data.lastIndex.toFloat())
                    Text("Observed value at step ${step+1}: %.2f".format(data[step]), color = LabText, fontSize = 13.sp)
                    Slider(value = data[step].toFloat().coerceIn(0f,35f), onValueChange = { data[step] = it.toDouble() }, valueRange = 0f..35f)
                    Text(controlLabel(kind, control), color = LabText, fontSize = 13.sp)
                    Slider(value = control.toFloat(), onValueChange = { control = it.toInt().coerceIn(2, 9) }, valueRange = 2f..9f)
                    if (kind == ForecastVisualization.Arima || kind == ForecastVisualization.Sarima) {
                        Text("Differencing d = $differenceOrder", color = LabText, fontSize = 13.sp)
                        Slider(value = differenceOrder.toFloat(), onValueChange = { differenceOrder = it.toInt().coerceIn(0, 2) }, valueRange = 0f..2f)
                        Text("Error lag order q = $errorOrder", color = LabText, fontSize = 13.sp)
                        Slider(value = errorOrder.toFloat(), onValueChange = { errorOrder = it.toInt().coerceIn(0, 4) }, valueRange = 0f..4f)
                    }
                    SegmentedOption("New deterministic series", false, Modifier.fillMaxWidth()) { seed++ }
                }
            }
        }
    }
}

private fun controlLabel(kind: ForecastVisualization, value: Int): String = when (kind) {
    ForecastVisualization.ExponentialSmoothing, ForecastVisualization.Holt, ForecastVisualization.HoltWinters -> "Smoothing α = %.1f".format(value / 10.0)
    ForecastVisualization.Kalman, ForecastVisualization.StateSpace -> "Measurement noise R = %.1f".format(value / 2.0)
    ForecastVisualization.TemporalCnn -> "Dilation = $value"
    ForecastVisualization.Ar, ForecastVisualization.Ma, ForecastVisualization.Arma -> "Lag order = $value"
    ForecastVisualization.Arima, ForecastVisualization.Sarima -> "AR order p = $value"
    ForecastVisualization.Transformer -> "Attention window = $value"
    else -> "Window = $value"
}

@Composable
private fun ForecastChart(frame: ForecastFrame) {
    Canvas(Modifier.fillMaxWidth().height(225.dp).background(Color(0xFF071B37))) {
        val values = frame.observed + frame.estimate + frame.forecast
        val lo = values.minOrNull()!! - 1.0
        val hi = values.maxOrNull()!! + 1.0
        val count = frame.observed.size
        fun point(i: Int, v: Double) = Offset(14f + (size.width - 28f) * i / count,
            (size.height - 17f - (size.height - 34f) * ((v - lo) / (hi - lo)).toFloat()))
        for (i in 0..4) {
            val y = 17f + (size.height - 34f) * i / 4f
            drawLine(Color(0xFF1C365B), Offset(12f, y), Offset(size.width - 12f, y), 1f)
        }
        frame.selectedInputs.filter { it in frame.observed.indices }.forEach { i ->
            val p = point(i, frame.observed[i])
            drawLine(Color(0x354CDBFF), Offset(p.x, 15f), Offset(p.x, size.height - 15f), 3f)
            drawCircle(LabCyan, 5f, p)
        }
        fun trace(series: List<Double>, color: Color) {
            if (series.size < 2) return
            val path = Path().apply {
                val start = point(0, series[0]); moveTo(start.x, start.y)
                for (i in 1 until series.size) { val p = point(i, series[i]); lineTo(p.x, p.y) }
            }
            drawPath(path, color, style = Stroke(width = 3f))
        }
        trace(frame.observed, LabCyan)
        if (frame.estimate != frame.observed) trace(frame.estimate, LabOrange)
        val last = point(count - 1, frame.observed.last())
        val next = point(count, frame.forecast.first())
        drawLine(Color(0xFF8B69FF), last, next, 4f)
        drawCircle(Color(0xFF8B69FF), 7f, next)
        if (frame.uncertainty > 0) {
            val upper = point(count, frame.forecast.first() + frame.uncertainty)
            val lower = point(count, frame.forecast.first() - frame.uncertainty)
            drawLine(Color(0xFF8B69FF), upper, lower, 2f)
        }
    }
}

@Composable
private fun ForecastBarChart(values: List<Double>) {
    Canvas(Modifier.fillMaxWidth().height(100.dp).background(Color(0xFF071B37))) {
        if (values.isEmpty()) return@Canvas
        val extent = max(.01, values.maxOf { kotlin.math.abs(it) })
        val mid = size.height / 2f
        values.forEachIndexed { i, v ->
            val x = (i + .5f) * size.width / values.size
            val y = mid - (v / extent).toFloat() * (size.height * .4f)
            drawLine(if (v >= 0) LabCyan else LabOrange, Offset(x, mid), Offset(x, y),
                (size.width / values.size * .55f).coerceAtLeast(2f))
        }
        drawLine(Color(0xFF355176), Offset(0f, mid), Offset(size.width, mid), 1f)
    }
}
