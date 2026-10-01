package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

internal enum class ForecastVisualization(val title: String) {
    MovingAverage("Moving Average"), ExponentialSmoothing("Exponential Smoothing"),
    Holt("Holt's Method"), HoltWinters("Holt-Winters"), Ar("AR"), Ma("MA"),
    Arma("ARMA"), Arima("ARIMA"), Sarima("SARIMA"), Var("VAR"),
    StateSpace("State-Space Models"), Kalman("Kalman Filter"),
    Prophet("Prophet-style Forecasting"), Lstm("LSTM Forecasting"),
    Gru("GRU Forecasting"), TemporalCnn("Temporal CNN"),
    Transformer("Transformer-based Forecasting");

    companion object {
        val byTitle = entries.associateBy { it.title }
    }
}

internal data class ForecastFrame(
    val observed: List<Double>,
    val estimate: List<Double>,
    val forecast: List<Double>,
    val components: List<Pair<String, Double>>,
    val explanation: String,
    val equation: String,
    val selectedInputs: List<Int>,
    val auxiliary: List<Double> = emptyList(),
    val auxiliaryLabel: String = "",
    val uncertainty: Double = 0.0
)

/** Small, deterministic teaching models. Forecasts are recomputed from data and controls. */
internal object StageThreeForecastEngine {
    fun sample(seasonal: Boolean = true, length: Int = 60, seed: Int = 0): List<Double> =
        List(length) { i ->
            12.0 + i * .12 + (if (seasonal) 2.2 * sin(i * Math.PI / 6.0) else 0.0) +
                .55 * sin(i * 2.31 + seed * .79) + .3 * cos(i * 1.23 + seed)
        }

    fun movingAverage(data: List<Double>, window: Int): List<Double> = data.indices.map { i ->
        data.subList(max(0, i - window + 1), i + 1).average()
    }

    fun exponential(data: List<Double>, alpha: Double): List<Double> {
        if (data.isEmpty()) return emptyList()
        val output = mutableListOf(data.first())
        for (i in 1 until data.size) output += alpha * data[i] + (1 - alpha) * output.last()
        return output
    }

    fun difference(data: List<Double>, lag: Int = 1): List<Double> =
        (lag until data.size).map { data[it] - data[it - lag] }

    fun arPrediction(data: List<Double>, p: Int): Double {
        val n = p.coerceIn(1, data.size.coerceAtLeast(1))
        val weights = (1..n).map { 1.0 / it }.let { raw -> raw.map { it / raw.sum() } }
        return weights.indices.sumOf { data[data.lastIndex - it] * weights[it] }
    }

    fun maPrediction(data: List<Double>, q: Int): Pair<Double, List<Double>> {
        val level = movingAverage(data, 4)
        val errors = data.indices.map { data[it] - (if (it == 0) data[0] else level[it - 1]) }
        val count = q.coerceIn(1, errors.size)
        val prediction = level.last() + (0 until count).sumOf { errors[errors.lastIndex - it] * .38 / (it + 1) }
        return prediction to errors
    }

    fun kalman(data: List<Double>, processNoise: Double, measurementNoise: Double): Triple<List<Double>, List<Double>, List<Double>> {
        if (data.isEmpty()) return Triple(emptyList(), emptyList(), emptyList())
        var estimate = data.first()
        var variance = 1.0
        val estimates = mutableListOf(estimate)
        val gains = mutableListOf(1.0)
        val variances = mutableListOf(variance)
        for (measurement in data.drop(1)) {
            val priorVariance = variance + processNoise
            val gain = priorVariance / (priorVariance + measurementNoise)
            estimate += gain * (measurement - estimate)
            variance = (1 - gain) * priorVariance
            estimates += estimate
            gains += gain
            variances += variance
        }
        return Triple(estimates, gains, variances)
    }

    fun evaluate(kind: ForecastVisualization, data: List<Double>, step: Int, parameter: Double,
                 differenceOrder: Int = 1, errorOrder: Int = 1): ForecastFrame {
        require(data.size >= 20)
        val end = step.coerceIn(if (kind == ForecastVisualization.Sarima) 25 else 12, data.size - 1)
        val y = data.take(end + 1)
        val window = parameter.toInt().coerceIn(2, 12)
        val recent = (max(0, end - window + 1)..end).toList()
        val ma = movingAverage(y, window)
        val alpha = parameter.coerceIn(.05, .95)
        val smooth = exponential(y, alpha)
        fun frame(estimate: List<Double>, next: Double, parts: List<Pair<String, Double>>, why: String, equation: String,
                  inputs: List<Int> = recent, aux: List<Double> = emptyList(), auxLabel: String = "", spread: Double = 0.0) =
            ForecastFrame(y, estimate, listOf(next), parts, why, equation, inputs, aux, auxLabel, spread)
        return when (kind) {
            ForecastVisualization.MovingAverage -> frame(ma, ma.last(), listOf("Window" to window.toDouble(), "Mean" to ma.last()),
                "The highlighted observations have equal weight. A wider window smooths short changes.", "mean = Σ recent values / window")
            ForecastVisualization.ExponentialSmoothing -> frame(smooth, smooth.last(), listOf("α" to alpha, "Level" to smooth.last()),
                "Recent observations get exponentially more weight.", "levelₜ = αyₜ + (1−α)levelₜ₋₁", listOf(end))
            ForecastVisualization.Holt, ForecastVisualization.HoltWinters -> {
                val beta = (alpha * .65).coerceIn(.05, .7)
                var level = y.first()
                var trend = y[1] - y[0]
                val seasonal = kind == ForecastVisualization.HoltWinters
                val period = 12
                val seasonalIndex = MutableList(period) { i -> if (seasonal) 1.4 * sin(i * 2 * Math.PI / period) else 0.0 }
                val fitted = mutableListOf(level)
                for (i in 1 until y.size) {
                    val previous = level
                    val s = seasonalIndex[i % period]
                    level = alpha * (y[i] - s) + (1 - alpha) * (level + trend)
                    trend = beta * (level - previous) + (1 - beta) * trend
                    if (seasonal && i >= period) seasonalIndex[i % period] = .25 * (y[i] - level) + .75 * s
                    fitted += level + if (seasonal) seasonalIndex[i % period] else 0.0
                }
                val nextSeason = if (seasonal) seasonalIndex[(end + 1) % period] else 0.0
                frame(fitted, level + trend + nextSeason,
                    listOf("Level" to level, "Trend" to trend, "Season" to nextSeason),
                    if (seasonal) "The forecast combines level, trend, and a 12-step seasonal cycle." else "The forecast advances the estimated level by its trend.",
                    if (seasonal) "forecast = level + trend + seasonal effect" else "forecast = level + trend", (max(0, end - 11)..end).toList(),
                    if (seasonal) seasonalIndex else emptyList(), if (seasonal) "Seasonal pattern" else "")
            }
            ForecastVisualization.Ar -> {
                val p = window.coerceAtMost(6)
                val fit = y.indices.map { i -> if (i < p) y[i] else arPrediction(y.take(i), p) }
                frame(fit, arPrediction(y, p), (1..p).map { "Lag $it" to y[end + 1 - it] },
                    "AR uses earlier observed values. Highlighted lags contribute to the next value.", "ŷₜ = Σ φᵢyₜ₋ᵢ", (end - p + 1..end).toList())
            }
            ForecastVisualization.Ma -> {
                val (prediction, errors) = maPrediction(y, window.coerceAtMost(6))
                frame(smooth, prediction, (0 until window.coerceAtMost(6)).map { "Error ${it + 1}" to errors[end - it] },
                    "MA uses past prediction errors, shown below, rather than past observations as the lag terms.",
                    "ŷₜ = level + Σ θᵢ errorₜ₋ᵢ", recent, errors, "Past residuals")
            }
            ForecastVisualization.Arma -> {
                val ar = arPrediction(y, window.coerceAtMost(5))
                val (errorPart, errors) = maPrediction(y, window.coerceAtMost(5))
                val residualCorrection = errorPart - movingAverage(y, 4).last()
                frame(smooth, ar + residualCorrection, listOf("AR lags" to ar, "MA errors" to residualCorrection),
                    "The two contributions come from observations and residuals separately.", "ŷₜ = Σ φᵢyₜ₋ᵢ + Σ θᵢeₜ₋ᵢ", recent, errors, "Error lags")
            }
            ForecastVisualization.Arima, ForecastVisualization.Sarima -> {
                val seasonal = kind == ForecastVisualization.Sarima
                val d = differenceOrder.coerceIn(0, 2)
                val q = errorOrder.coerceIn(0, 4)
                val initial = if (seasonal) difference(y, 12) else y
                val levels = mutableListOf(initial)
                repeat(d) { levels += difference(levels.last()) }
                val modeled = levels.last()
                val arPart = arPrediction(modeled, window.coerceAtMost(5))
                val errorPart = if (q == 0) 0.0 else maPrediction(modeled, q).first - movingAverage(modeled, 4).last()
                var restored = arPart + errorPart
                for (i in levels.size - 2 downTo 0) restored += levels[i].last()
                if (seasonal) restored += y[y.size - 12]
                frame(y, restored,
                    listOf("p" to window.toDouble(), "d" to d.toDouble(), "q" to q.toDouble(),
                        "AR contribution" to arPart, "MA residual correction" to errorPart, "Restored forecast" to restored),
                    if (seasonal) "Seasonal Δ₁₂ first removes the 12-step pattern; ordinary differencing and ARMA then model the remainder."
                    else "Differencing removes the changing level; AR lags and MA residuals model the transformed series before integration.",
                    if (seasonal) "(p,d,q)(1,1,0)₁₂: seasonal difference → ARMA → integrate" else "ARIMA(p,d,q): difference → ARMA → integrate",
                    (end - 11..end).toList(), modeled, if (seasonal) "Seasonally and ordinarily differenced" else "Differenced series")
            }
            ForecastVisualization.Var -> {
                val price = y.indices.map { 5.0 - .035 * it + .5 * cos(it * .4) }
                val ads = y.indices.map { 2.0 + .6 * sin(it * .22) }
                val next = .7 * y.last() + .23 * y[y.lastIndex - 1] - .16 * price.last() + .35 * ads.last()
                frame(smooth, next, listOf("Sales lag" to y.last(), "Price lag" to price.last(), "Ads lag" to ads.last()),
                    "Each next value combines lagged values from every series; the arrows correspond to these coefficients.",
                    "salesₜ₊₁ = .70 salesₜ + .23 salesₜ₋₁ − .16 priceₜ + .35 adsₜ", listOf(end - 1, end), price, "Price series")
            }
            ForecastVisualization.StateSpace, ForecastVisualization.Kalman -> {
                val (filtered, gains, variance) = kalman(y, .06, parameter.coerceIn(.1, 3.0))
                frame(filtered, filtered.last(), listOf("Prior" to filtered[filtered.lastIndex - 1], "Measurement" to y.last(),
                    "Gain" to gains.last(), "Posterior" to filtered.last()),
                    if (kind == ForecastVisualization.Kalman) "Predict uncertainty, read the noisy measurement, then update using Kalman gain."
                    else "The hidden state evolves through a transition; observations measure it with noise.",
                    if (kind == ForecastVisualization.Kalman) "K = P⁻/(P⁻ + R); x⁺ = x⁻ + K(z − x⁻)" else "xₜ = Fxₜ₋₁ + w; zₜ = Hxₜ + v",
                    listOf(end), gains, "Kalman gain", sqrt(variance.last()))
            }
            ForecastVisualization.Prophet -> {
                val trend = 12.0 + (end + 1) * .12
                val season = 2.2 * sin((end + 1) * Math.PI / 6)
                val event = if ((end + 1) % 20 == 0) 1.5 else 0.0
                frame(y.indices.map { 12.0 + it * .12 + 2.2 * sin(it * Math.PI / 6) }, trend + season + event,
                    listOf("Trend" to trend, "Season" to season, "Event" to event),
                    "An additive decomposition example. This is an educational model, not Meta Prophet.",
                    "forecast = trend + seasonality + events", recent,
                    y.indices.map { 2.2 * sin(it * Math.PI / 6) }, "Seasonality")
            }
            ForecastVisualization.Lstm, ForecastVisualization.Gru -> {
                var hidden = 0.0
                var cell = 0.0
                val trace = y.map { value ->
                    val input = (value - 15.0) / 8.0
                    if (kind == ForecastVisualization.Lstm) {
                        val forget = sigmoid(.7 * hidden + .3)
                        val gate = sigmoid(.5 * input - .1)
                        cell = forget * cell + gate * kotlin.math.tanh(input)
                        hidden = sigmoid(.6 * input) * kotlin.math.tanh(cell)
                    } else {
                        val reset = sigmoid(.6 * input + .2 * hidden)
                        val update = sigmoid(.7 * input - .2 * hidden)
                        val candidate = kotlin.math.tanh(input + reset * hidden)
                        hidden = (1 - update) * hidden + update * candidate
                    }
                    hidden
                }
                frame(y, y.last() + hidden * .5,
                    if (kind == ForecastVisualization.Lstm) listOf("Cell state" to cell, "Hidden" to hidden)
                    else listOf("Update state" to hidden, "Separate cell" to 0.0),
                    if (kind == ForecastVisualization.Lstm) "The sliding window updates a gated cell memory before predicting."
                    else "Reset and update gates mix the window into one hidden state; there is no separate cell memory.",
                    if (kind == ForecastVisualization.Lstm) "cₜ = fₜcₜ₋₁ + iₜgₜ" else "hₜ = (1−zₜ)hₜ₋₁ + zₜh̃ₜ", recent, trace, "Hidden state")
            }
            ForecastVisualization.TemporalCnn -> {
                val dilation = window.coerceIn(2, 6)
                val taps = listOf(end, end - dilation, end - 2 * dilation).filter { it >= 0 }
                val weights = listOf(.55, .3, .15)
                val next = taps.indices.sumOf { y[taps[it]] * weights[it] }
                frame(y, next, taps.mapIndexed { i, at -> "Tap ${i + 1}" to y[at] },
                    "A causal 1D filter reads only past positions. Dilation expands the receptive field.",
                    "ŷₜ₊₁ = Σ kernelᵢ × yₜ₋ᵢ·d", taps)
            }
            ForecastVisualization.Transformer -> {
                val candidates = recent.map { y[it] }
                fun embedding(index: Int) = listOf((y[index]-15.0)/8.0,
                    sin(index*.3),cos(index*.3))
                val query = embedding(end)
                val logits = recent.map { index ->
                    query.zip(embedding(index)).sumOf { it.first*it.second } / sqrt(3.0)
                }
                val exponentials=logits.map { exp(it-logits.max()) }
                val total = exponentials.sum()
                val weights = exponentials.map { it / total }
                val next = candidates.indices.sumOf { candidates[it] * weights[it] }
                frame(y, next, weights.mapIndexed { i, w -> "t${recent[i]}" to w },
                    "Attention weights reveal which past time steps influence the forecast; positional order remains visible on the chart.",
                    "attention = softmax(QKᵀ/√d)V", recent, weights, "Historical attention")
            }
        }
    }

    private fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))
}
