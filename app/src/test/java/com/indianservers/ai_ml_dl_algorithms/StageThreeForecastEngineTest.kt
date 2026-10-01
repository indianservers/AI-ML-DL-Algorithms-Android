package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.ForecastVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeForecastEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageThreeForecastEngineTest {
    @Test fun movingAverageUsesOnlyCurrentWindow() {
        assertEquals(listOf(2.0, 3.0, 5.0, 7.0),
            StageThreeForecastEngine.movingAverage(listOf(2.0, 4.0, 6.0, 8.0), 2))
    }

    @Test fun exponentialSmoothingRespectsAlpha() {
        assertEquals(listOf(2.0, 3.0, 4.5),
            StageThreeForecastEngine.exponential(listOf(2.0, 4.0, 6.0), .5))
    }

    @Test fun arAndMaUseDifferentSources() {
        val data = listOf(1.0, 2.0, 5.0, 7.0, 3.0, 8.0)
        val ar = StageThreeForecastEngine.arPrediction(data, 2)
        val (ma, errors) = StageThreeForecastEngine.maPrediction(data, 2)
        assertTrue(ar != ma)
        assertEquals(data.size, errors.size)
    }

    @Test fun seasonalDifferenceAndKalmanAreComputed() {
        val data = (0..26).map { it.toDouble() }
        assertEquals(12.0, StageThreeForecastEngine.difference(data, 12).first(), 1e-9)
        val (filtered, gains, variance) = StageThreeForecastEngine.kalman(data, .05, .5)
        assertEquals(data.size, filtered.size)
        assertTrue(gains.drop(1).all { it in 0.0..1.0 })
        assertTrue(variance.all { it > 0 })
    }

    @Test fun everyForecastVariantProducesFiniteOutput() {
        val data = StageThreeForecastEngine.sample()
        ForecastVisualization.entries.forEach { kind ->
            val result = StageThreeForecastEngine.evaluate(kind, data, 40, 4.0)
            assertTrue(kind.name, result.forecast.single().isFinite())
            assertEquals(41, result.observed.size)
            assertTrue(kind.name, result.selectedInputs.isNotEmpty())
        }
    }

    @Test fun arimaOrdersChangeForecastAndSeasonalSeriesRemainFinite() {
        val data=StageThreeForecastEngine.sample()
        val first=StageThreeForecastEngine.evaluate(ForecastVisualization.Arima,data,40,3.0,1,0)
        val second=StageThreeForecastEngine.evaluate(ForecastVisualization.Arima,data,40,3.0,2,3)
        assertTrue(first.forecast.first()!=second.forecast.first())
        val seasonal=StageThreeForecastEngine.evaluate(ForecastVisualization.Sarima,data,25,3.0,2,2)
        assertTrue(seasonal.forecast.first().isFinite())
    }

    @Test fun forecastAttentionWeightsAreNormalized() {
        val data=StageThreeForecastEngine.sample()
        val result=StageThreeForecastEngine.evaluate(ForecastVisualization.Transformer,data,40,5.0)
        assertEquals(1.0,result.auxiliary.sum(),1e-9)
        assertTrue(result.auxiliary.all{it>=0})
    }

    @Test fun editingAnObservationChangesTheForecast() {
        val data=StageThreeForecastEngine.sample().toMutableList()
        val before=StageThreeForecastEngine.evaluate(ForecastVisualization.MovingAverage,data,40,4.0).forecast.first()
        data[40]+=10.0
        val after=StageThreeForecastEngine.evaluate(ForecastVisualization.MovingAverage,data,40,4.0).forecast.first()
        assertEquals(2.5,after-before,1e-9)
    }
}
