package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.Detection
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.EmbeddingClassifier
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.GridWorld
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.PracticalAlgorithms
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.QuantizationInfo
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.TensorElementType
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.TensorSpec
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.movingAverageForecast
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class PhaseFiveEngineTest {
    @Test
    fun stableSoftmaxAndQuantizationRoundTripAreNumericallyValid() {
        val probabilities = PracticalAlgorithms.softmax(floatArrayOf(1_000f, 999f, 998f))
        assertEquals(1f, probabilities.sum(), 1e-6f)
        assertTrue(probabilities.all { it.isFinite() })
        val original = floatArrayOf(-1f, -.4f, 0f, .6f, 1f)
        val quantized = PracticalAlgorithms.quantize(original, .01f, 0)
        val restored = PracticalAlgorithms.dequantize(quantized, .01f, 0)
        assertTrue(PracticalAlgorithms.meanAbsoluteError(original, restored) <= .01f)
    }

    @Test
    fun iouAndClassAwareNmsSuppressOnlyOverlappingSameClassBoxes() {
        val first = Detection(0, .9f, 0f, 0f, 1f, 1f)
        val overlap = Detection(0, .8f, .1f, .1f, .9f, .9f)
        val otherClass = Detection(1, .7f, .1f, .1f, .9f, .9f)
        assertTrue(PracticalAlgorithms.iou(first, overlap) > .5f)
        val kept = PracticalAlgorithms.nonMaximumSuppression(listOf(overlap, otherClass, first), .5f)
        assertEquals(listOf(first, otherClass), kept)
    }

    @Test
    fun spectrogramFindsEnergyAndRemainsFinite() {
        val samples = FloatArray(512) { sin(2 * PI * 8 * it / 64).toFloat() }
        val spectrogram = PracticalAlgorithms.spectrogram(samples, 64, 32)
        assertEquals(15, spectrogram.size)
        assertTrue(spectrogram.flatMap { it.asIterable() }.all { it.isFinite() && it >= 0f })
        assertTrue(spectrogram.first()[8] > spectrogram.first()[2])
    }

    @Test
    fun tokenizerEmbeddingSimilarityAndClassifierUseActualVectors() {
        val vocabulary = mapOf("[unk]" to 1, "private" to 2, "ai" to 3)
        val tokens = PracticalAlgorithms.tokenize("Private AI!", vocabulary)
        assertEquals(listOf("private" to 2, "ai" to 3, "!" to 1), tokens)
        assertEquals(1f, PracticalAlgorithms.cosineSimilarity(floatArrayOf(1f, 2f), floatArrayOf(1f, 2f)), 1e-6f)
        val classifier = EmbeddingClassifier(2)
        classifier.add(floatArrayOf(1f, 0f), "local")
        classifier.add(floatArrayOf(0f, 1f), "cloud")
        assertEquals("local", classifier.predict(floatArrayOf(.9f, .1f)).first)
    }

    @Test
    fun benchmarkPercentilesAndTensorMemoryAreExact() {
        val summary = PracticalAlgorithms.benchmark(12.0, listOf(1.0, 2.0, 3.0, 4.0, 5.0))
        assertEquals(3.0, summary.p50, 0.0)
        assertEquals(4.6, summary.p90, 1e-9)
        assertEquals(333.3333333333333, summary.throughputPerSecond, 1e-9)
        val tensor = TensorSpec("image", longArrayOf(1, 224, 224, 3), TensorElementType.Float32, QuantizationInfo(.1f, 0))
        assertEquals(150_528, tensor.elementCount)
        assertEquals(602_112L, tensor.memoryBytes)
    }

    @Test
    fun confusionCalibrationAndTimeSeriesMetricsAreDeterministic() {
        val confusion = PracticalAlgorithms.confusionMatrix(intArrayOf(0, 0, 1, 1), intArrayOf(0, 1, 1, 1), 2)
        assertEquals(listOf(1, 1), confusion[0].toList())
        assertEquals(listOf(0, 2), confusion[1].toList())
        val calibration = PracticalAlgorithms.expectedCalibrationError(floatArrayOf(.9f, .8f, .6f, .55f), booleanArrayOf(true, false, true, false), 4)
        assertTrue(calibration in 0f..1f)
        assertEquals(6f, movingAverageForecast(floatArrayOf(1f, 3f, 5f, 7f), 2), 0f)
    }

    @Test
    fun qLearningBuildsFinitePolicyTowardGoal() {
        val world = GridWorld(4, 7)
        val history = world.train(250)
        assertTrue(history.isNotEmpty())
        assertTrue(world.qValues.flatMap { it.asIterable() }.all { it.isFinite() })
        assertTrue(world.qValues[0].max() > world.qValues[0].min())
    }
}
