package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.data

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Activation
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NeuralSample
import kotlin.math.cos
import kotlin.math.sin

data class NetworkPreset(val name: String, val sizes: List<Int>, val activation: Activation)

object DeepLearningContent {
    val presets = listOf(
        NetworkPreset("Linear", listOf(2, 1), Activation.Linear),
        NetworkPreset("Tiny", listOf(2, 3, 1), Activation.Tanh),
        NetworkPreset("XOR", listOf(2, 4, 4, 1), Activation.Tanh),
        NetworkPreset("Classifier", listOf(2, 8, 4, 1), Activation.ReLU),
        NetworkPreset("Deep demo", listOf(2, 8, 8, 8, 1), Activation.ReLU)
    )

    val xor = listOf(
        NeuralSample(floatArrayOf(0f, 0f), floatArrayOf(0f)),
        NeuralSample(floatArrayOf(0f, 1f), floatArrayOf(1f)),
        NeuralSample(floatArrayOf(1f, 0f), floatArrayOf(1f)),
        NeuralSample(floatArrayOf(1f, 1f), floatArrayOf(0f))
    )

    val linearlySeparable = List(40) { index ->
        val x = ((index * 37) % 100) / 50f - 1f
        val y = ((index * 61 + 13) % 100) / 50f - 1f
        NeuralSample(floatArrayOf(x, y), floatArrayOf(if (x + y > 0f) 1f else 0f))
    }

    val spiral = buildList {
        repeat(2) { label ->
            repeat(40) { i ->
                val radius = i / 40f
                val angle = radius * 5f + label * 3.14159f
                add(NeuralSample(floatArrayOf(radius * cos(angle), radius * sin(angle)), floatArrayOf(label.toFloat())))
            }
        }
    }
}
