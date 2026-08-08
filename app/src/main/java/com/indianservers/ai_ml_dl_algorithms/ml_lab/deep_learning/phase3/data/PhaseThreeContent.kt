package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.data

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.ImageSample
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine.TensorImage
import kotlin.math.abs
import kotlin.random.Random

data class KernelPreset(val name: String, val values: FloatArray)

object PhaseThreeContent {
    val kernels = listOf(
        KernelPreset("Vertical edge", floatArrayOf(-1f, 0f, 1f, -2f, 0f, 2f, -1f, 0f, 1f)),
        KernelPreset("Horizontal edge", floatArrayOf(-1f, -2f, -1f, 0f, 0f, 0f, 1f, 2f, 1f)),
        KernelPreset("Sharpen", floatArrayOf(0f, -1f, 0f, -1f, 5f, -1f, 0f, -1f, 0f)),
        KernelPreset("Blur", FloatArray(9) { 1f / 9f }),
        KernelPreset("Emboss", floatArrayOf(-2f, -1f, 0f, -1f, 1f, 1f, 0f, 1f, 2f)),
        KernelPreset("Identity", floatArrayOf(0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f))
    )

    val classNames = listOf("Vertical", "Horizontal", "Diagonal")
    val shapeDataset: List<ImageSample> = buildList {
        val random = Random(103)
        repeat(3) { label ->
            repeat(24) { sampleIndex ->
                val image = TensorImage.zeros(1, 8, 8)
                val offset = 2 + sampleIndex % 4
                for (row in 0 until 8) for (column in 0 until 8) {
                    val active = when (label) {
                        0 -> abs(column - offset) <= if (sampleIndex % 3 == 0) 1 else 0
                        1 -> abs(row - offset) <= if (sampleIndex % 3 == 0) 1 else 0
                        else -> abs(column - row - (sampleIndex % 3 - 1)) <= 0
                    }
                    val noise = if (random.nextFloat() < 0.035f) random.nextFloat() * 0.5f else 0f
                    image[0, row, column] = if (active) 0.75f + random.nextFloat() * 0.25f else noise
                }
                add(ImageSample(image, label))
            }
        }
    }

    val sequence = listOf(1f, 0f, 1f, 1f, 0f).map { floatArrayOf(it) }
    val memorySequence = listOf(1f, 0f, 0f, 0f, 0f, 1f).map { floatArrayOf(it) }
    val vocabulary = listOf("cat", "dog", "bird", "run", "walk", "fly")

    val reconstructionPatterns = listOf(
        floatArrayOf(1f, 1f, 0f, 0f, 0f, 0f, 1f, 1f),
        floatArrayOf(0f, 0f, 1f, 1f, 1f, 1f, 0f, 0f),
        floatArrayOf(1f, 0f, 1f, 0f, 1f, 0f, 1f, 0f),
        floatArrayOf(0f, 1f, 0f, 1f, 0f, 1f, 0f, 1f),
        floatArrayOf(1f, 0f, 0f, 1f, 1f, 0f, 0f, 1f),
        floatArrayOf(0f, 1f, 1f, 0f, 0f, 1f, 1f, 0f)
    )
}
