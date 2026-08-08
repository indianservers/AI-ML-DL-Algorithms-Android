package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine

import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

data class ImageSample(val image: TensorImage, val label: Int)
data class CnnPrediction(val probabilities: FloatArray, val featureMaps: TensorImage, val pooled: TensorImage) {
    val predictedClass: Int get() = probabilities.indices.maxBy { probabilities[it] }
}
data class CnnEpoch(val epoch: Int, val loss: Float, val accuracy: Float)

class TinyCnnClassifier(val inputSize: Int = 8, val classes: Int = 3, filters: Int = 4, seed: Int = 23) {
    val convolution = Conv2D(1, filters, 3, padding = 1, seed = seed)
    val pooling = Pool2D(2, 2, PoolingType.Max)
    private val featureCount = filters * (inputSize / 2) * (inputSize / 2)
    val denseWeights = FloatArray(classes * featureCount)
    val denseBiases = FloatArray(classes)

    init {
        val random = Random(seed + 1)
        denseWeights.indices.forEach { denseWeights[it] = (random.nextFloat() * 2f - 1f) * 0.18f }
    }

    fun predict(image: TensorImage): CnnPrediction {
        val featureMaps = relu(convolution.forward(image))
        val pooled = pooling.forward(featureMaps).output
        val logits = FloatArray(classes) { output ->
            denseBiases[output] + pooled.values.indices.sumOf { (denseWeights[output * featureCount + it] * pooled.values[it]).toDouble() }.toFloat()
        }
        val peak = logits.maxOrNull() ?: 0f
        val exps = FloatArray(classes) { exp(logits[it] - peak) }
        val total = exps.sum().coerceAtLeast(1e-7f)
        return CnnPrediction(FloatArray(classes) { exps[it] / total }, featureMaps, pooled)
    }

    fun train(samples: List<ImageSample>, epochs: Int, learningRate: Float = 0.025f): List<CnnEpoch> {
        val history = mutableListOf<CnnEpoch>()
        repeat(epochs + 1) { epoch ->
            if (epoch > 0) samples.shuffled(Random(epoch + 71)).forEach { trainOne(it, learningRate) }
            if (epoch % 2 == 0 || epoch == epochs) {
                var loss = 0f; var correct = 0
                samples.forEach { sample ->
                    val prediction = predict(sample.image)
                    loss -= ln(prediction.probabilities[sample.label].coerceAtLeast(1e-7f))
                    if (prediction.predictedClass == sample.label) correct++
                }
                history += CnnEpoch(epoch, loss / samples.size, correct.toFloat() / samples.size)
            }
        }
        return history
    }

    private fun trainOne(sample: ImageSample, learningRate: Float) {
        val prediction = predict(sample.image)
        val dLogits = prediction.probabilities.copyOf().also { it[sample.label] -= 1f }
        val pooledGradient = TensorImage.zeros(prediction.pooled.channels, prediction.pooled.height, prediction.pooled.width)
        for (output in 0 until classes) {
            prediction.pooled.values.indices.forEach { feature ->
                val index = output * featureCount + feature
                pooledGradient.values[feature] += denseWeights[index] * dLogits[output]
                denseWeights[index] -= learningRate * dLogits[output] * prediction.pooled.values[feature]
            }
            denseBiases[output] -= learningRate * dLogits[output]
        }
        val featureGradient = pooling.backward(pooledGradient)
        featureGradient.values.indices.forEach { if (prediction.featureMaps.values[it] <= 0f) featureGradient.values[it] = 0f }
        convolution.backward(featureGradient)
        convolution.step(learningRate)
    }

    fun parameterCount() = convolution.parameterCount() + denseWeights.size + denseBiases.size
    fun memoryBytes() = parameterCount() * 8L
    fun flopEstimate() = inputSize * inputSize * convolution.outputChannels * 3 * 3 * 2 + classes * featureCount * 2
}
