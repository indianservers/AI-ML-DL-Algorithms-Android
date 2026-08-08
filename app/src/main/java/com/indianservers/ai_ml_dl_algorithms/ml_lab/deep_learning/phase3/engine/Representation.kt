package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase3.engine

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Activation
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Initializer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.LossFunction
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NeuralNetwork
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NeuralSample
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.OptimizerType
import kotlin.math.sqrt
import kotlin.random.Random

class EmbeddingLayer(val vocabulary: List<String>, val dimensions: Int, seed: Int = 61) {
    val values = FloatArray(vocabulary.size * dimensions)
    init { val random = Random(seed); values.indices.forEach { values[it] = (random.nextFloat() * 2f - 1f) * 0.5f } }
    fun lookup(token: String): FloatArray { val id = vocabulary.indexOf(token).coerceAtLeast(0); return FloatArray(dimensions) { values[id * dimensions + it] } }
    fun cosine(first: String, second: String): Float {
        val a = lookup(first); val b = lookup(second)
        val dot = a.indices.sumOf { (a[it] * b[it]).toDouble() }.toFloat()
        val an = sqrt(a.sumOf { (it * it).toDouble() }.toFloat()); val bn = sqrt(b.sumOf { (it * it).toDouble() }.toFloat())
        return dot / (an * bn).coerceAtLeast(1e-7f)
    }
}

class DenseAutoencoder(val inputSize: Int, val latentSize: Int, seed: Int = 71) {
    val network = NeuralNetwork(listOf(inputSize, (inputSize / 2).coerceAtLeast(latentSize + 1), latentSize, (inputSize / 2).coerceAtLeast(latentSize + 1), inputSize), Activation.Tanh, Activation.Sigmoid, LossFunction.MSE, Initializer.Xavier, seed)
    fun reconstruct(input: FloatArray) = network.predict(input)
    fun encode(input: FloatArray): FloatArray { val trace = network.trace(NeuralSample(input, input)); return trace.layers[1].output }
    fun decode(latent: FloatArray): FloatArray {
        var value = latent.copyOf()
        network.layers.drop(2).forEach { value = it.forward(value) }
        return value
    }
    fun train(inputs: List<FloatArray>, epochs: Int, learningRate: Float = 0.02f) = network.train(inputs.map { NeuralSample(it, it) }, epochs, learningRate, OptimizerType.Adam, inputs.size, 5f, 10)
}
