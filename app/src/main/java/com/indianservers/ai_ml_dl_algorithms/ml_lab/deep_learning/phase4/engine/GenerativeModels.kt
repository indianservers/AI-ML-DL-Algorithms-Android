package com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.phase4.engine

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

private fun stableSigmoid(x: Float) = if (x >= 0f) 1f / (1f + exp(-x)) else exp(x) / (1f + exp(x))

data class VaeSample(val mean: FloatArray, val logVariance: FloatArray, val epsilon: FloatArray, val latent: FloatArray, val reconstruction: FloatArray, val reconstructionLoss: Float, val klLoss: Float) { val totalLoss get() = reconstructionLoss + klLoss }
class TinyVae(val inputSize: Int = 8, val latentSize: Int = 2, seed: Int = 161) {
    private val meanWeights = MatrixOps.random(inputSize, latentSize, seed)
    private val varianceWeights = MatrixOps.random(inputSize, latentSize, seed + 1)
    private val decoderWeights = MatrixOps.random(latentSize, inputSize, seed + 2)
    fun sample(input: FloatArray, sampleSeed: Int = 1): VaeSample {
        val x = Matrix(1, inputSize, input.copyOf())
        val mean = MatrixOps.matmul(x, meanWeights).row(0)
        val logVariance = MatrixOps.matmul(x, varianceWeights).row(0).also { it.indices.forEach { i -> it[i] = it[i].coerceIn(-5f, 5f) } }
        val random = Random(sampleSeed); val epsilon = FloatArray(latentSize) { gaussian(random) }
        val latent = FloatArray(latentSize) { mean[it] + exp(0.5f * logVariance[it]) * epsilon[it] }
        val reconstruction = MatrixOps.matmul(Matrix(1, latentSize, latent), decoderWeights).row(0).map(::stableSigmoid).toFloatArray()
        val reconstructionLoss = input.indices.sumOf { ((input[it] - reconstruction[it]) * (input[it] - reconstruction[it])).toDouble() }.toFloat() / inputSize
        val kl = -0.5f * mean.indices.sumOf { (1f + logVariance[it] - mean[it] * mean[it] - exp(logVariance[it])).toDouble() }.toFloat() / latentSize
        return VaeSample(mean, logVariance, epsilon, latent, reconstruction, reconstructionLoss, kl)
    }
    fun decode(latent: FloatArray) = MatrixOps.matmul(Matrix(1, latentSize, latent), decoderWeights).row(0).map(::stableSigmoid).toFloatArray()
    private fun gaussian(random: Random): Float { val u1 = random.nextFloat().coerceAtLeast(1e-7f); val u2 = random.nextFloat(); return sqrt(-2f * ln(u1)) * kotlin.math.cos(2f * Math.PI.toFloat() * u2) }
    fun parameterCount() = inputSize * latentSize * 3
}

data class GanSnapshot(val step: Int, val discriminatorLoss: Float, val generatorLoss: Float, val generatedMean: Float, val generatedStd: Float)
class TinyGan(seed: Int = 171) {
    private val random = Random(seed)
    var generatorWeight = 0.35f; var generatorBias = -0.8f
    var discriminatorWeight = 0.2f; var discriminatorBias = 0f
    fun generate(noise: Float) = generatorWeight * noise + generatorBias
    fun discriminate(value: Float) = stableSigmoid(discriminatorWeight * value + discriminatorBias)
    fun stepDiscriminator(learningRate: Float = 0.025f): GanSnapshot = trainInternal(1, 0, learningRate)
    fun stepGenerator(learningRate: Float = 0.025f): GanSnapshot = trainInternal(0, 1, learningRate)
    fun modeCollapse() { generatorWeight = 0.01f }
    fun train(steps: Int, learningRate: Float = 0.025f): List<GanSnapshot> {
        val history = mutableListOf<GanSnapshot>()
        repeat(steps + 1) { step ->
            val noise = gaussian(); val real = 2f + 0.6f * gaussian(); val fake = generate(noise)
            val realP = discriminate(real).coerceIn(1e-6f, 1f - 1e-6f); val fakeP = discriminate(fake).coerceIn(1e-6f, 1f - 1e-6f)
            val dWeightGrad = (realP - 1f) * real + fakeP * fake; val dBiasGrad = (realP - 1f) + fakeP
            discriminatorWeight -= learningRate * dWeightGrad; discriminatorBias -= learningRate * dBiasGrad
            val updatedFakeP = discriminate(fake).coerceIn(1e-6f, 1f - 1e-6f)
            val throughInput = (updatedFakeP - 1f) * discriminatorWeight
            generatorWeight -= learningRate * throughInput * noise; generatorBias -= learningRate * throughInput
            if (step % 10 == 0 || step == steps) {
                val generated = FloatArray(80) { generate(gaussian()) }; val mean = generated.average().toFloat(); val std = sqrt(generated.sumOf { ((it - mean) * (it - mean)).toDouble() }.toFloat() / generated.size)
                history += GanSnapshot(step, -ln(realP) - ln(1f - fakeP), -ln(updatedFakeP), mean, std)
            }
        }
        return history
    }
    private fun trainInternal(discriminatorSteps: Int, generatorSteps: Int, learningRate: Float): GanSnapshot {
        var realP = .5f; var fakeP = .5f
        repeat(discriminatorSteps) {
            val real = 2f + .6f * gaussian(); val fake = generate(gaussian())
            realP = discriminate(real).coerceIn(1e-6f, 1f - 1e-6f); fakeP = discriminate(fake).coerceIn(1e-6f, 1f - 1e-6f)
            discriminatorWeight -= learningRate * ((realP - 1f) * real + fakeP * fake)
            discriminatorBias -= learningRate * ((realP - 1f) + fakeP)
        }
        repeat(generatorSteps) {
            val noise = gaussian(); val fake = generate(noise); fakeP = discriminate(fake).coerceIn(1e-6f, 1f - 1e-6f)
            val gradient = (fakeP - 1f) * discriminatorWeight
            generatorWeight -= learningRate * gradient * noise; generatorBias -= learningRate * gradient
        }
        val generated = FloatArray(80) { generate(gaussian()) }; val mean = generated.average().toFloat(); val std = sqrt(generated.sumOf { ((it - mean) * (it - mean)).toDouble() }.toFloat() / generated.size)
        return GanSnapshot(discriminatorSteps + generatorSteps, -ln(realP) - ln(1f - fakeP), -ln(fakeP), mean, std)
    }
    private fun gaussian(): Float { val u1 = random.nextFloat().coerceAtLeast(1e-7f); val u2 = random.nextFloat(); return sqrt(-2f * ln(u1)) * kotlin.math.cos(2f * Math.PI.toFloat() * u2) }
}

data class DiffusionSample(val step: Int, val alphaBar: Float, val clean: FloatArray, val noise: FloatArray, val noisy: FloatArray)
class TinyDiffusion(val steps: Int = 50, seed: Int = 181) {
    init { require(steps in 2..200) }
    val betas = FloatArray(steps) { 0.0005f + (0.08f - 0.0005f) * it / (steps - 1f) }
    val alphaBars = FloatArray(steps).also { values -> var product = 1f; betas.indices.forEach { product *= 1f - betas[it]; values[it] = product } }
    private val random = Random(seed)
    var weightX = 0.1f; var weightT = 0f; var bias = 0f
    fun forward(clean: FloatArray, step: Int, sampleSeed: Int = 1): DiffusionSample {
        val safe = step.coerceIn(0, steps - 1); val local = Random(sampleSeed + safe); val noise = FloatArray(clean.size) { gaussian(local) }
        val alpha = alphaBars[safe]; val noisy = FloatArray(clean.size) { sqrt(alpha) * clean[it] + sqrt(1f - alpha) * noise[it] }
        return DiffusionSample(safe, alpha, clean.copyOf(), noise, noisy)
    }
    fun predictNoise(noisy: FloatArray, step: Int) = FloatArray(noisy.size) { weightX * noisy[it] + weightT * step / steps + bias }
    fun train(cleanPoints: List<FloatArray>, epochs: Int, learningRate: Float = 0.02f): List<Float> {
        val history = mutableListOf<Float>()
        repeat(epochs + 1) { epoch ->
            var loss = 0f
            cleanPoints.forEachIndexed { index, clean ->
                val step = (epoch * 7 + index * 11) % steps; val sample = forward(clean, step, epoch + index + 3); val prediction = predictNoise(sample.noisy, step)
                prediction.indices.forEach { dimension ->
                    val error = prediction[dimension] - sample.noise[dimension]; loss += error * error
                    weightX -= learningRate * 2f * error * sample.noisy[dimension] / cleanPoints.size
                    weightT -= learningRate * 2f * error * step / steps / cleanPoints.size
                    bias -= learningRate * 2f * error / cleanPoints.size
                }
            }
            if (epoch % 10 == 0 || epoch == epochs) history += loss / (cleanPoints.size * cleanPoints.first().size)
        }
        return history
    }
    fun reverseStep(noisy: FloatArray, step: Int): FloatArray {
        val safe = step.coerceIn(1, steps - 1); val alpha = 1f - betas[safe]; val predicted = predictNoise(noisy, safe)
        return FloatArray(noisy.size) { (noisy[it] - betas[safe] / sqrt(1f - alphaBars[safe]) * predicted[it]) / sqrt(alpha) }
    }
    fun reverseTrajectory(start: FloatArray, fromStep: Int): List<FloatArray> {
        var current = start.copyOf()
        return buildList {
            add(current.copyOf())
            for (step in fromStep.coerceIn(1, steps - 1) downTo 1) {
                current = reverseStep(current, step)
                require(current.all { it.isFinite() })
                add(current.copyOf())
            }
        }
    }
    private fun gaussian(random: Random): Float { val u1 = random.nextFloat().coerceAtLeast(1e-7f); val u2 = random.nextFloat(); return sqrt(-2f * ln(u1)) * kotlin.math.cos(2f * Math.PI.toFloat() * u2) }
}
