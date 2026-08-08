package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class PhaseNineConcept(val displayName: String) {
    Autoencoder("Autoencoder"),
    Vae("Variational Autoencoder"),
    Gan("Generative Adversarial Network"),
    Diffusion("Diffusion Model"),
    GenerativeMap("Generative AI Map")
}

enum class GenShape(val label: String) { Circle("Circle"), Square("Square"), X("X"), Vertical("Vertical"), Horizontal("Horizontal") }
enum class GanPreset(val label: String) { Balanced("Balanced"), ModeCollapse("Mode collapse"), DiscriminatorStrong("D too strong"), GeneratorStrong("G too strong") }
enum class GanPhase(val label: String) { Discriminator("D step"), Generator("G step") }

data class Point2(val x: Double, val y: Double)
data class AutoencoderState(
    val shape: GenShape,
    val latentDims: Int,
    val original: List<List<Double>>,
    val noisy: List<List<Double>>,
    val hidden: List<Double>,
    val latent: List<Double>,
    val reconstruction: List<List<Double>>,
    val denoised: List<List<Double>>,
    val loss: Double,
    val selectedPixel: Pair<Int, Int>,
    val selectedPixelError: Double
)
data class LatentPointState(val shape: GenShape, val point: Point2, val reconstruction: List<List<Double>>)
data class VaeState(
    val shape: GenShape,
    val mean: List<Double>,
    val logVariance: List<Double>,
    val sigma: List<Double>,
    val epsilon: List<Double>,
    val z: List<Double>,
    val reconstruction: List<List<Double>>,
    val reconstructionLoss: Double,
    val klLoss: Double,
    val beta: Double,
    val totalLoss: Double
)
data class GanState(
    val preset: GanPreset,
    val phase: GanPhase,
    val real: List<Point2>,
    val generated: List<Pair<Point2, Point2>>,
    val discriminatorField: List<List<Double>>,
    val discriminatorLoss: Double,
    val generatorLoss: Double,
    val inspectedNoise: Point2,
    val inspectedGenerated: Point2,
    val inspectedConfidence: Double,
    val timeline: List<GanPhase>
)
data class DiffusionPixelState(val row: Int, val col: Int, val original: Double, val noise: Double, val alphaBar: Double, val noisy: Double)
data class DiffusionState(
    val shape: GenShape,
    val step: Int,
    val totalSteps: Int,
    val clean: List<List<Double>>,
    val noise: List<List<Double>>,
    val noisy: List<List<Double>>,
    val predictedNoise: List<List<Double>>,
    val denoised: List<List<Double>>,
    val reverseTimeline: List<List<List<Double>>>,
    val pixel: DiffusionPixelState,
    val loss: Double
)

object PhaseNineTopicMatcher {
    fun kindFor(title: String, domain: String): PhaseNineConcept? = if (domain != "Deep Learning") null else when (title) {
        "Autoencoders", "Basic Autoencoder", "Denoising Autoencoder" -> PhaseNineConcept.Autoencoder
        "Variational Autoencoder", "Variational Autoencoders" -> PhaseNineConcept.Vae
        "GAN" -> PhaseNineConcept.Gan
        "Diffusion Models" -> PhaseNineConcept.Diffusion
        else -> null
    }
}

object PhaseNineEngines {
    val anchors = mapOf(
        GenShape.Circle to Point2(-.75, .65),
        GenShape.Square to Point2(.72, .66),
        GenShape.X to Point2(-.63, -.68),
        GenShape.Vertical to Point2(.55, -.45),
        GenShape.Horizontal to Point2(.08, -.92)
    )

    fun shapeImage(shape: GenShape, size: Int = 8): List<List<Double>> = List(size) { r ->
        List(size) { c ->
            val x = (c - (size - 1) / 2.0) / (size / 2.0)
            val y = (r - (size - 1) / 2.0) / (size / 2.0)
            when (shape) {
                GenShape.Circle -> if (sqrt(x * x + y * y) in .45.. .82) 1.0 else 0.0
                GenShape.Square -> if (kotlin.math.abs(x) < .7 && kotlin.math.abs(y) < .7 && (kotlin.math.abs(x) > .42 || kotlin.math.abs(y) > .42)) 1.0 else 0.0
                GenShape.X -> if (kotlin.math.abs(kotlin.math.abs(x) - kotlin.math.abs(y)) < .18) 1.0 else 0.0
                GenShape.Vertical -> if (kotlin.math.abs(x) < .18) 1.0 else 0.0
                GenShape.Horizontal -> if (kotlin.math.abs(y) < .18) 1.0 else 0.0
            }
        }
    }

    fun autoencoder(shape: GenShape = GenShape.Circle, latentDims: Int = 2, selectedRow: Int = 3, selectedCol: Int = 3): AutoencoderState {
        val original = shapeImage(shape)
        val noisy = addNoise(original, .22, seed = shape.ordinal + 31)
        val hidden = encoderHidden(original)
        val latent = encode(original, latentDims)
        val decoded = decode(latent)
        val reconstruction = compressionBlend(original, decoded, latentDims)
        val denoised = compressionBlend(original, decode(encode(noisy, latentDims)), (latentDims + 2).coerceAtMost(8))
        val row = selectedRow.coerceIn(0, 7)
        val col = selectedCol.coerceIn(0, 7)
        return AutoencoderState(shape, latentDims, original, noisy, hidden, latent, reconstruction, denoised, mse(original, reconstruction), row to col, (original[row][col] - reconstruction[row][col]).pow(2))
    }

    fun encode(image: List<List<Double>>, latentDims: Int = 2): List<Double> {
        val flat = image.flatten()
        val total = flat.sum().coerceAtLeast(1e-9)
        val cx = image.indices.sumOf { r -> image[r].indices.sumOf { c -> image[r][c] * ((c / 7.0) * 2.0 - 1.0) } } / total
        val cy = image.indices.sumOf { r -> image[r].indices.sumOf { c -> image[r][c] * ((r / 7.0) * 2.0 - 1.0) } } / total
        val vertical = image.indices.sumOf { r -> image[r][3] + image[r][4] } / total
        val horizontal = image[3].sum() + image[4].sum()
        val diag = image.indices.sumOf { image[it][it] + image[it][7 - it] } / total
        val density = total / flat.size
        val raw = listOf(cx, cy, vertical, horizontal / total, diag, density, flat.maxOrNull() ?: 0.0, flat.average())
        return raw.take(latentDims.coerceIn(1, 8))
    }

    fun encoderHidden(image: List<List<Double>>): List<Double> {
        val flat = image.flatten()
        return List(16) { i -> sigmoid(flat.indices.sumOf { flat[it] * deterministic(90, i, it) } / 4.0) }
    }

    fun decode(latent: List<Double>): List<List<Double>> {
        val p = Point2(latent.getOrElse(0) { 0.0 }.coerceIn(-1.2, 1.2), latent.getOrElse(1) { 0.0 }.coerceIn(-1.2, 1.2))
        val weighted = anchors.mapValues { (_, anchor) -> exp(-distance2(p, anchor) * 3.2) }
        val total = weighted.values.sum().coerceAtLeast(1e-9)
        return List(8) { r -> List(8) { c -> weighted.entries.sumOf { (shape, w) -> shapeImage(shape)[r][c] * w / total }.coerceIn(0.0, 1.0) } }
    }

    fun latentPoints(): List<LatentPointState> = GenShape.entries.map { shape ->
        val point = anchors.getValue(shape)
        LatentPointState(shape, point, decode(listOf(point.x, point.y)))
    }

    fun decodeDragged(x: Double, y: Double): List<List<Double>> = decode(listOf(x, y))

    fun interpolate(first: GenShape, second: GenShape, percent: Double): LatentPointState {
        val a = anchors.getValue(first)
        val b = anchors.getValue(second)
        val t = percent.coerceIn(0.0, 1.0)
        val p = Point2(a.x * (1.0 - t) + b.x * t, a.y * (1.0 - t) + b.y * t)
        return LatentPointState(second, p, decode(listOf(p.x, p.y)))
    }

    fun vae(shape: GenShape = GenShape.Circle, beta: Double = 1.0, seed: Int = 4): VaeState {
        val original = shapeImage(shape)
        val base = anchors.getValue(shape)
        val mean = listOf(base.x * .78, base.y * .78)
        val logVariance = listOf(-.55 + beta * .18 + shape.ordinal * .03, -.72 + beta * .14)
        val sigma = logVariance.map { exp(.5 * it) }
        val epsilon = listOf(gaussian(seed, 0), gaussian(seed, 1))
        val z = mean.indices.map { mean[it] + sigma[it] * epsilon[it] }
        val reconstruction = decode(z)
        val reconstructionLoss = mse(original, reconstruction)
        val kl = -.5 * mean.indices.sumOf { 1.0 + logVariance[it] - mean[it].pow(2) - exp(logVariance[it]) } / mean.size
        return VaeState(shape, mean, logVariance, sigma, epsilon, z, reconstruction, reconstructionLoss, kl, beta, reconstructionLoss + beta * kl)
    }

    fun sampleVaePrior(seed: Int = 12): LatentPointState {
        val z = Point2(gaussian(seed, 0), gaussian(seed, 1))
        return LatentPointState(GenShape.Circle, z, decode(listOf(z.x, z.y)))
    }

    fun gan(preset: GanPreset = GanPreset.Balanced, steps: Int = 12, seed: Int = 7): GanState {
        val real = realRing(48)
        val timeline = List((steps * 2).coerceAtLeast(2)) { if (it % 2 == 0) GanPhase.Discriminator else GanPhase.Generator }
        val phase = timeline[(steps * 2 - 1).coerceAtLeast(0) % timeline.size]
        val generated = List(48) { i ->
            val z = Point2(gaussian(seed + i, 0), gaussian(seed + i, 1))
            z to generatePoint(z, preset, steps)
        }
        val field = List(9) { r -> List(9) { c -> discriminatorConfidence(Point2(c / 4.0 - 1.0, r / 4.0 - 1.0), preset, steps) } }
        val inspected = generated[5]
        val dLoss = generated.take(12).sumOf { -ln((1.0 - discriminatorConfidence(it.second, preset, steps)).coerceIn(1e-6, .999999)) } / 12.0
        val gLoss = generated.take(12).sumOf { -ln(discriminatorConfidence(it.second, preset, steps).coerceIn(1e-6, .999999)) } / 12.0
        return GanState(preset, phase, real, generated, field, dLoss, gLoss, inspected.first, inspected.second, discriminatorConfidence(inspected.second, preset, steps), timeline.take(10))
    }

    fun binaryLoss(prediction: Double, target: Double): Double =
        -(target * ln(prediction.coerceIn(1e-6, .999999)) + (1.0 - target) * ln((1.0 - prediction).coerceIn(1e-6, .999999)))

    fun diffusion(shape: GenShape = GenShape.Circle, step: Int = 10, totalSteps: Int = 24, seed: Int = 2, selectedRow: Int = 3, selectedCol: Int = 3, denoiseSteps: Int = 6): DiffusionState {
        val clean = shapeImage(shape)
        val safeStep = step.coerceIn(0, totalSteps - 1)
        val alpha = alphaBar(safeStep, totalSteps)
        val noise = List(8) { r -> List(8) { c -> gaussian(seed + r * 13 + c, safeStep) } }
        val noisy = List(8) { r -> List(8) { c -> sqrt(alpha) * clean[r][c] + sqrt(1.0 - alpha) * noise[r][c] } }
        val predictedNoise = List(8) { r -> List(8) { c -> noise[r][c] * .82 + deterministic(seed, r, c) * .08 } }
        val denoised = denoise(noisy, predictedNoise, alpha)
        val timeline = reverseTimeline(noisy, clean, denoiseSteps)
        val row = selectedRow.coerceIn(0, 7)
        val col = selectedCol.coerceIn(0, 7)
        val loss = mse(noise, predictedNoise)
        return DiffusionState(shape, safeStep, totalSteps, clean, noise, noisy, predictedNoise, denoised, timeline, DiffusionPixelState(row, col, clean[row][col], noise[row][col], alpha, noisy[row][col]), loss)
    }

    fun alphaBar(step: Int, totalSteps: Int): Double {
        val t = step.coerceIn(0, totalSteps - 1) / (totalSteps - 1).toDouble()
        return (1.0 - .92 * t).coerceIn(.08, 1.0)
    }

    fun denoise(noisy: List<List<Double>>, predictedNoise: List<List<Double>>, alphaBar: Double): List<List<Double>> =
        noisy.indices.map { r ->
            noisy[r].indices.map { c ->
                ((noisy[r][c] - sqrt(1.0 - alphaBar) * predictedNoise[r][c]) / sqrt(alphaBar).coerceAtLeast(1e-6)).coerceIn(0.0, 1.0)
            }
        }

    fun mse(a: List<List<Double>>, b: List<List<Double>>): Double =
        a.indices.sumOf { r -> a[r].indices.sumOf { c -> (a[r][c] - b[r][c]).pow(2) } } / (a.size * a.first().size)

    private fun compressionBlend(original: List<List<Double>>, decoded: List<List<Double>>, latentDims: Int): List<List<Double>> {
        val keep = when (latentDims.coerceIn(1, 8)) { 1 -> .18; 2 -> .42; 4 -> .66; else -> .86 }
        return original.indices.map { r -> original[r].indices.map { c -> (original[r][c] * keep + decoded[r][c] * (1.0 - keep)).coerceIn(0.0, 1.0) } }
    }

    private fun addNoise(image: List<List<Double>>, amount: Double, seed: Int): List<List<Double>> =
        image.indices.map { r -> image[r].indices.map { c -> (image[r][c] + deterministic(seed, r, c) * amount).coerceIn(0.0, 1.0) } }

    private fun realRing(count: Int): List<Point2> = List(count) { i ->
        val angle = 2.0 * PI * i / count
        val radius = .72 + .08 * sin(i * 2.1)
        Point2(cos(angle) * radius, sin(angle) * radius)
    }

    private fun generatePoint(z: Point2, preset: GanPreset, steps: Int): Point2 {
        val progress = (steps / 24.0).coerceIn(0.0, 1.0)
        return when (preset) {
            GanPreset.ModeCollapse -> Point2(.65 + z.x * .04, z.y * .04)
            GanPreset.DiscriminatorStrong -> Point2(z.x * .22 - .25, z.y * .22 - .25)
            GanPreset.GeneratorStrong -> Point2(z.x * .95, z.y * .95)
            GanPreset.Balanced -> {
                val angle = kotlin.math.atan2(z.y, z.x)
                val radius = .28 + .44 * progress + .04 * sin(z.x * 3.0)
                Point2(cos(angle) * radius, sin(angle) * radius)
            }
        }
    }

    private fun discriminatorConfidence(point: Point2, preset: GanPreset, steps: Int): Double {
        val radius = sqrt(point.x * point.x + point.y * point.y)
        val ringScore = 1.0 - kotlin.math.abs(radius - .72) * 3.0
        val strength = when (preset) {
            GanPreset.DiscriminatorStrong -> 8.0
            GanPreset.GeneratorStrong -> 2.2
            else -> 4.5 + steps * .04
        }
        return sigmoid(ringScore * strength)
    }

    private fun reverseTimeline(start: List<List<Double>>, clean: List<List<Double>>, steps: Int): List<List<List<Double>>> =
        List(steps.coerceIn(3, 10)) { i ->
            val t = i / (steps.coerceIn(3, 10) - 1).toDouble()
            start.indices.map { r -> start[r].indices.map { c -> (start[r][c] * (1.0 - t) + clean[r][c] * t).coerceIn(0.0, 1.0) } }
        }

    private fun distance2(a: Point2, b: Point2): Double = (a.x - b.x).pow(2) + (a.y - b.y).pow(2)
    private fun sigmoid(x: Double) = 1.0 / (1.0 + exp(-x))
    private fun gaussian(seed: Int, index: Int): Double {
        val u1 = (deterministic(seed, index, 11) * .5 + .5).coerceIn(1e-6, .999999)
        val u2 = (deterministic(seed, index, 23) * .5 + .5).coerceIn(1e-6, .999999)
        return sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
    }
    private fun deterministic(seed: Int, a: Int, b: Int): Double {
        val raw = sin(seed * 91.0 + a * 37.0 + b * 17.0) * 1009.0
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}
