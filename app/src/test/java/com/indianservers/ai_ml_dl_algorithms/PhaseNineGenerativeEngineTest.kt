package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.GanPhase
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.GanPreset
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.GenShape
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseNineEngines
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PhaseNineGenerativeEngineTest {
    @Test
    fun autoencoderEncodesDecodesAndReconstructsTinyShapes() {
        val state = PhaseNineEngines.autoencoder(GenShape.Circle, latentDims = 2)
        assertEquals(8, state.original.size)
        assertEquals(16, state.hidden.size)
        assertEquals(2, state.latent.size)
        assertEquals(8, state.reconstruction.size)
        assertTrue(state.loss.isFinite())
        assertTrue(state.loss >= 0.0)
    }

    @Test
    fun bottleneckDimensionChangesReconstructionLoss() {
        val small = PhaseNineEngines.autoencoder(GenShape.X, latentDims = 1)
        val larger = PhaseNineEngines.autoencoder(GenShape.X, latentDims = 8)
        assertTrue(larger.loss < small.loss)
        assertTrue(small.selectedPixelError >= 0.0)
    }

    @Test
    fun latentSelectionAndInterpolationDecodeImages() {
        val points = PhaseNineEngines.latentPoints()
        val decoded = PhaseNineEngines.decodeDragged(.25, -.35)
        val interpolated = PhaseNineEngines.interpolate(GenShape.Circle, GenShape.Square, .5)
        assertEquals(GenShape.entries.size, points.size)
        assertEquals(8, decoded.size)
        assertTrue(interpolated.reconstruction.flatten().all { it in 0.0..1.0 })
    }

    @Test
    fun vaeProducesMeanLogVarianceReparameterizedSampleAndLosses() {
        val first = PhaseNineEngines.vae(GenShape.Square, beta = 1.4, seed = 9)
        val second = PhaseNineEngines.vae(GenShape.Square, beta = 1.4, seed = 9)
        assertEquals(2, first.mean.size)
        assertEquals(2, first.logVariance.size)
        assertEquals(first.epsilon[0], second.epsilon[0], 0.0)
        assertEquals(first.mean[0] + first.sigma[0] * first.epsilon[0], first.z[0], 1e-12)
        assertTrue(first.klLoss >= 0.0)
        assertEquals(first.reconstructionLoss + first.beta * first.klLoss, first.totalLoss, 1e-12)
    }

    @Test
    fun vaePriorSamplingProducesDecodedVisualState() {
        val sample = PhaseNineEngines.sampleVaePrior(seed = 3)
        assertTrue(sample.point.x.isFinite())
        assertEquals(8, sample.reconstruction.size)
        assertTrue(sample.reconstruction.flatten().all { it in 0.0..1.0 })
    }

    @Test
    fun ganGeneratorDiscriminatorLossAndAlternationAreDeterministic() {
        val state = PhaseNineEngines.gan(GanPreset.Balanced, steps = 10, seed = 5)
        val repeat = PhaseNineEngines.gan(GanPreset.Balanced, steps = 10, seed = 5)
        assertEquals(48, state.real.size)
        assertEquals(48, state.generated.size)
        assertEquals(state.generated[0].second.x, repeat.generated[0].second.x, 0.0)
        assertTrue(state.inspectedConfidence in 0.0..1.0)
        assertTrue(state.discriminatorLoss.isFinite())
        assertTrue(state.generatorLoss.isFinite())
        assertTrue(state.timeline.contains(GanPhase.Discriminator))
        assertTrue(state.timeline.contains(GanPhase.Generator))
    }

    @Test
    fun ganModeCollapseOccupiesLessSpaceThanBalancedPreset() {
        val balanced = PhaseNineEngines.gan(GanPreset.Balanced, steps = 20)
        val collapsed = PhaseNineEngines.gan(GanPreset.ModeCollapse, steps = 20)
        val balancedSpread = balanced.generated.maxOf { it.second.x } - balanced.generated.minOf { it.second.x }
        val collapsedSpread = collapsed.generated.maxOf { it.second.x } - collapsed.generated.minOf { it.second.x }
        assertTrue(collapsedSpread < balancedSpread)
        assertEquals(9, collapsed.discriminatorField.size)
    }

    @Test
    fun binaryLossMatchesDiscriminatorObjective() {
        val realLoss = PhaseNineEngines.binaryLoss(.9, 1.0)
        val fakeLoss = PhaseNineEngines.binaryLoss(.1, 0.0)
        assertTrue(realLoss < PhaseNineEngines.binaryLoss(.4, 1.0))
        assertTrue(fakeLoss < PhaseNineEngines.binaryLoss(.6, 0.0))
    }

    @Test
    fun diffusionForwardEquationScheduleAndPixelStateAreConsistent() {
        val state = PhaseNineEngines.diffusion(GenShape.Vertical, step = 12, seed = 6, selectedRow = 2, selectedCol = 3)
        val pixel = state.pixel
        val expected = kotlin.math.sqrt(pixel.alphaBar) * pixel.original + kotlin.math.sqrt(1.0 - pixel.alphaBar) * pixel.noise
        assertEquals(expected, pixel.noisy, 1e-12)
        assertTrue(PhaseNineEngines.alphaBar(0, 24) > PhaseNineEngines.alphaBar(23, 24))
        assertEquals(8, state.noisy.size)
        assertTrue(state.loss >= 0.0)
    }

    @Test
    fun diffusionDenoisingUpdateAndGenerationLoopStayFinite() {
        val state = PhaseNineEngines.diffusion(GenShape.Horizontal, step = 20, denoiseSteps = 7)
        val denoised = PhaseNineEngines.denoise(state.noisy, state.predictedNoise, state.pixel.alphaBar)
        assertEquals(state.denoised.flatten().first(), denoised.flatten().first(), 1e-12)
        assertEquals(7, state.reverseTimeline.size)
        assertTrue(state.reverseTimeline.flatten().flatten().all { it.isFinite() })
    }
}
