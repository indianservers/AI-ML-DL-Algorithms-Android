package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.data.DeepLearningContent
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Activation
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.Initializer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.LossFunction
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.NeuralNetwork
import com.indianservers.ai_ml_dl_algorithms.ml_lab.deep_learning.engine.OptimizerType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class DeepLearningEngineTest {
    @Test
    fun activationFunctionsAreNumericallyStable() {
        assertEquals(0.5f, Activation.Sigmoid.apply(0f), 1e-6f)
        assertEquals(0f, Activation.ReLU.apply(-4f), 0f)
        assertTrue(Activation.Softplus.apply(100f).isFinite())
        assertEquals(1f, Activation.Softmax.vector(floatArrayOf(1f, 2f, 3f)).sum(), 1e-5f)
    }

    @Test
    fun backpropGradientMatchesFiniteDifference() {
        val network = NeuralNetwork(listOf(2, 2, 1), Activation.Tanh, Activation.Sigmoid, LossFunction.BinaryCrossEntropy, Initializer.Xavier, 5)
        val sample = DeepLearningContent.xor[1]
        network.zeroGrad()
        network.forward(sample.input)
        network.backward(sample.target)
        val parameter = network.layers.first().weights
        val analytic = parameter.gradients[0]
        val original = parameter.values[0]
        val epsilon = 1e-3f
        parameter.values[0] = original + epsilon
        val plus = network.lossFunction.value(network.forward(sample.input), sample.target)
        parameter.values[0] = original - epsilon
        val minus = network.lossFunction.value(network.forward(sample.input), sample.target)
        parameter.values[0] = original
        val numerical = (plus - minus) / (2f * epsilon)
        assertTrue("analytic=$analytic numerical=$numerical", abs(analytic - numerical) < 2e-3f)
    }

    @Test
    fun xorNetworkLearnsNonlinearBoundary() {
        val network = NeuralNetwork(listOf(2, 4, 4, 1), Activation.Tanh, Activation.Sigmoid, LossFunction.BinaryCrossEntropy, Initializer.Xavier, 42)
        val before = network.evaluate(DeepLearningContent.xor).first
        network.train(DeepLearningContent.xor, 1200, 0.03f, OptimizerType.Adam, 4, 5f, 100)
        val after = network.evaluate(DeepLearningContent.xor)
        assertTrue("loss $before -> ${after.first}", after.first < before * 0.25f)
        assertEquals(1f, after.second, 0f)
    }
}
