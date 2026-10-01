package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.runtime.Composable
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.SupervisedVisualizationScreen

@Composable
internal fun StageThreeDeepScreen(topic: LearnTopic, kind: DeepVisualization) {
    when (kind.section) {
        "Neural Network Fundamentals" -> when (kind) {
            DeepVisualization.Perceptron -> SupervisedVisualizationScreen(topic, SupervisedVisualization.Perceptron)
            DeepVisualization.Activations -> StageThreeActivationScreen(topic)
            DeepVisualization.GradientDescent, DeepVisualization.Sgd, DeepVisualization.MiniBatch,
            DeepVisualization.Loss -> StageThreeOptimizationScreen(topic, kind)
            else -> StageThreeFundamentalsScreen(topic, kind)
        }
        "Convolutional Neural Networks" -> StageThreeCnnArchitectureScreen(topic, kind)
        "Sequence Models" -> StageThreeSequenceScreen(topic, kind)
        "Transformers" -> when (kind) {
            DeepVisualization.Bert -> StageThreeNeuralLanguageScreen(topic, NeuralLanguageVisualization.Bert)
            DeepVisualization.Gpt -> StageThreeNeuralLanguageScreen(topic, NeuralLanguageVisualization.Gpt)
            DeepVisualization.T5 -> StageThreeNeuralLanguageScreen(topic, NeuralLanguageVisualization.T5)
            DeepVisualization.Vit -> StageThreeVisionScreen(topic, VisionVisualization.Vit)
            DeepVisualization.Swin -> StageThreeSwinScreen(topic)
            else -> error("Unknown Transformer variant")
        }
        "Autoencoders", "Generative Models" -> StageThreeGenerativeVariantsScreen(topic, kind)
        "Graph Neural Networks" -> StageThreeGraphScreen(topic, kind)
        else -> error("Unknown Deep Learning section ${kind.section}")
    }
}
