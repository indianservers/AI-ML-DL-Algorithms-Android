package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LearnCatalogTest {
    @Test
    fun catalogContainsCompleteUniqueTaxonomy() {
        assertEquals(14, LearnCatalog.domains.size)
        assertTrue(LearnCatalog.topics.size > 240)
        assertEquals(LearnCatalog.topics.size, LearnCatalog.topics.map { it.id }.toSet().size)
        assertTrue(LearnCatalog.domains.all { domain ->
            domain.sections.isNotEmpty() && domain.sections.all { it.topics.isNotEmpty() }
        })
    }

    @Test
    fun everyTopicProducesACompleteLearningProfile() {
        LearnCatalog.topics.forEach { topic ->
            LearningDepth.entries.forEach { depth ->
                val profile = LearnCatalog.profile(topic, depth)
                assertTrue(profile.definition.isNotBlank())
                assertTrue(profile.steps.size >= 5)
                assertTrue(profile.equation.isNotBlank())
                assertTrue(profile.advantages.isNotEmpty())
                assertTrue(profile.limitations.isNotEmpty())
                assertTrue(profile.hyperparameters.isNotEmpty())
            }
        }
    }

    @Test
    fun flagshipAlgorithmsUseSpecificVisualizations() {
        fun kind(title: String) = LearnCatalog.profile(
            LearnCatalog.topics.first { it.title == title },
            LearningDepth.University
        ).kind

        assertEquals(VisualizationKind.Regression, kind("Simple Linear Regression"))
        assertEquals(VisualizationKind.Clustering, kind("K-Means"))
        assertEquals(VisualizationKind.Density, kind("DBSCAN"))
        assertEquals(VisualizationKind.Attention, kind("Transformer"))
        assertEquals(VisualizationKind.Reinforcement, kind("Q-Learning"))
    }

    @Test
    fun topAlgorithmTheoryUsesSpecificEquationsAndKinds() {
        fun profile(title: String) = LearnCatalog.profile(
            LearnCatalog.topics.first { it.title == title },
            LearningDepth.University
        )

        assertTrue(profile("Simple Linear Regression").equation.contains("wx + b"))
        assertEquals(VisualizationKind.Classification, profile("Logistic Regression").kind)
        assertTrue(profile("Polynomial Regression").equation.contains("x^2"))
        assertTrue(profile("Ridge Regression").equation.contains("sum_j w_j^2"))
        assertTrue(profile("Lasso Regression").equation.contains("|w_j|"))
        assertTrue(profile("K-Nearest Neighbors").equation.contains("N_K"))
        assertTrue(profile("Decision Tree").equation.contains("impurity"))
        assertTrue(profile("Random Forest").definition.contains("bootstrap"))
        assertTrue(profile("Extra Trees").definition.contains("random"))
        assertTrue(profile("Support Vector Machine").equation.contains("C * sum"))
        assertTrue(profile("Gaussian Naive Bayes").equation.contains("log P"))
        assertTrue(profile("Linear Discriminant Analysis").equation.contains("Sigma"))
        assertTrue(profile("Gradient Boosting").equation.contains("F_m"))
        assertTrue(profile("XGBoost").equation.contains("gain"))
        assertTrue(profile("LightGBM").definition.contains("histogram"))
        assertTrue(profile("CatBoost").definition.contains("categorical"))
        assertTrue(profile("K-Means").equation.contains("mu"))
        assertTrue(profile("DBSCAN").equation.contains("MinPts"))
        assertTrue(profile("Gaussian Mixture Models").equation.contains("N(x"))
        assertTrue(profile("PCA").equation.contains("eigenvectors"))
        assertTrue(profile("Multi-Layer Perceptron").equation.contains("W_l"))
        assertTrue(profile("CNN").equation.contains("feature_map"))
        assertTrue(profile("Recurrent Neural Network").equation.contains("h_t"))
        assertTrue(profile("LSTM").equation.contains("c_t"))
        assertTrue(profile("GRU").equation.contains("z_t"))
        assertTrue(profile("Transformer").equation.contains("softmax"))
        assertTrue(profile("Self-Attention").equation.contains("softmax"))
    }
}
