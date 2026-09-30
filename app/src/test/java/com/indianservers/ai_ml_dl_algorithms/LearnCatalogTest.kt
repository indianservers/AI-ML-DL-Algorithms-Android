package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.searchLabAssetPath
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LearnCatalogTest {
    @Test
    fun phaseEightCompletesTwentyFourDistinctLabs() {
        assertEquals("phase8_labs/index.html?lab=sma", searchLabAssetPath("Simplified Memory-Bounded A Star Search"))
        assertEquals(searchLabAssetPath("Simplified Memory-Bounded A Star Search"), searchLabAssetPath("Simplified Memory-Bounded A* Search"))
        assertEquals("phase8_labs/index.html?lab=hill", searchLabAssetPath("Hill Climbing Search"))
        assertEquals("phase8_labs/index.html?lab=local-beam", searchLabAssetPath("Local Beam Search"))
    }

    @Test
    fun phaseSevenAddsThreeDistinctSearchRoutes() {
        assertEquals("phase7_labs/index.html?lab=ucs", searchLabAssetPath("Uniform Cost Search"))
        assertEquals("phase7_labs/index.html?lab=iddfs", searchLabAssetPath("Iterative Deepening Depth First Search"))
        assertEquals("phase7_labs/index.html?lab=astar", searchLabAssetPath("A Star Search"))
        assertEquals(searchLabAssetPath("A Star Search"), searchLabAssetPath("A* Search"))
    }

    @Test
    fun phaseSixHasDistinctRoutesAndReusesBayesianNetworkTopic() {
        val topics = LearnCatalog.domains.first { it.title == "AI Algorithms" }.sections.flatMap { it.topics }
        val construction = topics.first { it.title == "Construction of Bayesian Network" }
        assertEquals(LearnCatalog.topics.first { it.title == "Bayesian Networks" }.id, construction.id)
        assertEquals(searchLabAssetPath("Bayesian Networks"), searchLabAssetPath(construction.title))
        assertEquals("phase6_labs/index.html?lab=minimax", searchLabAssetPath("Minimax Search"))
        assertEquals("phase6_labs/index.html?lab=inference", searchLabAssetPath("Inference from Bayesian Network"))
        assertEquals(null, searchLabAssetPath("Bayesian Inference"))
    }

    @Test
    fun phaseThreeKeepsExistingTopicIdsAndDistinctLabRoutes() {
        val aiTopics = LearnCatalog.domains.first { it.title == "AI Algorithms" }.sections.flatMap { it.topics }
        assertTrue(aiTopics.any { it.title == "Alpha-Beta Pruning" })
        for ((original, alias) in listOf("SARSA" to "SARSA Learning", "Markov Decision Process" to "Markov Decision Process Explorer")) {
            assertEquals(LearnCatalog.topics.first { it.title == original }.id, aiTopics.first { it.title == alias }.id)
            assertEquals(searchLabAssetPath(original), searchLabAssetPath(alias))
        }
        val routes = aiTopics.map { searchLabAssetPath(it.title) }
        assertTrue(routes.all { it != null })
        assertEquals(24, routes.toSet().size)
        assertEquals(null, searchLabAssetPath("Expected SARSA"))
    }

    @Test
    fun phaseFiveReusesQTopicAndProvidesDistinctSearchRoutes() {
        val aiTopics = LearnCatalog.domains.first { it.title == "AI Algorithms" }.sections.flatMap { it.topics }
        val q = aiTopics.first { it.title == "Q Learning" }
        assertEquals(LearnCatalog.topics.first { it.title == "Q-Learning" }.id, q.id)
        assertEquals(searchLabAssetPath("Q-Learning"), searchLabAssetPath(q.title))
        assertEquals("phase5_labs/index.html?lab=dfs", searchLabAssetPath("AI Depth First Search"))
        assertEquals("phase5_labs/index.html?lab=greedy", searchLabAssetPath("Greedy Best First Search"))
    }

    @Test
    fun phaseFourReusesHmmAndAddsTwoDistinctIterationRoutes() {
        val aiTopics = LearnCatalog.domains.first { it.title == "AI Algorithms" }.sections.flatMap { it.topics }
        val hmm = aiTopics.first { it.title == "Hidden Markov Model – Forward & Viterbi Algorithms" }
        assertEquals(LearnCatalog.topics.first { it.title == "Hidden Markov Models" }.id, hmm.id)
        assertEquals(searchLabAssetPath("Hidden Markov Models"), searchLabAssetPath(hmm.title))
        assertEquals("phase4_labs/index.html?lab=policy", searchLabAssetPath("Policy Iteration"))
        assertEquals("phase4_labs/index.html?lab=value", searchLabAssetPath("Value Iteration"))
    }

    @Test
    fun aiAlgorithmsMenuReusesExistingEvolutionaryTopics() {
        val aiTopics = LearnCatalog.domains.first { it.title == "AI Algorithms" }
            .sections.flatMap { it.topics }
        assertTrue(aiTopics.map { it.title }.containsAll(listOf(
            "Breadth First Search", "Bidirectional Search", "Beam Search",
            "Simulated Annealing", "Genetic Algorithm", "Monte Carlo Tree Search"
        )))
        for (name in listOf("Simulated Annealing", "Genetic Algorithm")) {
            val original = LearnCatalog.domains.first { it.title == "Evolutionary Algorithms" }
                .sections.flatMap { it.topics }.first { it.title == name }
            assertEquals(original.id, aiTopics.first { it.title == name }.id)
        }
    }

    @Test
    fun catalogContainsCompleteUniqueTaxonomy() {
        assertEquals(15, LearnCatalog.domains.size)
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
