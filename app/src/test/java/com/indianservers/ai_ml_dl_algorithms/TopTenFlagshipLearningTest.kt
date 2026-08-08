package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFiveConcept
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseFiveTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseOneAlgorithmKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseOneTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenConcept
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSevenTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSixCnnConcept
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseSixTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeAlgorithmKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseThreeTopicMatcher
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoAlgorithmKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoTopicMatcher
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TopTenFlagshipLearningTest {
    @Test
    fun topTenFlagshipCatalogContainsRequestedAlgorithmsInOrder() {
        val names = LearnCatalog.flagshipTopics.map { it.title }
        assertEquals(
            listOf(
                "Simple Linear Regression",
                "Logistic Regression",
                "K-Nearest Neighbors",
                "Decision Tree",
                "Random Forest",
                "Support Vector Machine",
                "K-Means",
                "Multi-Layer Perceptron",
                "CNN",
                "LSTM"
            ),
            names
        )
        assertEquals(10, LearnCatalog.flagshipTopics.map { it.id }.distinct().size)
        assertTrue(LearnCatalog.flagshipTopics.all { LearnCatalog.isFlagship(it) })
    }

    @Test
    fun topTenFlagshipTopicsRouteToInteractiveLabs() {
        assertEquals(PhaseOneAlgorithmKind.SimpleLinearRegression, PhaseOneTopicMatcher.kindFor("Simple Linear Regression", "Regression"))
        assertEquals(PhaseOneAlgorithmKind.LogisticRegression, PhaseOneTopicMatcher.kindFor("Logistic Regression", "Classification"))
        assertEquals(PhaseOneAlgorithmKind.Knn, PhaseOneTopicMatcher.kindFor("K-Nearest Neighbors", "Classification"))
        assertEquals(PhaseOneAlgorithmKind.DecisionTreeClassification, PhaseOneTopicMatcher.kindFor("Decision Tree", "Classification"))
        assertEquals(PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoTopicMatcher.kindFor("Random Forest", "Classification", "Supervised Learning"))
        assertEquals(PhaseTwoAlgorithmKind.LinearSvm, PhaseTwoTopicMatcher.kindFor("Support Vector Machine", "Classification", "Supervised Learning"))
        assertEquals(PhaseThreeAlgorithmKind.KMeans, PhaseThreeTopicMatcher.kindFor("K-Means", "Clustering", "Unsupervised Learning"))
        assertEquals(PhaseFiveConcept.Mlp, PhaseFiveTopicMatcher.kindFor("Multi-Layer Perceptron", "Deep Learning"))
        assertEquals(PhaseSixCnnConcept.Architecture, PhaseSixTopicMatcher.kindFor("CNN", "Deep Learning"))
        assertEquals(PhaseSevenConcept.Lstm, PhaseSevenTopicMatcher.kindFor("LSTM", "Deep Learning"))
    }

    @Test
    fun randomForestRoutesFromEnsembleAndSupervisedEntries() {
        assertEquals(PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoTopicMatcher.kindFor("Random Forest", "Classification", "Supervised Learning"))
        assertEquals(PhaseTwoAlgorithmKind.RandomForestClassifier, PhaseTwoTopicMatcher.kindFor("Random Forest", "Bagging", "Ensemble Learning"))
    }
}
