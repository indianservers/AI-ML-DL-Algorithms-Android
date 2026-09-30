package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.AlgorithmIconKind
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.algorithmIconKind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AlgorithmIconTest {
    @Test
    fun everyCatalogTopicGetsARepresentativeIcon() {
        val resolved = LearnCatalog.topics.associateWith(::algorithmIconKind)
        assertEquals(LearnCatalog.topics.size, resolved.size)
        assertTrue(resolved.values.all { it in AlgorithmIconKind.entries })
    }

    @Test
    fun flagshipAlgorithmsUseConceptSpecificIcons() {
        val kinds = LearnCatalog.flagshipTopics.associate { it.title to algorithmIconKind(it) }
        assertEquals(AlgorithmIconKind.Regression, kinds["Simple Linear Regression"])
        assertEquals(AlgorithmIconKind.Neighbours, kinds["K-Nearest Neighbors"])
        assertEquals(AlgorithmIconKind.Tree, kinds["Decision Tree"])
        assertEquals(AlgorithmIconKind.Forest, kinds["Random Forest"])
        assertEquals(AlgorithmIconKind.Clustering, kinds["K-Means"])
        assertEquals(AlgorithmIconKind.Convolution, kinds["CNN"])
        assertEquals(AlgorithmIconKind.Sequence, kinds["LSTM"])
    }
}
