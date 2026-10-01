package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LanguageVisualization
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeLanguageEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageThreeLanguageEngineTest {
    @Test fun bagOfWordsIgnoresOrder() {
        val words = StageThreeLanguageEngine.vocabulary(StageThreeLanguageEngine.corpus)
        assertEquals(StageThreeLanguageEngine.counts("dog bites man", words),
            StageThreeLanguageEngine.counts("man bites dog", words))
    }

    @Test fun idfDropsWhenWordAppearsAcrossMoreDocuments() {
        val corpus = StageThreeLanguageEngine.corpus
        val dog = StageThreeLanguageEngine.tfidf(corpus, 0, "dog")
        val bites = StageThreeLanguageEngine.tfidf(corpus, 0, "bites")
        assertTrue(dog.second < bites.second || dog.second == bites.second)
        assertTrue(dog.third > 0)
    }

    @Test fun ngramsAndCosineAreCorrect() {
        assertEquals(listOf("a b", "b c"), StageThreeLanguageEngine.ngrams(listOf("a", "b", "c"), 2))
        assertEquals(1.0, StageThreeLanguageEngine.cosine(listOf(1.0, 0.0), listOf(2.0, 0.0)), 1e-9)
    }

    @Test fun everyClassicalAndWordViewHasComputedRows() {
        LanguageVisualization.entries.forEach { kind ->
            val frame = StageThreeLanguageEngine.frame(kind, 0, 1)
            assertTrue(kind.name, frame.rows.isNotEmpty())
            assertTrue(kind.name, frame.steps.isNotEmpty())
        }
    }

    @Test fun hmmForwardAndViterbiUseDifferentRecurrences() {
        val forward=StageThreeLanguageEngine.frame(LanguageVisualization.Hmm,0,1,false)
        val viterbi=StageThreeLanguageEngine.frame(LanguageVisualization.Hmm,0,1,true)
        assertTrue(viterbi.steps.last().contains("Best path"))
        assertTrue(forward.rows.last().second.sum()>viterbi.rows.last().second.sum())
    }

    @Test fun textNaiveBayesUsesSharedSmoothedLikelihoods() {
        val frame=StageThreeLanguageEngine.frame(LanguageVisualization.NaiveBayes,0,0)
        assertTrue(frame.conclusion.contains("Promotional"))
        assertTrue(frame.rows.first().second[1].isFinite())
    }
}
