package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.AlgorithmLessonSeedFactory
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonQuizScorer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.McqOptionRecord
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.McqQuestionRecord
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LessonSeedFactoryTest {
    @Test
    fun everyAlgorithmGetsFiveHtmlLessonPagesAndFiveMcqs() {
        val lessons = AlgorithmLessonSeedFactory.buildAll(LearningDepth.Beginner)

        assertEquals(LearnCatalog.topics.size, lessons.size)
        lessons.forEach { seed ->
            assertEquals(5, seed.pages.size)
            assertEquals(5, seed.questions.size)
            assertTrue(seed.lesson.algorithmTitle.isNotBlank())
            assertTrue(seed.lesson.expertNote.isNotBlank())

            seed.pages.forEachIndexed { index, page ->
                assertEquals(index + 1, page.pageNumber)
                assertTrue(page.htmlContent.contains("<article"))
                assertTrue(page.title.contains(seed.lesson.algorithmTitle))
                assertTrue(page.story.isNotBlank())
                assertTrue(page.explanation.isNotBlank())
                assertTrue(page.realtimeExample.isNotBlank())
                assertTrue(page.teacherTip.isNotBlank())
            }

            seed.questions.forEach { question ->
                assertTrue(question.question.question.contains(seed.lesson.algorithmTitle))
                assertEquals(4, question.options.size)
                assertEquals(1, question.options.count { it.isCorrect })
                assertTrue(question.options.all { it.optionText.isNotBlank() })
            }
        }
    }

    @Test
    fun generatedContentPassesPhaseFiveQualityAudit() {
        val bannedDummyWords = listOf("lorem", "todo", "placeholder", "dummy")
        val lessons = AlgorithmLessonSeedFactory.buildAll(LearningDepth.Beginner)

        lessons.forEach { seed ->
            val lessonText = seed.pages.joinToString(" ") {
                "${it.title} ${it.story} ${it.explanation} ${it.realtimeExample} ${it.realtimeApplications} ${it.teacherTip}"
            }.lowercase()
            val htmlText = seed.pages.joinToString(" ") { it.htmlContent }.lowercase()

            bannedDummyWords.forEach { banned ->
                assertTrue("${seed.lesson.algorithmTitle} contains banned dummy word $banned", banned !in lessonText)
            }
            assertTrue(lessonText.contains(seed.lesson.algorithmTitle.lowercase()))
            assertTrue(lessonText.contains("real"))
            assertTrue(htmlText.contains("teacher"))
            assertTrue(lessonText.contains("example"))

            seed.pages.forEach { page ->
                assertTrue(page.htmlContent.contains("<header>"))
                assertTrue(page.htmlContent.contains("<section class=\"story\">"))
                assertTrue(page.htmlContent.contains("<section class=\"simple\">"))
                assertTrue(page.htmlContent.contains("<section class=\"realtime\">"))
                assertTrue(page.htmlContent.contains("<section class=\"applications\">"))
                assertTrue(page.htmlContent.contains("<aside class=\"teacher-tip\">"))
                assertTrue(page.htmlContent.length > 900)
            }
        }
    }

    @Test
    fun quizScorerCountsOnlyCorrectSelectedOptions() {
        val questions = listOf(
            McqQuestionRecord(id = 10, algorithmId = "demo", questionNumber = 1, question = "Q1", explanation = "E1") to listOf(
                McqOptionRecord(id = 101, questionId = 10, optionNumber = 1, optionText = "Wrong", isCorrect = false),
                McqOptionRecord(id = 102, questionId = 10, optionNumber = 2, optionText = "Right", isCorrect = true)
            ),
            McqQuestionRecord(id = 20, algorithmId = "demo", questionNumber = 2, question = "Q2", explanation = "E2") to listOf(
                McqOptionRecord(id = 201, questionId = 20, optionNumber = 1, optionText = "Right", isCorrect = true),
                McqOptionRecord(id = 202, questionId = 20, optionNumber = 2, optionText = "Wrong", isCorrect = false)
            )
        )

        assertEquals(1, LessonQuizScorer.score(questions, mapOf(10L to 102L, 20L to 202L)))
        assertEquals(2, LessonQuizScorer.score(questions, mapOf(10L to 102L, 20L to 201L)))
        assertEquals(0, LessonQuizScorer.score(questions, emptyMap()))
        assertEquals(50.0, LessonQuizScorer.percentage(1, 2), 0.001)
        assertEquals(0.0, LessonQuizScorer.percentage(1, 0), 0.001)
    }

    @Test
    fun flagshipAlgorithmsUsePremiumNarratives() {
        val lessonsById = AlgorithmLessonSeedFactory.buildAll(LearningDepth.Beginner)
            .associateBy { it.lesson.algorithmId }

        LearnCatalog.flagshipTopics.forEach { topic ->
            val lesson = lessonsById.getValue(topic.id)
            val text = lesson.pages.joinToString(" ") { "${it.title} ${it.story} ${it.explanation}" }

            assertTrue(lesson.lesson.isAwardWinning)
            assertTrue(text.contains(topic.title))
            assertTrue("Expected premium narrative for ${topic.title}", "young inventor" !in text.lowercase())
            assertTrue("Expected realtime phrasing for ${topic.title}", lesson.pages.any { it.realtimeExample.contains("system", true) || it.realtimeExample.contains("team", true) || it.realtimeExample.contains("app", true) || it.realtimeExample.contains("store", true) })
        }
    }

    @Test
    fun allAlgorithmsAvoidOldGenericScaffolding() {
        val oldScaffoldPhrases = listOf(
            "young inventor",
            "box of messy clues",
            "patient teacher",
            "school science fair",
            "opens the toolbox",
            "formula is not a monster",
            "superpower and a warning label"
        )

        AlgorithmLessonSeedFactory.buildAll(LearningDepth.Beginner).forEach { lesson ->
            val text = lesson.pages.joinToString(" ") { "${it.title} ${it.story} ${it.explanation} ${it.realtimeExample} ${it.teacherTip}" }.lowercase()
            oldScaffoldPhrases.forEach { phrase ->
                assertTrue("${lesson.lesson.algorithmTitle} still uses old scaffold phrase: $phrase", phrase !in text)
            }
            assertTrue("${lesson.lesson.algorithmTitle} should name a concrete story character", lesson.pages.first().story.contains(" faces a real problem: ") || lesson.lesson.isAwardWinning)
            assertTrue("${lesson.lesson.algorithmTitle} should include production or deployment quality thinking", text.contains("validation") || text.contains("fresh") || text.contains("baseline"))
        }
    }

    @Test
    fun mcqAnswersAreDistributedAndDistractorsAreNotGenericRepeats() {
        val lessons = AlgorithmLessonSeedFactory.buildAll(LearningDepth.Beginner)
        val answerCounts = mutableMapOf("A" to 0, "B" to 0, "C" to 0, "D" to 0)
        val distractors = mutableSetOf<String>()
        var capacityAnswerCount = 0
        var genericFormulaAnswerCount = 0
        var trainingOnlyMistakeCount = 0

        lessons.forEach { lesson ->
            lesson.questions.forEach { question ->
                val correctIndex = question.options.indexOfFirst { it.isCorrect }
                val letter = ('A' + correctIndex).toString()
                answerCounts[letter] = answerCounts.getValue(letter) + 1
                val correctText = question.options[correctIndex].optionText
                if (correctText.contains("Capacity or complexity", true)) capacityAnswerCount++
                if (correctText.contains("It defines the signal or objective", true)) genericFormulaAnswerCount++
                if (correctText.contains("Evaluating on training data only", true)) trainingOnlyMistakeCount++
                question.options.filterNot { it.isCorrect }.forEach { distractors.add(it.optionText) }
            }
        }

        assertTrue(answerCounts.values.max() - answerCounts.values.min() <= 1)
        assertTrue("Expected many plausible distractors, found ${distractors.size}", distractors.size > 120)
        assertEquals(0, capacityAnswerCount)
        assertEquals(0, genericFormulaAnswerCount)
        assertEquals(0, trainingOnlyMistakeCount)
    }
}
