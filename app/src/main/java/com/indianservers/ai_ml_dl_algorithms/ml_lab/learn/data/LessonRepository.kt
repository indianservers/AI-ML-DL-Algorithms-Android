package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

import android.content.Context
import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth

class LessonRepository private constructor(context: Context) {
    private val dao = LessonDao(LessonDatabaseHelper(context.applicationContext))

    fun seedLessonsIfNeeded(depth: LearningDepth = LearningDepth.Beginner) {
        dao.seedIfEmpty(AlgorithmLessonSeedFactory.buildAll(depth))
    }

    fun lessonFor(algorithmId: String): AlgorithmLessonRecord? = dao.getLesson(algorithmId)

    fun pagesFor(algorithmId: String): List<LessonPageRecord> = dao.getPages(algorithmId)

    fun questionsFor(algorithmId: String): List<Pair<McqQuestionRecord, List<McqOptionRecord>>> = dao.getQuestions(algorithmId)

    fun progressFor(algorithmId: String): LessonProgressRecord? = dao.getProgress(algorithmId)

    fun updateProgress(algorithmId: String, lastPageNumber: Int, completed: Boolean = false) {
        dao.updateProgress(algorithmId, lastPageNumber, completed)
    }

    fun recordQuizAttempt(algorithmId: String, selectedOptionIdsByQuestionId: Map<Long, Long>): QuizAttemptRecord {
        val questions = questionsFor(algorithmId)
        val score = LessonQuizScorer.score(questions, selectedOptionIdsByQuestionId)
        return dao.recordQuizAttempt(algorithmId, score, questions.size)
    }

    companion object {
        @Volatile private var instance: LessonRepository? = null

        fun get(context: Context): LessonRepository {
            return instance ?: synchronized(this) {
                instance ?: LessonRepository(context).also { instance = it }
            }
        }
    }
}
