package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

data class AlgorithmLessonRecord(
    val algorithmId: String,
    val algorithmTitle: String,
    val domain: String,
    val section: String,
    val isAwardWinning: Boolean,
    val expertNote: String,
    val createdAt: Long,
    val updatedAt: Long
)

data class LessonPageRecord(
    val id: Long = 0,
    val algorithmId: String,
    val pageNumber: Int,
    val title: String,
    val htmlContent: String,
    val story: String,
    val explanation: String,
    val realtimeExample: String,
    val realtimeApplications: String,
    val teacherTip: String
)

data class McqQuestionRecord(
    val id: Long = 0,
    val algorithmId: String,
    val questionNumber: Int,
    val question: String,
    val explanation: String
)

data class McqOptionRecord(
    val id: Long = 0,
    val questionId: Long,
    val optionNumber: Int,
    val optionText: String,
    val isCorrect: Boolean
)

data class QuizAttemptRecord(
    val id: Long = 0,
    val algorithmId: String,
    val score: Int,
    val totalQuestions: Int,
    val percentage: Double,
    val attemptedAt: Long
)

data class LessonProgressRecord(
    val algorithmId: String,
    val lastPageNumber: Int,
    val completed: Boolean,
    val bestScore: Int,
    val bestScoreTotal: Int,
    val updatedAt: Long
)

data class SeedLesson(
    val lesson: AlgorithmLessonRecord,
    val pages: List<LessonPageRecord>,
    val questions: List<SeedQuestion>
)

data class SeedQuestion(
    val question: McqQuestionRecord,
    val options: List<SeedOption>
)

data class SeedOption(
    val optionText: String,
    val isCorrect: Boolean
)
