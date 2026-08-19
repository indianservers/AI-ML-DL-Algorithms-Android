package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

object LessonQuizScorer {
    fun score(
        questions: List<Pair<McqQuestionRecord, List<McqOptionRecord>>>,
        selectedOptionIdsByQuestionId: Map<Long, Long>
    ): Int = questions.count { (question, options) ->
        val selectedOptionId = selectedOptionIdsByQuestionId[question.id]
        options.any { option -> option.id == selectedOptionId && option.isCorrect }
    }

    fun percentage(score: Int, totalQuestions: Int): Double =
        if (totalQuestions == 0) 0.0 else score * 100.0 / totalQuestions
}
