package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

import android.content.ContentValues
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Attempts
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Lessons
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Options
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Pages
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Progress
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Questions

class LessonDao(private val helper: LessonDatabaseHelper) {
    fun lessonCount(): Int = helper.readableDatabase.rawQuery("SELECT COUNT(*) FROM ${Lessons.TABLE}", null).use { cursor ->
        if (cursor.moveToFirst()) cursor.getInt(0) else 0
    }

    fun seedIfEmpty(seedLessons: List<SeedLesson>) {
        if (lessonCount() > 0) return
        val db = helper.writableDatabase
        db.beginTransaction()
        try {
            seedLessons.forEach { seedLesson ->
                insertLesson(db, seedLesson.lesson)
                seedLesson.pages.forEach { page -> insertPage(db, page) }
                seedLesson.questions.forEach { seedQuestion ->
                    val questionId = insertQuestion(db, seedQuestion.question)
                    seedQuestion.options.forEachIndexed { index, option ->
                        insertOption(
                            db,
                            McqOptionRecord(
                                questionId = questionId,
                                optionNumber = index + 1,
                                optionText = option.optionText,
                                isCorrect = option.isCorrect
                            )
                        )
                    }
                }
            }
            db.setTransactionSuccessful()
        } finally {
            db.endTransaction()
        }
    }

    fun getLesson(algorithmId: String): AlgorithmLessonRecord? {
        return helper.readableDatabase.query(
            Lessons.TABLE,
            null,
            "${Lessons.ALGORITHM_ID} = ?",
            arrayOf(algorithmId),
            null,
            null,
            null
        ).use { cursor -> if (cursor.moveToFirst()) cursor.toLesson() else null }
    }

    fun getPages(algorithmId: String): List<LessonPageRecord> {
        return helper.readableDatabase.query(
            Pages.TABLE,
            null,
            "${Pages.ALGORITHM_ID} = ?",
            arrayOf(algorithmId),
            null,
            null,
            Pages.PAGE_NUMBER
        ).use { cursor -> cursor.mapRows { it.toPage() } }
    }

    fun getQuestions(algorithmId: String): List<Pair<McqQuestionRecord, List<McqOptionRecord>>> {
        val questions = helper.readableDatabase.query(
            Questions.TABLE,
            null,
            "${Questions.ALGORITHM_ID} = ?",
            arrayOf(algorithmId),
            null,
            null,
            Questions.QUESTION_NUMBER
        ).use { cursor -> cursor.mapRows { it.toQuestion() } }

        return questions.map { question ->
            val options = helper.readableDatabase.query(
                Options.TABLE,
                null,
                "${Options.QUESTION_ID} = ?",
                arrayOf(question.id.toString()),
                null,
                null,
                Options.OPTION_NUMBER
            ).use { cursor -> cursor.mapRows { it.toOption() } }
            question to options
        }
    }

    fun updateProgress(algorithmId: String, lastPageNumber: Int, completed: Boolean) {
        val now = System.currentTimeMillis()
        val existing = getProgress(algorithmId)
        val values = ContentValues().apply {
            put(Progress.ALGORITHM_ID, algorithmId)
            put(Progress.LAST_PAGE_NUMBER, maxOf(lastPageNumber, existing?.lastPageNumber ?: 0))
            put(Progress.COMPLETED, if (completed || existing?.completed == true) 1 else 0)
            put(Progress.BEST_SCORE, existing?.bestScore ?: 0)
            put(Progress.BEST_SCORE_TOTAL, existing?.bestScoreTotal ?: 0)
            put(Progress.UPDATED_AT, now)
        }
        helper.writableDatabase.insertWithOnConflict(Progress.TABLE, null, values, SQLiteDatabase.CONFLICT_REPLACE)
    }

    fun recordQuizAttempt(algorithmId: String, score: Int, totalQuestions: Int): QuizAttemptRecord {
        val percentage = if (totalQuestions == 0) 0.0 else score * 100.0 / totalQuestions
        val attempt = QuizAttemptRecord(
            algorithmId = algorithmId,
            score = score,
            totalQuestions = totalQuestions,
            percentage = percentage,
            attemptedAt = System.currentTimeMillis()
        )
        val db = helper.writableDatabase
        db.beginTransaction()
        try {
            val attemptId = db.insertOrThrow(Attempts.TABLE, null, ContentValues().apply {
                put(Attempts.ALGORITHM_ID, attempt.algorithmId)
                put(Attempts.SCORE, attempt.score)
                put(Attempts.TOTAL_QUESTIONS, attempt.totalQuestions)
                put(Attempts.PERCENTAGE, attempt.percentage)
                put(Attempts.ATTEMPTED_AT, attempt.attemptedAt)
            })
            val existing = getProgress(algorithmId)
            val isBest = existing == null ||
                totalQuestions > 0 && score * (existing.bestScoreTotal.coerceAtLeast(1)) >= existing.bestScore * totalQuestions
            val values = ContentValues().apply {
                put(Progress.ALGORITHM_ID, algorithmId)
                put(Progress.LAST_PAGE_NUMBER, existing?.lastPageNumber ?: 5)
                put(Progress.COMPLETED, if (existing?.completed == true) 1 else 0)
                put(Progress.BEST_SCORE, if (isBest) score else existing?.bestScore ?: 0)
                put(Progress.BEST_SCORE_TOTAL, if (isBest) totalQuestions else existing?.bestScoreTotal ?: 0)
                put(Progress.UPDATED_AT, attempt.attemptedAt)
            }
            db.insertWithOnConflict(Progress.TABLE, null, values, SQLiteDatabase.CONFLICT_REPLACE)
            db.setTransactionSuccessful()
            return attempt.copy(id = attemptId)
        } finally {
            db.endTransaction()
        }
    }

    fun getProgress(algorithmId: String): LessonProgressRecord? {
        return helper.readableDatabase.query(
            Progress.TABLE,
            null,
            "${Progress.ALGORITHM_ID} = ?",
            arrayOf(algorithmId),
            null,
            null,
            null
        ).use { cursor -> if (cursor.moveToFirst()) cursor.toProgress() else null }
    }

    private fun insertLesson(db: SQLiteDatabase, lesson: AlgorithmLessonRecord) {
        db.insertWithOnConflict(Lessons.TABLE, null, ContentValues().apply {
            put(Lessons.ALGORITHM_ID, lesson.algorithmId)
            put(Lessons.ALGORITHM_TITLE, lesson.algorithmTitle)
            put(Lessons.DOMAIN, lesson.domain)
            put(Lessons.SECTION, lesson.section)
            put(Lessons.IS_AWARD_WINNING, if (lesson.isAwardWinning) 1 else 0)
            put(Lessons.EXPERT_NOTE, lesson.expertNote)
            put(Lessons.CREATED_AT, lesson.createdAt)
            put(Lessons.UPDATED_AT, lesson.updatedAt)
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    private fun insertPage(db: SQLiteDatabase, page: LessonPageRecord) {
        db.insertWithOnConflict(Pages.TABLE, null, ContentValues().apply {
            put(Pages.ALGORITHM_ID, page.algorithmId)
            put(Pages.PAGE_NUMBER, page.pageNumber)
            put(Pages.TITLE, page.title)
            put(Pages.HTML_CONTENT, page.htmlContent)
            put(Pages.STORY, page.story)
            put(Pages.EXPLANATION, page.explanation)
            put(Pages.REALTIME_EXAMPLE, page.realtimeExample)
            put(Pages.REALTIME_APPLICATIONS, page.realtimeApplications)
            put(Pages.TEACHER_TIP, page.teacherTip)
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    private fun insertQuestion(db: SQLiteDatabase, question: McqQuestionRecord): Long {
        return db.insertWithOnConflict(Questions.TABLE, null, ContentValues().apply {
            put(Questions.ALGORITHM_ID, question.algorithmId)
            put(Questions.QUESTION_NUMBER, question.questionNumber)
            put(Questions.QUESTION, question.question)
            put(Questions.EXPLANATION, question.explanation)
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    private fun insertOption(db: SQLiteDatabase, option: McqOptionRecord) {
        db.insertWithOnConflict(Options.TABLE, null, ContentValues().apply {
            put(Options.QUESTION_ID, option.questionId)
            put(Options.OPTION_NUMBER, option.optionNumber)
            put(Options.OPTION_TEXT, option.optionText)
            put(Options.IS_CORRECT, if (option.isCorrect) 1 else 0)
        }, SQLiteDatabase.CONFLICT_REPLACE)
    }

    private fun Cursor.toLesson() = AlgorithmLessonRecord(
        algorithmId = getString(getColumnIndexOrThrow(Lessons.ALGORITHM_ID)),
        algorithmTitle = getString(getColumnIndexOrThrow(Lessons.ALGORITHM_TITLE)),
        domain = getString(getColumnIndexOrThrow(Lessons.DOMAIN)),
        section = getString(getColumnIndexOrThrow(Lessons.SECTION)),
        isAwardWinning = getInt(getColumnIndexOrThrow(Lessons.IS_AWARD_WINNING)) == 1,
        expertNote = getString(getColumnIndexOrThrow(Lessons.EXPERT_NOTE)),
        createdAt = getLong(getColumnIndexOrThrow(Lessons.CREATED_AT)),
        updatedAt = getLong(getColumnIndexOrThrow(Lessons.UPDATED_AT))
    )

    private fun Cursor.toPage() = LessonPageRecord(
        id = getLong(getColumnIndexOrThrow(Pages.ID)),
        algorithmId = getString(getColumnIndexOrThrow(Pages.ALGORITHM_ID)),
        pageNumber = getInt(getColumnIndexOrThrow(Pages.PAGE_NUMBER)),
        title = getString(getColumnIndexOrThrow(Pages.TITLE)),
        htmlContent = getString(getColumnIndexOrThrow(Pages.HTML_CONTENT)),
        story = getString(getColumnIndexOrThrow(Pages.STORY)),
        explanation = getString(getColumnIndexOrThrow(Pages.EXPLANATION)),
        realtimeExample = getString(getColumnIndexOrThrow(Pages.REALTIME_EXAMPLE)),
        realtimeApplications = getString(getColumnIndexOrThrow(Pages.REALTIME_APPLICATIONS)),
        teacherTip = getString(getColumnIndexOrThrow(Pages.TEACHER_TIP))
    )

    private fun Cursor.toQuestion() = McqQuestionRecord(
        id = getLong(getColumnIndexOrThrow(Questions.ID)),
        algorithmId = getString(getColumnIndexOrThrow(Questions.ALGORITHM_ID)),
        questionNumber = getInt(getColumnIndexOrThrow(Questions.QUESTION_NUMBER)),
        question = getString(getColumnIndexOrThrow(Questions.QUESTION)),
        explanation = getString(getColumnIndexOrThrow(Questions.EXPLANATION))
    )

    private fun Cursor.toOption() = McqOptionRecord(
        id = getLong(getColumnIndexOrThrow(Options.ID)),
        questionId = getLong(getColumnIndexOrThrow(Options.QUESTION_ID)),
        optionNumber = getInt(getColumnIndexOrThrow(Options.OPTION_NUMBER)),
        optionText = getString(getColumnIndexOrThrow(Options.OPTION_TEXT)),
        isCorrect = getInt(getColumnIndexOrThrow(Options.IS_CORRECT)) == 1
    )

    private fun Cursor.toProgress() = LessonProgressRecord(
        algorithmId = getString(getColumnIndexOrThrow(Progress.ALGORITHM_ID)),
        lastPageNumber = getInt(getColumnIndexOrThrow(Progress.LAST_PAGE_NUMBER)),
        completed = getInt(getColumnIndexOrThrow(Progress.COMPLETED)) == 1,
        bestScore = getInt(getColumnIndexOrThrow(Progress.BEST_SCORE)),
        bestScoreTotal = getInt(getColumnIndexOrThrow(Progress.BEST_SCORE_TOTAL)),
        updatedAt = getLong(getColumnIndexOrThrow(Progress.UPDATED_AT))
    )

    private inline fun <T> Cursor.mapRows(mapper: (Cursor) -> T): List<T> {
        val rows = mutableListOf<T>()
        while (moveToNext()) rows.add(mapper(this))
        return rows
    }
}
