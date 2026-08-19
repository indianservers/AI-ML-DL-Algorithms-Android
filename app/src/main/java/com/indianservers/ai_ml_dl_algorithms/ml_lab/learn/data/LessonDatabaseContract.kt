package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

internal object LessonDatabaseContract {
    const val DATABASE_NAME = "algorithm_lessons.db"
    const val DATABASE_VERSION = 1

    object Lessons {
        const val TABLE = "algorithm_lessons"
        const val ALGORITHM_ID = "algorithm_id"
        const val ALGORITHM_TITLE = "algorithm_title"
        const val DOMAIN = "domain"
        const val SECTION = "section"
        const val IS_AWARD_WINNING = "is_award_winning"
        const val EXPERT_NOTE = "expert_note"
        const val CREATED_AT = "created_at"
        const val UPDATED_AT = "updated_at"
    }

    object Pages {
        const val TABLE = "lesson_pages"
        const val ID = "id"
        const val ALGORITHM_ID = "algorithm_id"
        const val PAGE_NUMBER = "page_number"
        const val TITLE = "title"
        const val HTML_CONTENT = "html_content"
        const val STORY = "story"
        const val EXPLANATION = "explanation"
        const val REALTIME_EXAMPLE = "realtime_example"
        const val REALTIME_APPLICATIONS = "realtime_applications"
        const val TEACHER_TIP = "teacher_tip"
    }

    object Questions {
        const val TABLE = "mcq_questions"
        const val ID = "id"
        const val ALGORITHM_ID = "algorithm_id"
        const val QUESTION_NUMBER = "question_number"
        const val QUESTION = "question"
        const val EXPLANATION = "explanation"
    }

    object Options {
        const val TABLE = "mcq_options"
        const val ID = "id"
        const val QUESTION_ID = "question_id"
        const val OPTION_NUMBER = "option_number"
        const val OPTION_TEXT = "option_text"
        const val IS_CORRECT = "is_correct"
    }

    object Attempts {
        const val TABLE = "quiz_attempts"
        const val ID = "id"
        const val ALGORITHM_ID = "algorithm_id"
        const val SCORE = "score"
        const val TOTAL_QUESTIONS = "total_questions"
        const val PERCENTAGE = "percentage"
        const val ATTEMPTED_AT = "attempted_at"
    }

    object Progress {
        const val TABLE = "lesson_progress"
        const val ALGORITHM_ID = "algorithm_id"
        const val LAST_PAGE_NUMBER = "last_page_number"
        const val COMPLETED = "completed"
        const val BEST_SCORE = "best_score"
        const val BEST_SCORE_TOTAL = "best_score_total"
        const val UPDATED_AT = "updated_at"
    }
}
