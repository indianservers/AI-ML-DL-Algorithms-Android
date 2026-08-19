package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Attempts
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Lessons
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Options
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Pages
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Progress
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data.LessonDatabaseContract.Questions

class LessonDatabaseHelper(context: Context) : SQLiteOpenHelper(
    context,
    LessonDatabaseContract.DATABASE_NAME,
    null,
    LessonDatabaseContract.DATABASE_VERSION
) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL("PRAGMA foreign_keys=ON")
        db.execSQL(
            """
            CREATE TABLE ${Lessons.TABLE} (
                ${Lessons.ALGORITHM_ID} TEXT PRIMARY KEY,
                ${Lessons.ALGORITHM_TITLE} TEXT NOT NULL,
                ${Lessons.DOMAIN} TEXT NOT NULL,
                ${Lessons.SECTION} TEXT NOT NULL,
                ${Lessons.IS_AWARD_WINNING} INTEGER NOT NULL,
                ${Lessons.EXPERT_NOTE} TEXT NOT NULL,
                ${Lessons.CREATED_AT} INTEGER NOT NULL,
                ${Lessons.UPDATED_AT} INTEGER NOT NULL
            )
            """.trimIndent()
        )
        db.execSQL(
            """
            CREATE TABLE ${Pages.TABLE} (
                ${Pages.ID} INTEGER PRIMARY KEY AUTOINCREMENT,
                ${Pages.ALGORITHM_ID} TEXT NOT NULL,
                ${Pages.PAGE_NUMBER} INTEGER NOT NULL,
                ${Pages.TITLE} TEXT NOT NULL,
                ${Pages.HTML_CONTENT} TEXT NOT NULL,
                ${Pages.STORY} TEXT NOT NULL,
                ${Pages.EXPLANATION} TEXT NOT NULL,
                ${Pages.REALTIME_EXAMPLE} TEXT NOT NULL,
                ${Pages.REALTIME_APPLICATIONS} TEXT NOT NULL,
                ${Pages.TEACHER_TIP} TEXT NOT NULL,
                UNIQUE(${Pages.ALGORITHM_ID}, ${Pages.PAGE_NUMBER}),
                FOREIGN KEY(${Pages.ALGORITHM_ID}) REFERENCES ${Lessons.TABLE}(${Lessons.ALGORITHM_ID}) ON DELETE CASCADE
            )
            """.trimIndent()
        )
        db.execSQL(
            """
            CREATE TABLE ${Questions.TABLE} (
                ${Questions.ID} INTEGER PRIMARY KEY AUTOINCREMENT,
                ${Questions.ALGORITHM_ID} TEXT NOT NULL,
                ${Questions.QUESTION_NUMBER} INTEGER NOT NULL,
                ${Questions.QUESTION} TEXT NOT NULL,
                ${Questions.EXPLANATION} TEXT NOT NULL,
                UNIQUE(${Questions.ALGORITHM_ID}, ${Questions.QUESTION_NUMBER}),
                FOREIGN KEY(${Questions.ALGORITHM_ID}) REFERENCES ${Lessons.TABLE}(${Lessons.ALGORITHM_ID}) ON DELETE CASCADE
            )
            """.trimIndent()
        )
        db.execSQL(
            """
            CREATE TABLE ${Options.TABLE} (
                ${Options.ID} INTEGER PRIMARY KEY AUTOINCREMENT,
                ${Options.QUESTION_ID} INTEGER NOT NULL,
                ${Options.OPTION_NUMBER} INTEGER NOT NULL,
                ${Options.OPTION_TEXT} TEXT NOT NULL,
                ${Options.IS_CORRECT} INTEGER NOT NULL,
                UNIQUE(${Options.QUESTION_ID}, ${Options.OPTION_NUMBER}),
                FOREIGN KEY(${Options.QUESTION_ID}) REFERENCES ${Questions.TABLE}(${Questions.ID}) ON DELETE CASCADE
            )
            """.trimIndent()
        )
        db.execSQL(
            """
            CREATE TABLE ${Attempts.TABLE} (
                ${Attempts.ID} INTEGER PRIMARY KEY AUTOINCREMENT,
                ${Attempts.ALGORITHM_ID} TEXT NOT NULL,
                ${Attempts.SCORE} INTEGER NOT NULL,
                ${Attempts.TOTAL_QUESTIONS} INTEGER NOT NULL,
                ${Attempts.PERCENTAGE} REAL NOT NULL,
                ${Attempts.ATTEMPTED_AT} INTEGER NOT NULL,
                FOREIGN KEY(${Attempts.ALGORITHM_ID}) REFERENCES ${Lessons.TABLE}(${Lessons.ALGORITHM_ID}) ON DELETE CASCADE
            )
            """.trimIndent()
        )
        db.execSQL(
            """
            CREATE TABLE ${Progress.TABLE} (
                ${Progress.ALGORITHM_ID} TEXT PRIMARY KEY,
                ${Progress.LAST_PAGE_NUMBER} INTEGER NOT NULL,
                ${Progress.COMPLETED} INTEGER NOT NULL,
                ${Progress.BEST_SCORE} INTEGER NOT NULL,
                ${Progress.BEST_SCORE_TOTAL} INTEGER NOT NULL,
                ${Progress.UPDATED_AT} INTEGER NOT NULL,
                FOREIGN KEY(${Progress.ALGORITHM_ID}) REFERENCES ${Lessons.TABLE}(${Lessons.ALGORITHM_ID}) ON DELETE CASCADE
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX idx_pages_algorithm ON ${Pages.TABLE}(${Pages.ALGORITHM_ID})")
        db.execSQL("CREATE INDEX idx_questions_algorithm ON ${Questions.TABLE}(${Questions.ALGORITHM_ID})")
        db.execSQL("CREATE INDEX idx_attempts_algorithm ON ${Attempts.TABLE}(${Attempts.ALGORITHM_ID})")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        db.execSQL("DROP TABLE IF EXISTS ${Options.TABLE}")
        db.execSQL("DROP TABLE IF EXISTS ${Questions.TABLE}")
        db.execSQL("DROP TABLE IF EXISTS ${Pages.TABLE}")
        db.execSQL("DROP TABLE IF EXISTS ${Attempts.TABLE}")
        db.execSQL("DROP TABLE IF EXISTS ${Progress.TABLE}")
        db.execSQL("DROP TABLE IF EXISTS ${Lessons.TABLE}")
        onCreate(db)
    }

    override fun onConfigure(db: SQLiteDatabase) {
        super.onConfigure(db)
        db.setForeignKeyConstraintsEnabled(true)
    }
}
