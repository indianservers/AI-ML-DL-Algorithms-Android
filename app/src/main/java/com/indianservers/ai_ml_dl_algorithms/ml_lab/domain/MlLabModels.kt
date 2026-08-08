package com.indianservers.ai_ml_dl_algorithms.ml_lab.domain

enum class LearningDepth(val title: String, val description: String) {
    Beginner("Beginner", "Intuition first"),
    Intro("School", "Simple formulas"),
    University("University", "Derivations and assumptions"),
    Advanced("Advanced", "Numerics and edge cases")
}

enum class AlgorithmStatus {
    Interactive,
    LessonReady,
    Future
}

enum class AlgorithmFamily(val title: String) {
    Regression("Regression"),
    Classification("Classification"),
    Clustering("Clustering"),
    DimensionalityReduction("Dimensionality Reduction"),
    Ensemble("Ensemble Learning"),
    Anomaly("Anomaly Detection"),
    TimeSeries("Time Series"),
    Reinforcement("Reinforcement Learning"),
    DeepLearning("Deep Learning"),
    Association("Association")
}

data class Algorithm(
    val id: String,
    val title: String,
    val family: AlgorithmFamily,
    val subtitle: String,
    val status: AlgorithmStatus,
    val accent: Long
)

data class LessonSection(
    val title: String,
    val beginner: String,
    val intro: String,
    val university: String,
    val advanced: String
) {
    fun body(depth: LearningDepth): String = when (depth) {
        LearningDepth.Beginner -> beginner
        LearningDepth.Intro -> intro
        LearningDepth.University -> university
        LearningDepth.Advanced -> advanced
    }
}

data class Point2D(val x: Float, val y: Float, val label: Int = 0)

data class TrainingSnapshot(
    val epoch: Int,
    val loss: Float,
    val weight: Float,
    val bias: Float,
    val gradientWeight: Float,
    val gradientBias: Float
)

data class RegressionState(
    val points: List<Point2D>,
    val snapshots: List<TrainingSnapshot>,
    val selectedEpoch: Int
) {
    val current: TrainingSnapshot get() = snapshots[selectedEpoch.coerceIn(snapshots.indices)]
}
