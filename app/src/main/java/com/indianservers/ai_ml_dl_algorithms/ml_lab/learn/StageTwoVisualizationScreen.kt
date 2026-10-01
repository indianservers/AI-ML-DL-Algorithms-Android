package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.runtime.Composable
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.*

@Composable
internal fun StageTwoVisualizationScreen(topic: LearnTopic, kind: StageTwoVisualization) {
    val existing = when (kind) {
        StageTwoVisualization.RandomForest -> SupervisedVisualization.ClassificationForest
        StageTwoVisualization.ExtraTrees -> SupervisedVisualization.ClassificationExtraTrees
        StageTwoVisualization.AdaBoost -> SupervisedVisualization.ClassificationAdaBoost
        StageTwoVisualization.GradientBoosting -> SupervisedVisualization.ClassificationGradientBoosting
        StageTwoVisualization.Xgboost -> SupervisedVisualization.ClassificationXgboost
        StageTwoVisualization.Lightgbm -> SupervisedVisualization.ClassificationLightgbm
        StageTwoVisualization.Catboost -> SupervisedVisualization.ClassificationCatboost
        else -> null
    }
    if (existing != null) {
        SupervisedVisualizationScreen(topic, existing)
        return
    }
    when (kind.section) {
        "Clustering" -> StageTwoClusteringPanel(kind)
        "Dimensionality Reduction" -> StageTwoReductionPanel(kind)
        "Association Learning" -> StageTwoAssociationPanel(kind)
        "Anomaly Detection" -> StageTwoAnomalyPanel(kind)
        "Methods" -> StageTwoSemiSupervisedPanel(kind)
        "Bagging", "Combining Models" -> StageTwoEnsemblePanel(kind)
        else -> error("No Stage 2 view for " + kind.name)
    }
}
