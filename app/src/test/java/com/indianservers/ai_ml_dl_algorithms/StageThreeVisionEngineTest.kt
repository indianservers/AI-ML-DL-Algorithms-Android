package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.StageThreeVisionEngine
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisionBox
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisionVisualization
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class StageThreeVisionEngineTest {
    @Test fun syntheticSegmentationNormalizesPixelScoresAndChangesClassAcrossImage() {
        val cells = (0..3).flatMap { row -> (0..3).map { col -> StageThreeVisionEngine.segmentationCell(row, col) } }
        assertEquals(setOf("background", "cat", "dog"), cells.map { it.label }.toSet())
        cells.forEach { cell ->
            assertEquals(1.0, cell.probabilities.sum(), 1e-9)
            assertEquals(cell.probabilities.max(), cell.confidence, 1e-9)
        }
    }
    @Test fun intersectionOverUnionMatchesSimpleGeometry() {
        val a = VisionBox(0.0, 0.0, 1.0, 1.0, "cat", .9)
        val b = VisionBox(.5, .5, 1.0, 1.0, "cat", .8)
        assertEquals(1.0 / 7.0, StageThreeVisionEngine.iou(a, b), 1e-9)
    }

    @Test fun nmsSuppressesOnlyOverlappingSameClass() {
        val kept = StageThreeVisionEngine.nms(StageThreeVisionEngine.proposals, .5, .45)
        assertEquals(2, kept.size)
        assertEquals(setOf("cat", "dog"), kept.map { it.label }.toSet())
    }

    @Test fun cosineRanksSameClassImagesFirst() {
        val nearest = StageThreeVisionEngine.ranking("cat A").first()
        assertEquals("cat B", nearest.first)
        assertTrue(nearest.second > .9)
    }

    @Test fun everyVisionTopicHasDistinctPipeline() {
        val pipelines = VisionVisualization.entries.map { StageThreeVisionEngine.stages(it) }
        assertEquals(VisionVisualization.entries.size, pipelines.distinct().size)
    }
}
