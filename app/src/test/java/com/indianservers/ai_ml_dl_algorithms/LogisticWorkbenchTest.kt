package com.indianservers.ai_ml_dl_algorithms

import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.fitLogistic
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.LabPoint
import org.junit.Assert.assertTrue
import org.junit.Test

class LogisticWorkbenchTest {
    @Test
    fun fittedBoundaryFollowsChangedData() {
        val leftZero = listOf(
            LabPoint(-.9, -.4, 0), LabPoint(-.7, .1, 0),
            LabPoint(.6, -.1, 1), LabPoint(.9, .5, 1)
        )
        val model = fitLogistic(leftZero)
        assertTrue(model.probability(-.8, 0.0) < .5)
        assertTrue(model.probability(.8, 0.0) > .5)

        val flipped = fitLogistic(leftZero.map { it.copy(label = 1 - it.label) })
        assertTrue(flipped.probability(-.8, 0.0) > .5)
        assertTrue(flipped.probability(.8, 0.0) < .5)
    }
}
