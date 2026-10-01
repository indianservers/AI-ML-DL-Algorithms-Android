package com.indianservers.ai_ml_dl_algorithms

import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.test.espresso.Espresso
import org.junit.Rule
import org.junit.Test

class VisualizationNavigationTest {
    @get:Rule val compose = createAndroidComposeRule<MainActivity>()

    @Test fun searchOpensAlgorithmAndHomeReturnsToCatalog() {
        compose.onNodeWithText("Algorithm Categories").assertExists()
        compose.onNode(hasSetTextAction()).performTextInput("K-Nearest Neighbors")
        compose.onNode(hasText("K-Nearest Neighbors") and hasClickAction() and !hasSetTextAction()).performClick()
        compose.onNodeWithText("Visualization").performClick()
        compose.onNodeWithText("Home", substring = true).performClick()
        compose.onNodeWithText("Algorithm Categories").assertExists()
    }

    @Test fun backAtHomeRequiresExitConfirmationAndNoKeepsCatalogOpen() {
        compose.onNodeWithText("Algorithm Categories").assertExists()
        Espresso.pressBack()
        compose.onNodeWithText("Exit the app?").assertExists()
        compose.onNodeWithText("No").performClick()
        compose.onNodeWithText("Algorithm Categories").assertExists()
    }
}
