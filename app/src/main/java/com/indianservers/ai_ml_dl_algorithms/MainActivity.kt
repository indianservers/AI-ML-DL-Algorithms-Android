package com.indianservers.ai_ml_dl_algorithms

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.indianservers.ai_ml_dl_algorithms.ml_lab.presentation.MlLabApp
import com.indianservers.ai_ml_dl_algorithms.ui.theme.AIMLDLAlgorithmsTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AIMLDLAlgorithmsTheme(darkTheme = true, dynamicColor = false) {
                MlLabApp()
            }
        }
    }
}
