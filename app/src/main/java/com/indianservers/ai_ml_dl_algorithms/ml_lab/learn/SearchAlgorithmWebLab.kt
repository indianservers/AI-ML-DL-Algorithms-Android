package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.unit.dp

/** Stable asset routes, including aliases of existing reinforcement learning topics. */
fun searchLabAssetPath(title: String): String? = when (title) {
    "Breadth First Search" -> "search_labs/index.html?lab=bfs"
    "Bidirectional Search" -> "search_labs/index.html?lab=bidirectional"
    "Beam Search" -> "search_labs/index.html?lab=beam"
    "Simulated Annealing" -> "phase2_labs/index.html?lab=annealing"
    "Genetic Algorithm" -> "phase2_labs/index.html?lab=genetic"
    "Monte Carlo Tree Search" -> "phase2_labs/index.html?lab=mcts"
    "Alpha-Beta Pruning" -> "phase3_labs/index.html?lab=alpha-beta"
    "Markov Decision Process", "Markov Decision Process Explorer" -> "phase3_labs/index.html?lab=mdp"
    "SARSA", "SARSA Learning" -> "phase3_labs/index.html?lab=sarsa"
    "Hidden Markov Models", "Hidden Markov Model – Forward & Viterbi Algorithms" -> "phase4_labs/index.html?lab=hmm"
    "Policy Iteration" -> "phase4_labs/index.html?lab=policy"
    "Value Iteration" -> "phase4_labs/index.html?lab=value"
    "Q-Learning", "Q Learning" -> "phase5_labs/index.html?lab=qlearning"
    "AI Depth First Search", "Depth First Search" -> "phase5_labs/index.html?lab=dfs"
    "Greedy Best First Search" -> "phase5_labs/index.html?lab=greedy"
    "Minimax Search" -> "phase6_labs/index.html?lab=minimax"
    "Bayesian Networks", "Construction of Bayesian Network" -> "phase6_labs/index.html?lab=construction"
    "Inference from Bayesian Network" -> "phase6_labs/index.html?lab=inference"
    "Uniform Cost Search" -> "phase7_labs/index.html?lab=ucs"
    "Iterative Deepening Depth First Search" -> "phase7_labs/index.html?lab=iddfs"
    "A Star Search", "A* Search" -> "phase7_labs/index.html?lab=astar"
    "Simplified Memory-Bounded A Star Search", "Simplified Memory-Bounded A* Search" -> "phase8_labs/index.html?lab=sma"
    "Hill Climbing Search" -> "phase8_labs/index.html?lab=hill"
    "Local Beam Search" -> "phase8_labs/index.html?lab=local-beam"
    else -> null
}

/** Offline virtual labs matching the AI Algorithms mockups. All bundles use local GSAP. */
@Composable
fun SearchAlgorithmWebLab(topic: LearnTopic, onBack: () -> Unit, showHeader: Boolean = true) {
    val assetPath = searchLabAssetPath(topic.title) ?: return
    val target = "file:///android_asset/$assetPath"
    Column(Modifier.fillMaxSize()) {
        if (showHeader) TextButton(onClick = onBack, modifier = Modifier.padding(start = 8.dp)) { Text("‹  ${topic.domain}") }
        AndroidView(
            modifier = Modifier.fillMaxWidth().weight(1f),
            factory = { context ->
                WebView(context).apply {
                    webViewClient = WebViewClient()
                    settings.javaScriptEnabled = true
                    settings.domStorageEnabled = true
                    settings.allowFileAccess = true
                    setBackgroundColor(android.graphics.Color.WHITE)
                    loadUrl(target)
                }
            },
            onRelease = { view ->
                // Leaving a lab must also release its document, timers and GSAP state.
                view.stopLoading()
                view.loadUrl("about:blank")
                view.destroy()
            },
            update = { view ->
                if (view.url != target) view.loadUrl(target)
            }
        )
    }
}
