package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption

@Composable
internal fun StageThreeLanguageScreen(topic: LearnTopic, kind: LanguageVisualization) {
    var selectedDocument by remember(topic.id) { mutableIntStateOf(0) }
    var focus by remember(topic.id) { mutableIntStateOf(0) }
    var viterbi by remember(topic.id) { androidx.compose.runtime.mutableStateOf(false) }
    val frame = remember(kind, selectedDocument, focus, viterbi) { StageThreeLanguageEngine.frame(kind, selectedDocument, focus, viterbi) }
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(kind.title, when (kind) {
            LanguageVisualization.Bow -> "Count words in each document and see why word order disappears"
            LanguageVisualization.Tfidf -> "Compare local term frequency with rarity across the corpus"
            LanguageVisualization.Ngrams -> "Slide a token window to preserve nearby word order"
            LanguageVisualization.NaiveBayes -> "Compare class log scores from word likelihoods and priors"
            LanguageVisualization.Hmm -> "Compare forward probabilities with the most likely hidden path"
            LanguageVisualization.Crf -> "Score an entire label sequence with emission and transition terms"
            LanguageVisualization.Word2Vec -> "Inspect example word vectors and their geometric relationships"
            LanguageVisualization.Cbow -> "Predict the center word from its surrounding context"
            LanguageVisualization.SkipGram -> "Predict surrounding words from a selected center word"
            LanguageVisualization.Glove -> "Inspect global co-occurrence counts behind word vectors"
            LanguageVisualization.FastText -> "Build word features from shared character fragments"
        }) }
        if(StageThreeLanguageEngine.examples(kind)!=null) item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    Text(if(kind==LanguageVisualization.Bow||kind==LanguageVisualization.Tfidf||kind==LanguageVisualization.Ngrams)"Small corpus" else "Choose text example",
                        color = LabText, fontWeight = FontWeight.Bold)
                    val documents=StageThreeLanguageEngine.examples(kind)!!
                    if (documents.maxOf { it.length } <= 18) {
                        documents.chunked(2).forEachIndexed { row, pair ->
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                pair.forEachIndexed { column, document ->
                                    val index = row * 2 + column
                                    SegmentedOption(document, index == selectedDocument, Modifier.weight(1f)) {
                                        selectedDocument = index; focus = 0
                                    }
                                }
                            }
                        }
                    } else {
                        documents.forEachIndexed { i, document ->
                            SegmentedOption(document, i == selectedDocument, Modifier.fillMaxWidth()) {
                                selectedDocument = i; focus = 0
                            }
                        }
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    Text("Token flow", color = LabText, fontWeight = FontWeight.Bold)
                    if (kind == LanguageVisualization.Hmm) Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        SegmentedOption("Forward", !viterbi, Modifier.weight(1f)) { viterbi = false }
                        SegmentedOption("Viterbi", viterbi, Modifier.weight(1f)) { viterbi = true }
                    }
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                        frame.tokens.forEachIndexed { index, token ->
                            SegmentedOption(token, index in frame.highlights, Modifier) { focus = index }
                        }
                    }
                    Text(frame.conclusion, color = LabMuted, fontSize = 12.sp)
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                    Text(when (kind) {
                        LanguageVisualization.Hmm -> "Forward state probabilities"
                        LanguageVisualization.Crf -> "Global sequence terms"
                        LanguageVisualization.Word2Vec -> "Embedding coordinates"
                        LanguageVisualization.Cbow, LanguageVisualization.SkipGram -> "Training pairs"
                        LanguageVisualization.Glove -> "Global co-occurrence"
                        LanguageVisualization.FastText -> "Character 3-grams"
                        else -> "Document-term computation"
                    }, color = LabText, fontWeight = FontWeight.Bold)
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState())) {
                        Column {
                            Row {
                                Text("Item", Modifier.width(158.dp), color = LabCyan, fontSize = 11.sp)
                                frame.columns.forEach { Text(it, Modifier.width(76.dp), color = LabCyan, fontSize = 11.sp) }
                            }
                            frame.rows.forEachIndexed { i, (label, values) ->
                                Row(Modifier.background(if (i % 2 == 0) Color(0x223B69AA) else Color.Transparent).padding(vertical = 5.dp)) {
                                    Text(label, Modifier.width(158.dp), color = LabText, fontSize = 11.sp, maxLines = 1)
                                    values.forEach { value -> Text("%.2f".format(value), Modifier.width(76.dp), color = LabMuted, fontSize = 11.sp) }
                                }
                            }
                        }
                    }
                }
            }
        }
        item {
            GlassPanel(Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text("Step through", color = LabText, fontWeight = FontWeight.Bold)
                    frame.steps.forEachIndexed { i, step ->
                        Text("${i + 1}. $step", color = if (i == focus % frame.steps.size) LabPurple else LabMuted, fontSize = 12.sp)
                    }
                    Text(frame.equation, color = LabCyan, fontSize = 12.sp)
                    if (kind == LanguageVisualization.Ngrams) {
                        Text("N = ${(focus.coerceIn(1, 3))}", color = LabMuted, fontSize = 12.sp)
                        Slider(focus.toFloat().coerceIn(1f, 3f), { focus = it.toInt().coerceIn(1, 3) }, valueRange = 1f..3f)
                    }
                }
            }
        }
    }
}
