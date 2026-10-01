package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive.PhaseTwoEngines

internal enum class LanguageVisualization(val title: String) {
    Bow("Bag of Words"), Tfidf("TF-IDF"), Ngrams("N-Grams"), NaiveBayes("Naive Bayes Text Classification"),
    Hmm("Hidden Markov Models"), Crf("Conditional Random Fields"), Word2Vec("Word2Vec"),
    Cbow("CBOW"), SkipGram("Skip-Gram"), Glove("GloVe"), FastText("FastText");
    companion object { val byTitle = entries.associateBy { it.title } }
}

internal data class LanguageFrame(
    val tokens: List<String>, val columns: List<String>, val rows: List<Pair<String, List<Double>>>,
    val highlights: Set<Int>, val steps: List<String>, val conclusion: String, val equation: String
)

internal object StageThreeLanguageEngine {
    val corpus = listOf("dog bites man", "man bites dog", "cat sits on mat", "dog sits on rug")
    val textClassificationExamples = listOf("free money offer", "meeting project report", "free project report", "money meeting offer")
    val hmmExamples = listOf("walk shop clean", "shop walk clean", "clean shop walk", "walk walk shop")
    val crfExamples = listOf("John lives in Hyderabad", "Mary works in Delhi", "Alice moved to Mumbai", "Bob lives in Pune")
    val embeddingExamples = listOf("the cat sat on mat", "the dog ran in park", "birds fly over trees", "small cats play outside")
    val fastTextExamples = listOf("playing", "played", "player", "playful")
    fun examples(kind:LanguageVisualization):List<String>?=when(kind) {
        LanguageVisualization.NaiveBayes -> textClassificationExamples
        LanguageVisualization.Hmm -> hmmExamples
        LanguageVisualization.Crf -> crfExamples
        LanguageVisualization.Cbow,LanguageVisualization.SkipGram -> embeddingExamples
        LanguageVisualization.FastText -> fastTextExamples
        LanguageVisualization.Word2Vec,LanguageVisualization.Glove -> null
        else -> corpus
    }
    fun tokenize(text: String): List<String> = text.lowercase().split(Regex("[^a-z]+" )).filter { it.isNotBlank() }
    fun vocabulary(documents: List<String>): List<String> = documents.flatMap(::tokenize).distinct().sorted()
    fun counts(document: String, vocabulary: List<String>): List<Double> {
        val tokens = tokenize(document)
        return vocabulary.map { word -> tokens.count { it == word }.toDouble() }
    }
    fun tfidf(documents: List<String>, documentIndex: Int, word: String): Triple<Double, Double, Double> {
        val terms = tokenize(documents[documentIndex])
        val tf = terms.count { it == word }.toDouble() / terms.size.coerceAtLeast(1)
        val df = documents.count { word in tokenize(it) }
        val idf = ln((documents.size + 1.0) / (df + 1.0)) + 1.0
        return Triple(tf, idf, tf * idf)
    }
    fun ngrams(tokens: List<String>, n: Int): List<String> = tokens.windowed(n).map { it.joinToString(" ") }
    fun cosine(a: List<Double>, b: List<Double>): Double {
        val dot = a.zip(b).sumOf { it.first * it.second }
        return dot / (sqrt(a.sumOf { it * it }) * sqrt(b.sumOf { it * it })).coerceAtLeast(1e-9)
    }
    private val vectors = mapOf(
        "king" to listOf(.9, .8, .1), "queen" to listOf(.9, .2, .1),
        "man" to listOf(.2, .8, .1), "woman" to listOf(.2, .2, .1),
        "dog" to listOf(.1, .1, .9), "cat" to listOf(.15, .1, .82)
    )
    private val contextVectors = mapOf(
        "the" to listOf(.2, .1, .3), "cat" to listOf(.25, .35, .4),
        "sat" to listOf(.1, .6, .2), "on" to listOf(.4, .2, .1),
        "mat" to listOf(.3, .4, .5), "dog" to listOf(.2, .31, .38)
    )
    private fun contextVector(word:String):List<Double> = contextVectors[word] ?: List(3) { dimension ->
        (((word.hashCode().toLong() * (dimension+7) + dimension*37) and 255L).toDouble()/255.0).coerceIn(.05,.95)
    }
    private fun softmax(scores: List<Double>): List<Double> {
        val exponentials = scores.map { exp(it - scores.max()) }
        return exponentials.map { it / exponentials.sum() }
    }
    fun frame(kind: LanguageVisualization, selected: Int, n: Int, viterbi: Boolean = false): LanguageFrame {
        val chosen = selected.coerceIn(corpus.indices)
        val tokens = tokenize(corpus[chosen])
        val vocabulary = vocabulary(corpus)
        return when (kind) {
            LanguageVisualization.Bow -> LanguageFrame(tokens, vocabulary,
                corpus.map { it to counts(it, vocabulary) }, tokens.indices.toSet(),
                listOf("Tokenize", "Build vocabulary", "Count each word"),
                "The first two documents produce the same vector even though word order reverses.", "vectorⱼ = count(wordⱼ, document)")
            LanguageVisualization.Tfidf -> {
                val selectedWord = tokens[n.coerceIn(tokens.indices)]
                val stats = tfidf(corpus, chosen, selectedWord)
                LanguageFrame(tokens, vocabulary, corpus.mapIndexed { i, text ->
                    text to vocabulary.map { tfidf(corpus, i, it).third }
                }, setOf(n.coerceIn(tokens.indices)),
                    listOf("TF = %.3f".format(stats.first), "IDF = %.3f".format(stats.second),
                        "TF-IDF = %.3f".format(stats.third)),
                    "A term earns more weight when it appears in this document but in fewer documents overall.",
                    "TF-IDF(t,d) = TF(t,d) × [ln((N+1)/(DF+1))+1]")
            }
            LanguageVisualization.Ngrams -> {
                val order = n.coerceIn(1, 3)
                val grams = ngrams(tokens, order)
                LanguageFrame(tokens, listOf("Position", "$order-gram"), grams.mapIndexed { i, g ->
                    g to listOf(i.toDouble(), (i + order - 1).toDouble())
                }, (0 until order).toSet(), listOf("Window width $order", "${grams.size} overlapping windows"),
                    "The window preserves local word order, unlike bag of words.", "gramᵢ = tokens[i .. i+n−1]")
            }
            LanguageVisualization.NaiveBayes -> {
                val words=tokenize(textClassificationExamples[chosen])
                val counts=PhaseTwoEngines.textVocabulary.associateWith { word -> words.count { it==word } }
                val model=PhaseTwoEngines.multinomialNaiveBayes(counts)
                val labels=listOf("Promotional", "Work")
                LanguageFrame(words, listOf("count", "log P(w|promo)", "log P(w|work)"),
                    words.distinct().map { word -> word to listOf(counts.getValue(word).toDouble(),
                        PhaseTwoEngines.multinomialTokenLogLikelihood(word,0),
                        PhaseTwoEngines.multinomialTokenLogLikelihood(word,1)) }, words.indices.toSet(),
                    listOf("Promotional score = %.2f".format(model.classScores.getValue(0)),
                        "Work score = %.2f".format(model.classScores.getValue(1))),
                    "Prediction: ${labels[model.prediction]}. Uses the same add-one-smoothed Multinomial Naive Bayes engine as supervised learning.",
                    "log P(class|text) ∝ log P(class) + Σ log P(word|class)")
            }
            LanguageVisualization.Hmm -> {
                val observations = tokenize(hmmExamples[chosen])
                val observationCodes = mapOf("walk" to 0,"shop" to 1,"clean" to 2)
                val states = listOf("Sunny", "Rainy")
                val transition = listOf(listOf(.7, .3), listOf(.4, .6))
                val emission = listOf(listOf(.6, .3, .1), listOf(.1, .4, .5))
                val firstObservation=observationCodes.getValue(observations.first())
                val scores = mutableListOf(listOf(.6 * emission[0][firstObservation], .4 * emission[1][firstObservation]))
                val pointers = mutableListOf<List<Int>>()
                for (t in 1..2) {
                    val previous = scores.last()
                    pointers += List(2) { s -> (0..1).maxBy { prior -> previous[prior] * transition[prior][s] } }
                    scores += List(2) { s ->
                        val predecessors = (0..1).map { prior -> previous[prior] * transition[prior][s] }
                        (if (viterbi) predecessors.max() else predecessors.sum()) * emission[s][observationCodes.getValue(observations[t])]
                    }
                }
                val path = if (viterbi) {
                    var state = scores.last().indices.maxBy { scores.last()[it] }
                    val chosen = mutableListOf(state)
                    for (t in pointers.lastIndex downTo 0) {
                        state = pointers[t][state]
                        chosen += state
                    }
                    chosen.asReversed().map { states[it] }.joinToString(" → ")
                } else ""
                LanguageFrame(observations, states, scores.mapIndexed { i, values -> "t${i + 1}: ${observations[i]}" to values },
                    setOf(n.coerceIn(0, 2)), listOf("Hidden weather → observed activity",
                        if (viterbi) "Best path $path; score = %.4f".format(scores.last().max()) else "P(sequence) = %.4f".format(scores.last().sum())),
                    if (viterbi) "Viterbi keeps only the strongest predecessor for each hidden state."
                    else "Forward sums every predecessor path for each hidden state.",
                    if (viterbi) "δₜ(j) = emissionⱼ(oₜ) maxᵢ δₜ₋₁(i) transitionᵢⱼ"
                    else "αₜ(j) = emissionⱼ(oₜ) Σᵢ αₜ₋₁(i) transitionᵢⱼ")
            }
            LanguageVisualization.Crf -> {
                val sequence = crfExamples[chosen].split(" ")
                val labels = listOf("B-PER", "O", "O", "B-LOC")
                val emissions = listOf(2.0+sequence[0].length*.08, 1.5+sequence[1].length*.05,
                    1.6+sequence[2].length*.1, 2.0+sequence[3].length*.08)
                val transition = listOf(0.0, .3, .8, .2)
                LanguageFrame(sequence, listOf("Emission", "Transition", "Total"), sequence.indices.map { i ->
                    "${sequence[i]} → ${labels[i]}" to listOf(emissions[i], transition[i], emissions[i] + transition[i])
                }, setOf(n.coerceIn(0, 3)), listOf("Sequence score = %.2f".format(emissions.sum() + transition.sum()),
                    "Adjacent labels affect one another"),
                    "A CRF scores the whole label path, so independent best labels need not form the best sequence.",
                    "score(x,y) = Σ emission(xᵢ,yᵢ) + Σ transition(yᵢ₋₁,yᵢ)")
            }
            LanguageVisualization.Word2Vec -> {
                val analogy = vectors.getValue("king").indices.map { i ->
                    vectors.getValue("king")[i] - vectors.getValue("man")[i] + vectors.getValue("woman")[i]
                }
                LanguageFrame(vectors.keys.toList(), listOf("dim 1", "dim 2", "dim 3"),
                    vectors.map { it.key to it.value } + ("king − man + woman" to analogy), setOf(0, 1, 2, 3),
                    listOf("Cosine to queen = %.3f".format(cosine(analogy, vectors.getValue("queen")))),
                    "Illustrative fixed embeddings show vector arithmetic; CBOW and Skip-Gram below show training directions.",
                    "v(king) − v(man) + v(woman) ≈ v(queen)")
            }
            LanguageVisualization.Cbow, LanguageVisualization.SkipGram -> {
                val sentence=tokenize(embeddingExamples[chosen])
                val center = sentence[1]
                val context = sentence.filterIndexed { index,_ -> index!=1 }
                val contextMean = (0..2).map { dimension -> context.map { contextVector(it)[dimension] }.average() }
                val centerVector = contextVector(center)
                val candidates = listOf(center,"cat","dog").distinct()
                val scores = candidates.map { word -> contextMean.zip(contextVector(word)).sumOf { it.first*it.second } }
                val probabilities = softmax(scores)
                val rows = if (kind == LanguageVisualization.Cbow)
                    context.map { it to contextVector(it) } + ("mean context" to contextMean)
                else listOf(center to centerVector) + context.map { it to contextVector(it) }
                LanguageFrame(sentence, listOf("dim 1", "dim 2", "dim 3"), rows,
                    if (kind == LanguageVisualization.Cbow) sentence.indices.filter{it!=1}.toSet() else setOf(1),
                    listOf(if (kind == LanguageVisualization.Cbow) "Aggregate context embeddings" else "Start with the center embedding",
                        if (kind == LanguageVisualization.Cbow) "P($center | mean context) = %.3f".format(probabilities[0])
                        else "Center–context dot($center, ${context.first()}) = %.3f".format(centerVector.zip(contextVector(context.first())).sumOf{it.first*it.second})),
                    if (kind == LanguageVisualization.Cbow) "Many context words feed one center-word prediction."
                    else "One center word produces multiple context predictions.",
                    if (kind == LanguageVisualization.Cbow) "P(center | mean(context vectors))" else "Σ log P(context word | center)")
            }
            LanguageVisualization.Glove -> {
                val words = listOf("dog", "cat", "bites", "sits")
                val matrix = words.map { a -> words.map { b ->
                    corpus.sumOf { doc -> val t = tokenize(doc); t.windowed(2).count { pair -> pair.contains(a) && pair.contains(b) } }.toDouble()
                } }
                LanguageFrame(words, words, words.mapIndexed { i, word -> word to matrix[i] }, setOf(n.coerceIn(words.indices)),
                    listOf("Global co-occurrence matrix", "Factorize log counts into word and context vectors"),
                    "GloVe learns from corpus-wide word-pair counts rather than only one local prediction window.",
                    "loss = Σ f(Xᵢⱼ)(wᵢ·w̃ⱼ + bᵢ + b̃ⱼ − log Xᵢⱼ)²")
            }
            LanguageVisualization.FastText -> {
                val word = fastTextExamples[chosen]
                val grams = "<$word>".windowed(3)
                LanguageFrame(listOf(word), listOf("Length"), grams.map { it to listOf(it.length.toDouble()) }, setOf(0),
                    listOf("Character 3-grams: ${grams.joinToString(", ")}", "Sum subword vectors to form the word representation"),
                    "Shared character pieces give rare or unseen forms a meaningful representation.",
                    "v(word) = Σ v(character n-gram)")
            }
        }
    }
}
