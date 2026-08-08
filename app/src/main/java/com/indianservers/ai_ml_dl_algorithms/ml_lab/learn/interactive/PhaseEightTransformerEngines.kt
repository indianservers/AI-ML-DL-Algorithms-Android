package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.interactive

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

enum class PhaseEightConcept(val displayName: String) {
    Attention("Attention intuition"),
    Qkv("Query, Key, Value"),
    ScaledAttention("Scaled Dot-Product Attention"),
    SelfAttention("Self-Attention"),
    AttentionMatrix("Attention Matrix"),
    MultiHead("Multi-Head Attention"),
    PositionalEncoding("Positional Encoding"),
    EncoderBlock("Transformer Encoder Block"),
    DecoderCausal("Transformer Decoder / Causal Attention"),
    TokenPrediction("Small Offline Token Prediction Lab")
}

data class TokenVector(val token: String, val vector: List<Double>)
data class AttentionCell(val query: Int, val key: Int, val dot: Double, val scaled: Double, val weight: Double, val masked: Boolean = false)
data class AttentionState(
    val tokens: List<String>,
    val embeddings: List<List<Double>>,
    val queries: List<List<Double>>,
    val keys: List<List<Double>>,
    val values: List<List<Double>>,
    val cells: List<List<AttentionCell>>,
    val output: List<List<Double>>,
    val causal: Boolean
)
data class MultiHeadState(val heads: List<AttentionState>, val concatenated: List<List<Double>>, val projected: List<List<Double>>)
data class TransformerBlockState(val input: List<List<Double>>, val positioned: List<List<Double>>, val attention: AttentionState, val residualNorm: List<List<Double>>, val feedForward: List<List<Double>>, val output: List<List<Double>>)
data class TokenPredictionState(val sequence: List<String>, val expected: String, val logits: List<Double>, val probabilities: List<Double>, val predicted: String, val loss: Double, val snapshots: List<List<List<Double>>>)

object PhaseEightTopicMatcher {
    fun kindFor(title: String, domain: String): PhaseEightConcept? = if (domain != "Deep Learning") null else when (title) {
        "Attention Mechanism", "Attention" -> PhaseEightConcept.Attention
        "Self-Attention" -> PhaseEightConcept.SelfAttention
        "Multi-Head Attention" -> PhaseEightConcept.MultiHead
        "Positional Encoding" -> PhaseEightConcept.PositionalEncoding
        "Transformer", "Encoder Transformer" -> PhaseEightConcept.EncoderBlock
        "Decoder Transformer" -> PhaseEightConcept.DecoderCausal
        else -> null
    }
}

object PhaseEightEngines {
    val defaultTokens = listOf("The", "cat", "sat", "on", "mat")
    val vocab = listOf("A", "B", "C", "D", "red", "blue", "green", "yellow")

    fun embeddings(tokens: List<String>, dim: Int = 4): List<List<Double>> =
        tokens.mapIndexed { index, token -> List(dim) { d -> deterministic(token.hashCode() + index, d, 0) } }

    fun projection(input: List<List<Double>>, salt: Int): List<List<Double>> =
        input.map { vector ->
            List(vector.size) { out ->
                vector.indices.sumOf { i -> vector[i] * deterministic(salt, i, out) } / sqrt(vector.size.toDouble())
            }
        }

    fun attention(tokens: List<String> = defaultTokens, dim: Int = 4, causal: Boolean = false, temperature: Double = 1.0): AttentionState {
        val x = embeddings(tokens, dim)
        val q = projection(x, 11)
        val k = projection(x, 17)
        val v = projection(x, 23)
        val scale = sqrt(dim.toDouble()) * temperature.coerceAtLeast(.05)
        val scoreRows = q.mapIndexed { qi, qv ->
            k.mapIndexed { ki, kv ->
                if (causal && ki > qi) Double.NEGATIVE_INFINITY else dot(qv, kv) / scale
            }
        }
        val weights = scoreRows.map { row -> softmaxMasked(row) }
        val cells = weights.mapIndexed { qi, row ->
            row.mapIndexed { ki, weight ->
                val masked = causal && ki > qi
                val rawDot = if (masked) 0.0 else dot(q[qi], k[ki])
                AttentionCell(qi, ki, rawDot, if (masked) Double.NEGATIVE_INFINITY else rawDot / scale, weight, masked)
            }
        }
        val output = weights.map { row -> v.first().indices.map { d -> row.indices.sumOf { i -> row[i] * v[i][d] } } }
        return AttentionState(tokens, x, q, k, v, cells, output, causal)
    }

    fun multiHead(tokens: List<String> = defaultTokens, heads: Int = 2, dim: Int = 4): MultiHeadState {
        val states = List(heads.coerceIn(1, 4)) { h -> attention(tokens, dim, causal = false, temperature = 1.0 + h * .25) }
        val concat = tokens.indices.map { t -> states.flatMap { it.output[t] } }
        val projected = concat.map { row -> List(dim) { d -> row.indices.sumOf { i -> row[i] * deterministic(41, i, d) } / sqrt(row.size.toDouble()) } }
        return MultiHeadState(states, concat, projected)
    }

    fun positional(position: Int, dim: Int): List<Double> =
        List(dim) { i ->
            val denom = 10000.0.pow((2 * (i / 2)).toDouble() / dim)
            if (i % 2 == 0) sin(position / denom) else cos(position / denom)
        }

    fun addPosition(x: List<List<Double>>): List<List<Double>> =
        x.mapIndexed { pos, vector -> vector.indices.map { vector[it] + positional(pos, vector.size)[it] } }

    fun layerNorm(vector: List<Double>): List<Double> {
        val mean = vector.average()
        val variance = vector.map { (it - mean).pow(2) }.average()
        return vector.map { (it - mean) / sqrt(variance + 1e-6) }
    }

    fun feedForward(x: List<Double>): List<Double> {
        val hidden = List(x.size * 2) { h -> kotlin.math.max(0.0, x.indices.sumOf { x[it] * deterministic(53, it, h) }) }
        return List(x.size) { o -> hidden.indices.sumOf { hidden[it] * deterministic(59, it, o) } / sqrt(hidden.size.toDouble()) }
    }

    fun encoderBlock(tokens: List<String> = defaultTokens, dim: Int = 4): TransformerBlockState {
        val x = embeddings(tokens, dim)
        val positioned = addPosition(x)
        val attn = attention(tokens, dim)
        val residualNorm = positioned.indices.map { i -> layerNorm(positioned[i].indices.map { positioned[i][it] + attn.output[i][it] }) }
        val ff = residualNorm.map { feedForward(it) }
        val out = residualNorm.indices.map { i -> layerNorm(residualNorm[i].indices.map { residualNorm[i][it] + ff[i][it] }) }
        return TransformerBlockState(x, positioned, attn, residualNorm, ff, out)
    }

    fun tokenPrediction(sequence: List<String> = listOf("A", "B"), expected: String = "C"): TokenPredictionState {
        val attn = attention(sequence, dim = 4, causal = true)
        val last = attn.output.last()
        val logits = vocab.take(4).mapIndexed { i, _ -> last.indices.sumOf { d -> last[d] * deterministic(71, d, i) } + if (i == 2) 1.4 else 0.0 }
        val probs = softmaxMasked(logits)
        val predicted = vocab[probs.indices.maxBy { probs[it] }]
        val expectedIndex = vocab.indexOf(expected).coerceAtLeast(0)
        val loss = -ln(probs[expectedIndex].coerceAtLeast(1e-9))
        val snapshots = listOf(attention(sequence, 4).cells.map { row -> row.map { it.weight } }, attn.cells.map { row -> row.map { it.weight } })
        return TokenPredictionState(sequence, expected, logits, probs, predicted, loss, snapshots)
    }

    fun parameterCount(vocabSize: Int, dim: Int, heads: Int, ffHidden: Int): Map<String, Int> {
        val embedding = vocabSize * dim
        val qkv = 3 * dim * dim
        val output = dim * dim
        val ff = dim * ffHidden + ffHidden * dim + ffHidden + dim
        return mapOf("Embedding" to embedding, "QKV" to qkv, "Output projection" to output, "Feed-forward" to ff, "Total" to embedding + qkv + output + ff)
    }

    fun knownAttention(): Pair<List<Double>, List<Double>> {
        val q = listOf(1.0, 0.0)
        val keys = listOf(listOf(1.0, 0.0), listOf(0.0, 1.0))
        val values = listOf(listOf(2.0, 0.0), listOf(0.0, 3.0))
        val weights = softmaxMasked(keys.map { dot(q, it) / sqrt(2.0) })
        val output = values.first().indices.map { d -> weights.indices.sumOf { weights[it] * values[it][d] } }
        return weights to output
    }

    private fun dot(a: List<Double>, b: List<Double>): Double = a.indices.sumOf { a[it] * b[it] }
    private fun softmaxMasked(values: List<Double>): List<Double> {
        val finite = values.filter { it.isFinite() }
        val max = finite.maxOrNull() ?: 0.0
        val exps = values.map { if (it.isFinite()) exp(it - max) else 0.0 }
        val total = exps.sum().coerceAtLeast(1e-12)
        return exps.map { it / total }
    }
    private fun deterministic(seed: Int, a: Int, b: Int): Double {
        val raw = sin(seed * 131.0 + a * 37.0 + b * 19.0) * 10007.0
        return (raw - kotlin.math.floor(raw)) * 2.0 - 1.0
    }
}
