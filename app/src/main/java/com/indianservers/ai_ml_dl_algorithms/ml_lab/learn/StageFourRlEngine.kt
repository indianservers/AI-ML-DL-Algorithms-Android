package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

internal object StageFourRlEngine {
    private const val SIDE = 4
    private const val GOAL = 15
    private const val CLIFF = 11
    private val deltas = listOf(-SIDE, 1, SIDE, -1)

    fun next(state: Int, action: Int): Int {
        val row = state / SIDE
        val col = state % SIDE
        val candidate = when (action) {
            0 -> if (row > 0) state - SIDE else state
            1 -> if (col < SIDE - 1) state + 1 else state
            2 -> if (row < SIDE - 1) state + SIDE else state
            else -> if (col > 0) state - 1 else state
        }
        return if (candidate == 5) state else candidate
    }

    fun reward(nextState: Int, stepPenalty: Double = -.04, goalReward: Double = 1.0): Double =
        when (nextState) { GOAL -> goalReward; CLIFF -> -1.0; else -> stepPenalty }

    fun transition(state: Int, action: Int, slip: Double): List<Pair<Int, Double>> {
        val p = slip.coerceIn(0.0, .45)
        return listOf(action to (1 - p), (action + 3) % 4 to (p / 2), (action + 1) % 4 to (p / 2))
            .groupBy { next(state, it.first) }
            .map { (s, outcomes) -> s to outcomes.sumOf { it.second } }
    }

    fun bellman(values: List<Double>, state: Int, action: Int, gamma: Double, slip: Double,
                stepPenalty:Double=-.04,goalReward:Double=1.0): Double =
        transition(state, action, slip).sumOf { (s, p) ->
            p * (reward(s,stepPenalty,goalReward) + if (s == GOAL || s == CLIFF) 0.0 else gamma * values[s])
        }

    fun valueIteration(iterations: Int, gamma: Double = .9, slip: Double = .1,
                       stepPenalty:Double=-.04,goalReward:Double=1.0): List<Double> {
        var values = List(SIDE * SIDE) { 0.0 }
        repeat(iterations.coerceIn(0, 60)) {
            values = values.indices.map { s ->
                if (s == GOAL || s == CLIFF || s == 5) 0.0
                else (0..3).maxOf { action -> bellman(values, s, action, gamma, slip,stepPenalty,goalReward) }
            }
        }
        return values
    }

    fun policyIteration(iterations: Int, gamma: Double = .9): Pair<List<Double>, List<Int>> {
        var policy = List(16) { 1 }
        var values = List(16) { 0.0 }
        repeat(iterations.coerceIn(0, 24)) {
            repeat(4) {
                values = values.indices.map { s ->
                    if (s == GOAL || s == CLIFF || s == 5) 0.0 else bellman(values, s, policy[s], gamma, .1)
                }
            }
            policy = policy.indices.map { s -> (0..3).maxBy { a -> bellman(values, s, a, gamma, .1) } }
        }
        return values to policy
    }

    fun mcReturn(rewards: List<Double>, gamma: Double, from: Int = 0): Double =
        rewards.drop(from).foldRight(0.0) { r, acc -> r + gamma * acc }

    fun tdTarget(reward: Double, nextValue: Double, gamma: Double) = reward + gamma * nextValue
    fun sarsaTarget(reward: Double, nextActionValue: Double, gamma: Double) = reward + gamma * nextActionValue
    fun qTarget(reward: Double, nextValues: List<Double>, gamma: Double) = reward + gamma * nextValues.max()
    fun expectedSarsaTarget(reward: Double, nextValues: List<Double>, probabilities: List<Double>, gamma: Double) =
        reward + gamma * nextValues.zip(probabilities).sumOf { (q, p) -> q * p }
    fun doubleDqnTarget(reward: Double, online: List<Double>, target: List<Double>, gamma: Double) =
        reward + gamma * target[online.indices.maxBy { online[it] }]
    fun duelingQ(value: Double, advantages: List<Double>): List<Double> =
        advantages.map { value + it - advantages.average() }
    fun ppoClipped(ratio: Double, advantage: Double, epsilon: Double): Double =
        min(ratio * advantage, ratio.coerceIn(1 - epsilon, 1 + epsilon) * advantage)
    fun entropy(probabilities: List<Double>): Double =
        -probabilities.sumOf { p -> if (p <= 0) 0.0 else p * ln(p) }
    fun softmax(logits: List<Double>): List<Double> {
        val offset = logits.max()
        val exps = logits.map { exp(it - offset) }
        return exps.map { it / exps.sum() }
    }
    fun banditEstimate(rewards: List<Double>) = rewards.average()
    fun banditUcb(estimate: Double, pulls: Int, total: Int, scale: Double = 1.0) =
        estimate + scale * kotlin.math.sqrt(2 * ln(total.coerceAtLeast(2).toDouble()) / pulls.coerceAtLeast(1))
    fun discounted(rewards: List<Double>, gamma: Double): List<Double> = rewards.indices.map { mcReturn(rewards, gamma, it) }
    fun advantage(reward: Double, nextValue: Double, currentValue: Double, gamma: Double) =
        reward + gamma * nextValue - currentValue
    fun twinTarget(q1: Double, q2: Double, reward: Double, gamma: Double) = reward + gamma * min(q1, q2)
    fun sacObjective(q: Double, probabilities: List<Double>, alpha: Double) = q + alpha * entropy(probabilities)
    fun finite(values: List<Double>) = values.all { it.isFinite() }
}
