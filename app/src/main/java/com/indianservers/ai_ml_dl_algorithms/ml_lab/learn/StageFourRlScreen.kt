package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Slider
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import kotlin.math.max
import kotlin.math.min

internal data class RlSpec(val subtitle: String, val flow: List<String>, val equation: String, val distinction: String)

internal object StageFourRlSpecs {
    fun forTitle(name: String): RlSpec = when (name) {
        "Agent" -> RlSpec("Observe the decision loop", listOf("State", "Agent", "Action", "Environment", "Next state + reward"), "(s, a) → (s′, r)", "This page describes the decision maker; it does not train a policy.")
        "Environment" -> RlSpec("Inspect transition dynamics", listOf("State", "Chosen action", "Transition probabilities", "Next state"), "P(s′ | s, a)", "Slippery moves can lead to more than one next state.")
        "State" -> RlSpec("Inspect a sufficient state representation", listOf("Grid cell", "Coordinates", "Nearby wall", "State vector"), "s = [row, column, goal distance, wall flag]", "A state encodes the information the agent needs to decide.")
        "Action" -> RlSpec("Compare available moves and their effects", listOf("Current state", "Valid actions", "Selected move", "Next state"), "a ∈ {↑, →, ↓, ←}", "Boundary and wall actions leave the agent in the same cell.")
        "Reward" -> RlSpec("Change reward design and inspect return", listOf("Move", "Immediate reward", "Discounted return"), "Gₜ = rₜ₊₁ + γGₜ₊₁", "A goal bonus and step cost change which route is attractive.")
        "Policy" -> RlSpec("Inspect state-to-action choices", listOf("State", "Action probabilities", "Chosen direction"), "π(a | s)", "A policy maps each state to one action or a distribution of actions.")
        "Value Function" -> RlSpec("Back up future value on the grid", listOf("State", "Possible next states", "Reward + γV", "Expected value"), "V(s) = Σₛ′ P(s′|s,a)[r + γV(s′)]", "V is a state value. Q(s,a) would additionally condition on an action.")
        "Markov Decision Process", "Markov Decision Process Explorer" -> RlSpec("Link transitions, rewards, policy and value", listOf("State", "Action", "P(s′|s,a)", "R(s,a,s′)", "V(s′)"), "⟨S, A, P, R, γ⟩", "The grid and transition readout describe the same MDP.")
        "Multi-Armed Bandit" -> RlSpec("Explore without state transitions", listOf("Choose arm", "Sample reward", "Update estimate", "Regret"), "Q(a) ← Q(a) + [r − Q(a)] / N(a)", "Bandits compare arms; they have no next-state dynamics.")
        "Dynamic Programming", "Policy Iteration" -> RlSpec("Alternate policy evaluation and improvement", listOf("Known model", "Evaluate current policy", "Improve actions", "Repeat"), "Vπ(s) ← Σₛ′ P(s′|s,π(s))[r + γVπ(s′)]", "A full transition model is known; evaluation precedes improvement.")
        "Value Iteration" -> RlSpec("Apply Bellman optimality backups", listOf("Known model", "Max-action backup", "Update values", "Extract policy"), "V(s) ← maxₐ Σₛ′ P(s′|s,a)[r + γV(s′)]", "Values are optimized before the greedy policy is extracted.")
        "Monte Carlo Learning" -> RlSpec("Wait for a complete episode", listOf("Episode", "Terminal reward", "Return backward", "Update estimate"), "Gₜ = Σₖ γᵏrₜ₊ₖ₊₁", "No value update occurs until the sampled episode ends.")
        "Temporal Difference Learning" -> RlSpec("Bootstrap after each transition", listOf("Transition", "r + γV(s′)", "TD error", "Immediate update"), "δ = r + γV(s′) − V(s)", "The next-state estimate is used before the episode ends.")
        "SARSA", "SARSA Learning" -> RlSpec("Update using the sampled next action", listOf("S,A", "R,S′", "Choose actual A′", "Q(S′,A′)", "Update"), "Q ← Q + α[r + γQ(S′,A′) − Q]", "On-policy: the action actually selected by the behavior policy supplies the target.")
        "Q-Learning", "Q Learning" -> RlSpec("Use a greedy target even when behavior explores", listOf("Behavior action", "R,S′", "maxₐ Q(S′,a)", "Update"), "Q ← Q + α[r + γ maxₐ Q(S′,a) − Q]", "Off-policy: the target action can differ from the next behavior action.")
        "Expected SARSA" -> RlSpec("Average over next-policy actions", listOf("R,S′", "π(a|S′)", "Σ πQ", "Update"), "target = r + γΣₐ π(a|S′)Q(S′,a)", "Uses an expectation, neither a sampled action nor a maximum.")
        "Deep Q-Network" -> RlSpec("Replay experiences into an online Q network", listOf("State", "Online Q outputs", "Replay buffer", "Mini-batch", "Target network"), "L = [r + γ maxₐ Qtarget(s′,a) − Qonline(s,a)]²", "The target network and replay memory stabilize Q learning.")
        "Double DQN" -> RlSpec("Separate selecting from evaluating", listOf("Online selects argmax", "Target evaluates that action", "TD target"), "target = r + γQtarget(s′, argmaxₐ Qonline(s′,a))", "Online and target estimates play different roles.")
        "Dueling DQN" -> RlSpec("Split state value and action advantage", listOf("Shared features", "V(s) stream", "A(s,a) stream", "Combine Q"), "Q(s,a) = V(s) + A(s,a) − meanₐ A(s,a)", "The value and advantage streams recombine before action selection.")
        "REINFORCE" -> RlSpec("Weight log-policy gradients by full returns", listOf("Trajectory", "Discounted returns", "log π(a|s)", "Policy update"), "∇J ≈ Σₜ Gₜ∇log π(aₜ|sₜ)", "This Monte Carlo policy update waits for episode returns.")
        "Policy Gradient" -> RlSpec("Change action probabilities directly", listOf("State", "Policy probabilities", "Sample action", "Advantage", "Update policy"), "∇J ≈ A(s,a)∇log π(a|s)", "The learner optimizes a policy distribution rather than Q-value targets.")
        "Actor-Critic" -> RlSpec("Actor chooses; critic evaluates", listOf("Actor π(a|s)", "Action", "Critic V(s)", "Advantage", "Update both"), "A = r + γV(s′) − V(s)", "The critic supplies a baseline/error to train the actor.")
        "A2C" -> RlSpec("Synchronize several rollout workers", listOf("Four environments", "Collect rollouts", "Merge batch", "One shared update"), "g = meanᵢ gᵢ", "All workers wait and contribute to one synchronous update.")
        "A3C" -> RlSpec("Apply asynchronous worker updates", listOf("Worker 1", "Worker 2", "Worker 3", "Shared parameters"), "θglobal ← θglobal + Δθworker", "Workers do not wait for a joint batch; update order matters.")
        "DDPG" -> RlSpec("Control a continuous action with actor and critic", listOf("State", "Actor continuous a", "Critic Q(s,a)", "Replay", "Target actor/critic"), "a = μθ(s); ∇θJ ≈ ∇aQ(s,a)∇θμθ(s)", "A deterministic actor emits a continuous control value.")
        "TD3" -> RlSpec("Correct DDPG overestimation", listOf("Smoothed target action", "Twin target critics", "min(Q₁,Q₂)", "Delayed actor update"), "target = r + γ min(Q₁′,Q₂′)(s′, μ′(s′)+ε)", "Two critics, target smoothing, and delayed actor updates act together.")
        "PPO" -> RlSpec("Clip harmful policy-ratio updates", listOf("Old policy", "New/old ratio", "Clip to 1±ε", "Surrogate objective"), "Lclip = min(rA, clip(r,1−ε,1+ε)A)", "The clipped objective flattens incentives beyond the allowed ratio.")
        "SAC" -> RlSpec("Trade reward against entropy", listOf("Stochastic actor", "Twin Q critics", "Reward + α entropy", "Update"), "J ≈ E[Q(s,a) + αH(π(·|s))]", "An entropy bonus keeps the action distribution exploratory.")
        "Model-Based Reinforcement Learning" -> RlSpec("Plan through learned dynamics", listOf("Real transitions", "Fit P̂(s′|s,a)", "Simulate futures", "Plan", "Real action"), "a* = argmaxₐ Eₚ̂[Σγᵗrₜ]", "The model supplies imagined rollouts; the chosen action is executed in the real environment.")
        "Multi-Agent Reinforcement Learning" -> RlSpec("Inspect interacting policies", listOf("Agent A observation", "Agent B observation", "Joint actions", "Shared or opposing rewards"), "P(s′ | s, a₁, a₂)", "Two agents act in one environment; cooperative and competitive rewards differ.")
        "Hierarchical Reinforcement Learning" -> RlSpec("Follow manager options and primitive control", listOf("Manager chooses subgoal", "Option executes", "Primitive actions", "Option terminates"), "π(a|s,o), μ(o|s)", "A higher-level option persists across several primitive steps.")
        "Offline Reinforcement Learning" -> RlSpec("Learn only from fixed logged transitions", listOf("Static (s,a,r,s′) log", "Estimate values", "Constrain policy", "Evaluate offline"), "D = {(s,a,r,s′)} fixed", "Training never collects a new environment transition.")
        "Imitation Learning" -> RlSpec("Mimic an expert's actions", listOf("Expert trajectory", "State-action pairs", "Behavior cloning", "Learned trajectory"), "minθ −Σ log πθ(aexpert|s)", "The supervised target is expert action, not an inferred reward.")
        "Inverse Reinforcement Learning" -> RlSpec("Infer a reward from expert behavior", listOf("Expert trajectory", "Infer R(s)", "Optimize policy", "Compare behavior"), "R* = argmaxR P(trajectoryexpert | R)", "Expert actions are evidence about a hidden reward function.")
        else -> error("Unregistered RL topic: $name")
    }
}

@Composable
internal fun StageFourRlScreen(topic: LearnTopic) {
    val name = topic.title
    val spec = StageFourRlSpecs.forTitle(name)
    var step by remember(topic.id) { mutableIntStateOf(0) }
    var action by remember(topic.id) {
        mutableIntStateOf(if (name in setOf("SARSA", "SARSA Learning", "Q-Learning", "Q Learning", "Expected SARSA")) 0 else 1)
    }
    var control by remember(topic.id) { mutableIntStateOf(3) }
    var mdpReward by remember(topic.id) { mutableIntStateOf(3) }
    var mdpDiscount by remember(topic.id) { mutableIntStateOf(4) }
    val explorer=name=="Markov Decision Process Explorer"
    val gamma = if(explorer).55+mdpDiscount*.06 else .75 + control * .04
    val slip = control * .035
    val mdpGoal=if(explorer).5+mdpReward*.2 else 1.0
    val values = StageFourRlEngine.valueIteration(step + 1, gamma, slip,goalReward=mdpGoal)
    val state = (step * 3 + 2) % 14
    val transitions = StageFourRlEngine.transition(state, action, slip)
    val nextQ = listOf(.15 + .02 * step, .34 + .01 * step, .22, .09)
    val probabilities = StageFourRlEngine.softmax(nextQ.map { it * (2 + control) })
    val reward = -.04
    val target = when (name) {
        "SARSA", "SARSA Learning" -> StageFourRlEngine.sarsaTarget(reward, nextQ[action], gamma)
        "Q-Learning", "Q Learning" -> StageFourRlEngine.qTarget(reward, nextQ, gamma)
        "Expected SARSA" -> StageFourRlEngine.expectedSarsaTarget(reward, nextQ, probabilities, gamma)
        else -> StageFourRlEngine.tdTarget(reward, values[StageFourRlEngine.next(state, action)], gamma)
    }
    val isConcept = name in setOf("Agent", "Environment", "State", "Action", "Reward", "Policy", "Value Function", "Markov Decision Process", "Markov Decision Process Explorer")
    val goalReward = if (name == "Reward") .5 + control * .2 else 1.0
    val stepPenalty = if (name == "Reward") -.02 - control * .015 else -.04
    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item { SectionTitle(name, spec.subtitle) }
        item { FourPanel("Mechanism") {
            FourFlow(spec.flow, step % spec.flow.size)
            FourText(spec.distinction)
            FourText(spec.equation, LabCyan)
        } }
        item { FourPanel("Live visualization") {
            when (name) {
                "Multi-Armed Bandit" -> {
                    val arms = (0..3).map { i -> "Arm ${i+1}" to StageFourRlEngine.banditEstimate((1..max(1,step+1)).map { t -> ((t*17+i*13)%11)/10.0 }) }
                    FourBars(arms, step % 4)
                    FourMetric("Chosen arm UCB", "%.3f".format(StageFourRlEngine.banditUcb(arms[step%4].second, step+1, 4*(step+1))))
                    FourMetric("Cumulative regret", "%.3f".format((0..step).sumOf { .75-arms[it%4].second }.coerceAtLeast(0.0)))
                }
                "PPO" -> {
                    val ratios = (0..20).map { .5 + it*.05 }
                    FourLine(ratios.map { StageFourRlEngine.ppoClipped(it,.8,control*.03+.05) }, ratios.map { it*.8 })
                    FourMetric("Current ratio", "%.2f".format(.7+step*.065))
                    FourMetric("Clipped objective", "%.3f".format(StageFourRlEngine.ppoClipped(.7+step*.065,.8,control*.03+.05)))
                }
                "Deep Q-Network", "Double DQN", "Dueling DQN" -> {
                    FourBars(nextQ.indices.map { "a$it" to nextQ[it] }, action)
                    if (name == "Deep Q-Network") FourText("Replay ${(step*7+3)%24}: (s=$state, a=$action, r=$reward, s′=${StageFourRlEngine.next(state,action)}) → sampled mini-batch → target network")
                    if (name == "Double DQN") FourMetric("Online selects / target evaluates", "a=${nextQ.indices.maxBy { nextQ[it] }} / %.3f".format(StageFourRlEngine.doubleDqnTarget(reward,nextQ,nextQ.reversed(),gamma)))
                    if (name == "Dueling DQN") FourBars(StageFourRlEngine.duelingQ(.28,nextQ).mapIndexed { i,q -> "Q$i" to q },action,LabGreen)
                }
                "DDPG", "TD3", "SAC" -> {
                    val continuous = -.8 + step*.13
                    FourLine((0..20).map { i -> 1.0 - (i/10.0 - continuous).let { it*it } })
                    FourMetric("Actor continuous action", "%.2f".format(continuous))
                    if (name == "TD3") FourMetric("Twin critic target", "%.3f".format(StageFourRlEngine.twinTarget(.74,.56,reward,gamma)))
                    if (name == "SAC") FourMetric("Q + entropy bonus", "%.3f".format(StageFourRlEngine.sacObjective(.56,probabilities,.2)))
                }
                "REINFORCE", "Policy Gradient", "Actor-Critic", "A2C", "A3C" -> {
                    FourBars(probabilities.mapIndexed { i,p -> "a$i" to p }, action, LabGreen)
                    FourMetric("Discounted return", "%.3f".format(StageFourRlEngine.mcReturn(listOf(-.04,-.04,.2,1.0).take(min(4,step+1)),gamma)))
                    if (name == "Actor-Critic") FourMetric("Critic advantage", "%.3f".format(StageFourRlEngine.advantage(reward,.54,.31,gamma)))
                    if (name == "A2C") FourText("Workers 1–4: synchronous rollouts ${listOf(step,step,step,step).joinToString()} → one averaged update")
                    if (name == "A3C") FourText("Asynchronous arrivals: worker ${(step%3)+1} updates shared actor while others continue")
                }
                "Monte Carlo Learning" -> {
                    val rewards = listOf(-.04,-.04,-.04,1.0)
                    FourBars(StageFourRlEngine.discounted(rewards,gamma).mapIndexed { i,g -> "G$i" to g }, step.coerceAtMost(3),LabOrange)
                    FourText(if (step < 3) "Episode still running: the estimate has not been updated." else "Terminal reached: returns are backed up from the end.")
                }
                "SARSA", "SARSA Learning", "Q-Learning", "Q Learning", "Expected SARSA" -> {
                    val greedyAction = nextQ.indices.maxBy { nextQ[it] }
                    val targetAction = if (name == "Q-Learning" || name == "Q Learning") greedyAction else action
                    FourBars(nextQ.mapIndexed { i,q -> listOf("↑","→","↓","←")[i] to q },targetAction)
                    FourMetric("Behavior next action",listOf("↑","→","↓","←")[action])
                    FourMetric("Greedy target action",listOf("↑","→","↓","←")[greedyAction],LabOrange)
                    if(name=="Expected SARSA")FourBars(probabilities.mapIndexed{i,p->"π$i" to p},action,LabGreen)
                    FourText("These are Q(s′,a) values for the selected next state; the three targets use them differently.")
                }
                "Offline Reinforcement Learning" -> {
                    FourText("Fixed log: (2,→,−.04,3), (3,↓,−.04,7), (7,→,+1,15)")
                    FourBars(nextQ.mapIndexed { i,q -> "logged $i" to q }, action)
                    FourText("Only logged actions contribute training evidence. No environment step is added.")
                }
                "Imitation Learning", "Inverse Reinforcement Learning" -> {
                    FourGrid((0..3).map { row -> (0..3).map { col -> if (row==col) 1.0 else .1 } })
                    FourText(if (name == "Imitation Learning") "Expert path 0→5→10→15 provides action labels to the learner." else "Expert path 0→5→10→15 provides evidence for a reward heatmap, then a policy is optimized.")
                }
                "Model-Based Reinforcement Learning" -> {
                    FourBars((0..3).map { a -> "plan $a" to StageFourRlEngine.bellman(values,state,a,gamma,slip) },action)
                    FourText("Observed transitions fit a small dynamics table. The bars are one-step imagined returns through that model.")
                }
                "Multi-Agent Reinforcement Learning" -> {
                    val other=(state+7)%16
                    FourGrid((0..3).map { r -> (0..3).map { c -> when(r*4+c){state->1.0;other->.65;15->.9;else->.08} } })
                    FourMetric("Agent A / B cells","$state / $other")
                    FourMetric("Cooperative / competitive return","%.2f / %.2f".format(.5+.03*step,.3-.02*step))
                }
                "Hierarchical Reinforcement Learning" -> {
                    FourFlow(listOf("Manager: corridor","Controller: steps","Manager: doorway","Controller: enter"),step%4)
                    FourBars(listOf("corridor" to .8,"doorway" to .6,"room" to .95),step%3)
                    FourText("The selected option lasts for several primitive actions before the manager selects another subgoal.")
                }
                else -> {
                    FourGrid((if (name in setOf("Agent","Environment","State","Action","Reward"))
                        List(16) { cell -> when (cell) { 15 -> goalReward; 11 -> -1.0; 5 -> -.5; state -> .45; else -> .08 } }
                    else values).chunked(4))
                    if (name == "Policy" || name == "Policy Iteration" || name == "Dynamic Programming") {
                        val policy = StageFourRlEngine.policyIteration(step+1,gamma).second
                        FourText("Policy arrows: " + policy.chunked(4).joinToString("  /  ") { row -> row.joinToString("") { listOf("↑","→","↓","←")[it] } })
                    }
                    if (name == "Environment" || name == "Markov Decision Process" || name == "Markov Decision Process Explorer") {
                        FourText("P(s′|s=$state,a=${listOf("↑","→","↓","←")[action]}) = " + transitions.joinToString { "${it.first}: %.2f".format(it.second) })
                        if(explorer)FourBars((0..3).map{a->listOf("↑","→","↓","←")[a] to StageFourRlEngine.bellman(values,state,a,gamma,slip,goalReward=mdpGoal)},action)
                    }
                }
            }
        } }
        item { FourPanel(if (isConcept) "Inspect this transition" else "Computed update") {
            FourMetric("State → next state", "$state → ${StageFourRlEngine.next(state,action)}")
            FourMetric("Immediate reward", "%.3f".format(StageFourRlEngine.reward(StageFourRlEngine.next(state,action),stepPenalty,goalReward)))
            if (!isConcept) FourMetric("Target / backup", "%.3f".format(target), LabGreen)
            when (name) {
                "SARSA", "SARSA Learning", "Q-Learning", "Q Learning", "Expected SARSA" -> FourText("Next Q: ${nextQ.joinToString { "%.2f".format(it) }}; policy π: ${probabilities.joinToString { "%.2f".format(it) }}")
                "Temporal Difference Learning" -> FourMetric("TD error", "%.3f".format(target-values[state]))
                "Reward" -> FourMetric("Four-step discounted return", "%.3f".format(StageFourRlEngine.mcReturn(listOf(stepPenalty,stepPenalty,stepPenalty,goalReward),gamma)))
                "State" -> FourText("Encoded state: [row=${state/4}, column=${state%4}, goal distance=${6-state/4-state%4}, wall nearby=${if(state==4||state==6) 1 else 0}]")
                "Action" -> FourText("An invalid move or obstacle returns the same cell; it is still a transition.")
                "TD3" -> FourText("Actor updates every second critic step; target action receives clipped smoothing noise.")
                else -> Unit
            }
        } }
        item { FourPanel("Explore") {
            FourSteps(step, 12) { step = it }
            FourText(when (name) {
                "Multi-Armed Bandit" -> "Exploration ε = %.2f".format(control/10.0)
                "PPO" -> "Clip ε = %.2f".format(control*.03+.05)
                "Environment", "Markov Decision Process", "Markov Decision Process Explorer" -> "Slip probability = %.2f".format(slip)
                "Reward" -> "Goal reward %.2f; step cost %.2f".format(goalReward,stepPenalty)
                else -> "Discount γ = %.2f".format(gamma)
            })
            Slider(control.toFloat(), { control = it.toInt().coerceIn(0,6) }, valueRange = 0f..6f)
            if(explorer){
                FourText("Goal reward %.2f".format(mdpGoal))
                Slider(mdpReward.toFloat(),{mdpReward=it.toInt().coerceIn(0,6)},valueRange=0f..6f)
                FourText("Discount γ = %.2f".format(gamma))
                Slider(mdpDiscount.toFloat(),{mdpDiscount=it.toInt().coerceIn(0,6)},valueRange=0f..6f)
            }
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                listOf("↑","→","↓","←").forEachIndexed { i,label ->
                    SegmentedOption(label, action==i, Modifier.weight(1f)) { action=i }
                }
            }
        } }
    }
}
