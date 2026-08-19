package com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.data

import com.indianservers.ai_ml_dl_algorithms.ml_lab.domain.LearningDepth
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnCatalog
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearningProfile
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.LearnTopic
import com.indianservers.ai_ml_dl_algorithms.ml_lab.learn.VisualizationKind

object AlgorithmLessonSeedFactory {
    fun buildAll(depth: LearningDepth = LearningDepth.Beginner): List<SeedLesson> {
        val now = System.currentTimeMillis()
        return LearnCatalog.topics.map { topic -> build(topic, depth, now) }
    }

    fun build(topic: LearnTopic, depth: LearningDepth, timestamp: Long = System.currentTimeMillis()): SeedLesson {
        val profile = LearnCatalog.profile(topic, depth)
        val title = topic.title
        val mainApplication = profile.applications.firstOrNull().orEmpty()
        val firstMistake = profile.mistakes.firstOrNull().orEmpty()
        val firstHyperparameter = profile.hyperparameters.firstOrNull().orEmpty()
        val flagshipNarrative = if (LearnCatalog.isFlagship(topic)) flagshipNarrative(topic) else null
        val teachingNarrative = teachingNarrative(topic, profile)
        val awardNote = if (LearnCatalog.isFlagship(topic)) {
            "$title is one of our flagship algorithms because it teaches a core idea that appears again and again in real ML systems."
        } else {
            "$title belongs to the ${topic.domain} toolkit and becomes powerful when students connect the idea to real data."
        }

        val pages = flagshipNarrative?.let { premiumPages(topic, profile, it) }
            ?: qualityPages(topic, profile, teachingNarrative)

        return SeedLesson(
            lesson = AlgorithmLessonRecord(
                algorithmId = topic.id,
                algorithmTitle = title,
                domain = topic.domain,
                section = topic.section,
                isAwardWinning = LearnCatalog.isFlagship(topic),
                expertNote = awardNote,
                createdAt = timestamp,
                updatedAt = timestamp
            ),
            pages = pages,
            questions = questions(topic, profile, application = mainApplication, hyperparameter = firstHyperparameter, mistake = firstMistake)
        )
    }

    private fun page(
        topic: LearnTopic,
        number: Int,
        title: String,
        story: String,
        explanation: String,
        realtimeExample: String,
        realtimeApplications: String,
        teacherTip: String
    ): LessonPageRecord {
        val html = """
            <article class="lesson-page algorithm-${topic.id.escapeHtml()}">
              <header>
                <p class="eyebrow">${topic.domain.escapeHtml()} / ${topic.section.escapeHtml()}</p>
                <h1>${title.escapeHtml()}</h1>
              </header>
              <section class="story">
                <h2>Story</h2>
                <p>${story.escapeHtml()}</p>
              </section>
              <section class="simple">
                <h2>Simple Explanation</h2>
                <p>${explanation.escapeHtml()}</p>
              </section>
              <section class="realtime">
                <h2>Real-World Example</h2>
                <p>${realtimeExample.escapeHtml()}</p>
              </section>
              <section class="applications">
                <h2>Applications</h2>
                <p>${realtimeApplications.escapeHtml()}</p>
              </section>
              <aside class="teacher-tip">
                <strong>ML Expert Teacher Tip:</strong> ${teacherTip.escapeHtml()}
              </aside>
            </article>
        """.trimIndent()
        return LessonPageRecord(
            algorithmId = topic.id,
            pageNumber = number,
            title = title,
            htmlContent = html,
            story = story,
            explanation = explanation,
            realtimeExample = realtimeExample,
            realtimeApplications = realtimeApplications,
            teacherTip = teacherTip
        )
    }

    private fun qualityPages(topic: LearnTopic, profile: LearningProfile, narrative: TeachingNarrative): List<LessonPageRecord> {
        val title = topic.title
        val steps = profile.steps.mapIndexed { index, step -> "${index + 1}. $step" }.joinToString(" ")
        val applications = profile.applications.joinToSentence()
        val limitations = profile.limitations.joinToSentence()
        val mistakes = profile.mistakes.joinToSentence()
        return listOf(
            page(
                topic = topic,
                number = 1,
                title = "$title: ${narrative.chapterTitle}",
                story = "${narrative.heroName} faces a real problem: ${narrative.problem}. $title enters the story because ${narrative.whyItFits}.",
                explanation = "${profile.definition} In kid-simple words, ${narrative.simpleExplanation}",
                realtimeExample = "$title worked example: ${narrative.realtimeExample}",
                realtimeApplications = "$title applications: $applications",
                teacherTip = "Start by naming the evidence, the decision, and the cost of being wrong. That makes $title feel practical instead of abstract."
            ),
            page(
                topic = topic,
                number = 2,
                title = "$title as a Simple Picture",
                story = narrative.analogy,
                explanation = "${profile.intuition} ${narrative.mentalModel}",
                realtimeExample = "$title classroom demo: use ${narrative.classroomDemo} to see the same idea without heavy math.",
                realtimeApplications = "$title review uses: real teams use this style of thinking for $applications.",
                teacherTip = "Ask the learner to draw the idea first. A good sketch often explains the algorithm before code does."
            ),
            page(
                topic = topic,
                number = 3,
                title = "How $title Works Step by Step",
                story = "${narrative.heroName} slows the problem down and follows a repeatable recipe instead of guessing.",
                explanation = steps,
                realtimeExample = "$title production workflow: ${narrative.productionWorkflow}",
                realtimeApplications = "$title mechanics in practice: ${narrative.mechanics}",
                teacherTip = "Every step should change either the data view, the model state, or the confidence. If nothing changes, check the pipeline."
            ),
            page(
                topic = topic,
                number = 4,
                title = "The Tiny Math Behind $title",
                story = "The math is a scoreboard for ${narrative.heroName}. It does not replace thinking; it keeps the thinking honest.",
                explanation = "${profile.equation}. ${profile.equationNote}",
                realtimeExample = "$title math example: ${narrative.mathMeaning}",
                realtimeApplications = "$title implementation setting: ${specificParameter(topic, profile)}.",
                teacherTip = "Read the equation like a sentence: what is measured, what is compared, and what should become smaller or stronger?"
            ),
            page(
                topic = topic,
                number = 5,
                title = "$title in Real Applications",
                story = "${narrative.heroName} finally ships the idea carefully: first on practice data, then validation data, then a small real trial.",
                explanation = "Superpower: ${profile.advantages.firstOrNull().orEmpty().ifBlank { narrative.superpower }}. Watch-outs: $limitations. Common mistakes: $mistakes.",
                realtimeExample = "$title failure case: ${narrative.failureCase}",
                realtimeApplications = "$title applied in: $applications. ${narrative.careerConnection}",
                teacherTip = "Expert ML is humble ML: compare against a baseline, test on fresh examples, explain limits, and improve only when evidence says so."
            )
        )
    }

    private data class TeachingNarrative(
        val heroName: String,
        val chapterTitle: String,
        val problem: String,
        val whyItFits: String,
        val simpleExplanation: String,
        val analogy: String,
        val mentalModel: String,
        val classroomDemo: String,
        val realtimeExample: String,
        val productionWorkflow: String,
        val mechanics: String,
        val mathMeaning: String,
        val failureCase: String,
        val superpower: String,
        val careerConnection: String
    )

    private fun teachingNarrative(topic: LearnTopic, profile: LearningProfile): TeachingNarrative = when (profile.kind) {
        VisualizationKind.Regression -> TeachingNarrative(
            "Asha",
            "The Number Predictor",
            "a neighborhood shop wants to estimate tomorrow's demand before ordering supplies",
            "it learns how clues connect to a numeric answer",
            "${topic.title} predicts a number, not a label. It studies past examples and learns how inputs push the output up or down.",
            "Picture a smooth measuring tape stretched across scattered dots. The tape is useful only if it stays close to most dots.",
            "The model is trying to make prediction errors smaller while staying simple enough to trust.",
            "prices, temperatures, house sizes, or study hours drawn as input-output pairs",
            "A planning system forecasts sales, traffic, energy use, or delivery time from historical signals.",
            "A team prepares numeric features, trains the model, checks residuals, and tests whether errors stay reasonable on new days.",
            "The important mechanics are feature quality, target definition, residual error, and validation on unseen examples.",
            "The equation is a promise: convert inputs into a predicted number, then measure how far that number is from reality.",
            "If a festival, storm, or holiday changes behavior, the model can look confident while being wrong.",
            "clear numeric prediction",
            "Jobs that use this thinking include demand planning, pricing, forecasting, and operations analytics."
        )
        VisualizationKind.Classification -> TeachingNarrative(
            "Kabir",
            "The Decision Gate",
            "a safety monitor must decide whether each case needs attention now or can wait",
            "it turns evidence into a class decision",
            "${topic.title} chooses a category. It studies labelled examples and learns the clues that separate one class from another.",
            "Imagine a gate with colored lanes. Each new example is guided into the lane that best matches its evidence.",
            "The model is not just saying yes or no; it is organizing evidence into a decision rule.",
            "cards labelled safe/risky, healthy/sick, spam/not spam, or pass/review",
            "A realtime alert system classifies transactions, messages, defects, or support tickets for faster review.",
            "A team balances precision, recall, fairness, and calibration before trusting the classifier.",
            "The mechanics are labels, features, decision boundary, threshold, and confusion-matrix tradeoffs.",
            "The formula creates a score or probability; the threshold turns that score into a final class.",
            "A model can score high overall while missing the rare class that matters most.",
            "fast decision support",
            "This powers fraud review, medical triage, moderation, quality inspection, and customer routing."
        )
        VisualizationKind.Neighbours -> TeachingNarrative(
            "Meera",
            "The Similarity Search",
            "a new item appears and nobody knows its answer yet",
            "it asks nearby known examples for advice",
            "${topic.title} predicts from similarity. The closest stored examples vote, average, or guide the answer.",
            "Think of standing in a library aisle: nearby books usually share a topic, but only if the shelves are organized well.",
            "Distance is the heart of the method. Bad distance means bad neighbors.",
            "points on a map where students draw circles around the nearest examples",
            "A recommendation or search system uses similarity to find useful neighbors for a new user, product, image, or case.",
            "A team scales features, chooses a distance rule, tests different neighbor counts, and watches speed on larger data.",
            "The mechanics are stored examples, distance metric, K, local voting, and feature scaling.",
            "The equation defines the neighborhood N_K(x), then combines the labels or values inside it.",
            "One irrelevant feature with large numbers can dominate distance and pick the wrong neighbors.",
            "intuitive local reasoning",
            "Similarity thinking appears in search, recommendations, deduplication, matching, and case-based decision support."
        )
        VisualizationKind.Tree -> TeachingNarrative(
            "Ishaan",
            "The Question Path",
            "a team needs a decision that people can inspect step by step",
            "it builds answers from clear branching questions",
            "${topic.title} asks useful questions about features and uses the answers to reach a prediction.",
            "It feels like a choose-your-path story where every branch makes the group cleaner.",
            "Each split should reduce confusion. Too many splits can memorize noise.",
            "paper cards sorted by yes/no questions like age, color, score, or size",
            "A business rule assistant explains why a case was approved, flagged, grouped, or predicted.",
            "A team controls depth, checks leaf sizes, compares validation results, and watches for leakage.",
            "The mechanics are split candidates, impurity, branches, leaves, ensembles when used, and overfit control.",
            "The equation scores how much a split improves the child groups compared with the parent group.",
            "A deep tree can explain training data beautifully and still fail on new messy cases.",
            "readable nonlinear decision logic",
            "Tree thinking is common in risk tools, diagnostics, operations rules, and tabular ML baselines."
        )
        VisualizationKind.Clustering, VisualizationKind.Density -> TeachingNarrative(
            "Nila",
            "The Hidden Groups",
            "a dataset arrives with no labels, but the team suspects useful groups are hiding inside",
            "it searches for structure without being told the answer",
            "${topic.title} discovers groups or dense regions. It helps people explore data before they know the labels.",
            "Imagine arranging mixed buttons on a table until similar buttons naturally sit together.",
            "The model is looking for shape, distance, density, or representative centers in the data.",
            "colored stickers placed on graph paper with students guessing natural groups",
            "A product team groups customers, documents, images, or events to understand behavior patterns.",
            "A team scales features, tries settings, checks group stability, and asks domain experts whether clusters mean anything.",
            "The mechanics are feature scale, distance or density, assignments, noise points, and cluster validation.",
            "The equation measures compactness, density reachability, or probability of belonging to a hidden group.",
            "The algorithm may create groups even when the real world has blurry or no meaningful groups.",
            "finding structure without labels",
            "Clustering supports customer segmentation, anomaly discovery, exploratory analytics, and data labeling strategy."
        )
        VisualizationKind.Projection -> TeachingNarrative(
            "Riya",
            "The Smart Map",
            "a dataset has too many columns for anyone to see clearly",
            "it compresses many clues into a smaller view",
            "${topic.title} reduces dimensions so important structure can be seen or used more easily.",
            "It is like making a map: you lose tiny details, but keep the shapes that help you travel.",
            "A good projection keeps useful neighbors, variation, or relationships while dropping less useful detail.",
            "students flattening a 3D object shadow onto paper from different angles",
            "An analytics team compresses images, text embeddings, or survey data before visualization or modeling.",
            "A team scales data, chooses a projection method, checks information loss, and avoids reading too much into pretty plots.",
            "The mechanics are compression, reconstruction, variance, neighborhoods, and information loss.",
            "The equation defines what structure should be preserved when high-dimensional data becomes smaller.",
            "A beautiful 2D picture can exaggerate separation or hide important uncertainty.",
            "making complex data visible",
            "Projection methods help dashboards, embedding inspection, preprocessing, visualization, and model debugging."
        )
        VisualizationKind.Convolution -> TeachingNarrative(
            "Tara",
            "The Visual Scanner",
            "a camera needs to notice small visual patterns before recognizing the whole object",
            "it scans local patches and reuses detectors across space",
            "${topic.title} learns visual features by looking at nearby pixels or patches in a structured way.",
            "Think of sliding a tiny stencil across a picture to find edges, corners, textures, or shapes.",
            "Local pattern detectors build feature maps; deeper layers combine simple clues into richer visual evidence.",
            "small image grids where students slide a 3x3 window and mark strong matches",
            "A vision system checks product defects, medical images, road scenes, or image categories.",
            "A team normalizes images, checks augmentations, validates on fresh image sources, and inspects failure examples.",
            "The mechanics are kernels, stride, padding, feature maps, pooling, and learned visual hierarchy.",
            "The equation shows one reusable filter scanning across positions and producing activations.",
            "A camera model can fail when lighting, angle, device, or background differs from training images.",
            "learning local visual patterns",
            "Computer vision roles use this for inspection, medical imaging, robotics, search, and accessibility tools."
        )
        VisualizationKind.Sequence, VisualizationKind.TimeSeries -> TeachingNarrative(
            "Arjun",
            "The Time Clue Keeper",
            "the answer depends on what happened before, not just what is happening now",
            "it models ordered information across time or tokens",
            "${topic.title} studies sequences. Earlier clues can change the meaning of later clues.",
            "It is like reading a story: the current sentence makes more sense when you remember the previous sentences.",
            "The model keeps, updates, or attends to context so the next prediction uses history.",
            "daily temperatures, word cards, music notes, or sensor readings arranged in order",
            "A monitoring system forecasts demand, detects machine trouble, or understands language from ordered signals.",
            "A team builds time windows, prevents future leakage, validates by time order, and checks drift.",
            "The mechanics are sequence order, hidden state or context, window length, leakage control, and forecast error.",
            "The equation updates memory or attention from one step to the next, then predicts from that context.",
            "If future values leak into training, the model looks brilliant in practice and weak in real deployment.",
            "using history to predict or understand",
            "Sequence thinking powers forecasting, language tools, speech, sensors, finance, and operations monitoring."
        )
        VisualizationKind.Attention -> TeachingNarrative(
            "Zoya",
            "The Focus Lens",
            "a model must decide which parts of a sentence, image, or document matter most right now",
            "it lets each item look at other items and mix useful information",
            "${topic.title} uses attention to compare pieces of context and decide what information to focus on.",
            "Imagine each word holding a tiny flashlight and shining it on the words that help explain its meaning.",
            "Attention creates weights between items; stronger weights contribute more to the mixed representation.",
            "word cards where students draw arrows to the words that explain each other",
            "A document system answers questions, summarizes text, searches passages, or connects image patches.",
            "A team checks masking, context length, tokenization, evaluation data, and whether outputs are grounded.",
            "The mechanics are queries, keys, values, weights, heads, context windows, and masks.",
            "The equation turns query-key similarity into weights, then uses those weights to mix values.",
            "Attention weights can be useful clues, but they are not guaranteed to be perfect explanations.",
            "global context modeling",
            "Attention is central to modern language, document, coding, retrieval, and vision systems."
        )
        VisualizationKind.NeuralNetwork, VisualizationKind.Autoencoder, VisualizationKind.Generative, VisualizationKind.Graph -> TeachingNarrative(
            "Ravi",
            "The Representation Builder",
            "raw data is too messy, so the system must build better internal clues",
            "it learns layered or structured representations from examples",
            "${topic.title} learns intermediate representations. Those hidden representations make hard patterns easier to use.",
            "It is like a workshop where each station improves the raw material before the final decision is made.",
            "Layers, bottlenecks, graph messages, or generators transform data into more useful forms.",
            "students passing cards through stations that each add, remove, or combine clues",
            "A model learns embeddings, reconstructions, generated samples, graph signals, or nonlinear predictions.",
            "A team defines the objective, watches training curves, validates outputs, and inspects failure cases.",
            "The mechanics are representation, loss, capacity, regularization, optimization, and validation.",
            "The equation describes how hidden states or samples are transformed and judged by an objective.",
            "A powerful representation can memorize training examples or create plausible but wrong outputs.",
            "learning useful hidden structure",
            "Representation learning supports deep learning, anomaly detection, generation, graph analytics, and embeddings."
        )
        VisualizationKind.Reinforcement -> TeachingNarrative(
            "Maya",
            "The Reward Explorer",
            "an agent must learn what to do by trying actions and seeing rewards",
            "it improves decisions through interaction, feedback, and long-term reward",
            "${topic.title} studies action. The learner tries, observes rewards, and improves its policy over time.",
            "It is like learning a maze: one step may look good now, but the best route considers what happens later.",
            "The algorithm balances exploration of uncertain actions with exploitation of actions that already seem useful.",
            "a grid game where students move a token, collect rewards, and avoid penalties",
            "A simulation system learns routing, control, resource allocation, or game strategy from repeated trials.",
            "A team defines rewards carefully, tests in simulation, watches unsafe behavior, and evaluates policies separately.",
            "The mechanics are state, action, reward, policy, value, discounting, and exploration.",
            "The equation updates beliefs about action value using reward plus expected future value.",
            "A badly designed reward can teach the agent to win the score while failing the real goal.",
            "learning decisions from feedback",
            "RL thinking appears in robotics, operations research, games, recommender systems, and control."
        )
        VisualizationKind.Probability -> TeachingNarrative(
            "Leela",
            "The Uncertainty Detective",
            "the team has clues but must admit what is still uncertain",
            "it represents beliefs and updates them with evidence",
            "${topic.title} reasons with uncertainty. It keeps track of how likely different explanations are.",
            "It is like adjusting a detective's notebook whenever new evidence arrives.",
            "The model combines prior beliefs, evidence, likelihood, and uncertainty instead of pretending everything is certain.",
            "mystery cards where each new clue changes which suspect is most likely",
            "A risk system updates probabilities for diagnosis, forecasting, reliability, or hidden states.",
            "A team checks assumptions, calibrates probabilities, compares likelihoods, and communicates uncertainty clearly.",
            "The mechanics are prior, likelihood, posterior, sampling or inference, and calibration.",
            "The equation shows how evidence updates belief while keeping probabilities consistent.",
            "Wrong assumptions can make probability numbers look precise while the story behind them is weak.",
            "honest uncertainty modeling",
            "Probabilistic thinking helps medicine, reliability, forecasting, experimentation, and decision analysis."
        )
        VisualizationKind.Optimization -> TeachingNarrative(
            "Omar",
            "The Better-Answer Climber",
            "a model has many adjustable knobs and needs a careful way to improve them",
            "it moves parameters toward a better objective",
            "${topic.title} is about improvement. It changes parameters to reduce loss or increase a useful score.",
            "Imagine walking downhill in fog: each step uses the local slope, and step size matters.",
            "Optimization is the engine that turns feedback into parameter updates.",
            "a simple hill drawing where students choose step sizes and watch overshoot or slow progress",
            "A training pipeline tunes model weights, schedules, or parameters until validation metrics improve.",
            "A team monitors loss curves, step size, stability, stopping rules, and overfitting.",
            "The mechanics are objective, gradient or search move, learning rate, momentum, and convergence.",
            "The equation says how to move from current parameters to hopefully better parameters.",
            "A step that is too large can jump past good solutions; a step too small can waste time.",
            "systematic improvement",
            "Optimization appears in every serious ML training system, from linear models to deep networks."
        )
        VisualizationKind.Recommendation -> TeachingNarrative(
            "Sana",
            "The Helpful Ranker",
            "a user has too many choices and needs the most useful items first",
            "it predicts relevance and ranks options",
            "${topic.title} helps choose what to show next. It learns from users, items, content, and feedback.",
            "It is like a librarian who learns your taste but must still show useful variety.",
            "A recommender estimates fit between a person, context, and item, then creates a ranked list.",
            "students rating books or songs, then finding similar users or items",
            "A media, shopping, or learning app ranks products, videos, lessons, or articles.",
            "A team checks relevance, diversity, cold start, feedback loops, and fairness before deployment.",
            "The mechanics are user signals, item features, similarity, ranking score, and evaluation metrics.",
            "The equation scores user-item fit or learns shared hidden factors.",
            "A recommender can trap users in narrow loops if it only repeats old behavior.",
            "personalized ranking",
            "Recommendation work appears in commerce, media, education, search, and personalization systems."
        )
        VisualizationKind.Explanation -> TeachingNarrative(
            "Naveen",
            "The Why Finder",
            "people need to understand why a model gave an answer",
            "it traces predictions back to influential evidence",
            "${topic.title} helps explain model behavior. It does not just ask what the answer is; it asks why.",
            "It is like highlighting the clues in a homework solution so another student can follow the reasoning.",
            "Explanation methods compare, perturb, trace, or visualize evidence that influenced a prediction.",
            "a model answer where students cover one clue at a time and watch the decision change",
            "A review tool explains risk scores, image decisions, text predictions, or model behavior to experts.",
            "A team compares explanations with domain knowledge and checks whether users interpret them correctly.",
            "The mechanics are feature influence, perturbation, attribution, local explanation, and trust calibration.",
            "The equation measures how changing or tracing evidence changes the prediction.",
            "An explanation can look convincing even when it is incomplete or misunderstood.",
            "making model behavior inspectable",
            "Explainability matters in healthcare, finance, safety, debugging, governance, and education."
        )
        VisualizationKind.Generic -> TeachingNarrative(
            "Diya",
            "The Pattern Workshop",
            "a team has data and needs a careful way to turn it into useful evidence",
            "it gives the team a repeatable method for learning from examples",
            "${topic.title} transforms observations into a signal, estimate, representation, or decision.",
            "It is like sorting a messy project table into labeled trays so each clue has a job.",
            "The method works best when the input, objective, validation, and limitations are all clear.",
            "small datasets where students define inputs, outputs, and a success metric",
            "A practical ML workflow uses ${topic.title} as one candidate method, then compares it with baselines.",
            "A team prepares data, fits the method, checks intermediate outputs, validates results, and documents limits.",
            "The mechanics are inputs, objective, preprocessing, model behavior, and evaluation.",
            "The equation defines what the method considers a good solution.",
            "A method can be mathematically correct but still solve the wrong real-world problem.",
            "repeatable evidence-building",
            "This kind of thinking supports experimentation, analytics, automation, and responsible ML delivery."
        )
    }

    private fun premiumPages(topic: LearnTopic, profile: LearningProfile, narrative: FlagshipNarrative): List<LessonPageRecord> {
        val title = topic.title
        return listOf(
            page(
                topic = topic,
                number = 1,
                title = "$title: ${narrative.chapterTitle}",
                story = narrative.story,
                explanation = "$title is the main tool in this story because ${narrative.whyItFits}. ${profile.definition}",
                realtimeExample = narrative.realtimeExample,
                realtimeApplications = narrative.applications,
                teacherTip = "Before touching code, ask: what is the input, what is the output, and what mistake would hurt a real person?"
            ),
            page(
                topic = topic,
                number = 2,
                title = "$title in Kid-Simple English",
                story = narrative.simpleAnalogy,
                explanation = narrative.simpleExplanation,
                realtimeExample = narrative.realtimeExample,
                realtimeApplications = "You see this idea in ${narrative.applications}.",
                teacherTip = "Teach it to a younger friend in one sentence. If they can repeat it, you understand the heart of $title."
            ),
            page(
                topic = topic,
                number = 3,
                title = "How $title Thinks",
                story = "Now ${narrative.heroName} slows down and watches the algorithm one move at a time.",
                explanation = narrative.steps,
                realtimeExample = "A production ML team would log each step, compare it against validation data, and check whether the model still behaves well on fresh examples.",
                realtimeApplications = narrative.mechanics,
                teacherTip = "Follow the data like a detective follows footprints. Each step should explain the next step."
            ),
            page(
                topic = topic,
                number = 4,
                title = "The Tiny Math of $title",
                story = "The math is the scoreboard. It tells ${narrative.heroName} whether the algorithm is getting warmer or colder.",
                explanation = "${profile.equation}. ${profile.equationNote}",
                realtimeExample = narrative.mathMeaning,
                realtimeApplications = "Important setting: ${narrative.keySetting}.",
                teacherTip = "Do not fear the equation. Point to each part and say what real thing it measures."
            ),
            page(
                topic = topic,
                number = 5,
                title = "$title in the Real World",
                story = narrative.finalStory,
                explanation = "Superpower: ${narrative.superpower}. Careful: ${narrative.caution}.",
                realtimeExample = narrative.realtimeExample,
                realtimeApplications = narrative.applications,
                teacherTip = "Award-winning ML thinking is honest thinking: test on new data, explain limits, and improve carefully."
            )
        )
    }

    private data class FlagshipNarrative(
        val heroName: String,
        val chapterTitle: String,
        val story: String,
        val whyItFits: String,
        val simpleAnalogy: String,
        val simpleExplanation: String,
        val steps: String,
        val realtimeExample: String,
        val applications: String,
        val mechanics: String,
        val mathMeaning: String,
        val keySetting: String,
        val finalStory: String,
        val superpower: String,
        val caution: String
    )

    private fun flagshipNarrative(topic: LearnTopic): FlagshipNarrative? = when (topic.title) {
        "Simple Linear Regression" -> FlagshipNarrative(
            "Anaya",
            "The Lemonade Line",
            "Anaya runs a tiny lemonade stall. Every hot afternoon she writes down the temperature and how many cups she sells. Soon she wants tomorrow's answer before tomorrow arrives.",
            "it learns a straight relationship between one clue and one number to predict",
            "It is like drawing the fairest straight road through scattered dots on graph paper.",
            "Simple Linear Regression predicts a number by learning a line. If temperature goes up, cup sales may go up too. The line turns that pattern into a prediction.",
            "1. Collect pairs like temperature and cups sold. 2. Draw a starting line. 3. Measure how far the line is from each real sale. 4. Move the line to reduce total error. 5. Use the final line for a new temperature.",
            "A store estimates demand from weather so it can prepare enough stock without wasting food.",
            "price estimation, demand forecasting, school science trend lines, energy usage prediction",
            "Slope tells how strongly the input changes the output; intercept is the starting value when the input is zero.",
            "The equation turns every x into y_hat, and the loss measures how far y_hat is from the real y.",
            "learning rate, slope, intercept, and residual error",
            "Anaya learns that the line is helpful, but it cannot predict a festival day unless festival data was part of the clues.",
            "clear, explainable numeric prediction",
            "a straight line can miss curves, outliers, and hidden causes"
        )
        "Logistic Regression" -> FlagshipNarrative(
            "Kabir",
            "The Email Gatekeeper",
            "Kabir builds a school email helper. It must decide whether a message is safe or spam before anyone clicks a risky link.",
            "it converts clues into a probability for a yes-or-no decision",
            "It is like a confidence meter that slides from almost no to almost yes.",
            "Logistic Regression predicts a class by calculating a score, squeezing it into a probability, and comparing it with a threshold.",
            "1. Turn message clues into numbers. 2. Add weighted clues into a score. 3. Convert the score into probability. 4. Compare probability with a threshold. 5. Check false alarms and missed spam.",
            "A security system estimates whether an email is spam, fraud, or safe.",
            "spam detection, churn prediction, medical risk screening, fraud alerts",
            "Weights show which clues push the answer toward yes or no; the threshold controls the final decision.",
            "The sigmoid turns a raw score into a probability between 0 and 1.",
            "decision threshold, regularization, class weights",
            "Kabir learns that 0.50 is not always the best threshold. A hospital alert may prefer catching more risky cases, even with extra checks.",
            "interpretable probability-based classification",
            "bad thresholds and imbalanced data can create unfair or risky decisions"
        )
        "K-Nearest Neighbors" -> FlagshipNarrative(
            "Meera",
            "The Nearest Friends Vote",
            "Meera finds a mystery fruit in the lunch basket. Instead of guessing, she compares it with fruits she already knows.",
            "it predicts by asking the most similar saved examples to vote",
            "It is like asking the nearest classmates who have seen similar examples before.",
            "K-Nearest Neighbors stores examples. For a new item, it finds the K closest examples and lets them vote for the answer.",
            "1. Scale features fairly. 2. Store labelled examples. 3. Choose K. 4. Measure distance from the new item. 5. Vote using the closest examples.",
            "A shopping app recommends products by finding users or items that look similar.",
            "small classification tools, similarity search, recommendation, handwriting recognition demos",
            "Distance decides who counts as a neighbor; K decides how many neighbors get a vote.",
            "The prediction comes from N_K(x), the neighborhood around the new point.",
            "K, distance metric, feature scaling",
            "Meera learns that one neighbor can be noisy, but too many neighbors can ignore local detail.",
            "simple local reasoning with no heavy training",
            "unscaled features and irrelevant columns can ruin distance"
        )
        "Decision Tree" -> FlagshipNarrative(
            "Ishaan",
            "The Question Tree",
            "Ishaan helps the school library sort books for younger readers. He asks one clear question at a time until every book lands on the right shelf.",
            "it makes predictions through readable yes-or-no questions",
            "It is like a choose-your-own-adventure path where each answer sends you to the next question.",
            "A Decision Tree splits data into smaller groups. Each split tries to make the groups cleaner and easier to predict.",
            "1. Look at all possible questions. 2. Pick the split that reduces label mixing most. 3. Repeat on each branch. 4. Stop before the tree memorizes noise. 5. Predict from the final leaf.",
            "A bank explains a simple approval rule by showing the path through a decision tree.",
            "rule explanations, risk screening, tabular classification, triage tools",
            "Impurity measures how mixed a node is; depth controls how many questions the tree may ask.",
            "The best split is the one that improves purity the most.",
            "max depth, min samples per leaf, impurity criterion",
            "Ishaan learns that a tree with too many questions may memorize one messy day instead of learning library rules.",
            "human-readable nonlinear decisions",
            "deep trees overfit unless they are controlled"
        )
        "Random Forest" -> FlagshipNarrative(
            "Sara",
            "The Council of Trees",
            "Sara does not trust one noisy judge. She asks many different decision trees and lets the forest vote.",
            "it reduces single-tree mistakes by combining many trees",
            "It is like asking a classroom of careful students instead of one student who may be tired.",
            "Random Forest trains many decision trees on slightly different data and feature choices. Their votes make the final answer more stable.",
            "1. Create bootstrap samples. 2. Train many trees. 3. Randomize features at splits. 4. Let every tree vote. 5. Average or majority-vote the result.",
            "A fraud team combines many tree opinions to flag suspicious transactions more reliably.",
            "fraud detection, credit scoring, churn prediction, robust tabular baselines",
            "Bootstrapping and feature randomness make trees disagree in useful ways.",
            "The final prediction is an average or vote across many trees.",
            "number of trees, max depth, max features",
            "Sara learns the forest is powerful, but still needs clean data and honest validation.",
            "strong, reliable tabular predictions",
            "less transparent than one tree and still vulnerable to leakage"
        )
        "Support Vector Machine" -> FlagshipNarrative(
            "Dev",
            "The Widest Safety Lane",
            "Dev paints a line between two playground teams. He wants the widest safe lane, not a line that barely misses the players.",
            "it finds a boundary with the largest margin around hard examples",
            "It is like drawing a road between two groups and keeping the road as wide as possible.",
            "Support Vector Machine finds a separating boundary. The closest points, called support vectors, decide where the boundary sits.",
            "1. Scale features. 2. Choose linear or kernel view. 3. Find the widest margin. 4. Allow controlled mistakes using C. 5. Validate margin behavior.",
            "A quality-control system separates acceptable and defective parts from measured features.",
            "classification with clear margins, quality control, bioinformatics, text classification",
            "Support vectors are the important edge cases; C controls how strict the boundary is.",
            "The objective rewards a wide margin and penalizes violations.",
            "C, kernel, gamma, feature scaling",
            "Dev learns that a perfect-looking boundary on training data can fail if the playground changes.",
            "strong margin-based classification",
            "kernel choices and scaling matter a lot"
        )
        "K-Means" -> FlagshipNarrative(
            "Nila",
            "The Sticker Club Finder",
            "Nila has a pile of stickers and no labels. She wants to discover natural clubs: animals, planets, cartoons, and sports.",
            "it discovers groups by moving centers toward nearby points",
            "It is like placing club leaders in a room and asking every sticker to join the nearest leader.",
            "K-Means groups data into K clusters. It assigns points to the nearest center, then moves each center to the middle of its assigned points.",
            "1. Choose K centers. 2. Assign each point to the nearest center. 3. Move centers to group averages. 4. Repeat until movement is small. 5. Inspect whether the groups make sense.",
            "A retail team groups customers by buying behavior to design better offers.",
            "customer segmentation, image color compression, document grouping, exploratory data analysis",
            "K decides the number of groups; initialization decides where the centers start.",
            "The objective reduces the distance from points to their assigned centers.",
            "K, initialization, max iterations",
            "Nila learns that K-Means will always make K groups, even if the real world does not have exactly K clubs.",
            "fast unlabeled grouping",
            "wrong K or bad scaling can create artificial clusters"
        )
        "Multi-Layer Perceptron" -> FlagshipNarrative(
            "Ravi",
            "The Tiny Brain Workshop",
            "Ravi builds a small number brain that learns XOR, a puzzle where simple straight lines are not enough.",
            "it stacks layers so simple neurons can build complex patterns together",
            "It is like a team where early students notice tiny clues and later students combine them into a smart answer.",
            "A Multi-Layer Perceptron sends numbers through layers of neurons. Each neuron mixes inputs, applies an activation, and passes a signal forward.",
            "1. Encode inputs. 2. Run a forward pass. 3. Compute loss. 4. Backpropagate gradients. 5. Update weights and test again.",
            "A tabular ML system predicts risk from many interacting features that a straight line cannot capture.",
            "nonlinear tabular prediction, pattern recognition, neural-network foundations, function approximation",
            "Hidden layers create intermediate representations; backpropagation tells each weight how to improve.",
            "Layer equations transform h_l into h_(l+1), and loss guides weight updates.",
            "hidden units, activation, learning rate, regularization",
            "Ravi learns that neural networks can learn powerful patterns, but they need enough data and careful validation.",
            "learning nonlinear feature interactions",
            "overfitting and poor scaling can make training unreliable"
        )
        "CNN" -> FlagshipNarrative(
            "Tara",
            "The Sliding Window Detective",
            "Tara teaches a camera to spot shapes. Instead of looking at the whole image at once, it scans small windows like a detective with a magnifying glass.",
            "it learns local visual patterns and reuses the same detector across an image",
            "It is like sliding a tiny stencil over a picture to find edges, corners, and textures.",
            "A CNN uses convolution filters to create feature maps. Early filters find simple patterns; deeper layers combine them into objects.",
            "1. Normalize the image. 2. Slide kernels across pixels. 3. Build feature maps. 4. Pool or downsample. 5. Classify from learned visual evidence.",
            "A medical imaging app highlights suspicious regions for a trained doctor to review.",
            "image classification, medical imaging, factory defect detection, visual search",
            "Kernel size controls local view; stride controls movement; filters learn useful visual detectors.",
            "The feature map equation shows one filter scanning the image and producing activations.",
            "filters, kernel size, stride, padding, pooling",
            "Tara learns that a CNN can see patterns, but it should never replace careful human review in high-stakes work.",
            "excellent visual pattern learning",
            "distribution shift and poor labels can make image models fail"
        )
        "LSTM" -> FlagshipNarrative(
            "Arjun",
            "The Memory Backpack",
            "Arjun reads a long weather diary. To predict tomorrow, he must remember useful old clues and forget noisy details.",
            "it uses gates to keep, erase, and reveal sequence memory",
            "It is like carrying a backpack where gates decide what notes stay inside.",
            "LSTM is a sequence model with a cell state and gates. The gates control what information enters memory, what is forgotten, and what becomes output.",
            "1. Read one time step. 2. Compute input, forget, and output gates. 3. Update cell memory. 4. Emit hidden state. 5. Repeat through the sequence.",
            "A sensor system forecasts machine failure from a stream of readings over time.",
            "time-series forecasting, text sequence modeling, sensor monitoring, speech sequences",
            "The forget gate removes stale clues; the input gate writes new clues; the output gate reveals useful memory.",
            "The cell-state equation mixes old memory with new candidate memory.",
            "hidden size, sequence length, dropout, learning rate",
            "Arjun learns that memory helps, but future data must never leak into training windows.",
            "handling longer sequence context than plain RNNs",
            "training is slower than attention and still needs careful windowing"
        )
        else -> null
    }

    private data class QuestionDraft(
        val text: String,
        val explanation: String,
        val correct: String,
        val distractors: List<String>
    )

    private fun questions(
        topic: LearnTopic,
        profile: LearningProfile,
        application: String,
        hyperparameter: String,
        mistake: String
    ): List<SeedQuestion> {
        val title = topic.title
        val support = quizSupport(topic, profile)
        val drafts = listOf(
            QuestionDraft(
                "Concept: what does $title mainly learn or compute?",
                "$title is used here for ${profile.purpose}",
                support.coreAnswer,
                support.coreDistractors
            ),
            QuestionDraft(
                "Scenario: which use case is the best fit for $title?",
                "$title fits this case because ${support.scenarioReason}",
                support.scenarioAnswer(application),
                support.scenarioDistractors
            ),
            QuestionDraft(
                "Formula/output: in $title, what does this expression focus on: ${profile.equation.takeMcqText()}?",
                support.formulaExplanation(profile),
                support.formulaAnswer(profile),
                support.formulaDistractors
            ),
            QuestionDraft(
                "Implementation: which parameter or setting is most important to inspect for $title?",
                "For $title, ${support.parameterName} changes ${support.parameterEffect}.",
                "${support.parameterName}: ${support.parameterEffect}",
                support.parameterDistractors
            ),
            QuestionDraft(
                "Debugging: what problem should you watch for when using $title?",
                "A realistic failure mode for $title is ${support.failureMode}.",
                support.failureMode,
                support.failureDistractors(mistake)
            )
        )
        val topicIndex = LearnCatalog.topics.indexOfFirst { it.id == topic.id }.coerceAtLeast(0)
        return drafts.mapIndexed { index, draft ->
            val correctIndex = (topicIndex * 5 + index) % 4
            question(topic, index + 1, draft.text, draft.explanation, draft.toSeedOptions(topic, correctIndex))
        }
    }

    private fun QuestionDraft.toSeedOptions(topic: LearnTopic, correctIndex: Int): List<SeedOption> {
        val wrongs = distractors.distinct()
            .filter { it != correct }
            .map { it.asAlgorithmSpecificDistractor(topic) }
            .take(3)
            .toMutableList()
        while (wrongs.size < 3) wrongs.add("A tempting shortcut that does not match this algorithm's assumptions")
        val result = mutableListOf<SeedOption>()
        var wrongIndex = 0
        repeat(4) { index ->
            if (index == correctIndex) result.add(SeedOption(correct, true)) else result.add(SeedOption(wrongs[wrongIndex++], false))
        }
        return result
    }

    private fun String.asAlgorithmSpecificDistractor(topic: LearnTopic): String =
        if (contains(topic.title, ignoreCase = true)) this else "$this for ${topic.title}"

    private data class QuizSupport(
        val coreAnswer: String,
        val coreDistractors: List<String>,
        val scenarioReason: String,
        val scenarioAnswer: (String) -> String,
        val scenarioDistractors: List<String>,
        val formulaAnswer: (LearningProfile) -> String,
        val formulaExplanation: (LearningProfile) -> String,
        val formulaDistractors: List<String>,
        val parameterName: String,
        val parameterEffect: String,
        val parameterDistractors: List<String>,
        val failureMode: String,
        val failureDistractors: (String) -> List<String>
    )

    private fun quizSupport(topic: LearnTopic, profile: LearningProfile): QuizSupport = when (profile.kind) {
        VisualizationKind.Regression -> QuizSupport(
            "A numeric prediction rule that maps features to a continuous target",
            listOf("A cluster label for every point without using targets", "A policy that learns by reward after actions", "A token-to-token attention map for language context"),
            "the output is a real number that can be compared with residual error",
            { "${topic.title} for ${it.ifBlank { "forecasting a continuous value from historical features" }}" },
            listOf("Grouping customers with no target column", "Choosing the next game action from rewards", "Explaining image pixels with a heatmap"),
            { "It connects predicted numeric values with residuals or objective error" },
            { "${topic.title}'s formula ${it.equation.takeMcqText()} measures how predictions relate to the numeric target." },
            listOf("It counts neighbor votes for a class label", "It assigns each point to the nearest centroid", "It masks future tokens in a decoder"),
            specificParameter(topic, profile),
            "bias, variance, smoothness, or penalty strength in the fitted numeric relationship",
            listOf("n_clusters: number of unlabeled groups", "epsilon: density-neighborhood radius", "temperature: token sampling randomness"),
            specificMistake(topic, profile, "ignoring residual plots, outliers, or multicollinearity"),
            { listOf("Using K-Means inertia as the main regression metric", "Treating a high training R2 as proof of causation", it.ifBlank { "Forgetting to inspect residuals on fresh data" }) }
        )
        VisualizationKind.Classification -> QuizSupport(
            "A rule that separates examples into classes using labelled training data",
            listOf("A method that always predicts a continuous amount", "A compression method that removes the target label", "A generator that creates new samples from noise"),
            "the goal is to decide a category and check precision, recall, or calibration",
            { "${topic.title} for ${it.ifBlank { "classifying cases such as fraud/not fraud or pass/review" }}" },
            listOf("Estimating a house price as a continuous value", "Reducing 300 features to 2 plotting axes", "Finding unlabeled clusters without a target"),
            { "It turns evidence into a class score, probability, margin, or vote" },
            { "${topic.title}'s formula/output should be interpreted as class evidence, not as a guaranteed truth." },
            listOf("It minimizes reconstruction error for an autoencoder", "It updates Q-values from future reward", "It computes centroids for unlabeled groups"),
            specificParameter(topic, profile),
            "decision boundary shape, regularization, class weighting, or classification threshold",
            listOf("n_components: number of projection axes", "sequence_length: how many time steps to read", "latent_dim: size of generated sample code"),
            specificMistake(topic, profile, "ignoring class imbalance, threshold choice, or calibration"),
            { listOf("Reporting accuracy only when the rare class matters", "Training with labels that leak the answer", it.ifBlank { "Using a default threshold without checking precision and recall" }) }
        )
        VisualizationKind.Neighbours -> QuizSupport(
            "A prediction made from the closest stored examples",
            listOf("A global linear equation fitted once with coefficients", "A generated image sampled from random noise", "A future reward estimate updated after actions"),
            "the new case should be solved by similarity to known cases",
            { "${topic.title} for ${it.ifBlank { "matching a query case to similar stored examples" }}" },
            listOf("Learning convolution filters from image patches", "Estimating a posterior from a prior and likelihood", "Choosing actions through exploration rewards"),
            { "It identifies N_K(x), the neighborhood used for voting or averaging" },
            { "${topic.title}'s formula is about which examples count as nearest neighbors and how their values are combined." },
            listOf("It computes entropy reduction at a tree split", "It multiplies Q, K, and V matrices", "It samples from a latent diffusion process"),
            specificParameter(topic, profile),
            "how local or smooth the neighbor vote/average becomes",
            listOf("max_depth: maximum tree path length", "learning_rate: optimizer step size", "heads: number of attention subspaces"),
            specificMistake(topic, profile, "using an inappropriate distance metric or unscaled features"),
            { listOf("Letting one large-scale feature dominate distance", "Choosing K after looking at the test score", it.ifBlank { "Using irrelevant features in the distance calculation" }) }
        )
        VisualizationKind.Tree -> QuizSupport(
            "A sequence of feature splits, or an ensemble of such splits, that forms predictions",
            listOf("A pure distance vote from stored examples", "A matrix factorization of users and items", "A recurrent memory cell for ordered data"),
            "the data has tabular rules, interactions, or split-based explanations",
            { "${topic.title} for ${it.ifBlank { "explaining tabular decisions with split paths or tree ensembles" }}" },
            listOf("Compressing images into a latent vector", "Sampling actions from a reward policy", "Using token self-attention over a paragraph"),
            { "It measures split quality, impurity reduction, or aggregated tree votes" },
            { "${topic.title}'s formula/output explains how splits or tree votes reduce uncertainty in the target." },
            listOf("It averages the K closest labels by distance", "It updates a hidden state across time", "It estimates a posterior by multiplying prior and likelihood"),
            specificParameter(topic, profile),
            "tree depth, leaf size, feature randomness, or ensemble strength",
            listOf("gamma: RBF kernel width for SVM", "n_neighbors: local vote count", "context_length: transformer input window"),
            specificMistake(topic, profile, "letting trees grow too deep or trusting leakage-heavy feature importance"),
            { listOf("Interpreting an unstable split as a permanent rule", "Using impurity importance without checking bias", it.ifBlank { "Overfitting noisy pockets with deep split paths" }) }
        )
        VisualizationKind.Clustering, VisualizationKind.Density -> QuizSupport(
            "Unlabeled structure such as groups, dense regions, or noise points",
            listOf("A supervised class boundary using known labels", "A regression line for a numeric target", "A policy optimized by reward feedback"),
            "there are no target labels and the goal is structure discovery",
            { "${topic.title} for ${it.ifBlank { "discovering groups or unusual structure before labels exist" }}" },
            listOf("Predicting a labelled spam class", "Forecasting tomorrow's numeric demand", "Training a language decoder with masks"),
            { "It focuses on assignments, compactness, density reachability, or mixture responsibility" },
            { "${topic.title}'s formula/output should be read as structure in feature space, not as supervised accuracy." },
            listOf("It estimates precision and recall from true labels", "It backpropagates through convolution kernels", "It computes discounted future reward"),
            specificParameter(topic, profile),
            "number of groups, density radius, minimum points, or initialization behavior",
            listOf("C: margin violation cost for SVM", "dropout: neural-network regularization", "threshold: converting probability to class"),
            specificMistake(topic, profile, "using unscaled features or forcing clusters that do not exist"),
            { listOf("Treating every cluster as a real business segment without validation", "Choosing K only because the picture looks neat", it.ifBlank { "Ignoring noise points and feature scaling" }) }
        )
        VisualizationKind.Attention -> QuizSupport(
            "Context mixing where each token or patch weighs other tokens or patches",
            listOf("A nearest-neighbor vote over stored rows", "A tree path made of yes/no feature tests", "A centroid update from unlabeled points"),
            "the task needs long-range context or relationships between parts",
            { "${topic.title} for ${it.ifBlank { "understanding documents, tokens, or image patches with context" }}" },
            listOf("Estimating a single straight-line slope", "Assigning K-Means clusters to customers", "Using one decision stump for tabular rules"),
            { "It turns query-key similarity into attention weights over values" },
            { "${topic.title}'s formula/output explains which context pieces are mixed for a selected query." },
            listOf("It counts Gini impurity in child nodes", "It averages labels of nearest neighbors", "It computes bootstrap votes from random trees"),
            specificParameter(topic, profile),
            "context window, number of heads, masking, or attention sharpness",
            listOf("epsilon: DBSCAN radius", "max_depth: tree length", "n_clusters: centroid count"),
            specificMistake(topic, profile, "forgetting positional information, masking, or context limits"),
            { listOf("Using a decoder without causal masking", "Treating attention maps as complete causal explanations", it.ifBlank { "Ignoring context-window truncation" }) }
        )
        else -> genericQuizSupport(topic, profile)
    }

    private fun genericQuizSupport(topic: LearnTopic, profile: LearningProfile): QuizSupport {
        val kindName = profile.kind.name
        return QuizSupport(
            "A ${topic.section.lowercase()} method that turns data into a tested ${kindName.lowercase()} signal",
            listOf("A random answer generator that ignores validation", "A dashboard color setting unrelated to data", "A way to use test labels during training"),
            "the method's objective matches the stated data and decision goal",
            { "${topic.title} for ${it.ifBlank { "a realistic ${topic.domain.lowercase()} workflow" }}" },
            listOf("Using test-set leakage to improve apparent results", "Replacing the objective with a visual preference", "Skipping preprocessing because the method name sounds advanced"),
            { "It defines the algorithm-specific state, objective, or output that must be interpreted" },
            { "${topic.title}'s expression ${it.equation.takeMcqText()} should be connected to its inputs, objective, and validation result." },
            listOf("It proves the model works on every future dataset", "It removes the need for assumptions", "It changes labels after evaluation"),
            specificParameter(topic, profile),
            "model behavior, stability, complexity, or evaluation quality",
            listOf("phone wallpaper: not an ML parameter", "test label access: leakage, not a setting", "random final score: not a reproducible decision"),
            specificMistake(topic, profile, "mismatching preprocessing, objective, or validation to the deployment task"),
            { listOf("Changing preprocessing between training and inference", "Optimizing a metric that does not match the real goal", it.ifBlank { "Skipping a baseline comparison" }) }
        )
    }

    private fun question(
        topic: LearnTopic,
        number: Int,
        text: String,
        explanation: String,
        options: List<SeedOption>
    ) = SeedQuestion(
        question = McqQuestionRecord(
            algorithmId = topic.id,
            questionNumber = number,
            question = text,
            explanation = explanation
        ),
        options = options
    )

    private fun specificParameter(topic: LearnTopic, profile: LearningProfile): String {
        val name = topic.title.lowercase()
        return when {
            "k-means" in name || name == "k-means" -> "n_clusters / K: the number of centroids to learn"
            name == "dbscan" || name == "hdbscan" || name == "optics" -> "epsilon and min_samples: the density neighborhood rules"
            "nearest" in name || "knn" in name -> "n_neighbors / K: how many nearby examples vote or average"
            "random forest" in name || "extra trees" in name -> "n_estimators and max_features: how many trees vote and how different they are"
            "decision tree" in name -> "max_depth and min_samples_leaf: how complex each split path may become"
            "support vector" in name || name == "one-class svm" -> "C, gamma, and kernel: margin cost and boundary shape"
            "logistic regression" in name -> "C, penalty, class_weight, and threshold: regularization and decision tradeoff"
            "ridge" in name -> "alpha / lambda: L2 shrinkage strength"
            "lasso" in name -> "alpha / lambda: L1 sparsity strength"
            "elastic net" in name -> "alpha and l1_ratio: total penalty and L1/L2 mix"
            "polynomial" in name -> "degree: how much curve the expanded features can express"
            "gradient boost" in name || "xgboost" in name || "lightgbm" in name || "catboost" in name || "adaboost" in name -> "learning_rate, n_estimators, and max_depth: boosting speed and weak-learner complexity"
            "pca" in name || "svd" in name || "factor analysis" in name || "component" in name -> "n_components: how many compressed dimensions to keep"
            "t-sne" in name -> "perplexity and learning_rate: local-neighborhood balance and optimization speed"
            "umap" in name -> "n_neighbors and min_dist: local structure and embedding compactness"
            "cnn" in name || "convolution" in name || name in listOf("lenet", "alexnet", "vgg", "resnet", "densenet", "mobilenet", "efficientnet", "convnext") -> "filters, kernel_size, stride, and padding: what local visual patterns are scanned"
            "lstm" in name || "gru" in name || "rnn" in name || "recurrent" in name -> "hidden_size, sequence_length, dropout, and learning_rate: memory capacity and training stability"
            "attention" in name || "transformer" in name || name in listOf("bert", "gpt", "t5") -> "num_heads, context_length, d_model, and masking: how context is mixed"
            "autoencoder" in name -> "latent_dim and reconstruction loss: bottleneck size and what must be rebuilt"
            "gan" in name || "diffusion" in name || "generative" in name -> "latent_dim, noise schedule, guidance, and learning_rate: sample diversity and stability"
            "q-learning" in name || "sarsa" in name || "bandit" in name || "policy" in name || "actor" in name || "ppo" in name || "sac" in name || "dqn" in name -> "alpha, gamma, and epsilon: learning speed, future reward, and exploration"
            "bayes" in name || "gaussian mixture" in name || "markov" in name || "mcmc" in name -> "prior, likelihood assumptions, covariance type, or sample count: uncertainty behavior"
            "adam" in name || "gradient descent" in name || "momentum" in name || "rmsprop" in name || "adagrad" in name -> "learning_rate and momentum/beta values: update size and smoothing"
            "recommend" in topic.domain.lowercase() || "collaborative" in name || "matrix factorization" in name -> "embedding_dim, regularization, and top_k: preference representation and ranking depth"
            "shap" in name || "lime" in name || "importance" in name || "grad-cam" in name || "saliency" in name -> "background sample, perturbation size, or attribution layer: explanation stability"
            else -> profile.hyperparameters.firstOrNull { it.isNotBlank() && "capacity or complexity" !in it.lowercase() }
                ?: "regularization, tolerance, or iteration budget: the setting that most changes model behavior"
        }
    }

    private fun specificMistake(topic: LearnTopic, profile: LearningProfile, fallback: String): String {
        val name = topic.title.lowercase()
        return when {
            "linear regression" in name || "ridge" in name || "lasso" in name || "elastic net" in name -> "missing multicollinearity, outlier, residual, or scaling checks"
            "logistic regression" in name -> "using a 0.50 threshold without checking imbalance, calibration, precision, and recall"
            "nearest" in name || "knn" in name -> "using raw unscaled features so distance is dominated by the largest numeric column"
            "tree" in name || "forest" in name || "boost" in name -> "letting split logic overfit leakage or noisy pockets in the training data"
            "k-means" in name || "cluster" in topic.section.lowercase() -> "choosing K or density settings without scaling features or validating cluster meaning"
            "pca" in name || "svd" in name || "t-sne" in name || "umap" in name -> "over-interpreting a projection plot as proof of true separation"
            "svm" in name || "support vector" in name -> "forgetting scaling or choosing C/gamma from test performance"
            "cnn" in name || "convolution" in name -> "training on images whose lighting, camera, or labels do not match deployment"
            "lstm" in name || "gru" in name || "rnn" in name || "time" in topic.domain.lowercase() -> "leaking future time steps into training windows"
            "attention" in name || "transformer" in name || name in listOf("bert", "gpt", "t5") -> "forgetting masks, context limits, or positional information"
            "autoencoder" in name -> "using reconstruction error without checking whether normal and abnormal cases overlap"
            "gan" in name -> "missing mode collapse or unstable discriminator-generator training"
            "diffusion" in name -> "using a sampling schedule or guidance setting that creates plausible but incorrect samples"
            "q-learning" in name || "sarsa" in name || "policy" in name || "dqn" in name || "ppo" in name -> "reward hacking from a poorly designed reward function"
            "bayes" in name || "markov" in name || "mcmc" in name -> "trusting precise-looking probabilities from weak priors or bad likelihood assumptions"
            "adam" in name || "gradient descent" in name || "momentum" in name -> "using a learning rate that causes unstable convergence or painfully slow learning"
            "recommend" in topic.domain.lowercase() || "collaborative" in name -> "creating feedback loops that only repeat old user behavior"
            "shap" in name || "lime" in name || "importance" in name || "grad-cam" in name -> "treating an explanation as causal proof instead of a diagnostic clue"
            else -> profile.mistakes.firstOrNull { it.isNotBlank() && "training data only" !in it.lowercase() } ?: fallback
        }
    }

    private fun String.joinToSentence(): String = this.ifBlank { "real-world learning problems" }

    private fun List<String>.joinToSentence(): String = filter { it.isNotBlank() }
        .take(4)
        .joinToString(", ")
        .ifBlank { "real-world learning problems" }

    private fun String.takeMcqText(): String {
        val compact = replace(Regex("\\s+"), " ").trim()
        return if (compact.length <= 120) compact else compact.take(117).trimEnd() + "..."
    }

    private fun String.escapeHtml(): String = buildString(length) {
        this@escapeHtml.forEach { char ->
            when (char) {
                '<' -> append("&lt;")
                '>' -> append("&gt;")
                '&' -> append("&amp;")
                '"' -> append("&quot;")
                '\'' -> append("&#39;")
                else -> append(char)
            }
        }
    }
}
