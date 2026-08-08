# Phase 4 Interactive Algorithms Report

| Algorithm | Core Visual Mechanism | Interactive Parameters | Step Mode | Math | Failure Case | Comparison | Tests | Status |
| --------- | --------------------- | ---------------------- | --------- | ---- | ------------ | ---------- | ----- | ------ |
| Isolation Forest | random splits -> path length -> anomaly score | trees, selected point | Yes | path length score | clustered anomaly | vs LOF/envelope | Yes | Implemented |
| LOF | neighbors -> density -> local density ratio | k | Yes | LOF | bad k/local anomaly | vs Isolation Forest | Yes | Implemented |
| One-Class SVM | normal region boundary concept | nu/gamma concept | Concept | region score | non-normal shape | vs LOF | Yes | Implemented conceptually |
| Elliptic Envelope | covariance ellipse -> Mahalanobis distance | threshold | Yes | Mahalanobis | two moons | vs Isolation Forest | Yes | Implemented |
| Z-Score | mean/sigma threshold | sigma threshold | Yes | z score | skew/outlier masking | vs IQR | Yes | Implemented |
| IQR Outlier | quartiles/fences | multiplier | Yes | IQR fences | multimodal data | vs Z-score | Yes | Implemented |
| Apriori | levels -> candidate pruning | min support | Yes | support | low support explosion | vs FP-Growth | Yes | Implemented |
| FP-Growth | transaction compression -> tree counts | min support concept | Yes | counts | dense baskets | vs Apriori | Yes | Implemented |
| ECLAT | item -> TID sets -> intersections | selected items concept | Yes | set support | large TID sets | vs Apriori | Yes | Implemented |
| Association Rules | support/confidence/lift | thresholds concept | Yes | rule metrics | high confidence low lift | rule graph concept | Yes | Implemented |
| Popularity | item averages/counts | sorting concept | N/A | averages | cold start personalization | recommender comparison | Yes | Implemented |
| Content-Based | user profile vs item vector | feature weights concept | Yes | cosine | sparse features | vs CF | Yes | Implemented |
| User-CF | similar users -> weighted rating | selected user/item | Yes | cosine weighted average | new user | vs item-CF | Yes | Implemented |
| Item-CF | similar items -> weighted rating | selected user/item | Yes | cosine weighted average | new item | vs user-CF | Yes | Implemented |
| Matrix Factorization | R ~= P Q^T | latent factors | Yes | dot product | sparse matrix | vs CF | Yes | Implemented |
| SVD Recommendation | low-rank reconstruction concept | rank concept | Yes | low-rank approximation | sparse missing values | matrix comparison | Yes | Implemented conceptually |
| ALS | alternate user/item factor updates | iteration concept | Yes | alternating objective | sparsity | vs SGD factors | Yes | Implemented conceptually |
| Neural CF Concepts | embeddings -> interaction -> score | embedding concept | Concept | dot/MLP concept | cold start | vs MF | Yes | Implemented conceptually |
| Bayes Theorem | prior/evidence -> posterior | prior/sensitivity/specificity | Yes | Bayes rule | base-rate effect | MLE/MAP/posterior | Yes | Implemented |
| Bayesian Inference | Beta prior -> observations -> posterior | H/T steps | Yes | Beta-Bernoulli | strong wrong prior | MLE/MAP | Yes | Implemented |
| MLE | likelihood peak | observations concept | Yes | likelihood | small samples | vs MAP | Yes | Implemented |
| MAP | likelihood x prior | prior strength | Yes | posterior mode | wrong prior | vs MLE | Yes | Implemented |
| Bayesian Linear Regression | lines/uncertainty concept | observations concept | Concept | posterior uncertainty | weak data | GP/regression comparison | Yes | Implemented conceptually |
| Bayesian Networks | DAG/CPT evidence concept | evidence concept | Concept | conditional probability | ambiguous evidence | HMM | Yes | Implemented conceptually |
| HMM | states -> emissions -> forward/Viterbi | observations | Yes | forward recursion | ambiguous emissions | forward vs Viterbi | Yes | Implemented |
| Gaussian Processes | kernel -> mean/uncertainty | length/noise | Yes | RBF covariance | bad length scale | Bayesian regression | Yes | Implemented |
| MCMC Concepts | proposal -> accept/reject -> samples | proposal width | Yes | acceptance ratio | poor scale | vs VI | Yes | Implemented |
| Metropolis-Hastings | density ratio and u decision | proposal width | Yes | min(1,p'/p) | poor mixing | vs Gibbs | Yes | Implemented |
| Gibbs Sampling | alternate conditionals | steps | Yes | conditional updates | correlation | vs MH | Yes | Implemented |
| Variational Inference | q distribution -> optimized approximation | steps | Yes | ELBO proxy | underfit posterior | vs MCMC | Yes | Implemented conceptually |

## Components Reused

- Phased algorithm routing, cards, buttons, metric pills, Canvas-based visuals, deterministic offline datasets.

## New Reusable Components

- Anomaly canvas, association metrics panel, user-item matrix panel, probability update panel, GP uncertainty canvas, Gibbs path canvas.

## Performance Limits

- Offline bounded toy datasets only.
- No cloud/API calls.
- No heavy matrix factorization or GP inversion on large data.

## Recommendations For Phase 5

- Extract common card/slider/canvas primitives from phase-local files.
- Add ViewModel/coroutine cancellation before heavier models.
- Promote probability matrix and sequence timeline into shared components.
