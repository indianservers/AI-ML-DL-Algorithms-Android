# ML Lab Phase 3 Implementation Report

## Phase 2 systems reused

Phase 3 extends the existing Deep navigation, primitive-array math, dense layers, activations, loss functions, optimizer-backed training, Canvas visualizations, metrics, local preferences, deterministic seeds, and offline-first architecture. Phase 1 classical ML and all Phase 2 MLP labs remain available.

## Files added

- `phase3/engine/TensorImage.kt`: compact channel-first image tensor, Conv2D forward/backward, and max/average pooling forward/backward.
- `phase3/engine/TinyCnn.kt`: trainable Conv-ReLU-Pool-Dense-Softmax classifier and telemetry.
- `phase3/engine/SequenceModels.kt`: vanilla RNN, BPTT, LSTM gate/state traces, and GRU gate/state traces.
- `phase3/engine/Representation.kt`: embedding lookup/cosine similarity and a dense autoencoder.
- `phase3/data/PhaseThreeContent.kt`: deterministic synthetic images, kernels, sequences, vocabulary, and reconstruction patterns.
- `phase3/presentation/ArchitectureLabs.kt`: convolution, CNN, sequence, autoencoder, embedding, and architecture-inspection experiences.
- `PhaseThreeEngineTest.kt`: convolution gradients, pooling routes, CNN convergence, recurrent-state, BPTT, embedding, and autoencoder tests.

## CNN engine and visualization

Conv2D performs multi-channel cross-correlation with configurable channels, filters, kernel size, stride, padding, and bias. Backpropagation computes input, kernel, and bias gradients. Pooling supports max-index routing and average-gradient distribution. The tiny classifier trains convolution filters and dense weights using categorical cross-entropy and Softmax.

The Vision lab includes editable kernel presets, cell-by-cell patch calculation, stride/padding shape changes, pooling comparison, synthetic image inspection, live feature maps, class probabilities, training metrics, filter-backed inference, and occlusion sensitivity.

## Image data and preprocessing

The bundled dataset is generated locally and contains 72 noisy 8 x 8 vertical, horizontal, and diagonal samples. Values are normalized to 0-1 during generation. The explorer demonstrates zero padding, stride, filtering, pooling, normalization, and occlusion. No large asset or download is required.

## Sequence models

The vanilla RNN implements recurrent forward propagation and BPTT with shared-weight gradient accumulation. LSTM implements forget, input, candidate, output, cell, and hidden-state equations. GRU implements reset, update, candidate, and hidden-state equations. Their traces feed the time-step and gate inspectors directly.

## Embeddings and autoencoder

The embedding layer performs trainable-table-style token lookup and cosine comparison. The dense autoencoder reuses the tested Phase 2 network and Adam optimizer for reconstruction, exposes its bottleneck values, and decodes user-controlled latent vectors.

## Persistence and model inspection

Phase 3 stores CNN architecture/accuracy metadata in local preferences. The inspector reports parameters, FLOPs, estimated parameter memory, measured local inference latency, and the future-compatible camera/backend pipeline contract.

## Tests and numerical gradients

The test suite checks a known convolution, finite-difference Conv2D kernel gradients, max-pool winner/backward routing, deterministic CNN convergence, recurrent state and BPTT gradients, finite LSTM/GRU state, embedding lookup, and autoencoder convergence.

## Performance and APK impact

Image models use 8 x 8 tensors and four filters; feature-map galleries show at most four maps. Training snapshots are sampled every two epochs. The implementation adds Kotlin code and generated data rather than bundled model or image assets, keeping APK growth small. The app remains fully offline.

## Known limitations and Phase 4 boundary

Phase 3 prioritizes transparent educational models. LSTM/GRU expose exact forward state but do not yet provide full trainable backward optimizers in the UI. Persistence stores model metadata rather than binary tensor checkpoints. Imported images, CameraX integration, TFLite models, background checkpoint workers, and large datasets are outside the compact starter implementation.

Attention, Transformers, Vision Transformers, GANs, variational autoencoders, diffusion, graph neural networks, and pretrained-model explainability remain explicitly deferred to Phase 4.
