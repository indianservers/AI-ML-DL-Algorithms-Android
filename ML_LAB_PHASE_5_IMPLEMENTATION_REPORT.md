# ML Lab Phase 5 Implementation Report

## 1. Phase 1-4 systems reused

Phase 5 keeps the existing Compose shell, premium dark theme, educational Kotlin engines, algorithm catalog, training telemetry, Canvas visualizers, local preferences, Phase 3 media concepts, and Phase 4 tensor/matrix experiments. The practical studio is isolated under `ml_lab/phase5`; Phases 1-4 remain independently navigable.

## 2. Files added

- `phase5/engine/InferenceContracts.kt`: backend-neutral model, tensor, execution and benchmark contracts.
- `phase5/engine/RuntimeBackends.kt`: LiteRT and ONNX Runtime adapters, capability detection and app-private model repository.
- `phase5/engine/MediaPipelines.kt`: image preprocessing, CameraX telemetry, microphone PCM, tensor builders and output decoding.
- `phase5/engine/PracticalAlgorithms.kt`: quantization, IoU/NMS, spectrogram, tokenizer, embeddings, metrics, personalization, Q-learning and time-series calculations.
- `phase5/presentation/AiEngineeringStudio.kt`: Models, Live AI, Tune, Inspect and Learn workspaces.
- `assets/models/tiny_double.tflite` and `tiny_double.onnx`: deterministic offline runtime smoke models.
- `PhaseFiveEngineTest.kt`, `app/proguard-rules.pro`, this report and `ML_LAB_FINAL_AUDIT.md`.

## 3. Files modified

The version catalog and app Gradle configuration now include LiteRT 1.4.1, ONNX Runtime Android 1.26.0, CameraX 1.6.1, lifecycle Compose, R8, packaging rules and a 64-bit ABI strategy. The manifest adds request-at-use camera and microphone permissions. The app shell exposes Phase 5 as Studio, and the catalog unlocks Q-learning/bandits while promoting time-series lessons.

## 4. Runtime backends

`InferenceBackend` separates load, run and close operations from UI code. `LoadedModel` always reports its backend and actual selected execution target. All native model/session/tensor/delegate handles are closed explicitly on a dedicated single-thread dispatcher.

## 5. LiteRT implementation

LiteRT loads mapped `.tflite` files, validates them with `Interpreter`, exposes input/output names, shapes, types and quantization parameters, allocates typed native-order buffers and runs multi-input/output inference. CPU, NNAPI and compatibility-checked GPU delegate configurations are supported. Unsupported GPU selection returns an exact error and leaves CPU available.

## 6. ONNX implementation

ONNX Runtime Mobile loads `.onnx` files from app-private storage, exposes `NodeInfo`/`TensorInfo`, creates typed `OnnxTensor` inputs, executes sessions and flattens typed outputs. CPU is the baseline. NNAPI is enabled only if the bundled runtime exposes the provider method; an unsupported provider produces an explicit failure. The Android R8 keep rule follows ONNX Runtime guidance.

## 7. CPU, GPU and NNAPI

CPU execution and thread configuration are available for both runtimes. LiteRT GPU is offered only when `CompatibilityList` detects delegate support. NNAPI is offered on Android API 28+ and validated during model load. The app never claims a specific NPU identity because Android does not expose one through these APIs, and delegate availability is not presented as proof of full graph delegation.

## 8. Model import, storage and metadata

The Storage Access Framework accepts only `.tflite` and `.onnx`, copies readable models up to 250 MB into app-private storage, hashes them for stable IDs, validates them through the selected runtime and displays exact failures. The library supports open, favorite, delete, search-ready tags and two bundled smoke models. Tensor inspectors show name, shape, type, quantization and theoretical memory.

## 9. Image and camera pipelines

Gallery images are decoded locally, resized to inspected rank-4 input dimensions, converted as NHWC or NCHW, normalized, quantized when required and passed to real inference. Preprocessing and inference timings remain separate. CameraX preview is lifecycle-bound, uses keep-latest backpressure and reports real frames, luma, analysis latency and dropped frames at a 10 FPS policy. Rotation remains CameraX-owned.

## 10. Detection, segmentation, pose and OCR

The detection tool implements real class-aware NMS and IoU with threshold control and thickness-coded boxes. Gallery luminance drives a real threshold-mask/opacity visualizer. Generic imported models can run through the raw/image pipeline. Model-specific detection decoding, semantic label maps, pose skeleton decoding and OCR text decoding remain dependent on compatible model metadata and are not fabricated by the bundled arithmetic smoke models.

## 11. Audio AI

Microphone access is requested at use time. `AudioRecord` captures real 16 kHz mono PCM, reports sample count/RMS and feeds a Hann-windowed DFT log-magnitude spectrogram. Imported audio models can be driven through the generic tensor runner; model-specific mel/filterbank metadata is not assumed.

## 12. Text AI, embeddings and tokenizers

The offline tokenizer explorer exposes token strings and IDs, including unknown-token handling. Deterministic local embeddings drive cosine similarity and the embedding classifier. Imported text models remain usable through typed raw tensors; model-specific WordPiece/BPE vocabulary and special-token metadata must accompany the model and are never guessed.

## 13. Quantization

The lab implements `real = scale * (quantized - zeroPoint)`, signed/unsigned clamping, dequantization and mean absolute error. FP32, FP16 and INT8 memory comparisons use exact byte counts. Imported tensor quantization scale and zero point are displayed where the runtime exposes them.

## 14. Benchmarking and profiling

The active model/backend runs one cold and 20 warm inferences. The app reports measured p50/p90/p95, mean and throughput, persists the latest p50/timestamp, and labels results device-specific. Profiling shows process heap used/max, theoretical tensor memory, active backend, thread control, current Android thermal status and battery-aware recommendations.

## 15. Transfer learning and personalization

The personalization lesson keeps the feature extractor frozen and trains an embedding-based nearest-similarity head from user-owned examples. It explains export requirements for labels, dimensions and preprocessing. Full backpropagation into imported native models is intentionally not claimed.

## 16. RL, time series and algorithm coverage

Phase 5 closes the curriculum gap with a deterministic Grid World Q-learning engine, epsilon decay, Q-value policy visualization, multi-armed-bandit concepts and a windowed moving-average forecasting lab with temporal-validation guidance. Advanced DQN/PPO and full ARIMA training remain lesson-level catalog topics.

## 17. Performance and memory results

On the Android emulator, the bundled ONNX model validated and executed in 1.645 ms and the bundled LiteRT model in 0.896 ms during smoke testing. These are individual emulator measurements, not cross-device claims. Process and tensor memory are measured/displayed in app; exhaustive native allocation attribution is not exposed by the runtimes.

## 18. APK-size impact and ABI strategy

The original Phase 4 debug APK was about 31.2 MB. Bundling full LiteRT, LiteRT GPU, ONNX Runtime and CameraX initially produced 184,572,792-byte debug and 145,831,287-byte release APKs across all ABIs. Restricting this minSdk-31 app to `arm64-v8a` plus `x86_64` reduced the final debug APK to 118,133,842 bytes and the R8-minified unsigned release APK to 79,720,689 bytes. Production delivery should use an Android App Bundle so each device receives only its ABI.

## 19. Tests and release verification

The full JVM suite contains 28 tests with zero failures. Phase 5 tests cover stable Softmax, signed quantization, NMS/IoU, spectrogram energy, tokenization, embeddings, personalization, percentiles, tensor memory, confusion matrices, calibration, time series and Q-learning. Debug and R8-minified release builds pass. Emulator tests validated both native backends, Phase 5 navigation and Android runtime stability.

## 20. Known limitations and future enhancements

- No pretrained vision/audio/text payload is bundled; only tiny arithmetic smoke models are included to keep licensing and model provenance clear.
- Detection/segmentation/pose/OCR postprocessors require model-specific metadata and are partial pipeline tools today.
- Public Java runtime APIs do not expose a complete portable operator graph or reliable per-op delegate assignment, so those views are marked unavailable rather than inferred.
- GPU was unavailable on the emulator and therefore was capability-gated, not execution-tested.
- Camera and microphone were lifecycle/permission audited on emulator; broad physical-device matrix testing remains future release work.
- A signed release/App Bundle, Play integrity checks, downloadable model packs and richer tokenizer metadata are deployment follow-ups outside this local build.

