package com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.presentation

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.os.PowerManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GlassPanel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.GradientButton
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBlue
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabBorder
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabCyan
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabGreen
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabMuted
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabOrange
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPink
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabPurple
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.LabText
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.MetricPill
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SectionTitle
import com.indianservers.ai_ml_dl_algorithms.ml_lab.components.SegmentedOption
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.BackendConfig
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.BenchmarkSummary
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.CameraTelemetry
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.CapabilityDetector
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.Detection
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.DeviceAiCapabilities
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.EmbeddingClassifier
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.ExecutionTarget
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.GridWorld
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.ImagePreprocessResult
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.ImageTensorPreprocessor
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.InferenceBackend
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.InferenceResult
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.LiteRtBackend
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.LoadedModel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.LumaFrameAnalyzer
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.MicrophoneSampler
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.ModelDescriptor
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.ModelFormat
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.ModelRepository
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.OnnxRuntimeBackend
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.PracticalAlgorithms
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.StoredModel
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.TensorData
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.TensorSpec
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.movingAverageForecast
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.outputFloats
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.sampleTensor
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.tensorPreview
import com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.zeroTensor
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private enum class StudioPage(val label: String) { Models("Models"), Live("Live AI"), Optimize("Tune"), Inspect("Inspect"), Curriculum("Learn") }

@Composable
fun AiEngineeringStudio() {
    val context = LocalContext.current
    val repository = remember { ModelRepository(context) }
    val scope = rememberCoroutineScope()
    val executor = remember { Executors.newSingleThreadExecutor() }
    val inferenceDispatcher = remember { executor.asCoroutineDispatcher() }
    var page by remember { mutableStateOf(StudioPage.Models) }
    var models by remember { mutableStateOf(repository.list()) }
    var selected by remember { mutableStateOf(models.firstOrNull()) }
    var loaded by remember { mutableStateOf<LoadedModel?>(null) }
    var backend by remember { mutableStateOf<InferenceBackend?>(null) }
    var execution by remember { mutableStateOf(ExecutionTarget.CPU) }
    var threads by remember { mutableIntStateOf(2) }
    var result by remember { mutableStateOf<InferenceResult?>(null) }
    var message by remember { mutableStateOf("Open a bundled model or import your own") }
    var benchmark by remember { mutableStateOf<BenchmarkSummary?>(null) }
    val capabilities = remember { CapabilityDetector.detect() }

    fun loadModel(model: StoredModel) {
        message = "Validating ${model.name}..."
        scope.launch {
            runCatching {
                withContext(inferenceDispatcher) {
                    backend?.let { old -> loaded?.let(old::close) }
                    val next = if (model.format == ModelFormat.LiteRT) LiteRtBackend() else OnnxRuntimeBackend()
                    val handle = next.loadModel(repository.source(model), BackendConfig(execution, threads))
                    next to handle
                }
            }.onSuccess { (next, handle) -> backend = next; loaded = handle; selected = model; result = null; message = "Model validated and ready" }
                .onFailure { message = it.message ?: "Unable to load model" }
        }
    }

    val importLauncher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@rememberLauncherForActivityResult
        runCatching { repository.import(uri) }.onSuccess { model -> models = repository.list(); selected = model; loadModel(model) }
            .onFailure { message = "Unable to import model: ${it.message}" }
    }

    DisposableEffect(Unit) { onDispose { backend?.let { current -> loaded?.let(current::close) }; inferenceDispatcher.close(); executor.shutdown() } }

    LazyColumn(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        item {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Column { Text("AI Engineering Studio", color = LabText, fontSize = 26.sp, fontWeight = FontWeight.Bold); Text("Run - inspect - optimize - measure", color = LabMuted) }
                Text("Phase 5", color = LabGreen, fontWeight = FontWeight.Bold)
            }
        }
        item { Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { StudioPage.entries.forEach { SegmentedOption(it.label, page == it, Modifier.weight(1f)) { page = it } } } }
        item {
            when (page) {
                StudioPage.Models -> ModelsPage(models, selected, loaded?.descriptor, message, capabilities, execution, threads,
                    onImport = { importLauncher.launch(arrayOf("application/octet-stream", "application/x-tflite", "application/onnx")) },
                    onSelect = { selected = it; loadModel(it) },
                    onFavorite = { repository.toggleFavorite(it); models = repository.list() },
                    onDelete = { repository.delete(it); models = repository.list(); if (selected?.id == it.id) { selected = null; loaded = null } },
                    onExecution = { execution = it; selected?.let(::loadModel) }, onThreads = { threads = it; selected?.let(::loadModel) },
                    onRun = {
                        val current = loaded; val runner = backend
                        if (current != null && runner != null) scope.launch { runCatching { withContext(inferenceDispatcher) { runner.run(current, current.descriptor.inputs.map(::sampleTensor)) } }.onSuccess { result = it; message = "Inference completed in %.3f ms".format(it.latencyMillis) }.onFailure { message = "Inference failed: ${it.message}" } }
                    }, result = result)
                StudioPage.Live -> LiveAiPage(loaded, backend, inferenceDispatcher, onResult = { result = it }, onMessage = { message = it })
                StudioPage.Optimize -> OptimizePage(context, loaded, backend, capabilities, benchmark, onBenchmark = {
                    val current = loaded; val runner = backend
                    if (current != null && runner != null) scope.launch {
                        runCatching { withContext(inferenceDispatcher) { val input = current.descriptor.inputs.map(::zeroTensor); val cold = runner.run(current, input).latencyMillis; val warm = List(20) { runner.run(current, input).latencyMillis }; PracticalAlgorithms.benchmark(cold, warm) } }
                            .onSuccess { benchmark = it; context.getSharedPreferences("phase5_benchmarks", Context.MODE_PRIVATE).edit().putFloat("last_p50", it.p50.toFloat()).putLong("saved_at", System.currentTimeMillis()).apply() }
                    }
                })
                StudioPage.Inspect -> InspectPage(loaded?.descriptor, result, capabilities)
                StudioPage.Curriculum -> CurriculumPage()
            }
        }
    }
}

@Composable
private fun ModelsPage(
    models: List<StoredModel>, selected: StoredModel?, descriptor: ModelDescriptor?, message: String, capabilities: DeviceAiCapabilities,
    execution: ExecutionTarget, threads: Int, onImport: () -> Unit, onSelect: (StoredModel) -> Unit, onFavorite: (StoredModel) -> Unit,
    onDelete: (StoredModel) -> Unit, onExecution: (ExecutionTarget) -> Unit, onThreads: (Int) -> Unit, onRun: () -> Unit, result: InferenceResult?,
) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Device AI Capabilities", "Only runtime-detectable capabilities are enabled")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { CapabilityPill("CPU", capabilities.cpu, Modifier.weight(1f)); CapabilityPill("LiteRT", capabilities.liteRt, Modifier.weight(1f)); CapabilityPill("ONNX", capabilities.onnxRuntime, Modifier.weight(1f)); CapabilityPill("NNAPI", capabilities.nnapi, Modifier.weight(1f)); CapabilityPill("GPU", capabilities.gpuDelegate, Modifier.weight(1f)) }
            Text(capabilities.acceleratorDetail, color = LabMuted, fontSize = 11.sp)
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Model Library", "App-private, offline model storage")
            GradientButton("Import .tflite / .onnx", Modifier.fillMaxWidth(), onImport)
            Text(message, color = if (message.contains("failed", true) || message.contains("unable", true)) LabPink else LabCyan, fontSize = 12.sp)
            if (models.isEmpty()) Text("No imported models. Use Android's document picker; broad storage permission is never requested.", color = LabMuted)
            models.forEach { model ->
                Row(Modifier.fillMaxWidth().background(if (selected?.id == model.id) LabPurple.copy(alpha = .15f) else Color.Transparent, RoundedCornerShape(6.dp)).padding(7.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Column(Modifier.weight(1f)) { Text(model.name, color = LabText, fontWeight = FontWeight.Bold); Text("${model.format} - ${formatBytes(model.bytes)}${if (model.favorite) " - Favorite" else ""}", color = LabMuted, fontSize = 11.sp) }
                    SegmentedOption("Open", false) { onSelect(model) }; SegmentedOption(if (model.favorite) "Unstar" else "Star", false) { onFavorite(model) }; SegmentedOption("Delete", false) { onDelete(model) }
                }
            }
        } }
        descriptor?.let { model ->
            GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
                SectionTitle(model.name, "${model.format} - ${formatBytes(model.sizeBytes)}")
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { ExecutionTarget.entries.forEach { target -> val available = target == ExecutionTarget.CPU || target == ExecutionTarget.NNAPI && capabilities.nnapi || target == ExecutionTarget.GPU && capabilities.gpuDelegate; SegmentedOption(if (available) target.name else "${target.name} N/A", execution == target, Modifier.weight(1f)) { if (available) onExecution(target) } } }
                Text("CPU threads: $threads", color = LabMuted); Slider(threads.toFloat(), { onThreads(it.toInt()) }, valueRange = 1f..8f, steps = 6)
                Text("Backend ${if (model.format == ModelFormat.LiteRT) "LiteRT" else "ONNX Runtime"} - execution ${execution.name}", color = LabGreen, fontWeight = FontWeight.Bold)
                GradientButton("Run sample tensor", Modifier.fillMaxWidth(), onRun)
                result?.let { inference -> Text("Measured inference %.3f ms; ${inference.outputs.size} output tensor(s).".format(inference.latencyMillis), color = LabCyan); inference.outputs.firstOrNull()?.let { Text("Output ${tensorPreview(it)}", color = LabGreen, fontSize = 11.sp) } }
            } }
        }
    }
}

@Composable private fun CapabilityPill(name: String, available: Boolean, modifier: Modifier) = MetricPill(name, if (available) "Yes" else "No", if (available) LabGreen else LabMuted, modifier)

@Composable
private fun LiveAiPage(loaded: LoadedModel?, backend: InferenceBackend?, inferenceDispatcher: kotlinx.coroutines.CoroutineDispatcher, onResult: (InferenceResult) -> Unit, onMessage: (String) -> Unit) {
    val context = LocalContext.current; val scope = rememberCoroutineScope()
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }; var preprocessing by remember { mutableStateOf<ImagePreprocessResult?>(null) }; var imageResult by remember { mutableStateOf<InferenceResult?>(null) }
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri -> uri?.let { bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(context.contentResolver, it)).copy(Bitmap.Config.ARGB_8888, false) } }
    var cameraPermission by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    var audioPermission by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) }
    val cameraPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { cameraPermission = it }
    val audioPermissionLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { audioPermission = it }
    var cameraEnabled by remember { mutableStateOf(false) }; var cameraTelemetry by remember { mutableStateOf<CameraTelemetry?>(null) }
    var audio by remember { mutableStateOf<FloatArray?>(null) }; var spectrogram by remember { mutableStateOf<Array<FloatArray>?>(null) }
    var text by remember { mutableStateOf("on-device ai keeps private data local") }
    var maskOpacity by remember { mutableFloatStateOf(.45f) }; var nmsThreshold by remember { mutableFloatStateOf(.5f) }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
            SectionTitle("Image AI", "Gallery preprocessing and real model inference")
            GradientButton("Choose image", Modifier.fillMaxWidth()) { imagePicker.launch("image/*") }
            bitmap?.let { selectedBitmap ->
                Text("Source ${selectedBitmap.width} x ${selectedBitmap.height}", color = LabMuted)
                loaded?.descriptor?.inputs?.firstOrNull()?.let { spec ->
                    SegmentedOption("Preprocess + Run", false, Modifier.fillMaxWidth()) {
                        val runner = backend ?: return@SegmentedOption
                        scope.launch { runCatching { withContext(inferenceDispatcher) { val prep = ImageTensorPreprocessor.preprocess(selectedBitmap, spec); val otherInputs = loaded.descriptor.inputs.drop(1).map(::zeroTensor); prep to runner.run(loaded, listOf(prep.tensor) + otherInputs) } }.onSuccess { (prep, inference) -> preprocessing = prep; imageResult = inference; onResult(inference); onMessage("Gallery inference complete") }.onFailure { onMessage("Image pipeline failed: ${it.message}") } }
                    }
                }
                preprocessing?.let { Text("Resize ${it.sourceWidth}x${it.sourceHeight} -> ${it.targetWidth}x${it.targetHeight}; normalize (x-${it.mean})/${it.standardDeviation}; %.3f ms".format(it.elapsedNanos / 1_000_000.0), color = LabCyan, fontSize = 11.sp) }
                imageResult?.outputs?.firstOrNull()?.let { output -> val values = outputFloats(output, loaded?.descriptor?.outputs?.firstOrNull()?.quantization); val probabilities = if (values.isNotEmpty()) PracticalAlgorithms.softmax(values) else values; Text("Top output ${probabilities.indices.sortedByDescending { probabilities[it] }.take(3).joinToString { "#$it %.2f%%".format(probabilities[it] * 100) }}", color = LabGreen) }
                Text("Segmentation mask opacity %.0f%%".format(maskOpacity * 100), color = LabMuted); Slider(maskOpacity, { maskOpacity = it }); ImageThresholdCanvas(selectedBitmap, maskOpacity)
            }
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Object Detection Tools", "IoU and class-aware non-maximum suppression")
            val detections = listOf(Detection(0, .91f, .1f, .15f, .62f, .72f), Detection(0, .78f, .18f, .2f, .68f, .75f), Detection(1, .74f, .66f, .12f, .93f, .55f))
            Text("NMS threshold %.2f".format(nmsThreshold), color = LabMuted); Slider(nmsThreshold, { nmsThreshold = it }, valueRange = .1f..1f)
            DetectionCanvas(detections, PracticalAlgorithms.nonMaximumSuppression(detections, nmsThreshold))
            Text("IoU(box 1, box 2) = %.3f; ${PracticalAlgorithms.nonMaximumSuppression(detections, nmsThreshold).size}/${detections.size} boxes retained. These boxes are an explicitly educational NMS input, not model detections.".format(PracticalAlgorithms.iou(detections[0], detections[1])), color = LabOrange, fontSize = 11.sp)
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Live Camera", "Lifecycle-bound CameraX analysis")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { SegmentedOption(if (cameraPermission) if (cameraEnabled) "Stop camera" else "Start camera" else "Grant camera", false, Modifier.weight(1f)) { if (!cameraPermission) cameraPermissionLauncher.launch(Manifest.permission.CAMERA) else cameraEnabled = !cameraEnabled }; MetricPill("Frame policy", "10 FPS", LabCyan, Modifier.weight(1f)) }
            if (cameraPermission && cameraEnabled) CameraPreview { cameraTelemetry = it }
            cameraTelemetry?.let { Text("Frames ${it.frames} - luma %.1f - analysis %.3f ms - dropped ${it.droppedFrames}".format(it.averageLuma, it.analysisMillis), color = LabGreen) }
            Text("Camera frames are analyzed with keep-latest backpressure. Rotation remains owned by CameraX; model inference requires a compatible imported image tensor.", color = LabMuted, fontSize = 11.sp)
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Audio AI", "Microphone PCM -> window -> spectrogram")
            GradientButton(if (audioPermission) "Capture 750 ms" else "Grant microphone", Modifier.fillMaxWidth()) { if (!audioPermission) audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO) else scope.launch { runCatching { withContext(Dispatchers.Default) { MicrophoneSampler.capture() } }.onSuccess { audio = it; spectrogram = PracticalAlgorithms.spectrogram(it) }.onFailure { onMessage("Audio capture failed: ${it.message}") } } }
            audio?.let { samples -> val rms = kotlin.math.sqrt(samples.sumOf { (it * it).toDouble() } / samples.size); Text("${samples.size} samples at 16 kHz - RMS %.4f".format(rms), color = LabCyan) }
            spectrogram?.let { SpectrogramCanvas(it); Text("Hann-windowed DFT; log magnitude. Bright columns indicate stronger frequency energy.", color = LabMuted, fontSize = 11.sp) }
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Text AI", "Tokenizer, embeddings and semantic similarity")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { listOf("private", "fast", "offline").forEach { SegmentedOption(it, false, Modifier.weight(1f)) { text += " $it" } } }
            Text(text, color = LabText)
            val vocabulary = remember { listOf("[pad]", "[unk]", "on", "device", "ai", "keeps", "private", "data", "local", "fast", "offline").withIndex().associate { it.value to it.index } }
            val tokens = PracticalAlgorithms.tokenize(text, vocabulary)
            Text(tokens.joinToString { "${it.first}:${it.second}" }, color = LabCyan, fontSize = 11.sp)
            val first = textEmbedding(tokens.map { it.second }); val second = textEmbedding(PracticalAlgorithms.tokenize("private offline ai", vocabulary).map { it.second })
            Text("Cosine similarity to 'private offline ai': %.3f".format(PracticalAlgorithms.cosineSimilarity(first, second)), color = LabGreen)
            Text("Special tokens and model-specific token IDs come from imported model metadata when available; this local vocabulary is explicitly the offline tokenizer explorer.", color = LabMuted, fontSize = 11.sp)
        } }
    }
}

@Composable
private fun CameraPreview(onTelemetry: (CameraTelemetry) -> Unit) {
    val context = LocalContext.current; val lifecycleOwner = LocalLifecycleOwner.current
    val executor = remember { Executors.newSingleThreadExecutor() }
    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    DisposableEffect(lifecycleOwner) {
        val future = ProcessCameraProvider.getInstance(context)
        val listener = Runnable {
            runCatching {
                val provider = future.get(); val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
                val analysis = ImageAnalysis.Builder().setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build().also { it.setAnalyzer(executor, LumaFrameAnalyzer(100, onTelemetry)) }
                provider.unbindAll(); provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            }
        }
        future.addListener(listener, ContextCompat.getMainExecutor(context))
        onDispose { runCatching { future.get().unbindAll() }; executor.shutdown() }
    }
    AndroidView({ previewView }, Modifier.fillMaxWidth().height(280.dp).border(1.dp, LabBorder, RoundedCornerShape(6.dp)))
}

@Composable
private fun OptimizePage(context: Context, loaded: LoadedModel?, backend: InferenceBackend?, capabilities: DeviceAiCapabilities, benchmark: BenchmarkSummary?, onBenchmark: () -> Unit) {
    var scale by remember { mutableFloatStateOf(.02f) }; var threads by remember { mutableIntStateOf(2) }
    val source = floatArrayOf(-1f, -.5f, 0f, .3f, .9f, 1.4f); val quantized = PracticalAlgorithms.quantize(source, scale, 0); val restored = PracticalAlgorithms.dequantize(quantized, scale, 0)
    val power = remember { context.getSystemService(PowerManager::class.java) }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Quantization Lab", "real = scale x (quantized - zero point)")
            Text("Scale %.4f".format(scale), color = LabMuted); Slider(scale, { scale = it.coerceAtLeast(.001f) }, valueRange = .001f..1f)
            Text("FP32 ${source.joinToString { "%.2f".format(it) }}", color = LabCyan); Text("INT8 ${quantized.joinToString { it.toString() }}", color = LabOrange); Text("Restored ${restored.joinToString { "%.2f".format(it) }}", color = LabGreen)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(7.dp)) { MetricPill("FP32", "${source.size * 4} B", LabCyan, Modifier.weight(1f)); MetricPill("FP16", "${source.size * 2} B", LabPurple, Modifier.weight(1f)); MetricPill("INT8", "${source.size} B", LabOrange, Modifier.weight(1f)); MetricPill("MAE", "%.4f".format(PracticalAlgorithms.meanAbsoluteError(source, restored)), LabPink, Modifier.weight(1f)) }
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            SectionTitle("Benchmark", "Cold run + 20 warm runs on the active backend")
            Text("Thread comparison selection: $threads", color = LabMuted); Slider(threads.toFloat(), { threads = it.toInt() }, valueRange = 1f..8f, steps = 6)
            GradientButton("Run reproducible benchmark", Modifier.fillMaxWidth()) { if (loaded != null && backend != null) onBenchmark() }
            if (loaded == null) Text("Load a model first. Benchmark numbers are never synthesized.", color = LabMuted)
            benchmark?.let { Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { MetricPill("Cold", "%.2f ms".format(it.coldMillis), LabOrange, Modifier.weight(1f)); MetricPill("p50", "%.2f".format(it.p50), LabCyan, Modifier.weight(1f)); MetricPill("p90", "%.2f".format(it.p90), LabPurple, Modifier.weight(1f)); MetricPill("p95", "%.2f".format(it.p95), LabPink, Modifier.weight(1f)) }; LatencyCanvas(it.warmSamples) }
            Text("Warm-up, sample count, active backend and execution target are recorded. Results are device-specific and not universal hardware claims.", color = LabMuted, fontSize = 11.sp)
        } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
            SectionTitle("Memory and Thermal", "Runtime-observed process state")
            val runtime = Runtime.getRuntime(); val used = runtime.totalMemory() - runtime.freeMemory()
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) { MetricPill("Heap used", formatBytes(used), LabCyan, Modifier.weight(1f)); MetricPill("Heap max", formatBytes(runtime.maxMemory()), LabPurple, Modifier.weight(1f)); MetricPill("Thermal", thermalLabel(power.currentThermalStatus), LabOrange, Modifier.weight(1f)) }
            Text("Battery-aware mode recommendation: reduce camera FPS, threads and benchmark repetitions when thermal status is moderate or above.", color = LabMuted, fontSize = 11.sp)
            Text("GPU ${if (capabilities.gpuDelegate) "available" else "unavailable"}; NNAPI ${if (capabilities.nnapi) "available" else "unavailable"}. Availability does not prove a specific model is fully delegated.", color = LabMuted, fontSize = 11.sp)
        } }
    }
}

@Composable
private fun InspectPage(descriptor: ModelDescriptor?, result: InferenceResult?, capabilities: DeviceAiCapabilities) {
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        if (descriptor == null) GlassPanel(Modifier.fillMaxWidth()) { SectionTitle("Model Inspector", "Import and validate a model to inspect real tensors") }
        descriptor?.let { model ->
            GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("Model Metadata", model.name); Text("Format ${model.format} - ${formatBytes(model.sizeBytes)} - ${model.inputs.size} input(s) - ${model.outputs.size} output(s)", color = LabCyan); model.metadata.forEach { (key, value) -> Text("$key: $value", color = LabMuted, fontSize = 12.sp) } } }
            GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("Input Tensors", "Shapes, types, quantization and theoretical memory"); model.inputs.forEachIndexed { index, spec -> TensorInspector("Input $index", spec) } } }
            GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("Output Tensors", "Raw model outputs"); model.outputs.forEachIndexed { index, spec -> TensorInspector("Output $index", spec); result?.outputs?.getOrNull(index)?.let { Text(tensorPreview(it), color = LabGreen, fontSize = 11.sp) } } } }
            GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(7.dp)) { SectionTitle("Compatibility", "Runtime-level checks without invented operator claims"); Text("CPU: supported by bundled runtime", color = LabGreen); Text("NNAPI: ${if (capabilities.nnapi) "runtime available; model compatibility determined during load" else "not available"}", color = LabMuted); Text("GPU: ${if (capabilities.gpuDelegate) "delegate available; model compatibility determined during load" else "not available"}", color = LabMuted); Text("Operator graph enumeration is not exposed by the selected public Java runtime API. No operator list is fabricated.", color = LabPink, fontSize = 11.sp) } }
        }
    }
}

@Composable private fun TensorInspector(label: String, spec: TensorSpec) { Column(Modifier.fillMaxWidth().background(Color(0xFF081126), RoundedCornerShape(6.dp)).padding(8.dp), verticalArrangement = Arrangement.spacedBy(3.dp)) { Text("$label - ${spec.name}", color = LabText, fontWeight = FontWeight.Bold); Text("Shape ${spec.shapeText()} - ${spec.type} - ${formatBytes(spec.memoryBytes)}", color = LabCyan, fontSize = 11.sp); Text(spec.quantization?.let { "Quantization scale ${it.scale}, zero point ${it.zeroPoint}" } ?: "Quantization: none exposed", color = LabMuted, fontSize = 11.sp) } }

@Composable
private fun CurriculumPage() {
    var trained by remember { mutableStateOf(emptyList<com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine.QLearningSnapshot>()) }; val grid = remember { GridWorld() }
    var timeWindow by remember { mutableIntStateOf(3) }; val series = floatArrayOf(2f, 2.5f, 3.1f, 2.8f, 3.5f, 4.2f)
    val classifier = remember { EmbeddingClassifier(4).apply { add(floatArrayOf(1f, 0f, .2f, 0f), "local"); add(floatArrayOf(0f, 1f, 0f, .2f), "cloud") } }
    Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("On-Device AI Learning Path", "Train vs inference - latency vs throughput - precision - deployment"); listOf("1  Inspect model tensors and preprocessing", "2  Run CPU baseline and record cold/warm latency", "3  Compare supported delegates without hiding fallback", "4  Inspect quantization error and model memory", "5  Review confidence, calibration, OOD and domain shift").forEach { Text(it, color = LabMuted, fontSize = 12.sp) } } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("Reinforcement Learning", "Grid World Q-learning closes the curriculum gap"); GradientButton("Train Q-learning", Modifier.fillMaxWidth()) { trained = grid.train() }; if (trained.isNotEmpty()) { Text("Episode ${trained.last().episode} - reward %.2f - epsilon %.2f".format(trained.last().reward, trained.last().epsilon), color = LabGreen); QValueCanvas(grid.qValues) }; Text("Multi-armed bandits balance exploration and exploitation; Q-learning extends this idea to state-action values.", color = LabMuted, fontSize = 11.sp) } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("Time-Series Foundation", "Windowed forecasting and temporal validation"); Text("Window $timeWindow", color = LabMuted); Slider(timeWindow.toFloat(), { timeWindow = it.toInt() }, valueRange = 1f..5f, steps = 3); Text("Series ${series.joinToString()} -> next forecast %.3f".format(movingAverageForecast(series, timeWindow)), color = LabCyan); Text("Random train/test splits can leak future information. Temporal validation must preserve order.", color = LabPink, fontSize = 11.sp) } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(8.dp)) { SectionTitle("Personalization", "Embedding-based head trained on user-owned features"); val prediction = classifier.predict(floatArrayOf(.8f, .1f, .2f, 0f)); Text("2 stored examples - query predicts ${prediction.first} with cosine %.3f".format(prediction.second), color = LabGreen); Text("Only the lightweight classifier head changes; the feature extractor remains frozen. Export should include labels, dimensions and preprocessing metadata.", color = LabMuted, fontSize = 11.sp) } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(7.dp)) { SectionTitle("Reliability and Error Analysis", "Confidence is not correctness"); val ece = PracticalAlgorithms.expectedCalibrationError(floatArrayOf(.95f, .8f, .6f, .55f), booleanArrayOf(true, false, true, false), 4); Text("Calibration example ECE %.3f".format(ece), color = LabOrange); Text("Review high-confidence errors, class imbalance, domain shift and out-of-distribution inputs. Small perturbations can change predictions; this is a robustness lesson, not an attack toolkit.", color = LabMuted, fontSize = 11.sp) } }
        GlassPanel(Modifier.fillMaxWidth()) { Column(verticalArrangement = Arrangement.spacedBy(7.dp)) { SectionTitle("Final Coverage Audit", "Learn - Train - Understand - Deploy - Optimize"); Text("Classical ML, neural networks, CNN, sequence models, Transformers, ViT, GNN, generative models, practical deployment, reinforcement learning and time series are represented in the final taxonomy.", color = LabCyan); Text("Core learning and educational experiments remain offline. Imported models and media never leave the device; the application requests camera or microphone only at the moment of use.", color = LabGreen, fontSize = 11.sp) } }
    }
}

@Composable private fun ImageThresholdCanvas(bitmap: Bitmap, opacity: Float) { val sample = remember(bitmap) { Bitmap.createScaledBitmap(bitmap, 24, 24, true).let { scaled -> IntArray(24 * 24).also { scaled.getPixels(it, 0, 24, 0, 0, 24, 24) }.also { if (scaled !== bitmap) scaled.recycle() } } }; Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(6.dp))) { val cellW = size.width / 24; val cellH = size.height / 24; sample.forEachIndexed { index, color -> val luma = ((color shr 16 and 0xff) + (color shr 8 and 0xff) + (color and 0xff)) / 765f; drawRect(if (luma > .5f) LabCyan.copy(alpha = opacity) else LabPink.copy(alpha = opacity * .35f), Offset(index % 24 * cellW, index / 24 * cellH), androidx.compose.ui.geometry.Size(cellW + .5f, cellH + .5f)) } } }
@Composable private fun DetectionCanvas(all: List<Detection>, kept: List<Detection>) { Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(6.dp))) { all.forEach { box -> val color = if (box in kept) LabGreen else LabMuted; drawRect(color, Offset(box.left * size.width, box.top * size.height), androidx.compose.ui.geometry.Size((box.right - box.left) * size.width, (box.bottom - box.top) * size.height), style = Stroke(if (box in kept) 6f else 2f)) } } }
@Composable private fun SpectrogramCanvas(values: Array<FloatArray>) { Canvas(Modifier.fillMaxWidth().height(180.dp).background(Color(0xFF081126), RoundedCornerShape(6.dp))) { val columns = values.size; val rows = values.first().size; val peak = values.maxOf { frame -> frame.max() }.coerceAtLeast(1e-6f); val cellW = size.width / columns; val cellH = size.height / rows; values.forEachIndexed { column, frame -> frame.forEachIndexed { row, value -> drawRect(Color(value / peak, .2f, 1f - value / peak, 1f), Offset(column * cellW, size.height - (row + 1) * cellH), androidx.compose.ui.geometry.Size(cellW + 1f, cellH + 1f)) } } } }
@Composable private fun LatencyCanvas(values: List<Double>) { Canvas(Modifier.fillMaxWidth().height(140.dp).background(Color(0xFF081126), RoundedCornerShape(6.dp))) { val peak = values.max().coerceAtLeast(.001); val width = size.width / values.size; values.forEachIndexed { index, value -> drawRect(LabCyan, Offset(index * width, (size.height * (1 - value / peak)).toFloat()), androidx.compose.ui.geometry.Size(width - 2f, (size.height * value / peak).toFloat())) } } }
@Composable private fun QValueCanvas(qValues: Array<FloatArray>) { Canvas(Modifier.fillMaxWidth().height(220.dp).background(Color(0xFF081126), RoundedCornerShape(6.dp))) { val grid = kotlin.math.sqrt(qValues.size.toFloat()).toInt(); val width = size.width / grid; val height = size.height / grid; qValues.forEachIndexed { index, actions -> val center = Offset((index % grid + .5f) * width, (index / grid + .5f) * height); val action = actions.indices.maxBy { actions[it] }; val end = when (action) { 0 -> center + Offset(0f, -height * .3f); 1 -> center + Offset(width * .3f, 0f); 2 -> center + Offset(0f, height * .3f); else -> center + Offset(-width * .3f, 0f) }; drawRect(LabBorder, Offset(index % grid * width, index / grid * height), androidx.compose.ui.geometry.Size(width - 2f, height - 2f), style = Stroke(2f)); drawLine(LabGreen, center, end, 5f) } } }

private fun textEmbedding(tokenIds: List<Int>): FloatArray = FloatArray(8) { dimension -> if (tokenIds.isEmpty()) 0f else tokenIds.sumOf { (sin((it + 1) * (dimension + 1.0)) + cos((it + 1) / (dimension + 1.0))).toDouble() }.toFloat() / tokenIds.size }
private fun formatBytes(bytes: Long): String = when { bytes >= 1_048_576 -> "%.1f MB".format(bytes / 1_048_576.0); bytes >= 1024 -> "%.1f KB".format(bytes / 1024.0); else -> "$bytes B" }
private fun thermalLabel(status: Int) = when (status) { PowerManager.THERMAL_STATUS_NONE -> "None"; PowerManager.THERMAL_STATUS_LIGHT -> "Light"; PowerManager.THERMAL_STATUS_MODERATE -> "Moderate"; PowerManager.THERMAL_STATUS_SEVERE -> "Severe"; PowerManager.THERMAL_STATUS_CRITICAL -> "Critical"; PowerManager.THERMAL_STATUS_EMERGENCY -> "Emergency"; PowerManager.THERMAL_STATUS_SHUTDOWN -> "Shutdown"; else -> "Not exposed" }
