package com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine

import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.net.Uri
import android.os.Build
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.channels.FileChannel
import java.security.MessageDigest
import kotlin.system.measureNanoTime

private class LiteRtLoadedModel(
    override val descriptor: ModelDescriptor,
    override val execution: ExecutionTarget,
    val interpreter: Interpreter,
    val delegate: GpuDelegate?,
) : LoadedModel { override val backendName = "LiteRT" }

class LiteRtBackend : InferenceBackend {
    override val name = "LiteRT"

    override suspend fun loadModel(source: ModelSource, config: BackendConfig): LoadedModel {
        require(source.format == ModelFormat.LiteRT)
        val options = Interpreter.Options().setNumThreads(config.threads.coerceIn(1, 8))
        var delegate: GpuDelegate? = null
        val actualExecution = when (config.execution) {
            ExecutionTarget.CPU -> ExecutionTarget.CPU
            ExecutionTarget.NNAPI -> {
                options.setUseNNAPI(true)
                ExecutionTarget.NNAPI
            }
            ExecutionTarget.GPU -> {
                val compatibility = CompatibilityList()
                if (!compatibility.isDelegateSupportedOnThisDevice) throw IllegalStateException("LiteRT GPU delegate is not supported on this device")
                delegate = GpuDelegate(compatibility.bestOptionsForThisDevice)
                options.addDelegate(delegate)
                ExecutionTarget.GPU
            }
        }
        val buffer = FileInputStream(source.file).channel.use { channel -> channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()) }
        val interpreter = try { Interpreter(buffer, options) } catch (error: Throwable) { delegate?.close(); throw IllegalArgumentException("Unable to parse LiteRT model: ${error.message}", error) }
        val inputs = (0 until interpreter.inputTensorCount).map { interpreter.getInputTensor(it).toSpec() }
        val outputs = (0 until interpreter.outputTensorCount).map { interpreter.getOutputTensor(it).toSpec() }
        return LiteRtLoadedModel(
            descriptor = descriptor(source, inputs, outputs, mapOf("Runtime" to "LiteRT Interpreter", "Threads" to config.threads.toString())),
            execution = actualExecution,
            interpreter = interpreter,
            delegate = delegate,
        )
    }

    override suspend fun run(model: LoadedModel, inputs: List<TensorData>): InferenceResult {
        val loaded = model as? LiteRtLoadedModel ?: error("Model was not loaded by LiteRT")
        require(inputs.size == loaded.descriptor.inputs.size) { "Expected ${loaded.descriptor.inputs.size} input tensors, received ${inputs.size}" }
        val inputObjects = inputs.map(::toNativeBuffer).toTypedArray()
        val outputBuffers = loaded.descriptor.outputs.associateBy({ loaded.descriptor.outputs.indexOf(it) }, ::allocateBuffer).toMutableMap<Int, Any>()
        val elapsed = measureNanoTime { loaded.interpreter.runForMultipleInputsOutputs(inputObjects, outputBuffers) }
        val outputs = loaded.descriptor.outputs.mapIndexed { index, spec -> readBuffer(outputBuffers.getValue(index) as ByteBuffer, spec) }
        return InferenceResult(outputs, elapsed, name, loaded.execution)
    }

    override fun close(model: LoadedModel) {
        (model as? LiteRtLoadedModel)?.let { it.interpreter.close(); it.delegate?.close() }
    }

    private fun org.tensorflow.lite.Tensor.toSpec(): TensorSpec {
        val params = quantizationParams()
        val quantization = if (params.scale > 0f) QuantizationInfo(params.scale, params.zeroPoint) else null
        return TensorSpec(name(), shape().map(Int::toLong).toLongArray(), dataType().toElementType(), quantization)
    }
}

private class OnnxLoadedModel(
    override val descriptor: ModelDescriptor,
    override val execution: ExecutionTarget,
    val environment: OrtEnvironment,
    val session: OrtSession,
) : LoadedModel { override val backendName = "ONNX Runtime" }

class OnnxRuntimeBackend : InferenceBackend {
    override val name = "ONNX Runtime"

    override suspend fun loadModel(source: ModelSource, config: BackendConfig): LoadedModel {
        require(source.format == ModelFormat.ONNX)
        val environment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(config.threads.coerceIn(1, 8))
            if (config.execution == ExecutionTarget.NNAPI) {
                val method = javaClass.methods.firstOrNull { it.name == "addNnapi" && it.parameterCount == 0 }
                    ?: throw IllegalStateException("This ONNX Runtime build does not expose NNAPI")
                method.invoke(this)
            }
            if (config.execution == ExecutionTarget.GPU) throw IllegalStateException("ONNX GPU execution provider is not bundled in this Android build")
        }
        val session = try { environment.createSession(source.file.absolutePath, options) } catch (error: Throwable) { options.close(); throw IllegalArgumentException("Unable to parse ONNX model: ${error.message}", error) }
        options.close()
        val inputs = session.inputInfo.map { (name, info) -> info.toSpec(name) }
        val outputs = session.outputInfo.map { (name, info) -> info.toSpec(name) }
        return OnnxLoadedModel(
            descriptor(source, inputs, outputs, mapOf("Runtime" to (OrtEnvironment::class.java.`package`?.implementationVersion ?: "Bundled ONNX Runtime"), "Threads" to config.threads.toString())),
            config.execution,
            environment,
            session,
        )
    }

    override suspend fun run(model: LoadedModel, inputs: List<TensorData>): InferenceResult {
        val loaded = model as? OnnxLoadedModel ?: error("Model was not loaded by ONNX Runtime")
        require(inputs.size == loaded.descriptor.inputs.size)
        val tensors = loaded.descriptor.inputs.mapIndexed { index, spec -> spec.name to createOnnxTensor(loaded.environment, inputs[index]) }
        try {
            lateinit var result: OrtSession.Result
            val elapsed = measureNanoTime { result = loaded.session.run(tensors.toMap()) }
            result.use { values ->
                val outputs = (0 until values.size()).map { index ->
                    val spec = loaded.descriptor.outputs[index]
                    val flattened = mutableListOf<Number>()
                    flattenNumbers(values[index].value, flattened)
                    when (spec.type) {
                        TensorElementType.Int64 -> TensorData.Longs(flattened.map(Number::toLong).toLongArray(), spec.shape)
                        TensorElementType.Int32 -> TensorData.Ints(flattened.map(Number::toInt).toIntArray(), spec.shape)
                        TensorElementType.UInt8, TensorElementType.Int8, TensorElementType.Bool -> TensorData.Bytes(flattened.map(Number::toByte).toByteArray(), spec.shape)
                        else -> TensorData.Floats(flattened.map(Number::toFloat).toFloatArray(), spec.shape)
                    }
                }
                return InferenceResult(outputs, elapsed, name, loaded.execution)
            }
        } finally {
            tensors.forEach { it.second.close() }
        }
    }

    override fun close(model: LoadedModel) { (model as? OnnxLoadedModel)?.session?.close() }

    private fun NodeInfo.toSpec(name: String): TensorSpec {
        val tensor = info as? TensorInfo ?: return TensorSpec(name, longArrayOf(1), TensorElementType.Unknown)
        return TensorSpec(name, tensor.shape, tensor.type.toElementType())
    }
}

class ModelRepository(private val context: Context) {
    private val directory = File(context.filesDir, "models").apply { mkdirs() }
    private val preferences = context.getSharedPreferences("phase5_models", Context.MODE_PRIVATE)

    init { installBundled("tiny_double.tflite", ModelFormat.LiteRT); installBundled("tiny_double.onnx", ModelFormat.ONNX) }

    fun import(uri: Uri): StoredModel {
        val displayName = queryName(uri) ?: "model_${System.currentTimeMillis()}"
        val format = when (displayName.substringAfterLast('.', "").lowercase()) {
            "tflite" -> ModelFormat.LiteRT
            "onnx" -> ModelFormat.ONNX
            else -> throw IllegalArgumentException("Only .tflite and .onnx files are supported")
        }
        val temporary = File(directory, "importing_${System.nanoTime()}")
        context.contentResolver.openInputStream(uri)?.use { input -> temporary.outputStream().use(input::copyTo) }
            ?: throw IllegalArgumentException("The selected document is not readable")
        require(temporary.length() in 1..250_000_000) { "Model must be between 1 byte and 250 MB" }
        val id = sha256(temporary).take(16)
        val destination = File(directory, "$id.${if (format == ModelFormat.LiteRT) "tflite" else "onnx"}")
        if (!temporary.renameTo(destination)) { temporary.copyTo(destination, overwrite = true); temporary.delete() }
        val stored = StoredModel(id, displayName, format, destination.absolutePath, destination.length(), System.currentTimeMillis())
        save(stored)
        return stored
    }

    fun list(): List<StoredModel> = preferences.all.keys.mapNotNull(::read).sortedByDescending(StoredModel::importedAt)
    fun source(model: StoredModel) = ModelSource(File(model.path), model.name, model.format)
    fun toggleFavorite(model: StoredModel): StoredModel = model.copy(favorite = !model.favorite).also(::save)
    fun delete(model: StoredModel) { File(model.path).delete(); preferences.edit().remove(model.id).apply() }

    private fun save(model: StoredModel) {
        val encodedName = android.util.Base64.encodeToString(model.name.toByteArray(), android.util.Base64.NO_WRAP)
        preferences.edit().putString(model.id, listOf(encodedName, model.format.name, model.path, model.bytes, model.importedAt, model.favorite, model.tags.joinToString(",")).joinToString("|" )).apply()
    }

    private fun installBundled(name: String, format: ModelFormat) {
        val id = "bundled_${name.substringBeforeLast('.')}_${format.name.lowercase()}"
        if (read(id) != null) return
        val destination = File(directory, name)
        if (!destination.exists()) context.assets.open("models/$name").use { input -> destination.outputStream().use(input::copyTo) }
        save(StoredModel(id, "Tiny Double (${format.name})", format, destination.absolutePath, destination.length(), 0L, favorite = true, tags = setOf("bundled", "smoke-test")))
    }

    private fun read(id: String): StoredModel? = runCatching {
        val parts = preferences.getString(id, null)!!.split('|')
        StoredModel(id, String(android.util.Base64.decode(parts[0], android.util.Base64.NO_WRAP)), ModelFormat.valueOf(parts[1]), parts[2], parts[3].toLong(), parts[4].toLong(), parts[5].toBoolean(), parts.getOrElse(6) { "" }.split(',').filter(String::isNotBlank).toSet())
    }.getOrNull()?.takeIf { File(it.path).exists() }

    private fun queryName(uri: Uri): String? = context.contentResolver.query(uri, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }
    private fun sha256(file: File): String = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it) }
}

object CapabilityDetector {
    fun detect(): DeviceAiCapabilities {
        val gpu = runCatching { CompatibilityList().isDelegateSupportedOnThisDevice }.getOrDefault(false)
        return DeviceAiCapabilities(
            cpu = true,
            liteRt = runCatching { Class.forName("org.tensorflow.lite.Interpreter") }.isSuccess,
            onnxRuntime = runCatching { Class.forName("ai.onnxruntime.OrtEnvironment") }.isSuccess,
            nnapi = Build.VERSION.SDK_INT >= Build.VERSION_CODES.P,
            gpuDelegate = gpu,
            acceleratorDetail = "Dedicated NPU identity is not exposed by the current Android runtime",
        )
    }
}

private fun descriptor(source: ModelSource, inputs: List<TensorSpec>, outputs: List<TensorSpec>, metadata: Map<String, String>) = ModelDescriptor(
    id = source.file.nameWithoutExtension,
    name = source.displayName,
    format = source.format,
    sizeBytes = source.file.length(),
    inputs = inputs,
    outputs = outputs,
    metadata = metadata,
)

private fun DataType.toElementType() = when (this) {
    DataType.FLOAT32 -> TensorElementType.Float32
    DataType.INT32 -> TensorElementType.Int32
    DataType.INT64 -> TensorElementType.Int64
    DataType.UINT8 -> TensorElementType.UInt8
    DataType.INT8 -> TensorElementType.Int8
    DataType.BOOL -> TensorElementType.Bool
    else -> TensorElementType.Unknown
}

private fun OnnxJavaType.toElementType() = when (this) {
    OnnxJavaType.FLOAT -> TensorElementType.Float32
    OnnxJavaType.INT32 -> TensorElementType.Int32
    OnnxJavaType.INT64 -> TensorElementType.Int64
    OnnxJavaType.UINT8 -> TensorElementType.UInt8
    OnnxJavaType.INT8 -> TensorElementType.Int8
    OnnxJavaType.BOOL -> TensorElementType.Bool
    else -> TensorElementType.Unknown
}

private fun toNativeBuffer(data: TensorData): ByteBuffer {
    val bytes = when (data) {
        is TensorData.Floats -> data.values.size * 4
        is TensorData.Ints -> data.values.size * 4
        is TensorData.Longs -> data.values.size * 8
        is TensorData.Bytes -> data.values.size
    }
    return ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).also { buffer ->
        when (data) {
            is TensorData.Floats -> data.values.forEach(buffer::putFloat)
            is TensorData.Ints -> data.values.forEach(buffer::putInt)
            is TensorData.Longs -> data.values.forEach(buffer::putLong)
            is TensorData.Bytes -> buffer.put(data.values)
        }
        buffer.rewind()
    }
}

private fun allocateBuffer(spec: TensorSpec): ByteBuffer = ByteBuffer.allocateDirect((spec.memoryBytes.coerceAtLeast(1L)).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()).order(ByteOrder.nativeOrder())

private fun readBuffer(buffer: ByteBuffer, spec: TensorSpec): TensorData {
    buffer.rewind()
    return when (spec.type) {
        TensorElementType.Int64 -> TensorData.Longs(LongArray(spec.elementCount) { buffer.long }, spec.shape)
        TensorElementType.Int32 -> TensorData.Ints(IntArray(spec.elementCount) { buffer.int }, spec.shape)
        TensorElementType.UInt8, TensorElementType.Int8, TensorElementType.Bool -> TensorData.Bytes(ByteArray(spec.elementCount).also(buffer::get), spec.shape)
        else -> TensorData.Floats(FloatArray(spec.elementCount) { buffer.float }, spec.shape)
    }
}

private fun createOnnxTensor(environment: OrtEnvironment, data: TensorData): OnnxTensor = when (data) {
    is TensorData.Floats -> OnnxTensor.createTensor(environment, FloatBuffer.wrap(data.values), data.shape)
    is TensorData.Ints -> OnnxTensor.createTensor(environment, IntBuffer.wrap(data.values), data.shape)
    is TensorData.Longs -> OnnxTensor.createTensor(environment, LongBuffer.wrap(data.values), data.shape)
    is TensorData.Bytes -> OnnxTensor.createTensor(environment, ByteBuffer.wrap(data.values), data.shape, OnnxJavaType.UINT8)
}

private fun flattenNumbers(value: Any?, output: MutableList<Number>) {
    when (value) {
        null -> Unit
        is Number -> output += value
        is Boolean -> output += if (value) 1 else 0
        else -> if (value.javaClass.isArray) repeat(java.lang.reflect.Array.getLength(value)) { flattenNumbers(java.lang.reflect.Array.get(value, it), output) }
    }
}
