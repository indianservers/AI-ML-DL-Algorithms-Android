package com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine

import java.io.File

enum class ModelFormat { LiteRT, ONNX, Educational }
enum class ExecutionTarget { CPU, GPU, NNAPI }
enum class TensorElementType(val bytes: Int) { Float32(4), Float16(2), Int64(8), Int32(4), UInt8(1), Int8(1), Bool(1), Unknown(0) }

data class QuantizationInfo(val scale: Float, val zeroPoint: Int) {
    val enabled get() = scale > 0f
}

data class TensorSpec(
    val name: String,
    val shape: LongArray,
    val type: TensorElementType,
    val quantization: QuantizationInfo? = null,
) {
    val elementCount: Int get() = shape.fold(1L) { total, value -> total * value.coerceAtLeast(1L) }.coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
    val memoryBytes: Long get() = elementCount.toLong() * type.bytes
    fun shapeText() = shape.joinToString(prefix = "[", postfix = "]")
}

data class ModelSource(val file: File, val displayName: String, val format: ModelFormat)
data class BackendConfig(val execution: ExecutionTarget = ExecutionTarget.CPU, val threads: Int = 2)

sealed interface TensorData {
    val shape: LongArray
    data class Floats(val values: FloatArray, override val shape: LongArray) : TensorData
    data class Ints(val values: IntArray, override val shape: LongArray) : TensorData
    data class Longs(val values: LongArray, override val shape: LongArray) : TensorData
    data class Bytes(val values: ByteArray, override val shape: LongArray) : TensorData
}

data class ModelDescriptor(
    val id: String,
    val name: String,
    val format: ModelFormat,
    val sizeBytes: Long,
    val inputs: List<TensorSpec>,
    val outputs: List<TensorSpec>,
    val metadata: Map<String, String>,
)

interface LoadedModel {
    val descriptor: ModelDescriptor
    val backendName: String
    val execution: ExecutionTarget
}

data class InferenceResult(
    val outputs: List<TensorData>,
    val latencyNanos: Long,
    val backendName: String,
    val execution: ExecutionTarget,
) {
    val latencyMillis get() = latencyNanos / 1_000_000.0
}

interface InferenceBackend {
    val name: String
    suspend fun loadModel(source: ModelSource, config: BackendConfig): LoadedModel
    suspend fun run(model: LoadedModel, inputs: List<TensorData>): InferenceResult
    fun close(model: LoadedModel)
}

data class DeviceAiCapabilities(
    val cpu: Boolean,
    val liteRt: Boolean,
    val onnxRuntime: Boolean,
    val nnapi: Boolean,
    val gpuDelegate: Boolean,
    val acceleratorDetail: String,
)

data class StoredModel(
    val id: String,
    val name: String,
    val format: ModelFormat,
    val path: String,
    val bytes: Long,
    val importedAt: Long,
    val favorite: Boolean = false,
    val tags: Set<String> = emptySet(),
)

data class BenchmarkSummary(
    val coldMillis: Double,
    val warmSamples: List<Double>,
    val p50: Double,
    val p90: Double,
    val p95: Double,
    val mean: Double,
    val throughputPerSecond: Double,
)
