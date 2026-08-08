package com.indianservers.ai_ml_dl_algorithms.ml_lab.phase5.engine

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max
import kotlin.math.min

data class ImagePreprocessResult(
    val tensor: TensorData,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val targetWidth: Int,
    val targetHeight: Int,
    val mean: Float,
    val standardDeviation: Float,
    val elapsedNanos: Long,
)

object ImageTensorPreprocessor {
    fun preprocess(bitmap: Bitmap, spec: TensorSpec, mean: Float = 127.5f, standardDeviation: Float = 127.5f): ImagePreprocessResult {
        require(spec.shape.size == 4) { "Image input must be rank 4" }
        val channelFirst = spec.shape[1] in 1..4 && spec.shape[3] !in 1..4
        val height = (if (channelFirst) spec.shape[2] else spec.shape[1]).toInt().coerceIn(1, 2048)
        val width = (if (channelFirst) spec.shape[3] else spec.shape[2]).toInt().coerceIn(1, 2048)
        val channels = (if (channelFirst) spec.shape[1] else spec.shape[3]).toInt().coerceIn(1, 4)
        lateinit var tensor: TensorData
        val elapsed = kotlin.system.measureNanoTime {
            val resized = Bitmap.createScaledBitmap(bitmap, width, height, true)
            val pixels = IntArray(width * height); resized.getPixels(pixels, 0, width, 0, 0, width, height)
            val values = FloatArray(width * height * channels)
            pixels.forEachIndexed { pixelIndex, color ->
                val rgb = floatArrayOf((color shr 16 and 0xff).toFloat(), (color shr 8 and 0xff).toFloat(), (color and 0xff).toFloat(), (color ushr 24).toFloat())
                repeat(channels) { channel ->
                    val outputIndex = if (channelFirst) channel * width * height + pixelIndex else pixelIndex * channels + channel
                    values[outputIndex] = (rgb[channel] - mean) / standardDeviation.coerceAtLeast(1e-7f)
                }
            }
            tensor = when (spec.type) {
                TensorElementType.UInt8 -> TensorData.Bytes(PracticalAlgorithms.quantize(values, spec.quantization?.scale ?: (1f / 255f), spec.quantization?.zeroPoint ?: 0, signed = false), spec.shape)
                TensorElementType.Int8 -> TensorData.Bytes(PracticalAlgorithms.quantize(values, spec.quantization?.scale ?: (1f / 128f), spec.quantization?.zeroPoint ?: 0, signed = true), spec.shape)
                else -> TensorData.Floats(values, spec.shape)
            }
            if (resized !== bitmap) resized.recycle()
        }
        return ImagePreprocessResult(tensor, bitmap.width, bitmap.height, width, height, mean, standardDeviation, elapsed)
    }
}

data class CameraTelemetry(val frames: Long, val averageLuma: Float, val analysisMillis: Double, val droppedFrames: Long)

class LumaFrameAnalyzer(
    private val minIntervalMillis: Long,
    private val onTelemetry: (CameraTelemetry) -> Unit,
) : ImageAnalysis.Analyzer {
    private val frames = AtomicLong(0); private val dropped = AtomicLong(0); private var lastAccepted = 0L
    override fun analyze(image: ImageProxy) {
        val start = System.nanoTime(); val now = System.currentTimeMillis()
        if (now - lastAccepted < minIntervalMillis) { dropped.incrementAndGet(); image.close(); return }
        lastAccepted = now
        val buffer = image.planes.first().buffer
        var sum = 0L; var count = 0
        while (buffer.hasRemaining()) { sum += buffer.get().toInt() and 0xff; count++ }
        val elapsed = System.nanoTime() - start
        onTelemetry(CameraTelemetry(frames.incrementAndGet(), if (count == 0) 0f else sum.toFloat() / count, elapsed / 1_000_000.0, dropped.get()))
        image.close()
    }
}

object MicrophoneSampler {
    const val sampleRate = 16_000

    @SuppressLint("MissingPermission")
    fun capture(milliseconds: Int = 750): FloatArray {
        require(milliseconds in 100..5_000)
        val requestedSamples = sampleRate * milliseconds / 1000
        val minimum = AudioRecord.getMinBufferSize(sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        require(minimum > 0) { "Audio input configuration is unsupported" }
        val recorder = AudioRecord(MediaRecorder.AudioSource.DEFAULT, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, max(minimum, requestedSamples * 2))
        require(recorder.state == AudioRecord.STATE_INITIALIZED) { "Microphone could not be initialized" }
        return try {
            val pcm = ShortArray(requestedSamples); var offset = 0
            recorder.startRecording()
            while (offset < pcm.size) {
                val read = recorder.read(pcm, offset, pcm.size - offset, AudioRecord.READ_BLOCKING)
                if (read <= 0) throw IllegalStateException("Microphone read failed: $read")
                offset += read
            }
            FloatArray(pcm.size) { pcm[it] / 32768f }
        } finally {
            runCatching { recorder.stop() }; recorder.release()
        }
    }
}

fun zeroTensor(spec: TensorSpec): TensorData = when (spec.type) {
    TensorElementType.Int64 -> TensorData.Longs(LongArray(spec.elementCount), spec.shape)
    TensorElementType.Int32 -> TensorData.Ints(IntArray(spec.elementCount), spec.shape)
    TensorElementType.UInt8, TensorElementType.Int8, TensorElementType.Bool -> TensorData.Bytes(ByteArray(spec.elementCount), spec.shape)
    else -> TensorData.Floats(FloatArray(spec.elementCount), spec.shape)
}

fun sampleTensor(spec: TensorSpec): TensorData = when (spec.type) {
    TensorElementType.Int64 -> TensorData.Longs(LongArray(spec.elementCount) { it.toLong() }, spec.shape)
    TensorElementType.Int32 -> TensorData.Ints(IntArray(spec.elementCount) { it }, spec.shape)
    TensorElementType.UInt8, TensorElementType.Int8, TensorElementType.Bool -> TensorData.Bytes(ByteArray(spec.elementCount) { it.toByte() }, spec.shape)
    else -> TensorData.Floats(FloatArray(spec.elementCount) { it.toFloat() }, spec.shape)
}

fun tensorPreview(data: TensorData, limit: Int = 12): String = when (data) {
    is TensorData.Floats -> data.values.take(limit).joinToString(prefix = "[", postfix = if (data.values.size > limit) ", ...]" else "]") { "%.4f".format(it) }
    is TensorData.Ints -> data.values.take(limit).joinToString(prefix = "[", postfix = if (data.values.size > limit) ", ...]" else "]")
    is TensorData.Longs -> data.values.take(limit).joinToString(prefix = "[", postfix = if (data.values.size > limit) ", ...]" else "]")
    is TensorData.Bytes -> data.values.take(limit).joinToString(prefix = "[", postfix = if (data.values.size > limit) ", ...]" else "]") { (it.toInt() and 0xff).toString() }
}

fun outputFloats(data: TensorData, quantization: QuantizationInfo? = null): FloatArray = when (data) {
    is TensorData.Floats -> data.values
    is TensorData.Ints -> data.values.map(Int::toFloat).toFloatArray()
    is TensorData.Longs -> data.values.map(Long::toFloat).toFloatArray()
    is TensorData.Bytes -> if (quantization != null) PracticalAlgorithms.dequantize(data.values, quantization.scale, quantization.zeroPoint, signed = true) else FloatArray(data.values.size) { data.values[it].toInt().toFloat() }
}
