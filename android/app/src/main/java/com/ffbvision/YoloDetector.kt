package com.ffbvision

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

class YoloDetector(
    context: Context
) {

    companion object {
        private const val MODEL_NAME = "merged_ripeness_best.onnx"
        private const val INPUT_SIZE = 640
        private const val CONF_THRESHOLD = 0.25f
        private const val IOU_THRESHOLD = 0.45f
    }

    private val environment = OrtEnvironment.getEnvironment()

    private val session: OrtSession

    init {

        val modelBytes = context.assets
            .open(MODEL_NAME)
            .use { it.readBytes() }

        session = environment.createSession(
            modelBytes,
            OrtSession.SessionOptions()
        )
    }

    fun detect(bitmap: Bitmap): List<Detection> {

        val resized = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )

        val input = FloatArray(
            3 * INPUT_SIZE * INPUT_SIZE
        )

        var indexR = 0
        var indexG = INPUT_SIZE * INPUT_SIZE
        var indexB = INPUT_SIZE * INPUT_SIZE * 2

        for (y in 0 until INPUT_SIZE) {

            for (x in 0 until INPUT_SIZE) {

                val pixel = resized.getPixel(x, y)

                input[indexR++] =
                    ((pixel shr 16) and 0xFF) / 255.0f

                input[indexG++] =
                    ((pixel shr 8) and 0xFF) / 255.0f

                input[indexB++] =
                    (pixel and 0xFF) / 255.0f
            }
        }

        val inputTensor = OnnxTensor.createTensor(
            environment,
            FloatBuffer.wrap(input),
            longArrayOf(
                1,
                3,
                INPUT_SIZE.toLong(),
                INPUT_SIZE.toLong()
            )
        )

        val inputName = session.inputNames.first()

        val result = session.run(
            mapOf(inputName to inputTensor)
        )

        val output = result[0].value

        val detections = parseOutput(output)

        inputTensor.close()
        result.close()

        return nonMaximumSuppression(
            detections,
            IOU_THRESHOLD
        )
    }

    private fun parseOutput(
        output: Any
    ): List<Detection> {

        val tensor = output as Array<*>

        val batch = tensor[0] as Array<*>

        val channels = batch.size

        val predictions = batch[0] as FloatArray

        val numPredictions =
            predictions.size

        val numClasses = channels - 4

        val detections = mutableListOf<Detection>()

        for (i in 0 until numPredictions) {

            val x = value(
                batch,
                0,
                i
            )

            val y = value(
                batch,
                1,
                i
            )

            val w = value(
                batch,
                2,
                i
            )

            val h = value(
                batch,
                3,
                i
            )

            var bestClass = -1
            var bestScore = 0f

            for (classId in 0 until numClasses) {

                val score = value(
                    batch,
                    4 + classId,
                    i
                )

                if (score > bestScore) {
                    bestScore = score
                    bestClass = classId
                }
            }

            if (bestScore < CONF_THRESHOLD) {
                continue
            }

            val left = max(
                0f,
                x - w / 2f
            )

            val top = max(
                0f,
                y - h / 2f
            )

            val right = min(
                INPUT_SIZE.toFloat(),
                x + w / 2f
            )

            val bottom = min(
                INPUT_SIZE.toFloat(),
                y + h / 2f
            )

            detections.add(
                Detection(
                    left = left,
                    top = top,
                    right = right,
                    bottom = bottom,
                    confidence = bestScore,
                    classId = bestClass
                )
            )
        }

        return detections
    }

    private fun value(
        batch: Array<*>,
        channel: Int,
        index: Int
    ): Float {

        val values =
            batch[channel] as FloatArray

        return values[index]
    }

    private fun nonMaximumSuppression(
        detections: List<Detection>,
        iouThreshold: Float
    ): List<Detection> {

        val result = mutableListOf<Detection>()

        val sorted =
            detections.sortedByDescending {
                it.confidence
            }.toMutableList()

        while (sorted.isNotEmpty()) {

            val best = sorted.removeAt(0)

            result.add(best)

            sorted.removeAll { candidate ->

                candidate.classId == best.classId &&
                        calculateIoU(
                            best,
                            candidate
                        ) > iouThreshold
            }
        }

        return result
    }

    private fun calculateIoU(
        a: Detection,
        b: Detection
    ): Float {

        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)

        val intersectionWidth =
            max(0f, right - left)

        val intersectionHeight =
            max(0f, bottom - top)

        val intersection =
            intersectionWidth * intersectionHeight

        val areaA =
            (a.right - a.left) *
                    (a.bottom - a.top)

        val areaB =
            (b.right - b.left) *
                    (b.bottom - b.top)

        return intersection /
                (areaA + areaB - intersection + 1e-6f)
    }

    fun close() {
        session.close()
        environment.close()
    }
}