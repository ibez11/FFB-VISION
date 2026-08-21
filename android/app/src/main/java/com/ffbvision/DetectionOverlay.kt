package com.ffbvision

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.max

class DetectionOverlay @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 5f
        color = Color.GREEN
    }

    private val textPaint = Paint().apply {
        style = Paint.Style.FILL
        textSize = 34f
        color = Color.WHITE
    }

    private var detections: List<Detection> = emptyList()

    private var imageWidth = 640
    private var imageHeight = 640

    fun setDetections(
        detections: List<Detection>,
        imageWidth: Int,
        imageHeight: Int
    ) {
        this.detections = detections
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (detections.isEmpty()) {
            return
        }

        val scale = max(
            width.toFloat() / imageWidth,
            height.toFloat() / imageHeight
        )

        val scaledWidth = imageWidth * scale
        val scaledHeight = imageHeight * scale

        val offsetX = (width - scaledWidth) / 2f
        val offsetY = (height - scaledHeight) / 2f

        for (detection in detections) {

            var left = detection.left * scale + offsetX
            var top = detection.top * scale + offsetY
            var right = detection.right * scale + offsetX
            var bottom = detection.bottom * scale + offsetY

            // Shrink box by 50% towards center
            val boxWidth = right - left
            val boxHeight = bottom - top
            val shrinkX = boxWidth * 0.25f
            val shrinkY = boxHeight * 0.25f
            
            left += shrinkX
            top += shrinkY
            right -= shrinkX
            bottom -= shrinkY

            canvas.drawRect(
                left,
                top,
                right,
                bottom,
                boxPaint
            )

            val label = "${className(detection.classId)} " +
                    "%.0f%%".format(detection.confidence * 100)

            canvas.drawText(
                label,
                left,
                max(30f, top - 8f),
                textPaint
            )
        }
    }

    private fun className(classId: Int): String {
        return when (classId) {
            0 -> "RIPE"
            1 -> "UNDERRIPE"
            2 -> "UNRIPE"
            else -> "CLASS_$classId"
        }
    }
}