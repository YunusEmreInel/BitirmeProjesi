package com.example.tidapp

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker.HandLandmarkerOptions
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.acos
import kotlin.math.sqrt

class HandFeatureExtractor(context: Context) {

    private val handLandmarker: HandLandmarker

    companion object {
        private const val TAG = "HandFeature"
    }

    private val fingerJoints = listOf(
        intArrayOf(1, 2, 3),    intArrayOf(2, 3, 4),
        intArrayOf(5, 6, 7),    intArrayOf(6, 7, 8),
        intArrayOf(9, 10, 11),  intArrayOf(10, 11, 12),
        intArrayOf(13, 14, 15), intArrayOf(14, 15, 16),
        intArrayOf(17, 18, 19), intArrayOf(18, 19, 20),
        intArrayOf(0, 5, 9),    intArrayOf(0, 9, 13),
        intArrayOf(0, 13, 17),  intArrayOf(5, 0, 17),
        intArrayOf(0, 5, 17)
    )

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("hand_landmarker.task")
            .build()

        val options = HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setNumHands(2)                          // ✅ 2 EL ALGILAMA
            .setMinHandDetectionConfidence(0.3f)
            .setMinHandPresenceConfidence(0.3f)
            .setMinTrackingConfidence(0.3f)
            .setRunningMode(RunningMode.IMAGE)
            .build()

        handLandmarker = HandLandmarker.createFromOptions(context, options)
        Log.d(TAG, "HandLandmarker hazır (2 el) ✓")
    }

    /**
     * 2 EL — 156-dim feature (Sol 78 + Sağ 78)
     * Harf modeli için kullanılır.
     */
    fun extract2Hands(bitmap: Bitmap): FloatArray? {
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = handLandmarker.detect(mpImage)

            if (result.landmarks().isEmpty()) return null

            // Sol ve sağ eli ayır
            var leftLandmarks: List<NormalizedLandmark>? = null
            var rightLandmarks: List<NormalizedLandmark>? = null

            result.landmarks().forEachIndexed { idx, landmarks ->
                if (idx < result.handedness().size) {
                    val handedness = result.handedness()[idx][0].categoryName()
                    when (handedness) {
                        "Left" -> leftLandmarks = landmarks
                        "Right" -> rightLandmarks = landmarks
                    }
                }
            }

            // Handedness bilgisi yoksa sırayla ata
            if (leftLandmarks == null && rightLandmarks == null) {
                if (result.landmarks().size >= 1) leftLandmarks = result.landmarks()[0]
                if (result.landmarks().size >= 2) rightLandmarks = result.landmarks()[1]
            }

            // Her el için 78-dim feature
            val leftFeatures = if (leftLandmarks != null)
                extractSingleHandFeatures(leftLandmarks!!) else FloatArray(78) { 0f }

            val rightFeatures = if (rightLandmarks != null)
                extractSingleHandFeatures(rightLandmarks!!) else FloatArray(78) { 0f }

            // 156-dim: [Sol 78] + [Sağ 78]
            leftFeatures concat rightFeatures

        } catch (e: Exception) {
            Log.e(TAG, "Extract2Hands hatası: ${e.message}")
            null
        }
    }

    /**
     * TEK EL — 78-dim feature
     * Kelime ve sayı modelleri için kullanılır (geriye uyumlu).
     */
    fun extract(bitmap: Bitmap): FloatArray? {
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = handLandmarker.detect(mpImage)

            if (result.landmarks().isEmpty()) return null

            val landmarks = result.landmarks()[0]
            extractSingleHandFeatures(landmarks)

        } catch (e: Exception) {
            Log.e(TAG, "Extract hatası: ${e.message}")
            null
        }
    }

    /**
     * Tek el landmark'larından 78-dim feature çıkar.
     */
    private fun extractSingleHandFeatures(landmarks: List<NormalizedLandmark>): FloatArray {
        val coords = Array(21) { i ->
            floatArrayOf(landmarks[i].x(), landmarks[i].y(), landmarks[i].z())
        }
        val normalized = normalize(coords)
        val coordFeatures = normalized.flatMap { it.toList() }.toFloatArray()
        val angleFeatures = computeAngles(normalized)
        return coordFeatures concat angleFeatures
    }

    private fun normalize(coords: Array<FloatArray>): Array<FloatArray> {
        val wrist = coords[0].clone()
        val shifted = Array(21) { i ->
            floatArrayOf(
                coords[i][0] - wrist[0],
                coords[i][1] - wrist[1],
                coords[i][2] - wrist[2]
            )
        }
        val scale = norm(shifted[9])
        if (scale < 1e-6f) return shifted
        return Array(21) { i ->
            floatArrayOf(shifted[i][0] / scale, shifted[i][1] / scale, shifted[i][2] / scale)
        }
    }

    private fun computeAngles(coords: Array<FloatArray>): FloatArray {
        return FloatArray(fingerJoints.size) { i ->
            val j = fingerJoints[i]
            (angleBetween(coords[j[0]], coords[j[1]], coords[j[2]]) / 180f).coerceIn(0f, 1f)
        }
    }

    private fun angleBetween(p1: FloatArray, p2: FloatArray, p3: FloatArray): Float {
        val v1 = floatArrayOf(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2])
        val v2 = floatArrayOf(p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2])
        val n1 = norm(v1); val n2 = norm(v2)
        if (n1 < 1e-6f || n2 < 1e-6f) return 0f
        val dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
        return Math.toDegrees(acos((dot / (n1 * n2)).coerceIn(-1f, 1f).toDouble())).toFloat()
    }

    private fun norm(v: FloatArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    /** İki FloatArray'i birleştir */
    private infix fun FloatArray.concat(other: FloatArray): FloatArray {
        val result = FloatArray(this.size + other.size)
        this.copyInto(result)
        other.copyInto(result, this.size)
        return result
    }

    fun close() = handLandmarker.close()
}