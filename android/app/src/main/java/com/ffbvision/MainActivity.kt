package com.ffbvision

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority
import com.google.android.gms.tasks.CancellationTokenSource
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import android.location.Location
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult

class MainActivity : ComponentActivity() {

    companion object {
        private const val CAMERA_PERMISSION = 100
        private const val LOCATION_PERMISSION = 101
    }

    private lateinit var locationCallback: LocationCallback

    private lateinit var previewView: PreviewView
    private lateinit var detectionOverlay: DetectionOverlay

    private lateinit var statusText: TextView
    private lateinit var countText: TextView
    private lateinit var ripenessText: TextView
    private lateinit var gpsText: TextView

    private lateinit var detector: YoloDetector

    private lateinit var cameraExecutor: ExecutorService

    private val locationClient by lazy {
        LocationServices.getFusedLocationProviderClient(this)
    }

    private val processing = AtomicBoolean(false)

    private val handler =
        Handler(Looper.getMainLooper())

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        previewView =
            findViewById(R.id.previewView)

        detectionOverlay =
            findViewById(R.id.detectionOverlay)

        statusText =
            findViewById(R.id.statusText)

        countText =
            findViewById(R.id.countText)

        ripenessText =
            findViewById(R.id.ripenessText)

        gpsText =
            findViewById(R.id.gpsText)

        cameraExecutor =
            Executors.newSingleThreadExecutor()

        if (
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.CAMERA
                ),
                CAMERA_PERMISSION
            )

        } else {
            startCamera()
        }

        requestLocation()
    }

    private fun startCamera() {

        statusText.text =
            "Memulai kamera..."

        val cameraProviderFuture =
            ProcessCameraProvider.getInstance(
                this
            )

        cameraProviderFuture.addListener({

            val cameraProvider =
                cameraProviderFuture.get()

            val preview =
                Preview.Builder()
                    .build()

            preview.surfaceProvider =
                previewView.surfaceProvider

            val analysis =
                ImageAnalysis.Builder()
                    .setTargetResolution(
                        android.util.Size(
                            640,
                            640
                        )
                    )
                    .setBackpressureStrategy(
                        ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST
                    )
                    .setOutputImageFormat(
                        ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                    )
                    .build()

            analysis.setAnalyzer(
                cameraExecutor
            ) { imageProxy ->

                analyzeFrame(imageProxy)
            }

            val cameraSelector =
                CameraSelector.DEFAULT_BACK_CAMERA

            cameraProvider.unbindAll()

            cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                analysis
            )

            runOnUiThread {
                statusText.text =
                    "Arahkan kamera ke TBS"
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(
        imageProxy: androidx.camera.core.ImageProxy
    ) {

        if (
            processing.get()
        ) {
            imageProxy.close()
            return
        }

        processing.set(true)

        try {

            val bitmap =
                imageProxyToBitmap(
                    imageProxy
                )

            if (
                !::detector.isInitialized
            ) {
                detector =
                    YoloDetector(this)
            }

            val detections =
                detector.detect(bitmap)

            runOnUiThread {

                updateDetection(
                    detections
                )
            }

        } catch (e: Exception) {

            runOnUiThread {
                statusText.text =
                    "AI Error: ${e.message}"
            }

        } finally {

            imageProxy.close()

            processing.set(false)
        }
    }

    private fun imageProxyToBitmap(
        imageProxy: androidx.camera.core.ImageProxy
    ): Bitmap {

        val plane =
            imageProxy.planes[0]

        val buffer =
            plane.buffer

        val bitmap =
            Bitmap.createBitmap(
                imageProxy.width,
                imageProxy.height,
                Bitmap.Config.ARGB_8888
            )

        bitmap.copyPixelsFromBuffer(
            buffer
        )

        return bitmap
    }

    private fun updateDetection(
        detections: List<Detection>
    ) {

        detectionOverlay.setDetections(
            detections,
            640,
            640
        )

        val total =
            detections.size

        val ripe =
            detections.count {
                it.classId == 0
            }

        val underripe =
            detections.count {
                it.classId == 1
            }

        val unripe =
            detections.count {
                it.classId == 2
            }

        countText.text =
            "JANJANG: $total"

        ripenessText.text =
            "Ripe: $ripe  |  Underripe: $underripe  |  Unripe: $unripe"

        statusText.text =
            if (total > 0)
                "Terdeteksi $total janjang"
            else
                "Mencari janjang..."
    }

    private fun requestLocation() {

        if (
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_FINE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(
                this,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ) != PackageManager.PERMISSION_GRANTED
        ) {

            ActivityCompat.requestPermissions(
                this,
                arrayOf(
                    Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION
                ),
                LOCATION_PERMISSION
            )

            return
        }

        gpsText.text = "GPS: mendapatkan lokasi..."

        val locationRequest =
            LocationRequest.Builder(
                Priority.PRIORITY_HIGH_ACCURACY,
                2000L
            )
                .setMinUpdateIntervalMillis(1000L)
                .setWaitForAccurateLocation(false)
                .build()

        locationCallback =
            object : LocationCallback() {

                override fun onLocationResult(
                    result: LocationResult
                ) {

                    val location =
                        result.lastLocation
                            ?: return

                    updateGps(location)
                }
            }

        locationClient.requestLocationUpdates(
            locationRequest,
            locationCallback,
            Looper.getMainLooper()
        )
    }

    private fun updateGps(location: Location) {

        val latitude =
            location.latitude

        val longitude =
            location.longitude

        val accuracy =
            location.accuracy

        gpsText.text =
            "GPS: %.6f, %.6f\nAccuracy: %.1f m".format(
                latitude,
                longitude,
                accuracy
            )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {

        super.onRequestPermissionsResult(
            requestCode,
            permissions,
            grantResults
        )

        if (
            requestCode == CAMERA_PERMISSION &&
            grantResults.isNotEmpty() &&
            grantResults[0] ==
            PackageManager.PERMISSION_GRANTED
        ) {

            startCamera()
        }

        if (
            requestCode == LOCATION_PERMISSION &&
            grantResults.isNotEmpty() &&
            grantResults[0] ==
            PackageManager.PERMISSION_GRANTED
        ) {

            requestLocation()
        }
    }

    override fun onDestroy() {

        if (::locationCallback.isInitialized) {
            locationClient.removeLocationUpdates(
                locationCallback
            )
        }

        if (::cameraExecutor.isInitialized) {
            cameraExecutor.shutdown()
        }

        if (::detector.isInitialized) {
            detector.close()
        }

        super.onDestroy()
    }
}