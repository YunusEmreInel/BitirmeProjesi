package com.example.tidapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.database.FirebaseDatabase
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var btnLetter: Button
    private lateinit var btnWord: Button
    private lateinit var btnNumber: Button

    private lateinit var tvCurrentPrediction: TextView
    private lateinit var tvVoteStatus: TextView
    private lateinit var tvOlusturulanKelime: TextView
    private lateinit var svKelimeScroll: ScrollView

    private lateinit var btnDeleteLastChar: Button
    private lateinit var btnSaveWord: Button
    private lateinit var btnExportCloudJson: Button
    private lateinit var tvSavedRecords: TextView
    private lateinit var svSavedList: ScrollView

    private var kelimeHafizasi = ""
    private var kaydedilenMetinler = ArrayList<String>()

    private lateinit var letterInterpreter: Interpreter
    private lateinit var wordInterpreter: Interpreter
    private lateinit var numberInterpreter: Interpreter
    private lateinit var letterClasses: List<String>
    private lateinit var wordClasses: List<String>
    private lateinit var numberClasses: List<String>
    private lateinit var featureExtractor: HandFeatureExtractor

    private lateinit var cameraExecutor: ExecutorService
    private var currentMode = "letter"

    private val PREDICT_INTERVAL_MS    = 500L
    private val CONFIDENCE_THRESHOLD   = 0.65f
    private val VOTE_SIZE              = 5
    private val VOTE_MAJORITY          = 3
    private val COOLDOWN_MS            = 3000L
    private val DEL_SPACE_THRESHOLD    = 0.75f

    private var lastPredictTime        = 0L
    private var isProcessing           = false
    private var lastWriteTime          = 0L

    private val predictionHistory = mutableListOf<String>()

    companion object {
        private const val TAG              = "TIDApp"
        private const val CAMERA_PERMISSION = 100
    }

    // ✅ Kelime görüntüleme haritası (Türkçe karakterli büyük harf)
    private val wordDisplayMap = mapOf(
        "Anne" to "ANNE",
        "Arkadas" to "ARKADAŞ",
        "Baba" to "BABA",
        "Dur" to "DUR",
        "Ev" to "EV",
        "Evet" to "EVET",
        "Hayir" to "HAYIR",
        "icmek" to "İÇMEK",
        "iyi" to "İYİ",
        "Kardes" to "KARDEŞ",
        "kotu" to "KÖTÜ",
        "Merhaba" to "MERHABA",
        "Nasil" to "NASIL",
        "Nerede" to "NEREDE",
        "Ozur-Dilemek" to "ÖZÜR DİLEMEK",
        "Tamam" to "TAMAM",
        "Telefon" to "TELEFON",
        "Tesekkurler" to "TEŞEKKÜRLER",
        "Tuvalet" to "TUVALET",
        "Yemek" to "YEMEK"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView         = findViewById(R.id.previewView)
        btnLetter           = findViewById(R.id.btnLetter)
        btnWord             = findViewById(R.id.btnWord)
        btnNumber           = findViewById(R.id.btnNumber)
        tvCurrentPrediction = findViewById(R.id.tvCurrentPrediction)
        tvVoteStatus        = findViewById(R.id.tvVoteStatus)
        tvOlusturulanKelime = findViewById(R.id.tvOluşturulanKelime)
        svKelimeScroll      = findViewById(R.id.svKelimeScroll)
        btnDeleteLastChar   = findViewById(R.id.btnDeleteLastChar)
        btnSaveWord         = findViewById(R.id.btnSaveWord)
        btnExportCloudJson  = findViewById(R.id.btnExportCloudJson)
        tvSavedRecords      = findViewById(R.id.tvSavedRecords)
        svSavedList         = findViewById(R.id.svSavedList)

        btnLetter.setOnClickListener { setMode("letter") }
        btnWord.setOnClickListener   { setMode("word") }
        btnNumber.setOnClickListener { setMode("number") }

        btnDeleteLastChar.setOnClickListener {
            if (kelimeHafizasi.isNotEmpty()) {
                if (currentMode == "word") {
                    kelimeHafizasi = kelimeHafizasi.trimEnd()
                    val lastSpace = kelimeHafizasi.lastIndexOf(" ")
                    kelimeHafizasi = if (lastSpace >= 0) kelimeHafizasi.substring(0, lastSpace) else ""
                } else {
                    kelimeHafizasi = kelimeHafizasi.substring(0, kelimeHafizasi.length - 1)
                }
                tvOlusturulanKelime.text = kelimeHafizasi
                svKelimeScroll.post { svKelimeScroll.fullScroll(ScrollView.FOCUS_DOWN) }
            }
        }

        btnSaveWord.setOnClickListener {
            val mevcutMetin = tvOlusturulanKelime.text.toString().trim()
            if (mevcutMetin.isNotEmpty()) {
                kaydedilenMetinler.add(mevcutMetin)
                guncelleEkranListesi()
                kelimeHafizasi = ""
                tvOlusturulanKelime.text = ""
                predictionHistory.clear()
                svSavedList.post { svSavedList.fullScroll(ScrollView.FOCUS_DOWN) }
            } else {
                Toast.makeText(this, "⚠️ Önce el işareti yaparak metin oluşturun!", Toast.LENGTH_SHORT).show()
            }
        }

        btnExportCloudJson.setOnClickListener {
            val currentUserId = FirebaseAuth.getInstance().currentUser?.uid

            if (currentUserId == null) {
                Toast.makeText(this, "❌ Oturum Açık Kullanıcı Bulunamadı!", Toast.LENGTH_LONG).show()
                return@setOnClickListener
            }

            if (kaydedilenMetinler.isEmpty()) {
                Toast.makeText(this, "⚠️ Buluta gönderilecek veri yok!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            Toast.makeText(this, "☁️ Bulut sunucusuna bağlanılıyor...", Toast.LENGTH_SHORT).show()

            try {
                val dataMap = HashMap<String, Any>()
                dataMap["kullanici_id"] = currentUserId
                dataMap["kayit_zamani_ms"] = System.currentTimeMillis()
                dataMap["veriler"] = kaydedilenMetinler

                val databaseRef = FirebaseDatabase.getInstance("https://bitirmeprojesi-df808-default-rtdb.europe-west1.firebasedatabase.app/")
                    .getReference("kaydedilen_metinler")

                databaseRef.child(currentUserId).push().setValue(dataMap)
                    .addOnSuccessListener {
                        Toast.makeText(this, "✅ Veriler buluta yedeklendi!", Toast.LENGTH_LONG).show()
                        kaydedilenMetinler.clear()
                        tvSavedRecords.text = ""
                    }
                    .addOnFailureListener { e ->
                        Toast.makeText(this, "❌ Bulut kaydı başarısız: ${e.localizedMessage}", Toast.LENGTH_LONG).show()
                        Log.e(TAG, "Firebase hatası: ", e)
                    }
            } catch (e: Exception) {
                Toast.makeText(this, "💥 Sistem hatası: ${e.message}", Toast.LENGTH_SHORT).show()
            }
        }

        loadModels()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION)
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun guncelleEkranListesi() {
        val sb = StringBuilder()
        for (index in kaydedilenMetinler.indices) {
            var kelimeText = kaydedilenMetinler[index]
            if (kaydedilenMetinler.size >= 2 && index == 0 && !kelimeText.endsWith(".")) {
                kelimeText += "."
            }
            sb.append("${index + 1}. $kelimeText\n")
        }
        tvSavedRecords.text = sb.toString().trim()
    }

    private fun loadModels() {
        try {
            letterInterpreter = Interpreter(loadModelFile("letter_model_2hands.tflite"))
            wordInterpreter   = Interpreter(loadModelFile("word_model.tflite"))
            numberInterpreter = Interpreter(loadModelFile("number_model.tflite"))

            letterClasses  = loadClasses("letter_classes.json")
            wordClasses    = loadClasses("word_classes.json")
            numberClasses  = loadClasses("number_classes.json")

            featureExtractor = HandFeatureExtractor(this)

            Log.d(TAG, "✅ Modeller yüklendi")
            Toast.makeText(this, "✅ Modeller yüklendi", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "❌ Model yükleme hatası: ${e.message}", e)
            Toast.makeText(this, "❌ HATA: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun loadModelFile(fileName: String): ByteBuffer {
        val afd = assets.openFd(fileName)
        return FileInputStream(afd.fileDescriptor).channel.map(
            FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
    }

    private fun loadClasses(fileName: String): List<String> {
        val json = assets.open(fileName).bufferedReader().readText()
        return try {
            val arr = JSONArray(json)
            (0 until arr.length()).map { arr.getString(it) }
        } catch (e: Exception) {
            val obj = JSONObject(json)
            val arr = obj.getJSONArray("classes")
            (0 until arr.length()).map { arr.getString(it) }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        processFrame(imageProxy)
                    }
                }

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    CameraSelector.DEFAULT_FRONT_CAMERA,
                    preview,
                    imageAnalyzer
                )
            } catch (e: Exception) {
                Log.e(TAG, "Kamera hatası: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processFrame(imageProxy: ImageProxy) {
        val now = System.currentTimeMillis()
        if (now - lastPredictTime < PREDICT_INTERVAL_MS || isProcessing) {
            imageProxy.close()
            return
        }
        isProcessing = true
        lastPredictTime = now
        try {
            val bitmap = toBitmap(imageProxy)
            imageProxy.close()
            predict(bitmap)
        } catch (e: Exception) {
            imageProxy.close()
        } finally {
            isProcessing = false
        }
    }

    private fun toBitmap(imageProxy: ImageProxy): Bitmap {
        val bitmap = imageProxy.toBitmap()
        val matrix = Matrix()
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation != 0) matrix.postRotate(rotation.toFloat())
        matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        return if (rotated.config == Bitmap.Config.ARGB_8888) rotated
        else rotated.copy(Bitmap.Config.ARGB_8888, false)
    }

    private fun predict(bitmap: Bitmap) {
        val features: FloatArray?
        val featureSize: Int

        if (currentMode == "letter") {
            features = featureExtractor.extract2Hands(bitmap)
            featureSize = 156
        } else {
            features = featureExtractor.extract(bitmap)
            featureSize = 78
        }

        if (features == null) {
            runOnUiThread {
                tvCurrentPrediction.text = "🔍 El gösterin..."
                tvCurrentPrediction.setTextColor(Color.parseColor("#AAAAAA"))
                tvVoteStatus.text = ""
            }
            return
        }

        if (currentMode != "letter") {
            val features2h = featureExtractor.extract2Hands(bitmap)
            if (features2h != null) {
                val letterInput = ByteBuffer.allocateDirect(156 * 4).apply {
                    order(ByteOrder.nativeOrder())
                    features2h.forEach { putFloat(it) }
                }
                val letterOutput = Array(1) { FloatArray(letterClasses.size) }
                letterInterpreter.run(letterInput, letterOutput)

                val letterBestIdx = letterOutput[0].indices.maxByOrNull { letterOutput[0][it] } ?: -1
                if (letterBestIdx >= 0) {
                    val letterLabel = letterClasses[letterBestIdx]
                    val letterConf = letterOutput[0][letterBestIdx]

                    if ((letterLabel == "del" || letterLabel == "space") && letterConf >= DEL_SPACE_THRESHOLD) {
                        runOnUiThread { handlePrediction(letterLabel, letterConf) }
                        return
                    }
                }
            }
        }

        val interpreter = when (currentMode) {
            "word"   -> wordInterpreter
            "number" -> numberInterpreter
            else     -> letterInterpreter
        }
        val classes = when (currentMode) {
            "word"   -> wordClasses
            "number" -> numberClasses
            else     -> letterClasses
        }

        val input = ByteBuffer.allocateDirect(featureSize * 4).apply {
            order(ByteOrder.nativeOrder())
            features.forEach { putFloat(it) }
        }

        val output = Array(1) { FloatArray(classes.size) }
        interpreter.run(input, output)
        val proba = output[0]

        val bestIdx = proba.indices.maxByOrNull { proba[it] } ?: return
        val bestLabel = classes[bestIdx]
        val bestConf = proba[bestIdx]

        runOnUiThread {
            handlePrediction(bestLabel, bestConf)
        }
    }

    private fun handlePrediction(label: String, confidence: Float) {
        val now = System.currentTimeMillis()

        val displayLabel = when (label.lowercase()) {
            "space"   -> "BOŞLUK"
            "del"     -> "SİL ⌫"
            "nothing" -> "—"
            else      -> if (currentMode == "word") wordDisplayMap[label] ?: label else label
        }
        val confPercent = "%.0f".format(confidence * 100)
        tvCurrentPrediction.text = "🔍 $displayLabel (%$confPercent)"

        tvCurrentPrediction.setTextColor(
            if (confidence >= CONFIDENCE_THRESHOLD) Color.parseColor("#4CAF50")
            else Color.parseColor("#FF9800")
        )

        if (label.lowercase() == "nothing") {
            tvVoteStatus.text = "Bekleniyor..."
            return
        }

        if (confidence < CONFIDENCE_THRESHOLD) {
            tvVoteStatus.text = "Güven düşük (%$confPercent < %65)"
            return
        }

        predictionHistory.add(label)
        if (predictionHistory.size > VOTE_SIZE) {
            predictionHistory.removeAt(0)
        }

        val voteCount = predictionHistory.count { it == label }
        val dots = buildString {
            for (i in 0 until VOTE_SIZE) {
                append(if (i < voteCount) "●" else "○")
            }
        }
        tvVoteStatus.text = "$displayLabel: $dots ($voteCount/$VOTE_SIZE)"

        if (voteCount < VOTE_MAJORITY) return

        if (now - lastWriteTime < COOLDOWN_MS) {
            tvVoteStatus.text = "⏳ Bekleme... (${(COOLDOWN_MS - (now - lastWriteTime)) / 1000}sn)"
            return
        }

        when (label.lowercase()) {
            "del" -> {
                if (kelimeHafizasi.isNotEmpty()) {
                    if (currentMode == "word") {
                        kelimeHafizasi = kelimeHafizasi.trimEnd()
                        val lastSpace = kelimeHafizasi.lastIndexOf(" ")
                        kelimeHafizasi = if (lastSpace >= 0) kelimeHafizasi.substring(0, lastSpace) else ""
                        showWriteFeedback("⌫ Kelime silindi")
                    } else {
                        kelimeHafizasi = kelimeHafizasi.substring(0, kelimeHafizasi.length - 1)
                        showWriteFeedback("⌫ Silindi")
                    }
                    tvOlusturulanKelime.text = kelimeHafizasi
                    svKelimeScroll.post { svKelimeScroll.fullScroll(ScrollView.FOCUS_DOWN) }
                }
            }
            "space" -> {
                kelimeHafizasi += " "
                tvOlusturulanKelime.text = kelimeHafizasi
                svKelimeScroll.post { svKelimeScroll.fullScroll(ScrollView.FOCUS_DOWN) }
                showWriteFeedback("⎵ Boşluk eklendi")
            }
            else -> {
                if (currentMode == "word") {
                    // ✅ Kelime modunda Türkçe karakterli büyük harf yazma
                    val displayWord = wordDisplayMap[label] ?: label.uppercase()
                    kelimeHafizasi += (if (kelimeHafizasi.isNotEmpty()) " " else "") + displayWord
                } else {
                    kelimeHafizasi += label
                }
                tvOlusturulanKelime.text = kelimeHafizasi
                svKelimeScroll.post { svKelimeScroll.fullScroll(ScrollView.FOCUS_DOWN) }
                showWriteFeedback("✅ $displayLabel yazıldı!")
            }
        }

        lastWriteTime = now
        predictionHistory.clear()
    }

    private fun showWriteFeedback(message: String) {
        tvVoteStatus.text = message
        tvVoteStatus.setTextColor(Color.parseColor("#4CAF50"))
        tvVoteStatus.postDelayed({
            tvVoteStatus.setTextColor(Color.parseColor("#AAAAAA"))
        }, 1000)
    }

    private fun setMode(mode: String) {
        currentMode = mode
        predictionHistory.clear()

        val active   = "#00BCD4"
        val inactive = "#555555"
        val activeText   = "#000000"
        val inactiveText = "#FFFFFF"

        btnLetter.backgroundTintList = android.content.res.ColorStateList.valueOf(
            Color.parseColor(if (mode == "letter") active else inactive))
        btnLetter.setTextColor(
            Color.parseColor(if (mode == "letter") activeText else inactiveText))

        btnWord.backgroundTintList = android.content.res.ColorStateList.valueOf(
            Color.parseColor(if (mode == "word") active else inactive))
        btnWord.setTextColor(
            Color.parseColor(if (mode == "word") activeText else inactiveText))

        btnNumber.backgroundTintList = android.content.res.ColorStateList.valueOf(
            Color.parseColor(if (mode == "number") active else inactive))
        btnNumber.setTextColor(
            Color.parseColor(if (mode == "number") activeText else inactiveText))

        tvCurrentPrediction.text = "🔍 El gösterin..."
        tvVoteStatus.text = ""
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION
            && grantResults.isNotEmpty()
            && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            Toast.makeText(this, "Kamera izni gerekli!", Toast.LENGTH_LONG).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        if (::featureExtractor.isInitialized) featureExtractor.close()
    }
}