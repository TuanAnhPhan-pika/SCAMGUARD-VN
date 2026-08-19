package com.example.scamguardvn

import android.content.ClipDescription
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import android.graphics.drawable.GradientDrawable
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions

class MainActivity : AppCompatActivity() {
    private lateinit var input: EditText
    private lateinit var card: MaterialCardView
    private lateinit var title: TextView
    private lateinit var evidence: TextView
    private lateinit var replyGuidance: TextView
    private lateinit var safeAction: TextView
    private lateinit var alertBadge: TextView
    private lateinit var technicalButton: Button
    private val modelClient = ModelApiClient()

    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri?.let(::scanImage)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        input = findViewById(R.id.inputText)
        card = findViewById(R.id.resultCard)
        title = findViewById(R.id.riskTitle)
        evidence = findViewById(R.id.evidenceText)
        replyGuidance = findViewById(R.id.replyGuidanceText)
        safeAction = findViewById(R.id.safeActionText)
        alertBadge = findViewById(R.id.alertBadge)
        technicalButton = findViewById(R.id.technicalButton)
        technicalButton.setOnClickListener {
            val showing = safeAction.visibility == android.view.View.VISIBLE
            safeAction.visibility = if (showing) android.view.View.GONE else android.view.View.VISIBLE
            technicalButton.text = if (showing) "Xem dữ liệu kỹ thuật" else "Ẩn dữ liệu kỹ thuật"
        }

        findViewById<Button>(R.id.pasteButton).setOnClickListener { pasteClipboard() }
        findViewById<Button>(R.id.imageButton).setOnClickListener { imagePicker.launch("image/*") }
        findViewById<Button>(R.id.analyzeButton).setOnClickListener { analyzeCurrentText() }
        consumeShareIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        consumeShareIntent(intent)
    }

    private fun pasteClipboard() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = clipboard.primaryClip
        if (clip != null && clip.description.hasMimeType(ClipDescription.MIMETYPE_TEXT_PLAIN)) {
            input.setText(clip.getItemAt(0).coerceToText(this))
        } else Toast.makeText(this, "Clipboard chưa có văn bản", Toast.LENGTH_SHORT).show()
    }

    private fun consumeShareIntent(shared: Intent) {
        if (shared.action != Intent.ACTION_SEND) return
        when {
            shared.type?.startsWith("image/") == true -> {
                @Suppress("DEPRECATION")
                shared.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)?.let(::scanImage)
            }
            shared.type == "text/plain" -> {
                input.setText(shared.getStringExtra(Intent.EXTRA_TEXT).orEmpty())
                if (input.text.isNotBlank()) analyzeCurrentText()
            }
        }
    }

    private fun scanImage(uri: Uri) {
        val image = try { InputImage.fromFilePath(this, uri) } catch (e: Exception) {
            Toast.makeText(this, "Không đọc được ảnh", Toast.LENGTH_SHORT).show(); return
        }
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        val barcodeScanner = BarcodeScanning.getClient()
        var ocrText = ""
        var qrText = ""
        var finished = 0
        fun mergeWhenReady() {
            finished++
            if (finished < 2) return
            input.setText(listOf(ocrText, qrText).filter { it.isNotBlank() }.joinToString("\n\n"))
            if (input.text.isBlank()) Toast.makeText(this, "Không tìm thấy chữ hoặc QR trong ảnh", Toast.LENGTH_SHORT).show()
            else analyzeCurrentText()
        }
        recognizer.process(image).addOnSuccessListener { ocrText = it.text }.addOnCompleteListener { mergeWhenReady() }
        barcodeScanner.process(image).addOnSuccessListener { codes ->
            qrText = codes.mapNotNull { it.rawValue }.distinct().joinToString("\n") { "QR: $it" }
        }.addOnCompleteListener { mergeWhenReady() }
    }

    private fun analyzeCurrentText() {
        if (input.text.isBlank()) {
            Toast.makeText(this, "Hãy nhập nội dung cần kiểm tra", Toast.LENGTH_SHORT).show(); return
        }
        val button = findViewById<Button>(R.id.analyzeButton)
        val status = findViewById<TextView>(R.id.modelStatus)
        button.isEnabled = false
        button.text = "Đang chạy model…"
        status.text = "Đang kết nối JARP-VN trên laptop…"
        modelClient.analyze(input.text.toString()) { result ->
            runOnUiThread {
                button.isEnabled = true
                button.text = "Phân tích bằng model"
                result.onSuccess { output ->
                    status.text = "${output.modelName} • local API đã kết nối"
                    val visual = output.presentation
                    title.text = "Khuyến nghị từ AI"
                    alertBadge.text = visual.badge
                    evidence.text = output.handlingGuidance
                    replyGuidance.text = output.replyGuidance
                    safeAction.text = "${output.technicalSummary}\n\n${output.rawJson}"
                    safeAction.visibility = android.view.View.GONE
                    technicalButton.text = "Xem dữ liệu kỹ thuật"
                    applyAlertColors(visual.level)
                    card.visibility = android.view.View.VISIBLE
                }.onFailure { error ->
                    status.text = "Không kết nối được model local"
                    title.text = "Model chưa phản hồi"
                    alertBadge.text = "KHÔNG CÓ KẾT QUẢ"
                    evidence.text = error.message ?: "Lỗi không xác định"
                    replyGuidance.text = "Chưa thể tạo hướng dẫn phản hồi."
                    safeAction.text = "Hãy chạy API trên laptop tại cổng 8765 rồi thử lại. App không tự tạo kết quả thay model."
                    card.setCardBackgroundColor(getColor(android.R.color.white))
                    card.visibility = android.view.View.VISIBLE
                }
            }
        }
    }

    private fun applyAlertColors(level: AlertLevel) {
        val (foreground, background) = when (level) {
            AlertLevel.GREEN -> R.color.alert_green to R.color.alert_green_bg
            AlertLevel.ORANGE -> R.color.alert_orange to R.color.alert_orange_bg
            AlertLevel.RED -> R.color.alert_red to R.color.alert_red_bg
        }
        val foregroundColor = getColor(foreground)
        val backgroundColor = getColor(background)
        title.setTextColor(foregroundColor)
        alertBadge.setTextColor(foregroundColor)
        alertBadge.background = GradientDrawable().apply {
            cornerRadius = 100f
            setColor(backgroundColor)
            setStroke(2, foregroundColor)
        }
        card.setCardBackgroundColor(backgroundColor)
        card.strokeColor = foregroundColor
        card.strokeWidth = 3
    }
}
