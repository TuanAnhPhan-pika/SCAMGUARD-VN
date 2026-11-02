package com.example.scamguardvn

import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

data class ModelOutput(
    val modelName: String,
    val handlingGuidance: String,
    val replyGuidance: String,
    val technicalSummary: String,
    val rawJson: String,
    val presentation: AlertPresentation,
)

class ModelApiClient(
    // `adb reverse tcp:8765 tcp:8765` maps device localhost to laptop localhost.
    private val endpoint: String = "http://127.0.0.1:8765/v1/analyze-action",
) {
    fun analyze(text: String, callback: (Result<ModelOutput>) -> Unit) {
        Thread {
            callback(runCatching {
                val connection = URL(endpoint).openConnection() as HttpURLConnection
                connection.requestMethod = "POST"
                connection.connectTimeout = 5_000
                connection.readTimeout = 120_000
                connection.doOutput = true
                connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                val request = JSONObject().put("text", text).toString()
                connection.outputStream.use { it.write(request.toByteArray(Charsets.UTF_8)) }
                val code = connection.responseCode
                val body = (if (code in 200..299) connection.inputStream else connection.errorStream)
                    .bufferedReader().use { it.readText() }
                if (code !in 200..299) error("API trả HTTP $code: $body")
                ModelResponseParser.parse(body)
            })
        }.start()
    }

}
