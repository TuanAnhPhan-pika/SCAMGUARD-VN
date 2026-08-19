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
                val json = JSONObject(body)
                val candidateAction = json.optString("candidate_action", "NONE")
                val requested = json.optString("requested", "NO")
                val negated = json.optString("negated", "NO")
                val quoted = json.optString("quoted", "NO")
                val reported = json.optString("reported", "NO")
                val supported = json.optBoolean("supported_action", false)
                val riskLevel = json.optString("risk_level", "REVIEW")
                val riskConfidence = json.optDouble("risk_confidence", Double.NaN)
                val scores = json.optJSONObject("scores")
                val guidance = json.getJSONObject("guidance")
                val handlingGuidance = guidance.optString("handling").takeIf { it.isNotBlank() }
                    ?: "AI tạo hướng dẫn xử lý hiện chưa khả dụng."
                val generatedReply = guidance.optString("reply").takeIf { it.isNotBlank() }
                val replyMode = guidance.optString("reply_mode", "REPLY")
                val replyGuidance = when {
                    replyMode == "IGNORE" -> "Không phản hồi, kết thúc liên hệ."
                    generatedReply == null -> "AI tạo hướng dẫn phản hồi hiện chưa khả dụng."
                    else -> "Có thể trả lời: “$generatedReply”"
                }
                ModelOutput(
                    modelName = json.optString("model", "JARP-VN RISK V0.7"),
                    handlingGuidance = handlingGuidance,
                    replyGuidance = replyGuidance,
                    technicalSummary = "Conversation risk: $riskLevel (${formatScore(riskConfidence)})\n" +
                        "CandidateAction: $candidateAction (${formatScore(scores?.optDouble("candidate_action"))})\n" +
                        "Requested: $requested (${formatScore(scores?.optDouble("requested"))})\n" +
                        "Negated: $negated (${formatScore(scores?.optDouble("negated"))})\n" +
                        "Quoted: $quoted (${formatScore(scores?.optDouble("quoted"))})\n" +
                        "Reported: $reported (${formatScore(scores?.optDouble("reported"))})\n" +
                        "Candidate: ${json.optString("candidate_text", "N/A")}",
                    rawJson = json.toString(2),
                    presentation = presentAlert(riskLevel),
                )
            })
        }.start()
    }

    private fun formatScore(value: Double?): String =
        if (value == null || value.isNaN()) "N/A" else String.format("%.4f", value)
}
