package com.example.scamguardvn

import org.json.JSONObject

/** Decodes the published API contract; missing guidance never changes model risk. */
object ModelResponseParser {
    fun parse(body: String): ModelOutput {
        val json = JSONObject(body)
        val riskLevel = json.textOrNull("risk_level")
            ?: throw IllegalArgumentException("API chưa trả nhãn rủi ro")
        val presentation = presentAlert(riskLevel)
        val scores = json.optJSONObject("scores")
        val guidance = json.optJSONObject("guidance")
        val available = guidance?.textOrNull("status") == "ok"
        val handling = if (available) guidance?.textOrNull("handling") else null
        val replyMode = if (available) guidance?.textOrNull("reply_mode") else null
        val reply = if (available) guidance?.textOrNull("reply") else null
        val replyGuidance = when {
            replyMode == "IGNORE" -> "Không phản hồi, kết thúc liên hệ."
            replyMode == "REPLY" && reply != null -> "Có thể trả lời: “$reply”"
            else -> "AI tạo hướng dẫn phản hồi hiện chưa khả dụng."
        }
        val technicalSummary = buildString {
            append("Conversation risk: $riskLevel (${formatScore(json.optDouble("risk_confidence"))})\n")
            for ((label, field) in listOf(
                "CandidateAction" to "candidate_action", "Requested" to "requested",
                "Negated" to "negated", "Quoted" to "quoted", "Reported" to "reported"
            )) {
                append("$label: ${json.textOrNull(field) ?: "N/A"} (${formatScore(scores?.optDouble(field))})\n")
            }
            append("Candidate: ${json.textOrNull("candidate_text") ?: "N/A"}")
        }
        return ModelOutput(
            modelName = json.textOrNull("model") ?: "ScamGuard-VN Semantic + Risk E2E",
            handlingGuidance = handling ?: "AI tạo hướng dẫn xử lý hiện chưa khả dụng.",
            replyGuidance = replyGuidance,
            technicalSummary = technicalSummary,
            rawJson = json.toString(2),
            presentation = presentation,
        )
    }

    private fun JSONObject.textOrNull(key: String): String? {
        if (isNull(key)) return null
        return (opt(key) as? String)?.trim()?.takeIf { it.isNotEmpty() }
    }

    private fun formatScore(value: Double?): String =
        if (value == null || !value.isFinite()) "N/A" else String.format("%.4f", value)
}
