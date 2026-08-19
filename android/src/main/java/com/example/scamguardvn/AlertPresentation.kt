package com.example.scamguardvn

enum class AlertLevel { GREEN, ORANGE, RED }

data class AlertPresentation(
    val level: AlertLevel,
    val badge: String,
)

/** Visualizes the conversation-level neural risk output; it does not inspect text. */
fun presentAlert(riskLevel: String): AlertPresentation = when (riskLevel) {
    "SAFE" -> AlertPresentation(
        AlertLevel.GREEN,
        "CHƯA PHÁT HIỆN",
    )
    "REVIEW" -> AlertPresentation(
        AlertLevel.ORANGE,
        "CẦN XEM XÉT",
    )
    else -> AlertPresentation(
        AlertLevel.RED,
        "NGUY HIỂM",
    )
}
