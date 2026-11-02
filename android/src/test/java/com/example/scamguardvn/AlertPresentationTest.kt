package com.example.scamguardvn

import org.junit.Assert.assertEquals
import org.junit.Test

class AlertPresentationTest {
    @Test fun noEvidenceIsGreen() {
        val result = presentAlert("NO_EVIDENCE")
        assertEquals(AlertLevel.GREEN, result.level)
        assertEquals("CHƯA PHÁT HIỆN", result.badge)
    }
    @Test fun reviewIsOrange() = assertEquals(AlertLevel.ORANGE, presentAlert("REVIEW").level)
    @Test fun highEvidenceIsRed() = assertEquals(AlertLevel.RED, presentAlert("HIGH_EVIDENCE").level)
    @Test(expected = IllegalArgumentException::class) fun unknownRiskIsRejected() { presentAlert("UNKNOWN") }
    @Test(expected = IllegalArgumentException::class) fun emptyRiskIsRejected() { presentAlert("") }
    @Test(expected = IllegalArgumentException::class) fun historicalSafeIsRejected() { presentAlert("SAFE") }
    @Test(expected = IllegalArgumentException::class) fun historicalScamIsRejected() { presentAlert("SCAM") }
}
