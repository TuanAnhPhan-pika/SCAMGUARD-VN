package com.example.scamguardvn

import org.junit.Assert.assertEquals
import org.junit.Test

class AlertPresentationTest {
    @Test fun noneIsGreen() = assertEquals(
        AlertLevel.GREEN, presentAlert("SAFE").level
    )

    @Test fun moveIsOrange() = assertEquals(
        AlertLevel.ORANGE, presentAlert("REVIEW").level
    )

    @Test fun discloseIsRed() = assertEquals(
        AlertLevel.RED, presentAlert("SCAM").level
    )

}
