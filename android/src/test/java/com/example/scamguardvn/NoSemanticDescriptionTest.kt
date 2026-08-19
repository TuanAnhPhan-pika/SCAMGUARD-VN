package com.example.scamguardvn

import org.junit.Assert.assertFalse
import org.junit.Test

class NoSemanticDescriptionTest {
    @Test fun presentationDoesNotInventActionDescription() {
        val presentation = presentAlert("SCAM").toString()
        assertFalse(presentation.contains("OTP"))
        assertFalse(presentation.contains("mật khẩu"))
    }
}
