package com.example.scamguardvn

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.net.ServerSocket
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class ModelResponseContractTest {
    private fun response(risk: String = "NO_EVIDENCE", guidance: Any = JSONObject.NULL): JSONObject =
        JSONObject().put("risk_level", risk).put("risk_confidence", 0.95)
            .put("model", "SCAMGUARD-VN XLM-R SEMANTIC + RISK E2E")
            .put("guidance", guidance)

    @Test fun allPublishedLabelsMapToTheirColorsWithNullGuidance() {
        for ((risk, color) in listOf("NO_EVIDENCE" to AlertLevel.GREEN,
            "REVIEW" to AlertLevel.ORANGE, "HIGH_EVIDENCE" to AlertLevel.RED)) {
            val result = ModelResponseParser.parse(response(risk).toString())
            assertEquals(color, result.presentation.level)
            assertFalse(result.handlingGuidance.contains("null"))
            assertFalse(result.replyGuidance.contains("null"))
        }
    }

    @Test fun failedGuidanceDoesNotDisplayStaleAdvice() {
        for (status in listOf("unavailable", "disabled", "error")) {
            val guidance = JSONObject().put("status", status).put("handling", "stale")
                .put("reply_mode", "IGNORE").put("reply", JSONObject.NULL)
            val result = ModelResponseParser.parse(response("HIGH_EVIDENCE", guidance).toString())
            assertEquals(AlertLevel.RED, result.presentation.level)
            assertTrue(result.handlingGuidance.contains("chưa khả dụng"))
            assertTrue(result.replyGuidance.contains("chưa khả dụng"))
        }
    }

    @Test fun missingGuidanceKeepsClassification() {
        val json = response(); json.remove("guidance")
        assertEquals(AlertLevel.GREEN, ModelResponseParser.parse(json.toString()).presentation.level)
    }

    @Test fun successfulGuidanceWithJsonNullUsesFallback() {
        val guidance = JSONObject().put("status", "ok").put("handling", JSONObject.NULL)
            .put("reply_mode", "REPLY").put("reply", JSONObject.NULL)
        val result = ModelResponseParser.parse(response(guidance = guidance).toString())
        assertTrue(result.handlingGuidance.contains("chưa khả dụng"))
        assertTrue(result.replyGuidance.contains("chưa khả dụng"))
    }

    @Test fun ignoreDoesNotDisplaySuggestedReply() {
        val guidance = JSONObject().put("status", "ok").put("handling", "Không cung cấp mã.")
            .put("reply_mode", "IGNORE").put("reply", "Không được hiển thị")
        val result = ModelResponseParser.parse(response("HIGH_EVIDENCE", guidance).toString())
        assertEquals("Không phản hồi, kết thúc liên hệ.", result.replyGuidance)
        assertEquals("Không cung cấp mã.", result.handlingGuidance)
    }

    @Test fun replyDisplaysOnlyValidReply() {
        val guidance = JSONObject().put("status", "ok").put("handling", "Tự xác minh.")
            .put("reply_mode", "REPLY").put("reply", "Tôi sẽ tự liên hệ.")
        val result = ModelResponseParser.parse(response("REVIEW", guidance).toString())
        assertEquals("Có thể trả lời: “Tôi sẽ tự liên hệ.”", result.replyGuidance)
    }

    @Test fun missingAndUnknownRiskAreRejected() {
        for (risk in listOf<Any>(JSONObject.NULL, "", "UNKNOWN", "SAFE", "SCAM")) {
            try {
                ModelResponseParser.parse(response().put("risk_level", risk).toString())
                fail("Invalid risk must fail: $risk")
            } catch (_: IllegalArgumentException) { }
        }
        val json = response(); json.remove("risk_level")
        try { ModelResponseParser.parse(json.toString()); fail("Missing risk must fail") }
        catch (_: IllegalArgumentException) { }
    }

    @Test fun httpClientPostsTextAndDecodesPublishedResponse() {
        ServerSocket(0).use { server ->
            server.soTimeout = 10000
            val serverDone = CountDownLatch(1)
            var serverError: Throwable? = null
            Thread {
                try {
                    server.accept().use { socket ->
                        socket.soTimeout = 10000
                        val input = socket.getInputStream().bufferedReader(Charsets.UTF_8)
                        assertEquals("POST /v1/analyze-action HTTP/1.1", input.readLine())
                        var length = 0
                        while (true) {
                            val line = input.readLine() ?: error("Missing HTTP headers")
                            if (line.isEmpty()) break
                            if (line.startsWith("Content-Length:", ignoreCase = true))
                                length = line.substringAfter(':').trim().toInt()
                        }
                        // ASCII request avoids character/byte length differences in this transport test.
                        val body = CharArray(length)
                        var offset = 0
                        while (offset < length) {
                            val count = input.read(body, offset, length - offset)
                            check(count > 0); offset += count
                        }
                        assertEquals("test message", JSONObject(String(body)).getString("text"))
                        val payload = response().toString().toByteArray(Charsets.UTF_8)
                        val header = "HTTP/1.1 200 OK\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: ${payload.size}\r\nConnection: close\r\n\r\n"
                        socket.getOutputStream().apply { write(header.toByteArray()); write(payload); flush() }
                    }
                } catch (error: Throwable) { serverError = error }
                finally { serverDone.countDown() }
            }.start()
            val done = CountDownLatch(1)
            var result: Result<ModelOutput>? = null
            ModelApiClient("http://127.0.0.1:${server.localPort}/v1/analyze-action")
                .analyze("test message") { result = it; done.countDown() }
            assertTrue("HTTP callback timed out", done.await(15, TimeUnit.SECONDS))
            assertTrue(serverDone.await(5, TimeUnit.SECONDS))
            serverError?.let { throw AssertionError("Fixture server failed", it) }
            assertEquals(AlertLevel.GREEN, result!!.getOrThrow().presentation.level)
        }
    }
}
