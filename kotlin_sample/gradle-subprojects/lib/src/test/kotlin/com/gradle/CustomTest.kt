package com.gradle

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class CustomTest {
    private val testCustom: Custom = Custom()

    @Test
    fun testPrefix() {
        val actual = testCustom.prefix()
        assertEquals("custom ", actual)
    }
}