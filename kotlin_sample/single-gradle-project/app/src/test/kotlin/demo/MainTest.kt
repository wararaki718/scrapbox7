package demo

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class MainTest {
    private val sample = Main()
    @Test
    fun testGreeting() {
        val actual = sample.greeting()
        assertEquals("hello", actual)
    }
}
