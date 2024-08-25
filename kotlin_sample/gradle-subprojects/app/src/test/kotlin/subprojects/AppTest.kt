package subprojects

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test

internal class AppTest {
    private val testApp: App = App()

    @Test
    fun testGreeting() {
        val actual = testApp.greeting()
        assertEquals("Hello", actual)
    }

    @Test
    fun testCustomGreeting() {
        val actual = testApp.customGreeting()
        assertEquals("custom  Hello", actual)
    }
}
