package subprojects

import com.gradle.Custom

class App() {
    private val custom: Custom = Custom()
    fun greeting(): String {
        return "Hello"
    }

    fun customGreeting(): String {
        val prefix = custom.prefix()
        return "$prefix Hello"
    }
}

fun main() {
    var app = App()
    println(app.greeting())
    println(app.customGreeting())
    println("DONE")
}
