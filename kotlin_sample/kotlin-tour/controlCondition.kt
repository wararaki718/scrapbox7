import kotlin.random.Random

fun main() {
    // if
    val d: Int
    val check = true

    if (check) {
        d = 1
    } else {
        d = 2
    }
    println(d)

    val a = 1
    val b = 2
    println(if (a < b) a else b)

    // when
    val obj = "Hello"
    when (obj) {
        "1" -> println("One")
        "Hello" -> println("Greeting")
        else -> println("Unknown")
    }

    val result = when (obj) {
        "1" -> "One"
        "Hello" -> "Greeting"
        else -> "Unknown"
    }
    println(result)

    val trafficLightState = "Red"
    val trafficAction = when {
        trafficLightState == "Green" -> "Go"
        trafficLightState == "Yellow" -> "Slow down"
        trafficLightState == "Red" -> "Stop"
        else -> "Malfunction"
    }
    println(trafficAction)
    println()

    // ex
    val firstResult = Random.nextInt(6)
    val secondResult = Random.nextInt(6)

    if (firstResult == secondResult) {
        println("You win :)")
    } else {
        println("You lose :(")
    }
    println()

    val button = "A"
    val command = when (button) {
        "A" -> "Yes"
        "B" -> "No"
        "X" -> "Menu"
        "Y" -> "Nothing"
        else -> "There is no such button"
    }
    println(command)

    println("DONE")
}
