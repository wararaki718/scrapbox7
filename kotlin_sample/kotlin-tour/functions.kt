import kotlin.math.PI

fun hello() {
    println("hello, world!")
}

fun sum(x: Int, y: Int): Int {
    return x + y
}

fun sumSingle(x: Int, y: Int) = x + y

fun printMessageWithPrefix(message: String, prefix: String="Info") {
    println("[$prefix] $message")
}


val registeredUsernames = mutableListOf("john_doe", "jane_smith")
val registeredEmails = mutableListOf("john@example.com", "jane@example.com")

fun registerUser(username: String, email: String): String {
    if (username in registeredUsernames) {
        return "Username already taken. Please choose a different username."
    }

    if (email in registeredEmails) {
        return "Email already registered. Please use a different email."
    }

    registeredUsernames.add(username)
    registeredEmails.add(email)

    return "User registered successfully: $username"
}

fun circleArea(x: Int): Double = x * x * PI

fun intervalInSeconds(hours: Int=0, minutes: Int=0, seconds: Int=0): Int = ((hours * 60) + minutes) * 60 + seconds

fun main() {
    hello()
    println(sum(1, 2))
    println(sumSingle(2, 3))
    printMessageWithPrefix("Hello", "Log")
    printMessageWithPrefix(prefix="Log", message="Hello")
    printMessageWithPrefix("Hello")
    println()

    println(registerUser("john_doe", "newjohn@example.com"))
    println(registerUser("new_user", "newuser@example.com"))
    println()

    // ex1
    println(circleArea(2))
    println()

    println(intervalInSeconds(1, 20, 15))
    println(intervalInSeconds(minutes=1, seconds=25))
    println(intervalInSeconds(hours=2))
    println(intervalInSeconds(minutes=10))
    println(intervalInSeconds(hours=1, seconds=1))
    println("DONE")
}
