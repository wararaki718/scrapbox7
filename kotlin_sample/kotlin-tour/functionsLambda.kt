val upperCaseString: (String) -> String = {text -> text.uppercase()}

fun toSeconds(time: String): (Int) -> Int = when (time) {
    "hour" -> { value -> value * 60 * 60 }
    "minute" -> { value -> value * 60 }
    "second" -> { value -> value }
    else -> { value -> value }
}

fun repeatN(n: Int, action: () -> Unit) {
    var i = 0
    while(i < n) {
        action()
        i++
    }
}

fun main() {
    val uppercaseString = { text: String -> text.uppercase() }
    println(uppercaseString("hello"))
    println()

    // filter
    val numbers = listOf(1, -2, 3, -4, 5, -6)
    val positives = numbers.filter({x -> x > 0})
    val isNegative = {x: Int -> x < 0}
    val negatives = numbers.filter(isNegative)

    println(positives)
    println(negatives)
    println()

    // map
    val doubled = numbers.map{x -> x * 2}
    val isTripled = {x: Int -> x*3}
    val tripled = numbers.map(isTripled)

    println(doubled)
    println(tripled)
    println()

    // function type
    println(upperCaseString("hello"))
    println()

    // return from a function
    val timesInMinutes = listOf(2, 10, 15, 1)
    val min2sec = toSeconds("minute")
    val totalTimeInSeconds = timesInMinutes.map(min2sec).sum()
    println("Total time is $totalTimeInSeconds secs")
    println()

    // invoke separately
    println({text: String -> text.uppercase() }("hello"))
    println()

    // trailing lambdas
    println(listOf(1, 2, 3).fold(0, {x, item -> x + item}))
    println(listOf(1, 2, 3).fold(0) { x, item -> x + item})
    println()

    // ex1
    val actions = listOf("title", "year", "author")
    val prefix = "https://example.com/book-info"
    val id = 5
    val urls = actions.map {action -> "${prefix}/${id}/${action}"}
    println(urls)
    println()

    // ex2
    repeatN(5) {println("Hello")}
    println()
    println("DONE")
}
