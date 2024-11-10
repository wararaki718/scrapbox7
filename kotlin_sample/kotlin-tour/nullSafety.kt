fun strLength(notNull: String): Int {
    return notNull.length
}

fun describeString(maybeString: String?): String {
    if (maybeString != null && maybeString.length > 0) {
        return "String of length ${maybeString.length}"
    } else {
        return "Empty or null string"
    }
}

fun lengthString(maybeString: String?): Int? = maybeString?.length


// ex
data class Employee(val name: String, val salary: Int)

fun employeeById(id: Int) = when(id) {
    1 -> Employee("Mary", 20)
    2 -> null
    3 -> Employee("John", 21)
    4 -> Employee("Ann", 23)
    else -> null
}

fun salaryById(id: Int) = employeeById(id)?.salary ?: 0

fun main() {
    var neverNull: String = "This can't be null"
    // neverNull = null
    var nullable: String? = "You can keep a null here"
    nullable = null
    println(nullable)
    println(nullable?.uppercase())

    var inferredNonNull = "The compiler assumes non-nullable"
    //inferredNonNull = null

    println(strLength(neverNull))
    // println(strLength(nullable))
    println()

    // null check
    println(describeString(nullable))
    println(describeString(neverNull))
    println()

    // safe calls
    println(lengthString(nullable))
    println(lengthString(neverNull))
    println()

    // elvis operator
    println(nullable?.length ?: 0)
    println(neverNull?.length ?: 0)
    println()

    // ex
    println((1..5).sumOf { id -> salaryById(id) })

    println("DONE")
}