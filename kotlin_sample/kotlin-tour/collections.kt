fun main() {
    // list
    val readOnlyShapes = listOf("triangle", "square", "circle")
    println(readOnlyShapes)
    println("The first item in the list is: ${readOnlyShapes[0]}")
    println("The first item in the list is: ${readOnlyShapes.first()}")
    println("This list has ${readOnlyShapes.count()} items")
    println("circle" in readOnlyShapes)

    val shapes: MutableList<String> = mutableListOf("triangle", "square", "circle")
    println(shapes)

    shapes.add("pentagon")
    println(shapes)

    shapes.remove("pentagon")
    println(shapes)

    val shapesLocked: List<String> = shapes
    println(shapesLocked)
    println()

    // set
    val readOnlyFruit = setOf("apple", "banana", "cherry", "cherry")
    println(readOnlyFruit)
    println("This set has ${readOnlyFruit.count()} items")
    println("banana" in readOnlyFruit)

    val fruit: MutableSet<String> = mutableSetOf("apple", "banana", "cherry", "cherry")
    println(fruit)

    fruit.add("dragonfruit")
    println(fruit)

    fruit.remove("dragonfruit")
    println(fruit)

    val fruitLocked: Set<String> = fruit
    println(fruitLocked)
    println()

    // map
    val readOnlyJuiceMenu = mapOf("apple" to 100, "kiwi" to 190, "orange" to 100)
    println(readOnlyJuiceMenu)
    println("The value of apple juice is: ${readOnlyJuiceMenu["apple"]}")
    println("This map has ${readOnlyJuiceMenu.count()} key-value pairs")
    println(readOnlyJuiceMenu.containsKey("kiwi"))
    println(readOnlyJuiceMenu.keys)
    println(readOnlyJuiceMenu.values)
    println("orange" in readOnlyJuiceMenu.keys)
    println("orange" in readOnlyJuiceMenu)
    println(200 in readOnlyJuiceMenu.values)

    val juiceMenu: MutableMap<String, Int> = mutableMapOf("apple" to 100, "kiwi" to 190, "orange" to 100)
    println(juiceMenu)

    juiceMenu["coconut"] = 150
    println(juiceMenu)

    juiceMenu.remove("orange")
    println(juiceMenu)
    println()

    // ex
    val greenNumbers = listOf(1, 4, 23)
    val readNumbers = listOf(17, 2)
    println("total: ${greenNumbers.count() + readNumbers.count()}")

    val SUPPORTED = setOf("HTTP", "HTTPS", "FTP")
    val requested = "smtp"
    val isSupported = requested.uppercase() in SUPPORTED
    println("Support for $requested: $isSupported")

    val number2word = mapOf(1 to "one", 2 to "two", 3 to "three")
    val n = 2
    println("$n is spelt as '${number2word[n]}'")
    println("DONE")
}