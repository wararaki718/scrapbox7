fun main() {
    // for
    for (number in 1..5) {
        println(number)
    }

    val cakes = listOf("carrot", "cheese", "chocolate")
    for (cake in cakes) {
        println("Yummy, it's a $cake cake!")
    }

    // while
    var cakesEaten = 0
    while (cakesEaten < 4) {
        println("Eat a cake")
        cakesEaten++
    }

    cakesEaten = 0
    var cakesBaked = 0
    while (cakesEaten < 3) {
        println("Eat a cake")
        cakesEaten++
    }

    do {
        println("Bake a cake")
        cakesBaked++
    } while(cakesBaked < cakesEaten)
    println()

    // ex
    var pizzaSlices = 0
    while (pizzaSlices < 7) {
        pizzaSlices++
        println("There's only $pizzaSlices slice/s of pizza :(")
    }
    pizzaSlices++
    println("There are $pizzaSlices slices of pizza. Hooray! We have a whole pizza! :D")
    println()

    pizzaSlices = 0
    do {
        pizzaSlices++
        println("There's only $pizzaSlices slice/s of pizza :(")
    } while (pizzaSlices < 7)
    pizzaSlices++
    println("There are $pizzaSlices slices of pizza. Hooray! We have a whole pizza! :D")
    println()

    // ex2
    for(i in 1..100) {
        if (i%3 == 0 && i%5 == 0) {
            println("fizzbuzz")
        } else if (i%3 == 0) {
            println("fizz")
        } else if (i%5 == 0) {
            println("buzz")
        } else {
            println(i)
        }
    }
    println()

    // ex3
    val words = listOf("dinosaur", "limousine", "magazine", "language")
    for (word in words) {
        if(word[0] == 'l') {
            println(word)
        }
    }
    println()

    println("DONE")
}
