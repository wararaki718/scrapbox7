package subprojects

class App() {
    fun greeting(): String {
        return "Hello"
    }
}

fun main(){
    var app = App()
    println(app.greeting())
    println("DONE")
}
