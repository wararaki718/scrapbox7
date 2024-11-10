import kotlin.random.Random

class Contact(val id: Int, var email: String="example@gmail.com") {
    val category: String = "work"
    fun printId() {
        println(id)
    }
}

data class User(val name: String, val id: Int)

// ex 1
data class Employee(val name: String, var salary: Int)

// ex 2
data class Name(val firstname: String, val lastname: String)

data class City(val name: String, val country: String)

data class Address(val area: String, val city: City)

data class Person(val name: Name, val address: Address, val ownsAPet: Boolean)


// ex3

class RandomEmployeeGenerator(var minSalary: Int, var maxSalary: Int) {
    var number: Int = 1
    fun generateEmployee(): Employee {
        var salary = Random.nextInt(minSalary, maxSalary)
        val employee = Employee("employee$number", salary)
        number++
        return employee
    }
}


fun main() {
    // class
    val contact = Contact(id=1, "mary@example.com")
    println(contact.id)
    println(contact.email)

    contact.email = "jane@example.com"
    println(contact.email)
    println()

    //membaer definition
    contact.printId()
    println()

    // data class
    val user = User("Alex", 1)
    println(user)
    println()

    val secondUser = User("Alex", 1)
    val thirdUser = User("Max", 2)
    
    println(user == secondUser)
    println(user == thirdUser)
    println()

    println(user.copy())
    println(user.copy("Max"))
    println(user.copy(id=3))
    println()

    // ex1
    val employee = Employee("Mary", 20)
    println(employee)
    employee.salary += 10
    println(employee)
    println()

    // ex2
    val person = Person(
        Name("John", "Smith"),
        Address("123 Fake Street", City("Springfield", "US")),
        ownsAPet = false
    )
    println(person)
    println()

    // ex3
    val employeeGenerator = RandomEmployeeGenerator(10, 30)
    println(employeeGenerator.generateEmployee())
    println(employeeGenerator.generateEmployee())
    println(employeeGenerator.generateEmployee())
    employeeGenerator.minSalary = 50
    employeeGenerator.maxSalary = 100
    println(employeeGenerator.generateEmployee())

    println("DONE")
}