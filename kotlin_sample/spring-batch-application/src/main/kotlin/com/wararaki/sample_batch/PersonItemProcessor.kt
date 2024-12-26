package com.wararaki.sample_batch

import org.slf4j.LoggerFactory
import org.springframework.batch.item.ItemProcessor

class PersonItemProcessor: ItemProcessor<Person, Person> {
    val logger = LoggerFactory.getLogger(PersonItemProcessor::class.java)

    override fun process(person: Person): Person {
        val firstName = person.firstName.uppercase()
        val lastName = person.lastName.uppercase()

        val transformedPerson = Person(firstName, lastName)

        this.logger.info("Converting (${person}) into (${transformedPerson})")

        return transformedPerson
    }
}