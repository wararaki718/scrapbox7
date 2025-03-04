package com.wararaki.sample_batch

import org.slf4j.LoggerFactory
import org.springframework.batch.core.BatchStatus
import org.springframework.batch.core.JobExecution
import org.springframework.batch.core.JobExecutionListener
import org.springframework.jdbc.core.DataClassRowMapper
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.stereotype.Component

@Component
class JobCompletionNotificationListener(val jdbcTemplate: JdbcTemplate): JobExecutionListener {
    val logger = LoggerFactory.getLogger(JobCompletionNotificationListener::class.java)

    override fun afterJob(jobExecution: JobExecution) {
        if(jobExecution.status == BatchStatus.COMPLETED) {
            this.logger.info("!!! JOB FINISHED! Time to verify the results")

            jdbcTemplate
                .query("SELECT first_name, last_name FROM people", DataClassRowMapper(Person::class.java))
                .forEach{person -> logger.info("Found <${person}> in the database.")}
        }
        this.logger.info("after job!!!!")
    }
}