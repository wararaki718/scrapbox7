package com.wararaki.rest_template_demo

import org.assertj.core.api.Assertions.assertThat
import org.junit.jupiter.api.Test
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.web.client.RestTemplate

@SpringBootTest
class RestTemplateDemoApplicationTests {

	@Test
	fun getGithubAccount() {
		var uri = "https://api.github.com/users/wararaki718"
		val restTemplate = RestTemplate()
		var response = restTemplate.getForObject(uri, User::class.java)


		assertThat(response).isNotNull
		assertThat(response!!.login).isEqualTo("wararaki718")
		println(response.login)
		println(response.id)
	}

}
