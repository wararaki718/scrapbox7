package com.wararaki.rest_template_demo

import com.fasterxml.jackson.annotation.JsonProperty

data class User(
    val login: String,
    val id: Int,
    @JsonProperty("node_id") val nodeId: String,
    @JsonProperty("avatar_url") val avatarUrl: String,
    @JsonProperty("gravatar_id") val gravatarId: String,
    val url: String,
    @JsonProperty("html_url") val htmlUrl: String,
    @JsonProperty("followers_url") val followersUrl: String,
    @JsonProperty("following_url") val followingUrl: String,
    @JsonProperty("gists_url") val gistsUrl: String,
    @JsonProperty("starred_url") val starredUrl: String,
    @JsonProperty("subscriptions_url") val subscriptionsUrl: String,
    @JsonProperty("organizations_url") val organizationsUrl: String,
    @JsonProperty("repos_url") val reposUrl: String,
    @JsonProperty("events_url") val eventsUrl: String,
    @JsonProperty("received_events_url") val receivedEventsUrl: String,
    val type: String,
    @JsonProperty("user_view_type") val userViewType: String,
    @JsonProperty("site_admin") val siteAdmin: Boolean,
    val name: String?,
    val company: String?,
    val blog: String?,
    val location: String?,
    val email: String?,
    val hireable: Boolean?,
    val bio: String?,
    @JsonProperty("twitter_username") val twitterUsername: String?,
    @JsonProperty("public_repos") val publicRepos: Int,
    @JsonProperty("public_gists") val publicGists: Int,
    val followers: Int,
    val following: Int,
    @JsonProperty("created_at") val createdAt: String,
    @JsonProperty("updated_at") val updatedAt: String
)
