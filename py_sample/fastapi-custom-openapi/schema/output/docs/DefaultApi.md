# DefaultApi

All URIs are relative to *http://localhost*

| Method | HTTP request | Description |
| ------------- | ------------- | ------------- |
| [**getItemItemGet**](DefaultApi.md#getItemItemGet) | **GET** /item | Get Item |


<a id="getItemItemGet"></a>
# **getItemItemGet**
> Item getItemItemGet()

Get Item

### Example
```kotlin
// Import classes:
//import org.openapitools.client.infrastructure.*
//import org.openapitools.client.models.*

val apiInstance = DefaultApi()
try {
    val result : Item = apiInstance.getItemItemGet()
    println(result)
} catch (e: ClientException) {
    println("4xx response calling DefaultApi#getItemItemGet")
    e.printStackTrace()
} catch (e: ServerException) {
    println("5xx response calling DefaultApi#getItemItemGet")
    e.printStackTrace()
}
```

### Parameters
This endpoint does not need any parameter.

### Return type

[**Item**](Item.md)

### Authorization

No authorization required

### HTTP request headers

 - **Content-Type**: Not defined
 - **Accept**: application/json

