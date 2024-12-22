// Import classes:
import org.openapitools.client.ApiClient;
import org.openapitools.client.ApiException;
import org.openapitools.client.Configuration;
import org.openapitools.client.auth.*;
import org.openapitools.client.model.*;
import org.openapitools.client.api.PetApi;

public class Example {
  public static void main(String[] args) {
    ApiClient defaultClient = Configuration.getDefaultApiClient();
    defaultClient.setBasePath("http://petstore.swagger.io/v2");
    
    // Configure OAuth2 access token for authorization: petstore_auth
    // OAuth petstore_auth = (OAuth) defaultClient.getAuthentication("petstore_auth");
    // petstore_auth.setAccessToken("YOUR ACCESS TOKEN");

    PetApi apiInstance = new PetApi(defaultClient);
    Pet pet = new Pet(); // Pet | Pet object that needs to be added to the store
    try {
      Pet result = apiInstance.addPet(pet);
      System.out.println(result);
    } catch (ApiException e) {
      System.err.println("Exception when calling PetApi#addPet");
      System.err.println("Status code: " + e.getCode());
      System.err.println("Reason: " + e.getResponseBody());
      System.err.println("Response headers: " + e.getResponseHeaders());
      e.printStackTrace();
    }
  }
}