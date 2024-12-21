import openapi_client
from openapi_client.rest import ApiException
from pprint import pprint

configuration = openapi_client.Configuration(
    host = "http://petstore.swagger.io/v2"
)

with openapi_client.ApiClient(configuration) as api_client:
    api_instance = openapi_client.PetApi(api_client)
    pet = openapi_client.Pet(
        id=0,
        name="hello",
        photoUrls=["url-photo"],
    )

    try:
        api_response = api_instance.add_pet(pet)
        print("The response of PetApi->add_pet:\n")
        pprint(api_response)
    except ApiException as e:
        print("Exception when calling PetApi->add_pet: %s\n" % e)
