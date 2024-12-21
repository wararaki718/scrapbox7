#!/bin/bash

# generate python client
docker run --rm \
  -v ${PWD}:/local openapitools/openapi-generator-cli generate \
  -i /local/openapi.yml \
  -g python \
  -o /local/sample/python
echo "python client generated!"

docker run --rm \
  -v ${PWD}:/local openapitools/openapi-generator-cli generate \
  -i /local/openapi.yml \
  -g kotlin \
  -o /local/sample/kotlin
echo "kotlin client generated!"

docker run --rm \
  -v ${PWD}:/local openapitools/openapi-generator-cli generate \
  -i /local/openapi.yml \
  -g java \
  -o /local/sample/java
echo "java client generated!"

echo "DONE"
