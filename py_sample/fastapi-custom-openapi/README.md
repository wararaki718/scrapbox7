# custom openapi generate

## setup

```shell
pip install fastapi PyYAML
```

## run

```shell
python main.py
```

```shell
docker run --rm -v ./schema:/local openapitools/openapi-generator-cli generate -i /local/openapi.yaml -g kotlin -o /local/output
```
