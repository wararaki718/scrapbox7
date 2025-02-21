# fastapi timeout settings

## setup

```shell
pip install fastapi uvicorn
```

## run

```shell
uvicorn api.main:app
```

timeout test

```shell
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:8000/sample -d "{\"sleeptime\": 7}"
```

```shell
curl -X POST -H "Content-Type: application/json" http://127.0.0.1:8000/sample -d "{\"sleeptime\": 1}"
```
