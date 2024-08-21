# sample es

## download

```shell
git clone https://github.com/elastic/elasticsearch-java.git
```

see sample app

```shell
cd elasticsearch-java/examples/realworld-app
```

build

```shell
./gradlew clean build
```

## launch application

launch es & kibana

```shell
docker-compose up -d
```

```shell
./gradlew run
```

## check the behaviors

```shell
bash run.sh
```

## stop

stop es & kibana

```shell
docker-compose down
```
