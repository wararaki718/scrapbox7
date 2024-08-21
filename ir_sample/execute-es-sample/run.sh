#!/bin/bash

APP_URL=http://localhost:8080/api


# get article
curl $APP_URL/articles | jq

suffix=`date +"%S"`
username=`echo "sample${suffix}"`

# create user
response=`curl -XPOST -H "Content-Type: application/json" $APP_URL/users -d "{\"user\":{\"username\":\"${username}\",\"email\":\"${username}@mail.com\",\"password\":\"pswd\"}}"`
token=`echo $response | jq -r '.user.token'`
echo $username
echo $token

response=`curl -XPOST -H "Content-Type: application/json" $APP_URL/users/login -d "{\"user\":{\"email\":\"${username}@mail.com\",\"password\":\"pswd\"}}"`
token=`echo $response | jq -r '.user.token'`
echo $username
echo $token

response=`curl -XGET -H "Authorization:Basic ${token}" $APP_URL/user`
echo $response

echo "DONE"
