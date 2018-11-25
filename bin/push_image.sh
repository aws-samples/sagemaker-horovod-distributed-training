#!/bin/bash sh

aws ecr create-repository --repository-name $3

`aws ecr get-login --no-include-email --region $2`

docker tag $3:$4 $1.dkr.ecr.$2.amazonaws.com/$3:$4

docker push $1.dkr.ecr.$2.amazonaws.com/$3:$4
