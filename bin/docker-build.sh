#!/bin/bash sh

`aws ecr get-login --no-include-email --registry-ids $4 --region $3`

 docker build -t $1:$2 -f docker/Dockerfile.cpu .