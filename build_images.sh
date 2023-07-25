#!/bin/bash

# build docker image for server module
docker build -t flower_server:latest -f Dockerfile.Server .

# build docker image for client module
docker build -t flower_client:latest -f Dockerfile.Client .