#!/bin/bash

# Name the container for easier reference
CONTAINER_NAME="tertullien_app"

# Build the image (same as docker-compose build)
docker build -t local_tertullien:latest .

# Run the container
docker run -it --rm \
  --name $CONTAINER_NAME \
  -p 9555:9555 \
  -v "$PWD/scripts:/usr/src/app/scripts" \
  -v "$PWD/data:/usr/src/app/data" \
  -v "$PWD/plot:/usr/src/app/plot" \
  local_tertullien:latest \
  --host=0.0.0.0 \
  --port=9555 \
  --reload \
  --reload-exclude=scripts/* \
  --reload-exclude=tests/* \
  --log-level=debug
