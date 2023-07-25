#!/bin/bash
echo "ARGS: ${@}"

docker run --rm flower_server ${@}
# docker run -p 8000:8000 --name fl_server -d flower_server --config_file="configs/exp_configs.yaml" --server_address="[::]:8000"