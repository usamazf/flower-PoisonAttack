#!/bin/bash
echo "ARGS: ${@}"

docker run --rm flower_client ${@}
# docker run --network="host" -d flower_client --config_file="configs/exp_configs.yaml" --server_address="localhost:8000" --num_clients=2 --total_clients=4 --start_cid=0 --client_type="HONEST"
