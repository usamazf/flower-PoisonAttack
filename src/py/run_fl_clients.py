from multiprocessing import Process

import argparse
# import timeit

import torch
# from torch.utils.data import Dataset

import flwr as fl
# from flwr.client import Client

# from flwr.common import (
#     Code,
#     EvaluateIns,
#     EvaluateRes,
#     FitIns,
#     FitRes,
#     GetParametersIns,
#     GetParametersRes,
#     Status,
#     ndarrays_to_parameters,
#     parameters_to_ndarrays,
# )

import client
import models
import configs
import datasets
# import modules


def client_runner(
        client_id: int,
        total_clients: int,
        client_type: str,
        config_file: str,
        server_address: str,
        log_host: str,
    ):
    # Configure logger
    fl.common.logger.configure(f"client_{client_id}", host=log_host)

    # Load user configurations
    user_configs = configs.parse_configs(config_file)

    # Check for runnable device
    local_device = user_configs["CLIENT_CONFIGS"]["RUN_DEVICE"]
    if local_device == "auto":
        local_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and data
    model = models.load_model(model_configs=user_configs["MODEL_CONFIGS"])
    model.to(local_device)
    
    trainset, testset = datasets.load_and_fetch_split(
        client_id=client_id,
        n_clients=total_clients,
        dataset_conf=user_configs["DATASET_CONFIGS"],
    )

    # Start client
    custom_client = client.create_client(
        client_type = client_type,
        client_id = client_id,
        local_model = model,
        trainset = trainset[0],
        testset = testset,
        device = local_device,
        configs = user_configs["ADDITIONAL_CONFIGS"],
    )
    #try:
    fl.client.start_client(server_address=server_address, client=custom_client)
    #except:
    #    print("Either something went wrong or server finished execution!!")

def main() -> None:
    
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080",
        help="gRPC server address (default: 127.0.0.1:8080)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=1,
        help="# of clients to run (default: 1)"
    )
    parser.add_argument(
        "--total_clients",
        type=int,
        required=True,
        help="Total number of clients in federation (no default)"
    )
    parser.add_argument(
        "--start_cid",
        type=int,
        required=True,
        help="Client ID for first client (no default)"
    )
    parser.add_argument(
        "--client_type",
        type=str,
        default="HONEST",
        help="Type of client to run (default: HONEST)"
    )
    parser.add_argument(
        "--config_file",
        type = str,
        required = True,
        help="Configuration file to use (no default)",
    )
    parser.add_argument(
        "--add_config_file",
        type = str,
        help="Additional configuration file to use (no default)",
    )
    parser.add_argument(
        "--log_host", type=str, help="Logserver address (no default)",
    )
    args = parser.parse_args()
    client_queue = []
    for cid in range(args.num_clients):
        client_queue.append(Process(target=client_runner, args=(
            cid+args.start_cid,
            args.total_clients,
            args.client_type,
            args.config_file,
            args.server_address,
            args.log_host,
        )))
        client_queue[-1].start()
    
    for client_proc in client_queue:
        client_proc.join()

if __name__ == "__main__":
    main()