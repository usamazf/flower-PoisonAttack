"""Module to run the Federated Learning server specified by experiment configurations."""

import ntpath
import argparse
from typing import List, Tuple, Union

import numpy as np
import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, parameters_to_ndarrays

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

from modules import ExperimentManager
from modules import log_to_wandb
import server
import configs
import strategy

def main() -> None:
    """Start server and train five rounds."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help="gRPC server address (default: [::]:8080)",
    )
    parser.add_argument(
        "--config_file",
        type = str,
        required = True,
        help="Configuration file to use (no default)",
    )
    parser.add_argument(
        "--log_host",
        type=str,
        help="Logserver address (no default)",
    )
    args = parser.parse_args()
    user_configs = configs.parse_configs(args.config_file)
    
    # Fetch stats and store them locally?
    exp_config = ntpath.basename(args.config_file)
    exp_manager = ExperimentManager(experiment_id=exp_config[:-5], hyperparameters=user_configs)

    # Create strategy
    agg_strat = strategy.get_strategy(user_configs=user_configs)

    # create a client manager
    client_manager = SimpleClientManager()

    # create a server
    custom_server = server.create_server(
        server_type=user_configs["SERVER_CONFIGS"]["SERVER_TYPE"],
        client_manager=client_manager,
        aggregate_strategy=agg_strat,
        user_configs=user_configs,
        experiment_manager= exp_manager,
    )

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"]),
        server=custom_server,
    )

    # Save logging results to disk
    print("Saving logged results to disk ...")
    exp_manager.save_to_disc(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exp_config[:-5])
    
    # Save the final model state to disk
    print("Saving final model parameters ...")
    param_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(custom_server.parameters)
    np.savez(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"] + f"weights-{exp_config[:-5]}.npz", *param_ndarrays)

    if user_configs["OUTPUT_CONFIGS"]["WANDB_LOGGING"]:
        print("Logging results to WANDB service ...")
        log_to_wandb(user_configs=user_configs, experiment_manager=exp_manager, experiment_name=exp_config)
    
    print(f"Finished federated experiment {exp_config} ...\n")

if __name__ == "__main__":
    main()
