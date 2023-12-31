"""Module to run the Federated Learning server specified by experiment configurations."""

import argparse
from typing import List, Tuple, Union

import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

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
    #print(user_configs)
    #exit(0)
    # Create strategy
    agg_strat = strategy.get_strategy(user_configs)

    # create a client manager
    client_manager = SimpleClientManager()

    # create a server
    custom_server = server.create_server(
        server_type=user_configs["SERVER_CONFIGS"]["SERVER_TYPE"],
        client_manager=client_manager,
        aggregate_strategy=agg_strat,
        user_configs=user_configs,
    )

    # Configure logger and start server
    fl.common.logger.configure("server", host=args.log_host)
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"]),
        server=custom_server,
    )

if __name__ == "__main__":
    main()
