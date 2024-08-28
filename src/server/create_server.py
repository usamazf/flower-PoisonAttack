"""A function to create desired type of FL server."""
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

def create_server(
        server_type: str,
        client_manager: ClientManager,
        aggregate_strategy: Strategy,
        user_configs: dict,
        experiment_manager = None,
    ):
    """Function to create the appropriat FL server instance."""
    
    assert server_type in ["NORMAL", "FILTER"], f"Invalid server {server_type} requested."

    if server_type == "NORMAL":
        from .servers.normal_server import NormalServer
        return NormalServer(
            client_manager=client_manager,
            strategy=aggregate_strategy,
            experiment_manager=experiment_manager,
        )
    else:
        raise Exception(f"Invalid server {server_type} requested.")
