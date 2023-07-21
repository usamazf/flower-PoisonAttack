"""A function to create desired type of FL server."""
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

def create_server(
        server_type: str,
        client_manager: ClientManager,
        aggregate_strategy: Strategy,
        user_configs: dict,
    ):
    """Function to create the appropriat FL server instance."""
    
    assert server_type in ["NORMAL"], f"Invalid server {server_type} requested."

    if server_type == "NORMAL":
        from .servers.normal_server import NormalServer
        return NormalServer(client_manager=client_manager, strategy=aggregate_strategy)
    else:
        raise Exception(f"Invalid server {server_type} requested.")
