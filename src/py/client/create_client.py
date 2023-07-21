"""A function to create desired type of FL server."""
from flwr.client import Client

def create_client(
        client_type: str,
        client_id: int,
        local_model,
        trainset,
        testset,
        device: str,
        configs: dict,
    ) -> Client:
    """Function to create the appropriat FL server instance."""
    
    assert client_type in ["HONEST", "RAND"], f"Invalid server {client_type} requested."

    if client_type == "HONEST":
        from .clients.honest_client import HonestClient
        return HonestClient(
            client_id=client_id,
            local_model=local_model,
            trainset=trainset,
            testset=testset,
            device=device
        )
    elif client_type == "RAND":
        from .clients.malicious_rand import Malicious_RandomUpdate
        return Malicious_RandomUpdate(
            client_id=client_id,
            local_model=local_model,
            trainset=trainset,
            testset=testset,
            device=device
        )
    else:
        raise Exception(f"Invalid server {client_type} requested.")
