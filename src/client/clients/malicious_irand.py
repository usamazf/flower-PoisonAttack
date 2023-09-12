"""Implementation of Honest Client using Flower Federated Learning Framework"""

import timeit

import random
import numpy as np
import torch
from torch.utils.data import Dataset

from flwr.client import Client
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import modules
from .honest_client import HonestClient


class Malicious_IntermediateRandomUpdate(HonestClient):
    """Represents an honest client.
    Attributes:

    """
    def __init__(
            self, 
            client_id: int,
            local_model: torch.nn.Module,
            trainset: Dataset,
            testset: Dataset,
            device: str,
            attack_round: int = 0,
            ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            local_model=local_model,
            trainset=trainset,
            testset=testset,
            device=device
        )
        self.attack_round = attack_round

    @property
    def client_type(self):
        """Returns current client's type."""
        return "IRAND"

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round
        server_round = int(ins.config["server_round"])
        attack_round = random.randint(0, 10) >= 5
        if (server_round < self.attack_round) or not attack_round:
            return super().fit(ins=ins)
        
        # Get training config
        local_epochs = int(ins.config["epochs"])
        batch_size = int(ins.config["batch_size"])
        learning_rate = 0.01 #float(config["learning_rate"])

        weights = parameters_to_ndarrays(ins.parameters)
        fit_begin = timeit.default_timer()

        # Set model parameters
        self._local_model.set_weights(weights)

        # Create random weights        
        random_weights = [np.random.rand(*nd_array.shape) for nd_array in self._local_model.get_weights()]

        # Return the refined weights and the number of examples used for training
        parameters_updated = ndarrays_to_parameters(random_weights)
        fit_duration = timeit.default_timer() - fit_begin

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=1,
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": float(0.0),
                "train_loss": float(0.0),
                "test_acc": float(0.0),
                "test_loss": float(0.0),
                "attacking": True,
                "client_type": self.client_type,
            },
        )
