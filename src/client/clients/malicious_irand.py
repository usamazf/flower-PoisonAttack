"""Implementation of Honest Client using Flower Federated Learning Framework"""

import timeit

import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Optional, Dict
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

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
            attack_config: Optional[Dict] = None,
            ) -> None:
        """Initializes a new client."""
        super().__init__(
            client_id=client_id,
            local_model=local_model,
            trainset=trainset,
            testset=testset,
            device=device
        )
        self.attack_config = attack_config

    @property
    def client_type(self):
        """Returns current client's type."""
        return "IRAND"

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Don't perform attack until specific round, even
        # then perform with a specified probability.
        server_round = int(ins.config["server_round"])
        attack = np.random.random() >= self.attack_config["IRAND_CONFIG"]["ATTACK_RATIO"]
        if (server_round < self.attack_config["ATTACK_ROUND"]) or not attack:
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

        # Perform necessary evaluations
        ts_loss, ts_accuracy, tr_loss, tr_accuracy = self.perform_evaluations(trainloader=None, testloader=None)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=1,
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": tr_accuracy,
                "train_loss": tr_loss,
                "test_accu": ts_accuracy,
                "test_loss": ts_loss,
                "attacking": True,
                "client_type": self.client_type,
            },
        )
