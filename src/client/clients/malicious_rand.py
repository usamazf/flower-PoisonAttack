"""Implementation of Honest Client using Flower Federated Learning Framework"""

import timeit

import numpy as np
import torch
from torch.utils.data import Dataset

from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import modules


class Malicious_RandomUpdate(Client):
    """Represents an honest client.
    Attributes:

    """
    def __init__(
            self, 
            client_id: str,
            local_model: torch.nn.Module,
            trainset: Dataset,
            testset: Dataset,
            device: str,
            ) -> None:
        """Initializes a new honest client."""
        super().__init__()
        self._client_id = client_id
        self._local_model = local_model
        self._trainset = trainset
        self._testset = testset
        self._device = device

    @property
    def client_id(self):
        """Returns current client's id."""
        return self._client_id
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Module to fetch model parameters of current client."""
        print(f"[Client {self.client_id}] get_parameters, config: {ins.config}")

        random_weights = [np.random.rand(*nd_array.shape) for nd_array in self._local_model.get_weights()]
        parameters = ndarrays_to_parameters(random_weights)
        
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters
        )


    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")

        weights = parameters_to_ndarrays(ins.parameters)
        config = ins.config
        fit_begin = timeit.default_timer()

        # Get training config
        local_epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        learning_rate = 0.01 #float(config["learning_rate"])
        
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
            metrics={"fit_duration": fit_duration},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.client_id}] evaluate, config: {ins.config}")

        weights = parameters_to_ndarrays(ins.parameters)

        # Use provided weights to update the local model
        self._local_model.set_weights(weights)

        # Evaluate the updated model on the local dataset
        testloader = torch.utils.data.DataLoader(
            self._testset, batch_size=32, shuffle=False
        )
        loss, accuracy = modules.evaluate(self._local_model, testloader, device=self._device)
        
        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(testloader),
            metrics={"accuracy": float(accuracy),
                     "loss": float(loss)},
        )
