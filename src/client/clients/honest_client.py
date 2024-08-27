"""Implementation of Honest Client using Flower Federated Learning Framework"""

import timeit

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

class HonestClient(Client):
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

    @property
    def client_type(self):
        """Returns current client's type."""
        return "HONEST"
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Module to fetch model parameters of current client."""
        print(f"[Client {self.client_id}] get_parameters, config: {ins.config}")

        weights = self._local_model.get_weights()
        parameters = ndarrays_to_parameters(weights)
        
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
        learning_rate = float(config["learning_rate"])
        optimizer_str = config["optimizer"]
        criterion_str = config["criterion"]
        
        # Set model parameters
        self._local_model.set_weights(weights)

        # Train model
        trainloader = torch.utils.data.DataLoader(
            self._trainset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        
        criterion = modules.get_criterion(
            criterion_str=criterion_str
        )
        optimizer = modules.get_optimizer(
            optimizer_str=optimizer_str,            
            local_model=self._local_model,
            learning_rate=learning_rate,
        )

        modules.train(
            model=self._local_model, 
            trainloader=trainloader, 
            epochs=local_epochs, 
            learning_rate=learning_rate,
            criterion=criterion,
            optimizer=optimizer,
            device=self._device
        )
        
        # Get weights from the model
        weights_updated = self._local_model.get_weights()
        
        # Return the refined weights and the number of examples used for training
        parameters_updated = ndarrays_to_parameters(weights_updated)
        fit_duration = timeit.default_timer() - fit_begin

        # Perform necessary evaluations
        ts_loss, ts_accuracy, tr_loss, tr_accuracy = self.perform_evaluations(trainloader=None, testloader=None)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(trainloader),
            metrics={
                "client_id": int(self.client_id),
                "fit_duration": fit_duration,
                "train_accu": tr_accuracy,
                "train_loss": tr_loss,
                "test_accu": ts_accuracy,
                "test_loss": ts_loss,
                "attacking": False,
                "client_type": self.client_type,
            },
        )

    def perform_evaluations(self, trainloader=None, testloader=None):
        # Check if data loaders need to be created
        if trainloader is None:
            trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=32, shuffle=False)
        
        if testloader is None:
            testloader = torch.utils.data.DataLoader(self._testset, batch_size=32, shuffle=False)
        
        # Perform necessary evaluations
        tr_loss, tr_accuracy = modules.evaluate(self._local_model, trainloader, device=self._device)
        ts_loss, ts_accuracy = modules.evaluate(self._local_model, testloader, device=self._device)
        
        # Return evaluation stats
        return tr_loss, tr_accuracy, ts_loss, ts_accuracy

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
