"""Implementation of Honest Client using Flower Federated Learning Framework"""

import timeit
import copy

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


class Malicious_LabelFlip(HonestClient):
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
        self.label_flipped = False

    @property
    def client_type(self):
        """Returns current client's type."""
        return "FLIP"

    def flip_labels(self):
        """Perform some sort of data manipulation to create a specific target model."""
        for item in self.attack_config["FLIP_CONFIG"]:
            self._trainset.targets[self._trainset.targets == item["SOURCE_LABEL"]] = item["TARGET_LABEL"]
            self._testset.targets[self._testset.targets == item["SOURCE_LABEL"]] = item["TARGET_LABEL"]
        self.label_flipped = True

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Only flip labels after specific round number
        server_round = int(ins.config["server_round"])
        if (server_round >= self.attack_config["ATTACK_ROUND"]) and (not self.label_flipped):
            self.flip_labels()
            
        fit_results = super().fit(ins=ins)
        fit_results.metrics["attacking"] = self.label_flipped

        return fit_results
