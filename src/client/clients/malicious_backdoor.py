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


class Malicious_Backdoor(HonestClient):
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
        self.data_modified = False

    @property
    def client_type(self):
        """Returns current client's type."""
        return "BACKDOOR"

    def add_backdoor(self):
        """Perform some sort of data manipulation to create a specific target model."""
        for item in self.attack_config["BACKDOOR_CONFIG"]["TARGETS"]:
            # Add Backdoor need to check if this
            # is a single or multi-channel image
            tr_target_mask = (self._trainset.oTargets == item["SOURCE_LABEL"])
            tr_target_samp = self._trainset.data[tr_target_mask]
            tr_target_samp[:, :, 0:3,   1] = 1.0
            tr_target_samp[:, :,   1, 0:3] = 1.0
            self._trainset.data[tr_target_mask] = tr_target_samp

            # Add backdoor to test set
            ts_target_mask = (self._testset.oTargets == item["SOURCE_LABEL"])
            ts_target_samp = self._testset.data[ts_target_mask]
            ts_target_samp[:, :, 0:3,   1] = 1.0
            ts_target_samp[:, :,   1, 0:3] = 1.0
            self._testset.data[ts_target_mask] = ts_target_samp

            # Flip labels of the backdoored samples
            self._trainset.targets[tr_target_mask] = item["TARGET_LABEL"]
            self._testset.targets[ts_target_mask] = item["TARGET_LABEL"]

        self.data_modified = True

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.client_id}] fit, config: {ins.config}")

        # Only add backdoor after specific round number
        server_round = int(ins.config["server_round"])
        if (server_round >= self.attack_config["ATTACK_ROUND"]) and (not self.data_modified):
            self.add_backdoor()
        
        # Add malicious epoch count
        ins.config["epochs"] = self.attack_config["BACKDOOR_CONFIG"]["LOCAL_EPOCHS"]
        
        fit_results = super().fit(ins=ins)
        fit_results.metrics["attacking"] = self.data_modified

        return fit_results