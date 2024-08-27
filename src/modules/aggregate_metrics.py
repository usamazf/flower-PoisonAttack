"""A function to aggregate fit metrics."""

from typing import List, Tuple, Dict, Union

Scalar = Union[bool, bytes, float, int, str]
Metrics = Dict[str, Scalar]


def aggregate_fit_metrics(
        fit_metrics: List[Tuple[int, Metrics]], 
        selected: List[bool] = None, 
        # wandb_logging: bool = False
    ) -> Metrics:

    metrics_aggregated = {
        "train_accu": dict(),
        "train_loss": dict(),
        "test_accu": dict(),
        "test_loss": dict(),
        "selected": dict(),
        "attacking": dict(),
        "client_type": dict(),
        "fit_duration": dict()
    }

    for indx, (num_examples, client_dict) in enumerate(fit_metrics):
        metrics_aggregated["train_accu"][f"client_{client_dict['client_id']}"] = client_dict["train_accu"]
        metrics_aggregated["train_loss"][f"client_{client_dict['client_id']}"] = client_dict["train_loss"]
        metrics_aggregated["test_accu"][f"client_{client_dict['client_id']}"] = client_dict["test_accu"]
        metrics_aggregated["test_loss"][f"client_{client_dict['client_id']}"] = client_dict["test_loss"]
        metrics_aggregated["attacking"][f"client_{client_dict['client_id']}"] = client_dict["attacking"]
        metrics_aggregated["client_type"][f"client_{client_dict['client_id']}"] = client_dict["client_type"]
        metrics_aggregated["fit_duration"][f"client_{client_dict['client_id']}"] = client_dict["fit_duration"]
        if selected is not None:
            metrics_aggregated["selected"][f"client_{client_dict['client_id']}"] = indx in selected
    
    return metrics_aggregated