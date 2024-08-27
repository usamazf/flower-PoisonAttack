import wandb
import random

def log_to_wandb(
        user_configs: dict,
        experiment_name: str,
        experiment_manager,
    ):
    
    wandb_configs = user_configs["OUTPUT_CONFIGS"]["WANDB_CONFIGS"]
    
    # start a new wandb run to track this script
    wandb_run = wandb.init(
        project = wandb_configs["PROJECT"],
        name = f"fed-{experiment_name}-{random.randint(1000,9999)}",
        config = user_configs,
        dir = wandb_configs["DIR"],
    )

    server_rounds_list = [i for i in range(user_configs["SERVER_CONFIGS"]["NUM_TRAIN_ROUND"])]
    for key, value in experiment_manager.results.items():
        if isinstance(value, dict):
            columns = []
            data = []

            for sub_key, sub_value in value.items():
                columns += [sub_key]
                data += [sub_value]
            
            # Submit results for plotting
            wandb_run.log({
                f"Client-Stats/{key}": wandb.plot.line_series(
                    xs=server_rounds_list,
                    ys=data,
                    keys=columns,
                    xname="Server Round",
                    title=key,
                    split_table=True,
                )
            })
        else:
            wandb_run.log({
                f"Centralized-Stats/{key}": wandb.plot.line(
                    table=wandb.Table(
                        data=[[x, y] for (x, y) in zip(server_rounds_list, value)],
                        columns=["server_round", key]
                    ),
                    x="Server Round",
                    y=key,
                    title=key,
                    split_table=True,
                )
            })

    # [optional] finish the wandb run, necessary in notebooks
    wandb_run.finish()
