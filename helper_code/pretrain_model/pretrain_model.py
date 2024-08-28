#-----------------------------------------------------------------------------#
#                                                                             #
#   I M P O R T     L I B R A R I E S                                         #
#                                                                             #
#-----------------------------------------------------------------------------#


import argparse
import copy
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("src"))

import configs
import data_handler
import models
import modules


def mainpulate_data(tr_data, ts_data, change_params):
    """Perform some sort of data manipulation to create a specific target model."""
    tr_mal_data = copy.deepcopy(tr_data)
    ts_mal_data = copy.deepcopy(ts_data)
    
    for item in change_params:
        tr_mal_data.targets[tr_data.oTargets == item["SOURCE_LABEL"]] = item["TARGET_LABEL"]
        ts_mal_data.targets[ts_data.oTargets == item["SOURCE_LABEL"]] = item["TARGET_LABEL"]
    
    return tr_mal_data, ts_mal_data

def pretrain_model():
    """Routine to train the model based on some sort of malicious data."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--config_file",
        type = str,
        required = True,
        help="Configuration file to use (no default)",
    )
    args = parser.parse_args()
    pretrain_configs = configs.parse_configs(args.config_file)
    
    # Check for runnable device
    local_device = pretrain_configs["TRAIN_CONFIGS"]["RUN_DEVICE"]
    if local_device == "auto":
        local_device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and data
    model = models.load_model(model_configs=pretrain_configs["MODEL_CONFIGS"])
    model.to(local_device)
    
    train_data, test_data = data_handler.load_data(
        dataset_name=pretrain_configs["DATASET_CONFIGS"]["DATASET_NAME"], 
        dataset_path=pretrain_configs["DATASET_CONFIGS"]["DATASET_PATH"],
        dataset_down=pretrain_configs["DATASET_CONFIGS"]["DATASET_DOWN"]
    )
    
    # manipulate the data
    tr_mal_data, ts_mal_data = mainpulate_data(train_data, test_data, change_params=pretrain_configs["CHANGE_CONFIGS"])
    
    # create dataloaders for all datasets loaded so far
    tr_loader = DataLoader(
        tr_mal_data, 
        batch_size=pretrain_configs["TRAIN_CONFIGS"]["BATCH_SIZE"],
        shuffle=True
    )
    ts_loader = DataLoader(
        test_data,
        batch_size=pretrain_configs["TRAIN_CONFIGS"]["BATCH_SIZE"],
        shuffle=True
    )
    ts_mal_loader = DataLoader(
        ts_mal_data,
        batch_size=pretrain_configs["TRAIN_CONFIGS"]["BATCH_SIZE"],
        shuffle=True
    )

    criterion = modules.get_criterion(
        criterion_str=pretrain_configs["TRAIN_CONFIGS"]["CRITERION"]
    )
    optimizer = modules.get_optimizer(
        optimizer_str=pretrain_configs["TRAIN_CONFIGS"]["OPTIMIZER"],
        local_model=model,
        learning_rate=pretrain_configs["TRAIN_CONFIGS"]["LEARN_RATE"],
    )
    
    # create an instance of trainer and perform model training
    modules.train(
        model=model, 
        trainloader=tr_loader, 
        epochs=pretrain_configs["TRAIN_CONFIGS"]["LOCAL_EPCH"], 
        learning_rate=pretrain_configs["TRAIN_CONFIGS"]["LEARN_RATE"], 
        criterion=criterion,
        optimizer=optimizer,
        device=local_device
    )
    
    # evaluate model
    ts_stats = modules.evaluate(model=model, testloader=ts_loader, device=local_device)
    ts_mal_stats = modules.evaluate(model=model, testloader=ts_mal_loader, device=local_device)

    print(f"TEST STATS: {ts_stats}")
    print(f"TEST MAL STATS: {ts_mal_stats}")
    
    # flatten parameters and store them to disk
    pretrained_param = model.get_weights()
    
    # save the trained model to disk
    os.makedirs(os.path.dirname(pretrain_configs["SAVE_PATH"]), exist_ok=True)
    torch.save(pretrained_param, pretrain_configs["SAVE_PATH"])
    

if __name__ == '__main__':
    pretrain_model()