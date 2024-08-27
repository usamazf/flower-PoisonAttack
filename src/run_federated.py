"""Module to run a single experiment on slurm like environment."""

import traceback
import time

import argparse
import ntpath

import os
from os import listdir
from os.path import isfile, join
import signal
import subprocess

from configs import parse_configs

def main():
    parser = argparse.ArgumentParser(description="Run experiment for given configuration file.")
    parser.add_argument(
        "--port",
        type=int,
        default=59995,
        help="Port number to use for connectivity (default: 59995)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Number of Allocated GPUs (no default)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="Configuration file path (no default)",
    )
    args = parser.parse_args()
    print(f"Connect Port : {args.port}")
    print(f"# of GPUs    : {args.num_gpus}")
    print(f"Config File  : {args.config_file}")
    # all_configs = [join(args.configs_path, f) for f in listdir(args.configs_path) if isfile(join(args.configs_path, f))]

    print(f"Running experiments for: {args.config_file}")
    
    # Variables to hold subprocess information
    server_proc = None
    honest_client_proc = None
    malicious_client_proc = None

    # Extracting the name of experiment file
    # will be used to redirect stdout
    exp_name = ntpath.basename(args.config_file)[:-5]
        
    # Extract required user configurations
    user_configs = parse_configs(args.config_file)
    total_clients = user_configs["SERVER_CONFIGS"]["MIN_NUM_CLIENTS"]
    num_mal_clients = int(user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_FRAC"] * total_clients)
    num_hon_clients = total_clients - num_mal_clients
    mal_client_type = user_configs["EXPERIMENT_CONFIGS"]["MAL_CLIENT_TYPE"]

    # Create stdout re-direction files
    os.makedirs(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], exist_ok=True)
    server_log = open(join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"console_{exp_name}_server.log"), "w")
    honest_log = open(join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"console_{exp_name}_honest.log"), "w")
    malice_log = open(join(user_configs["OUTPUT_CONFIGS"]["RESULT_LOG_PATH"], f"console_{exp_name}_malice.log"), "w")
    
    try:
        # Run FL Server
        server_call = f'python -u src/run_fl_server.py --server_address="0.0.0.0:{args.port}" --config_file="{args.config_file}"'
        server_proc = subprocess.Popen([server_call], shell=True, stdout=server_log, stderr=server_log, preexec_fn=os.setsid)
        
        time.sleep(30)      # Allow server to setup and begin listening on the port

        # Run Honest Clients
        if num_hon_clients > 0:
            honest_client_call = f'python -u src/run_fl_clients.py --server_address="127.0.0.1:{args.port}" --client_type=HONEST --max_gpus={args.num_gpus} --total_clients={total_clients} --num_clients={num_hon_clients} --start_cid=0 --config_file="{args.config_file}"'
            honest_client_proc = subprocess.Popen([honest_client_call], shell=True, stdout=honest_log, stderr=honest_log, preexec_fn=os.setsid)

        # Run Malicious Clients
        if num_mal_clients > 0:
            malicious_client_call = f'python -u src/run_fl_clients.py --server_address="127.0.0.1:{args.port}" --client_type={mal_client_type} --max_gpus={args.num_gpus} --total_clients={total_clients} --num_clients={num_mal_clients} --start_cid={num_hon_clients} --config_file="{args.config_file}"'
            malicious_client_proc = subprocess.Popen([malicious_client_call], shell=True, stdout=malice_log, stderr=malice_log, preexec_fn=os.setsid)
        
        # Wait for processes to terminate
        if honest_client_proc:
            honest_client_proc.wait()
        if malicious_client_proc:
            malicious_client_proc.wait()
        if server_proc:
            server_proc.wait()
    except Exception as err:
        print(f"[ERROR]: Exception was raised, performing cleanups")
        
        # Clean up in case of an exception
        if server_proc is not None: 
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        if honest_client_proc is not None: 
            os.killpg(os.getpgid(honest_client_proc.pid), signal.SIGTERM)
        if malicious_client_proc is not None: 
            os.killpg(os.getpgid(malicious_client_proc.pid), signal.SIGTERM)

        # Print exception traceback            
        traceback.print_tb(err.__traceback__)
        print("================================================\n\n", flush=True)

    finally:    
        # Flush log files
        server_log.flush()
        honest_log.flush()
        malice_log.flush()

        # Close log files
        server_log.close()
        honest_log.close()
        malice_log.close()

if __name__=="__main__":
    main()
