# flower-PoisonAttack
A systematic implementation of Model Poison Attacks in Federated Machine Learning using flower framework. The code is designed to help researchers quickly deploy and run a FedML setup to straight away test already implemented attacks or to implement their own attacks / defenses quickly. 

## Install Dependencies

This implementation currently uses [PyTorch](https://github.com/pytorch/pytorch) as its primary deep learning framework along with [Flower](https://github.com/adap/flower) as the federated machine learning framework. In order to run this code, you need to install the required files given by the requirements.txt file. It is recommended to create a fresh virtual environment before carrying out these installations. 

Best way to do this is with 
[Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) virtual environment. 

```bash
conda create -n myEnv
conda activate myEnv
```

Once the new virtual environment is ready, the required dependencies can be installed using the following:

```bash
pip install -r requirements.txt
```

## Running the Code
There are two ways to run the code based by running directly from python scripts or by creating docker containers.

### Running with python scripts

#### Preparing Configurations
To run an attack or defense simulation you need a configurations file that specifies all the experiment details in ```YAML``` format. A sample file is provided under ```src/configs/exp_configs.yaml``` for reference you are welcome to modify it as per your needs.

#### Staring the Federated Server
Once the configuration file is ready, the next step isto start a Federated Server this can be done by running the ```run_fl_server.py``` file (with appropriate arguments) located under the ```src``` folder using:

```bash
python src/run_fl_server.py \
    --server_address=$SERVER_ADDRESS \
    --config_file="path/to/config/file.yaml" \
    --log_host=$LOG_SERVER_ADDRESS
```

Refer to this [running federated server](https://github.com/usamazf/flower-PoisonAttack/wiki/Running-Federated-Server) wiki for full explanation of all hyperparameters and how to use them.

#### Starting the Federated Clients

After the server is successfully up and running next step is to run the Federated Clients. We provide a bridge to deploy different types of clients with varying capabilities and functionalities. Each worker is either an honest worker or a malicious adversarial worker (currently we provide 1 type of honest and 2 types of malicious clients). 

We can run any number of instances of these clients using the ```run_fl_clients.py``` file located under ```src``` folder.(Note: in case you want to run multiple types of clients you need to run this file multiple times in different terminals.)

```bash
python src/run_fl_clients.py \
    --server_address=$SERVER_ADDRESS \
    --config_file="path/to/config/file.yaml" \
    --log_host=$LOG_SERVER_ADDRESS \
    --client_type=type_of_client_to_run[HONEST or MPAF or RAND] \
    --total_clients=total_clients_in_the_federation \
    --num_clients=number_of_client_instances_to_run \
    --start_cid=starting_client_id_of_these_set_of_clients
```

Refer to this [running federated clients](https://github.com/usamazf/flower-PoisonAttack/wiki/Running-Federated-Clients) wiki for full explanation of all hyperparameters and how to use them.

### Running with Docker

In order to run with docker you need to first build docker images for both server and client. The code does not come with prebuilt images and hence it is up to the user to build these images for deployment. Docker files for building server image as well as client image is provided as ```Dockerfile.Server``` and ```Dockerfile.Client``` respectively.

To build theses images you need to have docker engine installed and running. Once you are ready to build images use the following commands:

```bash
# build docker image for server module
docker build -t flower_server:latest -f Dockerfile.Server .

# build docker image for client module
docker build -t flower_client:latest -f Dockerfile.Client .
```

This step might take some time to finish as it needs to download all dependencies for the images. Once done you can deploy docker containers for both server and client side as follows:

```bash
# deploy a server container listening at port 8000
docker run -p 8000:8000 --name fl_server -d flower_server --config_file="configs/exp_configs.yaml" --server_address="[::]:8000"

# deploy a container for 2 honest clients in a federation of 4 total clients
docker run --network="host" -d flower_client --config_file="configs/exp_configs.yaml" --server_address="localhost:8000" --num_clients=2 --total_clients=4 --start_cid=0 --client_type="HONEST"
```

> **NOTE**
> The docker deployment is still under active development and might not work as intended. It is recommend to deploy experiments using python scripts as discussed above instead of docker containers while we work on improving the solution.

## Issues

Having issues? Just report in [the issue section](https://github.com/usamazf/flower-PoisonAttack/issues). **Thanks for the feedback!**


## Contribute

Fork this repository, make your changes and then issue a pull request. If you have new ideas that you do not want to implement, file a feature request and we will get to it as soon as possible.


## Support

Consider becoming a patron (**highly appreciated!**):

[![](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/usamazf)

... Or if you prefer a one-time tip:

[![](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/usamazfr)
