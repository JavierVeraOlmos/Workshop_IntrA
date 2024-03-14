# Workshop Point Transformer with IntrA

This repository aims to reproduce the results from https://github.com/radreports/IntrA-cerebral but using an simple adaptation of the Point transformer network from https://github.com/POSTECH-CVLab/point-transformer.


Install the requirements in a venv and change the wsl installation of cuda for your distribution

```console
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt 
```


Install Cuda, this is the WSL version. You may meed a different one.
```console
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

In order to run some Cuda functions it is mandatory to install Pointops.

```console
python IntrA_transformers/external_libs/pointops/setup.py install
```

And finally run in a different terminal the Mlflow server:

```console
mlflow server --host 127.0.0.1 --port 8080
```

To run the train use the local_train.py script and then run the local_test.py