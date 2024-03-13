# Workshop Point Transformer with IntrA

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
