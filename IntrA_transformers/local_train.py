import os
import pickle
import torch
from train_configs.train_config_maker import TrainConfigMaker
import random
from models.segmentation_model import SegPointTransformer
from trainer import runner
import mlflow
from mlflow import MlflowClient


LOG_MLFLOW = True

if LOG_MLFLOW:
    client = MlflowClient(tracking_uri="http://127.0.0.1:8080")
    experiment = mlflow.set_experiment("IntrA_workshop")
    mlflow.start_run(run_name = "run_train_2_deformation")

print(f"Is cuda available: {torch.cuda.is_available()}")
print(f"torch cuda device name: {torch.cuda.get_device_name(0)}")

output_model_folder = "output_model_folder"

if not os.path.exists("output_model_folder"):
    os.makedirs(output_model_folder)

input_data_path = "dataset/IntrA/annotated"

input_file_names = os.listdir(os.path.join(input_data_path,'obj'))

# Shuffle the data
seed_value = 42
random.seed(seed_value)
random.shuffle(input_file_names)

# Split into training and testing sets
train_size = int(0.75 * len(input_file_names))
train_data = input_file_names[:train_size]
val_data = input_file_names[train_size:]

BATCH_SIZE = 6

train_config = TrainConfigMaker(
    model_config_path = "IntrA_transformers/train_configs/segmentation_config.py",
    input_data_dir_path = input_data_path,
    train_data_split= train_data ,
    val_data_split = val_data,
    epochs = 10,
    iter_per_epoch = 5000,
    sampled_points = None,
    model_folder = output_model_folder,
    train_batch_size=BATCH_SIZE,
    val_batch_size=1,
)
print(train_config)


model = SegPointTransformer(train_config.model_config)
model.load_optimizer_config()

if LOG_MLFLOW:
    mlflow.log_params(train_config.model_config["model_parameter"])
    mlflow.log_params(train_config.model_config["tr_set"])
    mlflow.log_params(train_config.train_loop_config._asdict())
    mlflow.log_params({"batch_size": BATCH_SIZE})

runner(train_config, model=model)

if LOG_MLFLOW:

    # mlflow.log_artifact(os.path.join(train_config.model_config["model_folder"], "model_weights.h5"))
    # mlflow.log_artifact(os.path.join(train_config.model_config["model_folder"], "model_weights_val.h5"))

    mlflow.end_run()