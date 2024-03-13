import importlib.util
import sys
import pickle
from collections import namedtuple


DatasetConfig = namedtuple(
    "DatasetConfig",
    [
        "input_data_dir_path",
        "train_data_split",
        "val_data_split",
        "train_batch_size",
        "val_batch_size",
    ],
)

TrainLoopConfig = namedtuple(
    "TrainLoopConfig", ["epochs", "sampled_points", "iter_per_epoch"]
)


def load_model_config(config_path, model_folder="output_model"):
    if config_path is None:
        print("loading repository config")
        config_path = "IntrA_transformers.train_configs.segmentation_config.py"

    if config_path.endswith(".py"):
        print("loading .py config")
        spec = importlib.util.spec_from_file_location("module.name", config_path)
        loaded_model_config = importlib.util.module_from_spec(spec)
        sys.modules["module.name"] = loaded_model_config
        spec.loader.exec_module(loaded_model_config)
        config = loaded_model_config.config
    elif config_path.endswith(".pkl"):
        print("loading .pkl config")
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    else:
        loaded_model_config = importlib.import_module(config_path)
        config = loaded_model_config.config

    config["model_folder"] = model_folder
    return config


class TrainConfigMaker:
    def __init__(
        self,
        model_config_path,
        input_data_dir_path,
        train_data_split,
        val_data_split,
        epochs,
        iter_per_epoch,
        sampled_points,
        model_folder,
        train_batch_size,
        val_batch_size,
        **kwargs,
    ) -> None:
        self.dataset_config = DatasetConfig(
            input_data_dir_path,
            train_data_split,
            val_data_split,
            train_batch_size,
            val_batch_size,
        )
        self.train_loop_config = TrainLoopConfig(
            int(epochs),  sampled_points, iter_per_epoch
        )
        self.model_config = load_model_config(model_config_path, model_folder)

        # add all epochs to the scheduler if not set in config
        if (
            "scheduler" in self.model_config["tr_set"].keys()
            and "full_steps" not in self.model_config["tr_set"]["scheduler"].keys()
        ):
            self.model_config["tr_set"]["scheduler"]["full_steps"] = int(epochs)
        self.extra_params = kwargs
