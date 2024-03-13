from abc import ABCMeta, abstractmethod
import torch
from torch.optim.lr_scheduler import ExponentialLR
import os
import copy
from external_libs.scheduler import build_scheduler_from_cfg


class BaseModel(metaclass=ABCMeta):
    def __init__(self, config: dict, module):
        self.config = config
        self.module = module
        self.module.train()
        self.module.cuda()

    def load_optimizer_config(self):
        if self.config["tr_set"]["optimizer"]["NAME"] == "sgd":
            self.optimizer = torch.optim.SGD(
                self.module.parameters(),
                lr=self.config["tr_set"]["optimizer"]["lr"],
                momentum=self.config["tr_set"]["optimizer"]["momentum"],
                weight_decay=self.config["tr_set"]["optimizer"]["weight_decay"],
            )
        elif self.config["tr_set"]["optimizer"]["NAME"] == "adam":
            self.optimizer = torch.optim.Adam(
                self.module.parameters(),
                lr=self.config["tr_set"]["optimizer"]["lr"],
                weight_decay=self.config["tr_set"]["optimizer"]["weight_decay"],
            )

        if self.config["tr_set"]["scheduler"]["sched"] == "exp":
            self.scheduler = ExponentialLR(
                self.optimizer, self.config["tr_set"]["scheduler"]["step_decay"]
            )
        elif self.config["tr_set"]["scheduler"]["sched"] == "cosine":
            sched_config = copy.deepcopy(self.config)
            sched_config["tr_set"]["scheduler"]["full_steps"] = self.config["tr_set"]["scheduler"]["full_steps"]
            sched_config["tr_set"]["scheduler"]["lr"] = self.config["tr_set"][
                "optimizer"
            ]["lr"]
            sched_config["tr_set"]["scheduler"]["min_lr"] = self.config["tr_set"][
                "scheduler"
            ]["min_lr"]
            self.scheduler = build_scheduler_from_cfg(
                sched_config["tr_set"]["scheduler"], self.optimizer
            )

    def _set_model(self, phase):
        if phase == "train":
            self.module.train()
        elif phase == "val" or phase == "test":
            self.module.eval()
        else:
            raise "phase is something unknown"

    def load(self):
        self.module.load_state_dict(torch.load(os.path.join(self.config["model_folder"], "model_weights.h5")))

    def save(self, phase):
        if phase == "train":
            torch.save(self.module.state_dict(), os.path.join(self.config["model_folder"], "model_weights.h5"))
        elif phase == "val":
            torch.save(self.module.state_dict(), os.path.join(self.config["model_folder"], "model_weights_val.h5"))
        else:
            raise "phase is something unknown"

    @abstractmethod
    def get_loss(self):
        pass

    @abstractmethod
    def step(self, batch_idx, batch_item):
        pass

    @abstractmethod
    def infer(self, batch_idx, batch_item):
        pass
