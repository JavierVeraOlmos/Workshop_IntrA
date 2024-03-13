import torch
from math import inf
import mlflow
from loss_meter import MetricMeter, LossMap, MetricMap
from data_loader import get_generator_set
from train_configs.train_config_maker import TrainConfigMaker
from models.base_model import BaseModel

LOG_MLFLOW = True

class Trainer:
    def __init__(
        self, config: TrainConfigMaker = None, model: BaseModel = None, gen_set=None
    ):
        self.gen_set = gen_set
        self.config = config
        self.model = model

        self.val_count = 0
        self.train_count = 0
        self.step_count = 0
        self.best_val_loss = inf

    def train(self, epoch, data_loader):
        
        total_loss_meter = MetricMeter()
        step_loss_meter = MetricMeter()

        data_loader_iter = iter(data_loader)
        for batch_idx in range(self.config.train_loop_config.iter_per_epoch):

            try:
                batch_item = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(data_loader)
                batch_item = next(data_loader_iter)

            loss, metrics = self.model.step(batch_idx, batch_item, "train")
            torch.cuda.empty_cache()
            
            loss_map = LossMap()
            metrics_map = MetricMap()
            
            loss_map.add_loss_by_dict(loss)
        
            metrics_map.add_metric_by_dict(metrics)

            total_loss_meter.aggr(
                {
                    **metrics_map.get_metric_dict_for_print("train"),
                    **loss_map.get_loss_dict_for_print("train"),
                }
            )
            step_loss_meter.aggr(
                {
                    **metrics_map.get_metric_dict_for_print("step"),
                    **loss_map.get_loss_dict_for_print("step"),
                }
            )
            
            if batch_idx % int(self.config.train_loop_config.iter_per_epoch/10) == 0:
                print(step_loss_meter.get_avg_results())
                
            if LOG_MLFLOW:
                mlflow.log_metrics(step_loss_meter.get_avg_results(), step=self.step_count)
                mlflow.log_metric("learning_rate", self.model.scheduler.get_last_lr()[0], step=self.step_count)

            self.step_count+=1
            
        self.model.scheduler.step(self.train_count)
        # reset step loss meter
        step_loss_meter.init()
        
        if LOG_MLFLOW:
            mlflow.log_metrics(total_loss_meter.get_avg_results(), step=self.train_count)
        self.train_count += 1
        self.model.save("train")
        print("model train saved, epoch {}".format(epoch))

    def val(self, epoch, data_loader, save_best_model):
        total_loss_meter = MetricMeter()
        for batch_idx, batch_item in enumerate(data_loader):
            loss, metrics = self.model.step(batch_idx, batch_item, "val")
            
            loss_map = LossMap()
            metrics_map = MetricMap()
            
            loss_map.add_loss_by_dict(loss)
        
            metrics_map.add_metric_by_dict(metrics)
            
            
            total_loss_meter.aggr(
                {
                    **loss_map.get_loss_dict_for_print("val"),
                    **metrics_map.get_metric_dict_for_print("val"),
                }
            )

        avg_total_loss = total_loss_meter.get_avg_results()
        if LOG_MLFLOW:
            mlflow.log_metrics(avg_total_loss, step=self.val_count )
        print(avg_total_loss)
        self.val_count+=1
        if save_best_model:
            if self.best_val_loss > avg_total_loss["total_val"]:
                self.best_val_loss = avg_total_loss["total_val"]
                self.model.save("val")
                print("model validation saved, epoch {}".format(epoch))

    def run(self, epochs):
        train_data_loader = self.gen_set[0][0]
        val_data_loader = self.gen_set[0][1]
        for epoch in range(epochs):
            self.train(epoch, train_data_loader)
            self.val(epoch, val_data_loader, True)


def runner(config: TrainConfigMaker, model):
    gen_set = [get_generator_set(config)]
    print("train_set", len(gen_set[0][0]))
    print("validation_set", len(gen_set[0][1]))
    trainner = Trainer(config=config, model=model, gen_set=gen_set)
    trainner.run(config.train_loop_config.epochs)
