import torch
from .base_model import BaseModel
from .modules.point_transformer import PointTransformer


def class_loss(cls_pred, gt_cls, weight=None):

    gt_cls = gt_cls.type(torch.long).cuda().transpose(0,1)
    cls_pred = cls_pred.transpose(0,1).unsqueeze(0)
    if weight is None:
        loss = torch.nn.CrossEntropyLoss().type(torch.float).cuda()(cls_pred, gt_cls)
    else:
        loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight).type(torch.float).cuda())(cls_pred, gt_cls)

    return loss

def class_acc(cls_pred, gt_cls, label):

    acc = 0
    for pred, gt in zip(cls_pred, gt_cls):

        gt = gt.type(torch.long)

        aux_gt_cls = gt[gt==label].cuda()

        aux_cls_pred = torch.argmax(pred, dim=1, keepdim=True)[gt==label]

        acc+=float((torch.sum(aux_cls_pred == aux_gt_cls) / len(aux_gt_cls)).item())
    return acc/len(gt_cls)

class SegPointTransformer(BaseModel):
    def __init__(self, model_config):

        point_transformer_net = PointTransformer(**model_config["model_parameter"])

        super().__init__(model_config, point_transformer_net)


    def infer(self, in_feats):
        self._set_model("test")
        with torch.no_grad():
            output = self.module(in_feats)
        return output

    def get_loss(self, sem_1, gt_seg_label_1):

        batch_looses = []

        for sem, gt in zip(sem_1, gt_seg_label_1):

            batch_looses.append(class_loss(sem, gt))
        
        loss = torch.mean(torch.stack(batch_looses))

        return loss

    def get_metrics(self, sem_1, gt_seg_label_1):
        tooth_class_acc_0 = class_acc(sem_1, gt_seg_label_1, 0)
        tooth_class_acc_1 = class_acc(sem_1, gt_seg_label_1, 1)

        return {
            "class_acc_0": (tooth_class_acc_0, 1),
            "class_acc_1": (tooth_class_acc_1, 1),
        }

    def step(self, batch_idx, batch_item, phase):
        self._set_model(phase)

        in_feats = batch_item[0]
        gt_labels = batch_item[1]

        

        if phase == "train":
            output = self.module(in_feats)

        else:
            with torch.no_grad():
                output = self.module(in_feats)


        loss = self.get_loss(
                output,
                gt_labels,
            )


        if phase == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


        return {
            "loss": (
                loss,
                1,
            ),
        } , self.get_metrics(output, gt_labels)
