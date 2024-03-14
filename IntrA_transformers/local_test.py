import os
import torch
from train_configs.train_config_maker import load_model_config
import random
from models.segmentation_model import SegPointTransformer
from torch.utils.data import DataLoader
from data_loader import IntrADataGenerator, collate_fn
import trimesh
import numpy as np


print(f"Is cuda available: {torch.cuda.is_available()}")
print(f"torch cuda device name: {torch.cuda.get_device_name(0)}")


output_model_folder = "output_model_folder"

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
print(val_data)

model_config = load_model_config("IntrA_transformers/train_configs/segmentation_config.py",output_model_folder)


model = SegPointTransformer(model_config)
model.load()

vessel_color = [255, 0, 0, 255]
an_color = [0, 255, 0, 255]



val_loader =  DataLoader(IntrADataGenerator(input_data_path,
                                            val_data, data_aug=True),
                        shuffle=False,
                        batch_size=1,
                        collate_fn=collate_fn,
                        num_workers=1)

ann_acc_list = []
ves_acc_list = []
ann_IoU_val = []
ves_IoU_val = []
for val_item in val_loader:
    pred_seg = model.infer(val_item[0])
    pred_seg=torch.argmax(pred_seg[0], dim=1).unsqueeze(1)


    mesh = trimesh.Trimesh(vertices=val_item[0][0][:,0:3].numpy(),
                       faces=val_item[2][0].numpy())

    gt_cls = val_item[1][0]
    
    aux_gt_cls = gt_cls[gt_cls==0]

    aux_cls_pred = pred_seg[gt_cls==0].cpu()

    acc = torch.sum(aux_cls_pred == aux_gt_cls) / len(aux_gt_cls)
    print(f"accuracy 0:{acc}")
    ves_acc_list.append(acc)

    aux_gt_cls = gt_cls[gt_cls==1]

    aux_cls_pred = pred_seg[gt_cls==1].cpu()

    acc = torch.sum(aux_cls_pred == aux_gt_cls) / len(aux_gt_cls)
    print(f"accuracy 1:{acc}")

    ann_acc_list.append(acc)


    ann = pred_seg.clone()
    ann_gt = val_item[1][0].cuda()

    intersection = torch.logical_and(ann, ann_gt).sum()
    union = torch.logical_or(ann, ann_gt).sum()

    iou = intersection.float() / union.float()
    print(F"ANN IoU:{iou}")
    ann_IoU_val.append(iou)

    ves = pred_seg.clone()
    ves[pred_seg==0] = 1
    ves[pred_seg!=0] = 0
    ves_gt = val_item[1][0].clone().cuda()
    ves_gt[val_item[1][0]==0] = 1
    ves_gt[val_item[1][0]!=0] = 0

    intersection = torch.logical_and(ves, ves_gt).sum()
    union = torch.logical_or(ves, ves_gt).sum()

    iou = intersection.float() / union.float()
    print(F"VES IoU:{iou}")
    ves_IoU_val.append(iou)

    for index, annotation in enumerate(pred_seg):
        if annotation != 0:
            mesh.visual.vertex_colors[index] = np.array(an_color)
        else:
            mesh.visual.vertex_colors[index] = np.array(vessel_color)

    # mesh.show()

ves_IoU = torch.mean(torch.Tensor(ves_IoU_val))
ann_IoU = torch.mean(torch.Tensor(ann_IoU_val))
ves_acc = torch.mean(torch.Tensor(ves_acc_list))
ann_acc = torch.mean(torch.Tensor(ann_acc_list))            
print(f"Mean VES IoU:{ves_IoU}")
print(f"Mean ANN IoU:{ann_IoU}")
print(f"Mean VES ACC:{ves_acc}")
print(f"Mean ANN ACC:{ann_acc}")