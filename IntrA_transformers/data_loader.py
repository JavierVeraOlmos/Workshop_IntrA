from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import torch
from train_configs.train_config_maker import TrainConfigMaker
import os
import math
import random

class IntrADataGenerator(Dataset):
    def __init__(self, data_folder_path, file_names) -> None:
        super().__init__()
        self.data_folder_path = data_folder_path
        self.file_names = file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        name = self.file_names[idx].split("_")[0]
        data_array = np.loadtxt(os.path.join(self.data_folder_path, "ad",name + "-_norm.ad"))
        mesh = trimesh.load_mesh(os.path.join(self.data_folder_path, "obj", self.file_names[idx]))
        center = mesh.centroid
        faces = mesh.faces
        angle1 = random.uniform(0, math.pi)
        rot_mat1 = trimesh.transformations.rotation_matrix(angle1, [1,0,0], center)
        
        angle2 = random.uniform(0, math.pi)
        rot_mat2 = trimesh.transformations.rotation_matrix(angle2, [0,1,0], center)
        
        angle3 = random.uniform(0, math.pi)
        rot_mat3 = trimesh.transformations.rotation_matrix(angle3, [0,0,1], center)
        mesh.apply_transform(rot_mat1@rot_mat2@rot_mat3)

        tras_mat = trimesh.transformations.translation_matrix(-mesh.centroid)
        mesh.apply_transform(tras_mat)

        assert len(mesh.vertices) == data_array.shape[0]
        data_array[data_array[:,-1] == 2, -1] = 1
        feats = np.concatenate([np.array(mesh.vertices)/np.max(np.abs(mesh.vertices),axis=0), np.array(mesh.vertex_normals)], axis = 1)

        return [feats, data_array[:,[-1]],torch.from_numpy(np.array(faces))]



def collate_fn(batch):

    feats_input = []
    labels = []
    faces = []
    for batch_item in batch:
        feats_input.append(torch.Tensor(batch_item[0]))
        labels.append(torch.Tensor(batch_item[1]))  
        faces.append(batch_item[2])
    return [feats_input,labels,faces]



def get_generator_set(config: TrainConfigMaker):
    
    train_loader = DataLoader(IntrADataGenerator(config.dataset_config.input_data_dir_path,
                                                  config.dataset_config.train_data_split),
                            shuffle=True,
                            batch_size=config.dataset_config.train_batch_size,
                            collate_fn=collate_fn,
                            num_workers=8)

    val_loader =  DataLoader(IntrADataGenerator(config.dataset_config.input_data_dir_path,
                                                  config.dataset_config.val_data_split),
                            shuffle=True,
                            batch_size=config.dataset_config.train_batch_size,
                            collate_fn=collate_fn,
                            num_workers=2)

    return [train_loader, val_loader]




















# for data loader testing and demo
if __name__ == "__main__":
    import pandas as pd
    from gen_utils import (
        rodrigues_rotation_matrix,
        get_rotation_angles_from_rot_mat,
    )
    from time import sleep

    df = pd.read_csv("output\\movements.csv")

    data_generator = DentalMovModelGenerator(
        ".",
        df,
        augment=True,
    )
    LABEL = 16
    for batch in data_generator:
        gt_angles = batch["gt_movements"][LABEL, 0:3].reshape(1, 3)
        rot_mat_rodrigues = rodrigues_rotation_matrix(torch.Tensor(gt_angles))
        # rot_mat_rodrigues_roma = rotvec_to_rotmat(torch.Tensor(gt_angles[0]))
        # print(
        #     f"error in rodrigues owncode vs roma: \
        #     {np.mean(np.abs(rot_mat_rodrigues.numpy() - rot_mat_rodrigues_roma.numpy()))}"
        # )
        angles = get_rotation_angles_from_rot_mat(rot_mat_rodrigues.numpy()[0:3, 0:3])
        print(f"angles_gt: {gt_angles}")
        print(f"angles from rodrigues: {angles}")

        rot_mat_x = trimesh.transformations.rotation_matrix(gt_angles[0, 0], [1, 0, 0])
        rot_mat_y = trimesh.transformations.rotation_matrix(gt_angles[0, 1], [0, 1, 0])
        rot_mat_z = trimesh.transformations.rotation_matrix(gt_angles[0, 2], [0, 0, 1])

        rot_mat = np.dot(rot_mat_z, np.dot(rot_mat_y, rot_mat_x))

        gs_rot_mat = get_rotation_matrix_from_vecs(
            torch.Tensor(rot_mat[0, 0:3]), torch.Tensor(rot_mat[1, 0:3])
        )
        angles_gs = get_rotation_angles_from_rot_mat(gs_rot_mat.numpy()[0:3, 0:3])
        print(f"angles from gramsmitch: {angles_gs}")

        pre_coord = batch["feat"][batch["gt_labels"] == LABEL, 0:3]
        C = batch["feat"][batch["gt_labels"] == LABEL, -3::]

        centered_pre_coord = pre_coord - C

        post_coords_applying_gt = (
            centered_pre_coord.dot(rot_mat[0:3, 0:3].T)
            + batch["gt_movements"][LABEL, 3:6]
            + C
        )
        post_coords_from_rodrigues = (
            centered_pre_coord.dot(rot_mat_rodrigues[0:3, 0:3].T)
            + batch["gt_movements"][LABEL, 3:6]
            + C
        )
        post_coords_from_gs = (
            centered_pre_coord.dot(gs_rot_mat[0:3, 0:3].T)
            + batch["gt_movements"][LABEL, 3:6]
            + C
        )

        post_coords_in_gt = batch["gt_points"][batch["gt_labels"] == LABEL]

        error_using_gt_angles = np.mean(
            np.abs(post_coords_applying_gt - post_coords_in_gt)
        )
        error_using_rodrigues = np.mean(
            np.abs(post_coords_from_rodrigues - post_coords_in_gt)
        )
        error_using_gs = np.mean(np.abs(post_coords_from_gs - post_coords_in_gt))

        print(f"error_using_gt_angles: {error_using_gt_angles}")
        print(f"error_using_rodrigues: {error_using_rodrigues}")
        print(f"error_using_gs: {error_using_gs}")
        sleep(3)
