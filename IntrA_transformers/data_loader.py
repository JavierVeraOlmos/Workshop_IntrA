from torch.utils.data import Dataset, DataLoader
import numpy as np
import trimesh
import torch
from train_configs.train_config_maker import TrainConfigMaker
import os
import math
import random
import open3d as o3d


class IntrADataGenerator(Dataset):
    def __init__(self, data_folder_path, file_names, data_aug=True) -> None:
        super().__init__()
        self.data_folder_path = data_folder_path
        self.file_names = file_names
        self.data_aug = data_aug

    def process_mesh(self, trimesh_mesh):
        vertex_ls = np.array(trimesh_mesh.vertices)
        tri_ls = np.array(trimesh_mesh.faces) + 1

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls) - 1)
        mesh.compute_vertex_normals()
        return mesh




    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        name = self.file_names[idx].split("_")[0]
        data_array = np.loadtxt(os.path.join(self.data_folder_path, "ad",name + "-_norm.ad"))
        mesh = trimesh.load_mesh(os.path.join(self.data_folder_path, "obj", self.file_names[idx]), file_type="obj")
        center = mesh.centroid
        faces = mesh.faces



        if self.data_aug:
            angle1 = random.uniform(0, math.pi)
            rot_mat1 = trimesh.transformations.rotation_matrix(angle1, [1,0,0], center)

            angle2 = random.uniform(0, math.pi)
            rot_mat2 = trimesh.transformations.rotation_matrix(angle2, [0,1,0], center)
            
            angle3 = random.uniform(0, math.pi)
            rot_mat3 = trimesh.transformations.rotation_matrix(angle3, [0,0,1], center)
            mesh.apply_transform(rot_mat1@rot_mat2@rot_mat3)
        

        tras_mat = trimesh.transformations.translation_matrix(-mesh.centroid)
        mesh.apply_transform(tras_mat)


        if self.data_aug:
            if random.uniform(0,1)>0.1:
                scaler = np.array([random.uniform(0.75,1.15), random.uniform(0.75,1.15), random.uniform(0.75,1.15)]) 
                mesh.vertices = mesh.vertices*scaler
        
        norm_coords = np.max(np.abs(mesh.vertices))

        assert len(mesh.vertices) == data_array.shape[0]
        data_array[data_array[:,-1] == 2, -1] = 1
        mesh = self.process_mesh(mesh)
        feats = np.concatenate([np.array(mesh.vertices)/norm_coords, np.array(mesh.vertex_normals)], axis = 1)

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
                                                  config.dataset_config.val_data_split, data_aug=False),
                            shuffle=True,
                            batch_size=config.dataset_config.train_batch_size,
                            collate_fn=collate_fn,
                            num_workers=2)

    return [train_loader, val_loader]
