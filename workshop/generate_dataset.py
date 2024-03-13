import trimesh
import numpy as np
import open3d as o3d
import os
import glob

dataset_path = "dataset/IntrA/annotated"

annotated_objs = os.listdir(os.path.join(dataset_path,"obj"))
annotated_ad = os.listdir(os.path.join(dataset_path,"ad"))

npy_path = os.path.join(dataset_path, "npy_data")

if not os.path.exists(npy_path):
    os.mkdir(npy_path)

def process_mesh(trimesh_mesh):
    vertex_ls = np.array(trimesh_mesh.vertices)
    tri_ls = np.array(trimesh_mesh.faces) + 1

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls) - 1)
    mesh.compute_vertex_normals()
    return mesh


shapes_list = []

for obj in annotated_objs:

    name = obj.split("_")[0]
    print(name)
    data_array = np.loadtxt(os.path.join(dataset_path, "ad",name + "-_norm.ad"))
    mesh = trimesh.load_mesh(os.path.join(dataset_path, "obj", obj))
    mesh = process_mesh(mesh)
    assert len(mesh.vertices) == data_array.shape[0]
    data_array[data_array[:,-1] == 2, -1] = 1
    npy_data = np.concatenate([np.array(mesh.vertices), np.array(mesh.vertex_normals), data_array[:,[-1]]], axis = 1)
    print(npy_data.shape)
    shapes_list.append(npy_data.shape[0])
    # print(np.unique(npy_data[:,-1]))
    # np.save(os.path.join(npy_path, name + ".npy"), npy_data)
    
print(min(shapes_list))
