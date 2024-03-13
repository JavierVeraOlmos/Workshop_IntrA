import trimesh
import numpy as np
import open3d as o3d
import math
import random

vessel_color = [255, 0, 0, 255]
an_color = [0, 255, 0, 255]

def process_mesh(trimesh_mesh):
    vertex_ls = np.array(trimesh_mesh.vertices)
    tri_ls = np.array(trimesh_mesh.faces) + 1

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertex_ls)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tri_ls) - 1)
    mesh.compute_vertex_normals()
    return mesh


# Specify the path to your text file
file_path = "dataset/IntrA/annotated/ad/AN1-_norm.ad"

# Load the data from the text file into a NumPy array
data_array = np.loadtxt(file_path)


path_obj = "dataset/IntrA/annotated/obj/AN1_full.obj"

mesh = trimesh.load_mesh(path_obj)
center = mesh.centroid

angle1 = random.uniform(0, math.pi)
rot_mat1 = trimesh.transformations.rotation_matrix(angle1, [1,0,0], center)

angle2 = random.uniform(0, math.pi)
rot_mat2 = trimesh.transformations.rotation_matrix(angle2, [0,1,0], center)

angle3 = random.uniform(0, math.pi)
rot_mat3 = trimesh.transformations.rotation_matrix(angle3, [0,0,1], center)
mesh.apply_transform(rot_mat1@rot_mat2@rot_mat3)


for index, annotation in enumerate(data_array):
    if annotation[-1] != 0:
        mesh.visual.vertex_colors[index] = np.array(an_color)
    else:
        mesh.visual.vertex_colors[index] = np.array(vessel_color)


# new_mesh = process_mesh(mesh)

mesh.show()