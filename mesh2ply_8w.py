import open3d as o3d
import numpy as np
import fire

def obj_to_ply(obj_filename, ply_filename):
    """
    Convert .obj file to .ply point cloud using Open3D.
    """
    # Read the .obj file
    mesh = o3d.io.read_triangle_mesh(obj_filename)
    # print(mesh.vertices)
    # print(mesh.faces)

    # Check if the mesh contains vertex normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Extract point cloud from the mesh
    # pcd = mesh.sample_points_uniformly(number_of_points=2000)
    pcd = mesh.sample_points_poisson_disk(number_of_points=80000)

    # Save the point cloud to a .ply file
    o3d.io.write_point_cloud(ply_filename, pcd)
    print(f"Point cloud saved to {ply_filename}")

# Replace 'path_to_obj.obj' with the path to your .obj file and
# 'output_path.ply' with the desired output path for the .ply file.
# obj_filename = './outputs/magic123-coarse-sd/new.png-a_panda_dancing@20231022-112824/save/it10000-export/model.obj'
# ply_filename = 'a.ply'
if __name__ == '__main__':
    # obj_to_ply(obj_filename, ply_filename)
    fire.Fire(obj_to_ply)
# def load_obj(filename):
#     """
#     Load the OBJ file and return vertices and faces.
#     """
#     vertices = []
#     faces = []

#     with open(filename, 'r') as file:
#         for line in file:
#             if line.startswith('v '):
#                 vertices.append(list(map(float, line.strip().split()[1:4])))
#             elif line.startswith('f'):
#                 face = [int(i.split('/')[0]) for i in line.strip().split()[1:]]
#                 faces.append(face)

#     return np.array(vertices), np.array(faces)

# res = load_obj(obj_filename)
# print(res[0].shape)
# print(res[1].shape)