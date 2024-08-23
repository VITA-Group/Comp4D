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

if __name__ == '__main__':
    # obj_to_ply(obj_filename, ply_filename)
    fire.Fire(obj_to_ply)
