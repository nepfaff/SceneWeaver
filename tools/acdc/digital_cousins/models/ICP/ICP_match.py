# import open3d as o3d
# import numpy as np

# def scale_transform(source, target, threshold, init_trans=np.eye(4)):
#     """
#     Perform a custom ICP that adjusts for scaling.
#     """
#     source_temp = source
#     target_temp = target

#     # # Perform initial point-to-point ICP
#     # icp_result = o3d.pipelines.registration.registration_icp(
#     #     source_temp, target_temp, threshold, init_trans,
#     #     o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     # )
    
#     # # Compute scale factor
#     # source_points = np.asarray(source_temp.points)
#     # target_points = np.asarray(target_temp.points)
    
#     # # Apply initial transformation
#     # transformed_source_points = (
#     #     np.dot(icp_result.transformation[:3, :3], source_points.T).T +
#     #     icp_result.transformation[:3, 3]
#     # )
    
#     # # Calculate scale factor based on average distances
#     # scale = np.linalg.norm(target_points) / np.linalg.norm(transformed_source_points)
    
#     # # Apply scale
#     # scaled_source_points = transformed_source_points * scale

#     # # Create a new point cloud for the scaled source
#     # scaled_source = o3d.geometry.PointCloud()
#     # scaled_source.points = o3d.utility.Vector3dVector(scaled_source_points)

#     # Re-run ICP for fine-tuning after scaling
#     icp_result_scaled = o3d.pipelines.registration.registration_icp(
#         source_temp, target_temp, threshold, np.eye(4),
#         o3d.pipelines.registration.TransformationEstimationPointToPoint()
#     )

#     return icp_result_scaled

# # Load point clouds
# mesh = o3d.io.read_triangle_mesh("/home/yandan/Desktop/book3.obj")
# source = mesh.sample_points_uniformly(number_of_points=5000)

# # Load point clouds
# mesh = o3d.io.read_triangle_mesh("/home/yandan/Desktop/book3.obj")
# source1 = mesh.sample_points_uniformly(number_of_points=5000)

# mesh = o3d.io.read_triangle_mesh("/home/yandan/Desktop/book4.obj")
# target = mesh.sample_points_uniformly(number_of_points=5000)

# mesh = o3d.io.read_triangle_mesh("/home/yandan/Desktop/book4.obj")
# target1 = mesh.sample_points_uniformly(number_of_points=5000)


# # Threshold and initial transformation
# threshold = 0.02  # Adjust based on point cloud scale
# init_trans = np.eye(4)

# # Perform scale-aware ICP
# icp_result = scale_transform(source, target, threshold, init_trans)

# print("Final Transformation:\n", icp_result.transformation)
# # print("Scale Factor:", scale_factor)

# # Visualize results
# source_temp = source1
# source_temp.transform(icp_result.transformation)
# scaled_source = source_temp

# o3d.visualization.draw_geometries(
#     [scaled_source.paint_uniform_color([1, 0, 0]), 
#      target1.paint_uniform_color([0, 1, 0])]
# )



# # Optionally, downsample the point clouds for better performance
# source = source.voxel_down_sample(voxel_size=0.05)
# target = target.voxel_down_sample(voxel_size=0.05)

# # Preprocess the point clouds (normals calculation for better alignment)
# source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# # Initial alignment (using an identity matrix)
# threshold = 0.02  # Maximum distance between two points to be considered as correspondences
# trans_init = np.eye(4)  # Identity matrix, initial transformation (no transformation)
  
# # Apply ICP algorithm
# # max_iteration can be adjusted based on how much precision you want
# icp_result = o3d.pipelines.registration.registration_icp(
#     source, target, threshold, trans_init, 
#     o3d.pipelines.registration.TransformationEstimationPointToPoint())

# # Print the ICP transformation matrix
# print("Transformation matrix:")
# print(icp_result.transformation)

# # Apply the transformation to the source point cloud
# source.transform(icp_result.transformation)

# # Visualize the aligned point clouds
# o3d.visualization.draw_geometries([source, target])


import open3d as o3d
import numpy as np

def load_and_prepare_point_cloud(file_path, voxel_size=None):
    """
    Load a point cloud and optionally downsample it.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd

def apply_icp(source, target, threshold=0.02):
    """
    Perform ICP alignment on the source point cloud to match the target point cloud.
    """
    # Initial transformation matrix (identity matrix)
    trans_init = np.eye(4)

    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(
    #     source, target, threshold, trans_init)
    # print(evaluation)

    # Perform ICP registration
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    return icp_result

def visualize_point_clouds(pcd1, pcd2, title="Point Cloud Alignment"):
    """
    Visualize two point clouds together.
    """
    pcd1.paint_uniform_color([1, 0, 0])  # Source in red
    pcd2.paint_uniform_color([0, 1, 0])  # Target in green
    o3d.visualization.draw_geometries([pcd1, pcd2], window_name=title)

# File paths for source and target point clouds
source_file = "/home/yandan/Desktop/book3.obj"  # Replace with your source point cloud file path
target_file = "/home/yandan/Desktop/book4.obj"  # Replace with your target point cloud file path

# Step 1: Load and preprocess the point clouds
source_pcd = load_and_prepare_point_cloud(source_file, voxel_size=0.005)
target_pcd = load_and_prepare_point_cloud(target_file, voxel_size=0.005)

# Step 2: Perform ICP alignment
threshold = 0.02  # Set the max correspondence distance
icp_result = apply_icp(source_pcd, target_pcd, threshold)

# Step 3: Transform the source point cloud using the ICP result
print(icp_result.transformation)
source_pcd.transform(icp_result.transformation)

# Step 4: Visualize the aligned point clouds
visualize_point_clouds(source_pcd, target_pcd)

# Print ICP results
print("Transformation Matrix:")
print(icp_result.transformation)
print(f"Fitness: {icp_result.fitness:.4f}")
print(f"Inlier RMSE: {icp_result.inlier_rmse:.4f}")
