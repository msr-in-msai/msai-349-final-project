import numpy as np
import open3d as o3d
import time
import random
import json
import cv2
import math
import collections
import copy
import os
import matplotlib.pyplot as plt


# Pointcloud Visualizer
class VISUALIZER:
    def __init__(self, num_groups) -> None:
        # create visualizer and window.
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(height=480, width=640)

        # initialize pointcloud instance.
        self._pcds = []
        self._pcd_points = []
        self._pcd_colors = []

        # Set render options
        self._set_render_options()

        # Initizalize pointclouds with specific groups
        self._register_points(num_groups)

    def _register_points(self, num_points_cluster):
        for i in range(num_points_cluster):
            points = [[0,0,0]]
            _pcd = o3d.geometry.PointCloud()
            _pcd.points = o3d.utility.Vector3dVector(points)
            _pcd.paint_uniform_color([0,0,0])
            self._pcds.append(_pcd)
        self._pcd_points = []
        self._pcd_colors = []
        for _pcd_idx in range(len(self._pcds)):
            for _point_idx in range(len(self._pcds[_pcd_idx].points)):
                self._pcd_points.append(list(self._pcds[_pcd_idx].points[_point_idx]))
                self._pcd_colors.append(list(self._pcds[_pcd_idx].colors[_point_idx]))

    def _set_render_options(self):
        opt = self._vis.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([1, 1, 1])

    def set_points_uni_color(self, group_index, points, color):
        self._pcds[group_index].points = o3d.utility.Vector3dVector(points)
        self._pcds[group_index].paint_uniform_color(color)

    def set_points_ind_color(self, group_index, points, colors):
        self._pcds[group_index].points = o3d.utility.Vector3dVector(points)
        self._pcds[group_index].colors = o3d.utility.Vector3dVector(colors)

    def _element_in_list(self, element, list):
        for idx in range(len(list)):
            e = list[idx]
            if(collections.Counter(e) == collections.Counter(element)):
                return True, idx
        idx = -1
        return False, idx
    
    def visualize(self):
        for _pcd in self._pcds:
            self._vis.add_geometry(_pcd)
        self._vis.poll_events()
        self._vis.update_renderer()
        self._vis.run()

    def destroy(self):
        self._vis.destroy_window()


class DEPTH_TO_POINTCLOUD:
    def __init__(self, distance_to_camera_path, rgb_path, seg_path, camera_params_path) -> None:
        self._distance_to_camera_path = distance_to_camera_path
        self._rgb_path = rgb_path
        self._seg_path = seg_path
        self._camera_params_path = camera_params_path
        self._setup()

    def _setup(self):
        self._read_params()
        self._get_intrinsic_matrix()
        self._get_extrinsic_matrix()

    def _read_params(self):
        self._depth = np.load(self._distance_to_camera_path)
        depth_image_normalized = cv2.normalize(self._depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_image_normalized = depth_image_normalized.astype(np.uint8)  # Convert to 8-bit format
        cv2.imwrite('depth_image.png', depth_image_normalized)
        
        self._rgb = cv2.imread(self._rgb_path, cv2.COLOR_BGR2RGB)
        self._seg = cv2.imread(self._seg_path, cv2.COLOR_BGR2RGB)
        with open(self._camera_params_path) as f:
            self._camera_params = json.load(f)

        self._h_aperture = self._camera_params["cameraAperture"][0]
        self._v_aperture = self._camera_params["cameraAperture"][1]
        self._h_aperture_offset = self._camera_params["cameraApertureOffset"][0]
        self._v_aperture_offset = self._camera_params["cameraApertureOffset"][1]
        self._focal_length = self._camera_params["cameraFocalLength"]
        self._h_resolution = self._camera_params["renderProductResolution"][0]
        self._v_resolution = self._camera_params["renderProductResolution"][1]
        self._cam_t = self._camera_params["cameraViewTransform"]

    def _get_intrinsic_matrix(self):
        self._focal_x = self._h_resolution * self._focal_length / self._h_aperture
        self._focal_y = self._v_resolution * self._focal_length / self._v_aperture
        self._center_x = self._h_resolution / 2
        self._center_y = self._v_resolution / 2
        self.intrinsic_matrix = np.array([[self._focal_x, 0, self._center_x],
                                          [0, self._focal_y, self._center_y],
                                          [0, 0, 1]])
        return self.intrinsic_matrix
        
    def _get_extrinsic_matrix(self):
        mat = np.resize(self._cam_t, (4,4))
        self._cam_pose = np.linalg.inv(mat).T
        FUCKING_OFFSET_MATRIX = np.mat([[1,0,0,0],
                                        [0,-1,0,0],
                                        [0,0,-1,0],
                                        [0,0,0,1]])
        self._cam_pose = self._cam_pose.dot(FUCKING_OFFSET_MATRIX)
        return self._cam_pose

    
    def get_pcd_np(self):
        u_indices , v_indices = np.meshgrid(np.arange(self._h_resolution), np.arange(self._v_resolution))
        x_factors = (u_indices - self._center_x) / self._focal_x
        y_factors = (v_indices - self._center_y) / self._focal_y

        z_mat = self._depth
        x_mat = x_factors * z_mat
        y_mat = y_factors * z_mat

        points = []
        colors = []
        t1 = time.time()
        for h in range(self._h_resolution):
            for v in range(self._v_resolution):
                if(self._seg[v][h][2] == 140 and self._seg[v][h][1] == 25 and self._seg[v][h][0] == 255):
                    x = x_mat[v][h]
                    y = y_mat[v][h]
                    z = z_mat[v][h]
                    
                    r = self._rgb[v][h][2]/255
                    g = self._rgb[v][h][1]/255
                    b = self._rgb[v][h][0]/255

                    point_mat = np.mat([[1,0,0,x],
                                        [0,1,0,y],
                                        [0,0,1,z],
                                        [0,0,0,1]])
                    transformed_point_mat = self._cam_pose.dot(point_mat)
                    x = transformed_point_mat[0,3]
                    y = transformed_point_mat[1,3]
                    z = transformed_point_mat[2,3]

                    points.append([x,y,z])
                    colors.append([r,g,b])
        print(point_mat)
        print(transformed_point_mat)
        t2 = time.time()
        print(t2-t1)
        
        return points, colors

        

if __name__ == "__main__":
    distance2plane_path_1 = "distance_to_image_plane_0593.npy"
    rgb_path_1 = "rgb_0593.png"
    seg_path_1 = "semantic_segmentation_0593.png"
    cam_params_path_1 = "camera_params_0593.json"

    depth_to_pcd_1 = DEPTH_TO_POINTCLOUD(distance2plane_path_1, rgb_path_1, seg_path_1, cam_params_path_1)
    pcd_1, colors_1 = depth_to_pcd_1.get_pcd_np()

    vis = VISUALIZER(1)
    vis.set_points_ind_color(group_index=0, points=pcd_1, colors=colors_1)
    vis.visualize()
    vis.destroy()