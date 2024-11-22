"""Convert RGB, Depth, and Segmentation samples into pointcloud."""

import csv
import os
import json
import pickle

import cv2

import numpy as np

from sklearn.neighbors import NearestNeighbors


class ConvertDataset:
    """Convert RGB, Depth, and Segmentation samples into pointcloud."""

    def __init__(self):
        """
        Initialize the class.

        Args:
            None

        Returns:
            None
        """
        pass

    def convert_dataset(self,
                        dataset_path: str,
                        output_path: str,
                        minimum_num_of_points: int = 2000,
                        output_num_of_points: int = 10000) -> None:
        """
        Convert RGB, Depth, and Segmentation samples into pointcloud.

        Args:
            dataset_path (str): Path to the dataset.
            output_path (str): Path to the output directory.
            minimum_num_of_points (int): Minimum number of points in a
                pointcloud.
            output_num_of_points (int): Number of points in a pointcloud.

        Returns:
            None
        """
        # Read the dataset
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                # Entry point for a sample in a scene based on the file name
                if (path[-4:] == '.png' and path[-30:-8] ==
                   'semantic_segmentation_'):
                    # Get other file paths related to this sample
                    scene_folder_directory = path[:-30]
                    data_id = path[-8:-4]
                    print(scene_folder_directory, data_id)
                    segmantic_segmentation_path = path
                    segmantic_segmentation_labels_path =\
                        scene_folder_directory +\
                        'semantic_segmentation_labels_'+data_id+'.json'
                    rgb_image_path = scene_folder_directory+'rgb_' +\
                        data_id+'.png'
                    depth_image_path = scene_folder_directory +\
                        'distance_to_image_plane_'+data_id+'.npy'
                    camera_params_path = scene_folder_directory +\
                        'camera_params_'+data_id+'.json'
                    occluder_name_path = scene_folder_directory +\
                        'occluder.pickle'

                    # Read the files
                    occluder_name =\
                        self._read_pickle_file(occluder_name_path)[0]
                    occluder_label = self._get_obj_label(occluder_name)
                    rgb_image = self._read_image_file(rgb_image_path)
                    depth_image = np.load(depth_image_path)
                    seg_image = np.asanyarray(
                        self._read_image_file(segmantic_segmentation_path))
                    seg_labels = self._read_json_file(
                            segmantic_segmentation_labels_path)
                    camera_params = self._read_json_file(camera_params_path)

                    # Get camera intrinsics
                    cam_K = self._get_camera_intrinsics(camera_params)

                    # Get camera extrinsics
                    cam_T = self._get_camera_extrinsics(camera_params)

                    # Get a dict with labels as keys and pointclouds as values
                    classified_scene_sample = self._classify_pointcloud(
                        rgb_image, depth_image, seg_image, seg_labels,
                        cam_K, cam_T, camera_params, occluder_label,
                        minimum_num_of_points, output_num_of_points)

                    # Write the classified pointcloud to a CSV file
                    for label, pointcloud in classified_scene_sample.items():
                        pointcloud = ";".join([np.array2string(
                            point, separator=",", precision=3)
                            for point in pointcloud])
                        with open(output_path, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            if file.tell() == 0:
                                writer.writerow(["label", "point"])
                            writer.writerow([label, pointcloud])

    def _classify_pointcloud(self,
                             rgb_image: np.ndarray,
                             depth_image: np.ndarray,
                             seg_image: np.ndarray,
                             seg_labels: dict,
                             camera_K: np.array,
                             camera_T: np.array,
                             camera_params: dict,
                             occluder_label: str,
                             minimum_num_of_points: int,
                             output_num_of_points: int) -> dict:
        """
        Classify the pointcloud by labels.

        Args:
            rgb_image (numpy.ndarray): RGB image.
            depth_image (numpy.ndarray): Depth image.
            seg_image (numpy.ndarray): Segmentation image.
            seg_labels (dict): Dictionary of segmentation labels.
            camera_K (numpy.array): Camera intrinsics.
            camera_T (numpy.array): Camera extrinsics.
            camera_params (dict): Camera parameters returned by
                self._read_json_file().
            occluder_label (str): Label of the occluder.
            minimum_num_of_points (int): Minimum number of points in a
                pointcloud.
            output_num_of_points (int): Number of points in a pointcloud.

        Returns:
            dict: Classified pointcloud.
        """
        classified_pointcloud = {}

        x_resolution = camera_params["renderProductResolution"][0]
        y_resolution = camera_params["renderProductResolution"][1]

        u_indices, v_indices = np.meshgrid(np.arange(x_resolution),
                                           np.arange(y_resolution))
        x_factors = np.array((u_indices - camera_K[0, 2]) / camera_K[0, 0])\
            .astype(np.float32)
        y_factors = np.array((v_indices - camera_K[1, 2]) / camera_K[1, 1])\
            .astype(np.float32)

        valid_mask = np.isfinite(depth_image)
        z_mat = depth_image.astype(np.float32)
        z_mat[~valid_mask] = 0
        x_mat = x_factors * z_mat
        y_mat = y_factors * z_mat

        for x in range(int(x_resolution)):
            for y in range(int(y_resolution)):
                # Get pixel label from rgb and segmentation
                seg_r = seg_image[y, x][2]
                seg_g = seg_image[y, x][1]
                seg_b = seg_image[y, x][0]
                argb_string = '({}, {}, {}, 255)'.format(seg_r, seg_g, seg_b)
                obj_name = seg_labels[argb_string]['class']
                if (obj_name is None or
                    obj_name == 'BACKGROUND' or
                        obj_name == 'UNLABELLED'):
                    continue
                else:
                    obj_label = self._get_obj_label(obj_name)
                    if (obj_label == occluder_label):
                        continue
                    else:
                        if obj_label not in classified_pointcloud.keys():
                            classified_pointcloud[obj_label] = []
                        # Get pointcloud
                        pcd_x = x_mat[y, x]
                        pcd_y = y_mat[y, x]
                        pcd_z = z_mat[y, x]
                        # pcd_r = seg_image[y, x][2]
                        # pcd_g = seg_image[y, x][1]
                        # pcd_b = seg_image[y, x][0]
                        point = np.array([pcd_x, pcd_y, pcd_z])
                        classified_pointcloud[obj_label].append(point)
        # Convert pointclouds as a numpy arrays with shape (n, 3)
        labels_to_remove = []
        for label in classified_pointcloud.keys():
            classified_pointcloud[label] = np.vstack(
                classified_pointcloud[label])
            num_of_points = classified_pointcloud[label].shape[0]

            # Record pointclouds with less than minimum_num_of_points
            if num_of_points < minimum_num_of_points:
                labels_to_remove.append(label)

        # Remove pointclouds with less than minimum_num_of_points
        for label in labels_to_remove:
            classified_pointcloud.pop(label)

        # Resize pointclouds
        for label in classified_pointcloud.keys():
            num_of_points = classified_pointcloud[label].shape[0]

            # Downsample pointclouds with more than output_num_of_points
            if num_of_points > output_num_of_points:
                indices = np.random.choice(num_of_points,
                                           output_num_of_points,
                                           replace=False)
                classified_pointcloud[label] =\
                    classified_pointcloud[label][indices]

            # Interpolate pointclouds with less than output_num_of_points
            if num_of_points < output_num_of_points:
                num_new_points = output_num_of_points - num_of_points
                classified_pointcloud[label] =\
                    self._interpolate_point_cloud(
                        classified_pointcloud[label],
                        num_new_points)

        # Print the number of points in each pointcloud
        for label in classified_pointcloud.keys():
            print(f"{label}: {classified_pointcloud[label].shape[0]}")
        return classified_pointcloud

    def _interpolate_point_cloud(self,
                                 points: np.ndarray,
                                 num_new_points: int,
                                 num_knn_neighbours: int = 3):
        """
        Interpolate point cloud.

        Args:
            points (numpy.ndarray): Point cloud.
            num_new_points (int): Number of new points to interpolate.
            num_knn_neighbours (int): Number of nearest neighbors to
                interpolate.

        Returns:
            numpy.ndarray: Interpolated point cloud.
        """
        # Fit NearestNeighbors on existing points
        nbrs = NearestNeighbors(n_neighbors=num_knn_neighbours).fit(points)
        interpolated_points = []

        for _ in range(num_new_points):
            random_point = points[np.random.randint(0, points.shape[0])]
            distances, indices = nbrs.kneighbors([random_point])
            neighbors = points[indices[0]]
            new_point = neighbors.mean(axis=0)  # Average of neighbors
            interpolated_points.append(new_point)

        return np.vstack([points, interpolated_points])

    def _get_camera_extrinsics(self,
                               camera_params: dict) -> np.array:
        """
        Get the camera extrinsics.

        Args:
            camera_params (dict): Camera parameters returned by
                self._read_json_file().

        Returns:
            numpy.array: Camera extrinsics.
        """
        _cam_t = camera_params["cameraViewTransform"]
        _cam_t = np.resize(_cam_t, (4, 4))
        _cam_pose = np.linalg.inv(_cam_t).T
        _SURPRISING_OFFSET_MATRIX = np.array([[1, 0, 0, 0],
                                              [0, -1, 0, 0],
                                              [0, 0, -1, 0],
                                              [0, 0, 0, 1]])
        _cam_pose = _cam_pose.dot(_SURPRISING_OFFSET_MATRIX)
        return _cam_pose

    def _get_camera_intrinsics(self,
                               camera_params: dict) -> np.array:
        """
        Get the camera intrinsics.

        Args:
            camera_params (dict): Camera parameters returned by
                self._read_json_file().

        Returns:
            numpy.array: Camera intrinsics.
        """
        x_aperture = camera_params["cameraAperture"][0]
        y_aperture = camera_params["cameraAperture"][1]
        # aperture offsets are default to 0
        # _x_aperture_offset = camera_params["cameraApertureOffset"][0]
        # _y_aperture_offset = camera_params["cameraApertureOffset"][1]
        focal_length = camera_params["cameraFocalLength"]
        x_resolution = camera_params["renderProductResolution"][0]
        y_resolution = camera_params["renderProductResolution"][1]

        focal_x = x_resolution * focal_length / x_aperture
        focal_y = y_resolution * focal_length / y_aperture
        center_x = x_resolution / 2
        center_y = y_resolution / 2
        intrinsic_matrix = np.array([[focal_x, 0, center_x],
                                     [0, focal_y, center_y],
                                     [0, 0, 1]])
        return intrinsic_matrix

    def _get_obj_labels(self,
                        obj_names: list[str]) -> list:
        """
        Get the labels of the objects.

        Args:
            obj_names (list): Names of the objects.

        Returns:
            list: Labels of the objects.
        """
        obj_labels = []
        for obj_name in obj_names:
            obj_label = self._get_obj_label(obj_name)
            obj_labels.append(obj_label)
        return obj_labels

    def _get_obj_label(self,
                       obj_name: str) -> str:
        """
        Get the label of the object.

        Args:
            obj_name (str): Name of the object.

        Returns:
            str: Label of the object.
        """
        for i in range(len(obj_name)):
            if obj_name[i] == '0' or obj_name[i] == '1' or\
                obj_name[i] == '2' or obj_name[i] == '3' or\
                obj_name[i] == '4' or obj_name[i] == '5' or\
                obj_name[i] == '6' or obj_name[i] == '7' or\
                    obj_name[i] == '8' or obj_name[i] == '9':
                obj_label = obj_name[:i-1]
                return obj_label
        return None

    def _read_json_file(self,
                        json_file_path: str) -> dict:
        """
        Read a JSON file and return the data.

        Args:
            json_file_path (str): Path to the JSON file.

        Returns:
            dict: Data from the JSON file.
        """
        data = None
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return data

    def _read_pickle_file(self,
                          pickle_file_path: str) -> any:
        """
        Read a pickle file and return the data.

        Args:
            pickle_file_path (str): Path to the pickle file.

        Returns:
            Any: Data from the pickle file.
        """
        data = None
        with open(pickle_file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def _read_image_file(self,
                         image_path: str) -> np.ndarray:
        """
        Read an image file and return the image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: Image read from the file.
        """
        image = cv2.imread(image_path)
        return image


cd = ConvertDataset()
cd.convert_dataset(
    '/home/zhengxiao-han/Downloads/Datasets/msai_dataset/train/scene_4',
    'output.csv',
    minimum_num_of_points=2000,
    output_num_of_points=1024)
