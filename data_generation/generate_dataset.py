"""Convert .obj models into pointcloud."""

import csv
from object import Object

import numpy as np

import os

from tqdm import tqdm


class GenerateDataset:
    """Generate a dataset from .obj models"""
    def __init__(self,
                 obj_dataset_path: str,
                 csv_output_path: str,
                 minimum_num_of_points: int = 3000,
                 output_num_of_points: int = 1024):
        """
        Initialize the dataset generator.

        Args:
            obj_dataset_path: The path to the .obj dataset.
            csv_output_path: The path to the output CSV file.

        Returns:
            None
        """
        # Load parameters
        if (obj_dataset_path[-1] == '/'):
            obj_dataset_path = obj_dataset_path[:-1]
        self._obj_dataset_path = obj_dataset_path
        self._csv_output_path = csv_output_path
        self._trimesh_objects = []
        self._model_scale = [0.001, 0.001, 0.001]
        self._model_rotation = [1.57, 0, 0]
        self._minimum_num_of_points = minimum_num_of_points
        self._output_num_of_points = output_num_of_points

    def generate_dataset(self):
        # Count the number of models
        total_count = 0
        for root, dirs, _ in os.walk(self._obj_dataset_path, topdown=False):
            for name in dirs:
                path = os.path.join(root, name)
                label = self._extract_label(path)
                if (os.path.join(root, name)[-14:] == "Scan_converted"):
                    total_count += 1
        with tqdm(total=total_count, desc="Processing files") as pbar:
            for root, dirs, _ in os.walk(
                    self._obj_dataset_path, topdown=False):
                for name in dirs:
                    path = os.path.join(root, name)
                    label = self._extract_label(path)
                    if (os.path.join(root, name)[-14:] == "Scan_converted"):
                        pbar.update(1)
                        obj = Object(obj_path=path[:-10]+'/Simp.obj',
                                     scale=self._model_scale,
                                     eular_angles=self._model_rotation,
                                     label=label)
                        label = obj.get_label()
                        points = obj.get_points()
                        num_of_points = points.shape[0]

                        # Downsample pointclouds to the output_num_of_points
                        if num_of_points > self._minimum_num_of_points:
                            indices = np.random.choice(
                                num_of_points,
                                self._output_num_of_points,
                                replace=False)
                            points = points[indices]
                            pcd_str = ';'.join([np.array2string(
                                point, separator=",", precision=3)
                                for point in points])
                            with open(
                                    self._csv_output_path,
                                    mode='a',
                                    newline='') as file:
                                writer = csv.writer(file)
                                if file.tell() == 0:
                                    writer.writerow(["label", "point"])
                                writer.writerow([label, pcd_str])

    def _extract_label(self,
                       file_path: str):
        """
        Extract the label of a model from its path.

        Args:
            file_path: The path to the model.

        Returns:
            The label of the model.
        """
        # Specify it according to specific file directory
        NUM_SLASH_BEFORE_TYPE = 2

        num_slash = 0
        object_type_str = []

        for i in range(len(file_path)):
            index = len(file_path) - 1 - i
            char = file_path[index]

            if (num_slash == NUM_SLASH_BEFORE_TYPE):
                object_type_str.append(char)

            if (char == "/"):
                num_slash += 1

        object_type_str.reverse()
        object_type_str = ''.join(object_type_str[1:])
        return object_type_str


gd = GenerateDataset(
    '/home/zhengxiao-han/Downloads/Datasets/msai_dataset/objects/train',
    'train.csv',
    minimum_num_of_points=5000,
    output_num_of_points=1024)
gd.generate_dataset()

gd = GenerateDataset(
    '/home/zhengxiao-han/Downloads/Datasets/msai_dataset/objects/valid',
    'valid.csv',
    minimum_num_of_points=5000,
    output_num_of_points=1024)
gd.generate_dataset()

gd = GenerateDataset(
    '/home/zhengxiao-han/Downloads/Datasets/msai_dataset/objects/test',
    'test.csv',
    minimum_num_of_points=5000,
    output_num_of_points=1024)
gd.generate_dataset()
