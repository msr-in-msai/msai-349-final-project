"""Load a dataset and visualize it."""

import csv

import matplotlib.pyplot as plt
import numpy as np


class DatasetVisualizer:
    """Visualize a dataset."""

    def __init__(self,
                 dataset_path: str):
        """
        Initialize the dataset visualizer.

        Args:
            dataset_path: The path to the dataset.

        Returns:
            None
        """
        self.dataset_path = dataset_path

    def visualize_data(self):
        """
        Visualize a data sample.

        Returns:
            None
        """
        # Load data from CSV
        with open(self.dataset_path, mode='r') as file:
            reader = csv.DictReader(file)
            data_list = list(reader)
            num_samples = len(list(data_list))
            fig = plt.figure(figsize=(8, 6))
            for data_index in range(num_samples):
                label = data_list[data_index]['label']
                pointcloud_str = data_list[data_index]['point']
                # Convert string back to a numpy array
                pointcloud = [
                    np.array(eval(point_str))
                    for point_str in pointcloud_str.split(';')
                    ]
                pointcloud = np.vstack(pointcloud)

                # Make sure the shape is correct
                assert pointcloud.shape[1] == 3

                # Visualize the pointcloud
                x = pointcloud[:, 0]
                y = pointcloud[:, 1]
                z = pointcloud[:, 2]
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(x, y, z, c=z, cmap='viridis', s=10)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-0.1, 0.1])
                ax.set_ylim([-0.1, 0.1])
                ax.set_zlim([-0.1, 0.1])
                ax.set_title(label)
                plt.pause(0.5)
                plt.cla()
                plt.clf()
                # plt.close(fig)


vis = DatasetVisualizer(
    'train.csv')
vis.visualize_data()
