# Final Project Proposal

## 1. Task Description and Motivation

### Task:
Point cloud classification.  
* **Input:** A set of 3D points (point cloud) representing an object.  
* **Output:** The object's class label, chosen from a fixed number of predefined classes (e.g., 10 classes).

### Motivation:
This task is critical in robotics, where we often need to classify objects using 3D data from sensors like LiDAR. Being able to accurately classify objects based solely on point clouds is essential for tasks like autonomous navigation, object manipulation, and environment understanding.

## 2. Data Acquisition

I will generate the dataset using Isaac Sim, which provides a flexible environment for creating synthetic data. The dataset will include:
* Depth images.
* Segmentation masks for each object.
* Object labels for every pixel in the depth images.
* Camera intrinsics to convert depth images into point clouds.

This approach ensures that the data generation is well-structured, reproducible, and tailored to my project's needs.

## 3. Features and Attributes

The key feature for this task is the 3D point cloud representing each object. Each point cloud is a collection of points in 3D space that captures the object's shape and spatial distribution.

## 4. Initial Approach

### Data Preprocessing:
1. Convert depth images into point clouds using the provided camera intrinsics.
2. Segment the point cloud by extracting the portion corresponding to each object based on the segmentation mask.
3. Use PointNet to extract representative features from the point clouds.

### Machine Learning Techniques:
The extracted features from PointNet will serve as input to an MLP (Multi-Layer Perceptron), which will be trained to classify objects into their respective categories.

### Evaluation:
The primary evaluation metric will be the classification accuracy, measuring the percentage of correctly classified objects.
