See detailed explaination: [https://github.com/msr-in-msai/msai-349-final-project/tree/main/proposal](https://github.com/msr-in-msai/msai-349-final-project/tree/main/proposal)
### Task:
Point cloud classification with RGB information.
* Input: A set of 3D points (point cloud), where each point contains its 3D position and RGB values, representing an object.
* Output: The object's class label, chosen from a fixed number of predefined classes (e.g., 10 classes).
### Motivation:
This task is fundamental in robotics, particularly in areas such as autonomous navigation, and robotic manipulation. 
### Data Acquisition
We will generate the dataset using [Isaac Sim](https://docs.omniverse.nvidia.com/isaacsim/latest/index.html) including:
* RGB images.
* Depth images (to be converted into point clouds).
* Segmentation masks for each object.
* Object labels for every pixel in the RGB and depth images, mapping from  RGBA value to object class.
* Camera intrinsics for converting depth images into point clouds with corresponding RGB values.

### Features and Attributes
The primary feature for this task will be the 3D point cloud enriched with RGB information, representing each object. Each point cloud will consist of points in 3D space, capturing the object's shape, color, and spatial distribution.

### Data Preprocessing:
Convert depth images into point clouds using the camera intrinsics.
Add corresponding RGB values to each point in the cloud based on the RGB images.
Segment the point cloud by extracting the portion corresponding to each object based on the segmentation masks.

### Machine Learning Techniques:
Use [PointNet](https://github.com/charlesq34/pointnet) to extract representative features from the RGB-enhanced pointcloud.
The extracted feature will be passed through an MLP (Multi-Layer Perceptron), which will be trained to classify the objects into their respective categories. We will also explore alternatives to MLP for classification performance, such as decision trees, KNN and K-Means.

### Evaluation:
We will evaluate our model based on classification accuracy, measuring the percentage of correctly classified objects. Additionally, we will conduct ablation study, exploring the impact of incorporating RGB data alongside depth information to understand its contribution to classification performance.