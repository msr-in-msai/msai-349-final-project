# Discussion for final proposal

## 1. What task will you address, and why is it interesting? This should be as simple as a couple of sentences.

### What task will you address?
Pointcloud Classification. 
* Input: the pointcloud of an object (a set of 3d points).
* Output: the class label of the object, out of fixed amount of classes (like 10).

### why is it interesting?
It's relevant to our field of robotics. We can specify objects only using data from LiDAR. 

## 2. How will you acquire your data? This element is intended to serve as a check that your project is doable -- so if you plan to collect a new data set (which I discourage), be as specific as possible.

### How will you acquire your data?
Use a simulator (Isaac Sim) to generate the dataset. The dataset contains multiple depth images, segmentation masks, the object label of every pixel in the depth image, and camera intrinsics to convert depth images to pointclouds.

## 3. Which features/attributes will you use for your task?

Each feature is a list of 3D points (The point cloud).

## 4. What will your initial approach be? What data pre-processing will you do, which machine learning techniques (decision trees, KNN, K-Means, Gaussian mixture models, etc.) will you use, and how will you evaluate your success (Note: you must use a quantitative metric)? Generally, you will likely use mean-squared error for regression tasks and precision-recall for classification tasks. Think about how you will organize your model outputs to calculate these metrics.

### What will your initial approach be?
Use MLP to calssify the object.

### What data pre-processing will you do?
1. Convert depth image to pointcloud with camera intrinsics.
2. Get the pointcloud corresponding to every kind of obejct based on the segmentation image (mask).
3. Extract representation features from pointclouds using PointNet. 

### which machine learning techniques (decision trees, KNN, K-Means, Gaussian mixture models, etc.) will you use?
Use the representation features as input to train the MLP to classify objects.

### How will you evaluate your success?

Use the sucess rate of the classification.
