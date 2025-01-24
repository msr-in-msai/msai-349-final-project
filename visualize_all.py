import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def visualize_pointclouds(dataset_path, class_nth_occurrence):
    """
    Visualizes the N-th occurrence of point clouds for each class in the dataset in separate plots.
    Rotates the apple and bowl plots 180 degrees along the y-axis.
    
    Args:
        dataset_path (str): Path to the dataset CSV file.
        class_nth_occurrence (dict): A dictionary where keys are class names and values are the N-th occurrences
                                      (1-based index) of the instances to visualize for each class.
    """
    # Define the classes and corresponding color maps
    class_cmaps = {
        'apple': 'Reds',         # Red colormap for apple
        'banana': 'YlOrBr',      # Yellow-orange-brown for banana
        'bottle': 'Blues',       # Blue colormap for bottle
        'bowl': 'Greys',         # Grey colormap for bowl
        'cup': 'Purples'         # Purple colormap for cup (similar tone to others)
    }

    # Define the figure and axis for 5 subplots arranged in a grid (2 rows x 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), subplot_kw={'projection': '3d'})

    # Hardcoded axis limits
    x_lim = (-0.1, 0.1)
    y_lim = (-0.1, 0.0)
    z_lim = (-0.1, 0.1)

    # Load data from CSV
    with open(dataset_path, mode='r') as file:
        reader = csv.DictReader(file)
        data_list = list(reader)

        # Filter dataset by class and store occurrences of each class
        class_instances = {key: [] for key in class_cmaps.keys()}

        # Organize data instances by class
        for instance in data_list:
            class_name = instance['label']
            if class_name in class_cmaps:
                class_instances[class_name].append(instance)

        # Loop through each class and plot its point cloud
        for i, (class_name, cmap) in enumerate(class_cmaps.items()):
            # Get the N-th occurrence (1-based index)
            if class_name not in class_nth_occurrence:
                print(f"No occurrence specified for class {class_name}. Skipping this class.")
                continue
            
            nth_occurrence = class_nth_occurrence[class_name] - 1  # Convert to 0-based index

            # Check if the N-th occurrence is valid
            if nth_occurrence >= len(class_instances[class_name]):
                print(f"Invalid N-th occurrence {nth_occurrence + 1} for class {class_name}. Skipping this class.")
                continue

            # Get the specific instance for the class
            instance = class_instances[class_name][nth_occurrence]

            # Extract point cloud data and convert it to a numpy array
            pointcloud_str = instance['point']

            # Debugging: Print raw pointcloud data
            print(f"Class: {class_name}, N-th Occurrence: {nth_occurrence + 1}")
            print(f"Pointcloud String for {class_name}: {pointcloud_str}")

            # Split the point cloud string and check for any issues
            try:
                pointcloud = [
                    np.array(eval(point_str))
                    for point_str in pointcloud_str.split(';')
                ]
                pointcloud = np.vstack(pointcloud)

                # Make sure the shape is correct and print debug info
                if pointcloud.shape[0] == 0:
                    print(f"No point cloud data found for class {class_name} (instance {nth_occurrence + 1}). Skipping.")
                    continue
                assert pointcloud.shape[1] == 3  # Ensure 3D coordinates
            except Exception as e:
                print(f"Error parsing point cloud for class {class_name}, instance {nth_occurrence + 1}: {e}")
                continue

            # Extract X, Y, Z coordinates from the point cloud
            x = pointcloud[:, 0]
            y = pointcloud[:, 1]
            z = pointcloud[:, 2]

            # If the class is 'apple' or 'bowl', flip the point cloud upside down (rotate 180 degrees around y-axis)
            if class_name in ['apple', 'bowl']:
                # Define the rotation matrix for 180 degrees around the y-axis
                rotation_matrix = np.array([[np.cos(np.pi), 0, np.sin(np.pi)],
                                             [0, 1, 0],
                                             [-np.sin(np.pi), 0, np.cos(np.pi)]])

                # Apply the rotation
                pointcloud = np.dot(pointcloud, rotation_matrix.T)

                # Extract the new coordinates after the rotation
                x = pointcloud[:, 0]
                y = pointcloud[:, 1]
                z = pointcloud[:, 2]

            # Select the subplot (rows, columns layout)
            row = i // 3  # 0 or 1 (since we have 2 rows)
            col = i % 3   # 0, 1, or 2 (since we have 3 columns)
            
            ax = axs[row, col]
            
            # Scatter plot with points colored based on their distance from the origin
            distances = np.linalg.norm(pointcloud, axis=1)
            ax.scatter(x, y, z, c=distances, cmap=cmap, s=10)

            # Set labels for each plot
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"{class_name.capitalize()}")

            # Hide axis and grid
            ax.set_axis_off()  # Hide the axis
            ax.grid(False)  # Turn off the grid

        # Remove the last subplot (6th one) by setting it to `None`
        axs[1, 2].axis('off')

        # Adjust the layout to make the subplots closer to each other
        plt.tight_layout(pad=0.1)  # Reduce padding to bring plots closer
        plt.subplots_adjust(hspace=0.05, wspace=0.05)  # Reduce horizontal and vertical spacing

        # Adjust bottom row subplots (2nd row) to be centered
        axs[1, 0].set_position([0.2, 0.1, 0.27, 0.35])  # Left bottom plot
        axs[1, 1].set_position([0.5, 0.1, 0.27, 0.35])  # Right bottom plot

        # Function to update the view for animation (rotate each subplot)
        def update(frame):
            for ax in axs.flatten():
                ax.view_init(elev=30, azim=frame)
            return []

        # Create an animation for rotating the view
        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 1), interval=50)

        # Show the plot with rotation
        plt.show()


# Replace 'test_isaac_sim_3d.csv' with the actual path to your dataset
# Provide a dictionary where each class has the N-th occurrence to visualize (1-based index)
dataset_path = 'test_isaac_sim_3d.csv'

# Hardcode the N-th occurrence for each class you want to visualize
class_nth_occurrence = {
    'apple': 1,  # 1st occurrence of apple
    'banana': 2, # 2nd occurrence of banana
    'bottle': 3, # 3rd occurrence of bottle
    'bowl': 1,   # 1st occurrence of bowl
    'cup': 46     # 46th occurrence of cup
}

visualize_pointclouds(dataset_path, class_nth_occurrence)
