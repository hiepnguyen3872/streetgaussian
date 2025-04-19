import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

# Load the data from the specified path with allow_pickle=True
data = np.load('/media/ml4u/ExtremeSSD/datasets/waymo/processed/150/pointcloud.npz', allow_pickle=True)

# Access the point cloud data
pointcloud = data['pointcloud']  # Access the point cloud data

# Print the type and contents of pointcloud to understand its structure
print("Pointcloud type:", type(pointcloud))
print("Pointcloud contents:", pointcloud)

# If pointcloud is a single object, extract it
if pointcloud.ndim == 0:
    pointcloud = pointcloud.item()  # Extract the single object

# Check the type and shape of the extracted pointcloud
print("Extracted Pointcloud type:", type(pointcloud))
if hasattr(pointcloud, 'shape'):
    print("Extracted Pointcloud shape:", pointcloud.shape)

# Ensure pointcloud is 2D and has the expected number of columns
if isinstance(pointcloud, np.ndarray) and pointcloud.ndim == 2 and pointcloud.shape[1] >= 3:
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]
else:
    raise ValueError("Pointcloud data does not have the expected shape.")

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)
ax.set_xlabel('X Axis Label')
ax.set_ylabel('Y Axis Label')
ax.set_zlabel('Z Axis Label')
ax.set_title('LiDAR Point Cloud Visualization')
plt.savefig('lidar_point_cloud.png')  # Save the figure as a PNG file