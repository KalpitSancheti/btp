import airsim
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

lidar_data = client.getLidarData(lidar_name="LidarSensor1")

if len(lidar_data.point_cloud) < 3:
    print("No LiDAR data received")
else:
    # Convert to Nx3 points
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    
    # Front distance = smallest +Y in front hemisphere
    front_dist = np.min(points[points[:,1] > 0][:,1])
    print("Front distance:", front_dist)
