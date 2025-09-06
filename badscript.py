import airsim
import time
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

print("Starting tunnel navigation...")

while True:
    # get drone state
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    x, y, z = pos.x_val, pos.y_val, pos.z_val  # AirSim z is NEGATIVE for up

    # get depth image
    image_request = airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, pixels_as_float=True)
    responses = client.simGetImages([image_request], "")

    depth_img = airsim.list_to_2d_float_array(
        responses[0].image_data_float,
        responses[0].width,
        responses[0].height
    )

    h, w = responses[0].height, responses[0].width
    center_region = np.mean(depth_img[h//3:2*h//3, w//3:2*w//3])
    top_region    = np.mean(depth_img[:h//3, w//3:2*w//3])
    bottom_region = np.mean(depth_img[2*h//3:, w//3:2*w//3])

    print(f"y={y:.2f}, z={z:.2f}, center={center_region:.2f}, top={top_region:.2f}, bottom={bottom_region:.2f}")

    # --- decision logic ---
    vx, vy, vz = 0.0, 0.8, 0.0  # default: move forward

    # If wall too close ahead
    if center_region < 1.5:
        # If bottom has more free space → move down
        if bottom_region > center_region + 1.0:
            vz = 0.8   # move down (positive z in AirSim)
            vy = 0.2   # slow forward
        # If top has more free space → move up
        elif top_region > center_region + 1.0:
            vz = -0.8  # move up (negative z in AirSim)
            vy = 0.2
        else:
            # No vertical hole → try left/right adjustment
            vx = -0.3 if x > 0 else 0.3
            vy = 0.2

    # send velocity command
    client.moveByVelocityAsync(vx, vy, vz, 0.2)

    time.sleep(0.2)
