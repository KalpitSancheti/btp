# smoke_test.py
from airsim_sac_env import AirSimEnv
import time, numpy as np

env = AirSimEnv()
for ep in range(3):
    obs, info = env.reset()
    print("Reset obs shape", obs.shape, "info:", info)
    for i in range(12):
        # neutral action: no forward, no lateral
        act = np.array([-1.0, 0.0], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(act)
        print(f"Step {i+1} z={env.client.getMultirotorState().kinematics_estimated.position.z_val:.3f} reward={reward:.2f} term={term} info_maxy={info.get('max_y_distance')}")
        time.sleep(0.12)
        if term:
            print("Terminated:", info)
            break
env.close()
