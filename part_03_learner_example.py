import numpy as np
from ibvs_controller import ibvs_controller
from ibvs_simulation import ibvs_simulation
from dcm_from_rpy import dcm_from_rpy
import time
import matplotlib.pyplot as plt


# Camera intrinsics matrix - known.
K = np.array([[500.0, 0, 400.0], 
              [0, 500.0, 300.0], 
              [0,     0,     1]])

# Target points (in target/object frame).
pts = np.array([[-0.75,  0.75, -0.75,  0.75],
                [-0.50, -0.50,  0.50,  0.50],
                [ 0.00,  0.00,  0.00,  0.00]])

# Camera poses, last and first.
C_last = np.eye(3)
t_last = np.array([[ 0.0, 0.0, -4.0]]).T
# C_init = dcm_from_rpy([np.pi/10, -np.pi/8, -np.pi/8]) #given 
C_init = dcm_from_rpy([np.pi/6, -np.pi/4, np.pi/3])  # More extreme rotation

# t_init = np.array([[-0.2, 0.3, -5.0]]).T #initial
t_init = np.array([[0.8, -0.6, -9.0]]).T  # Larger offset from initial pose


Twc_last = np.eye(4)
Twc_last[0:3, :] = np.hstack((C_last, t_last))
Twc_init = np.eye(4)
Twc_init[0:3, :] = np.hstack((C_init, t_init))

gains = [0.2, 0.5, 0.8, 1.1, 1.3,1.5,1.6]  # Test different gains

# Store results to analyze later
convergence_times = []

for gain in gains:
    print(f"Testing with gain = {gain}")
    start_time = time.time()
    try:
        # Run simulation - estimate depths.
        ibvs_simulation(Twc_init, Twc_last, pts, K, gain, True)
        elapsed_time = time.time() - start_time
        convergence_times.append(elapsed_time)
        print(f"Simulation completed in {elapsed_time:.4f} seconds")
    except np.linalg.LinAlgError as e:
        convergence_times.append(None)
        print(f"Simulation failed for gain = {gain}: {e}")

# Plotting
plt.figure(figsize=(10, 6))
valid_gains = [gains[i] for i in range(len(gains)) if convergence_times[i] is not None]
valid_times = [convergence_times[i] for i in range(len(gains)) if convergence_times[i] is not None]

plt.plot(valid_gains, valid_times, marker='o', linestyle='-', color='b', label='Convergence Time')
plt.title('Gain vs. Convergence Time (Unknown Depth)')
plt.xlabel('Gain')
plt.ylabel('Convergence Time (seconds)')
plt.grid(True)
plt.legend()
plt.show()