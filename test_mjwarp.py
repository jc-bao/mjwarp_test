"""
Visualize MJWarp Simulation

Author: Chaoyi Pan
Date: 2025-06-30
"""

import os
import time
from typing import Sequence

from loop_rate_limiters import RateLimiter
import mujoco
import mujoco.viewer
import numpy as np
import warp as wp

import mujoco_warp as mjwarp

# Initialize Warp
wp.init()


def _compile_step(m: mjwarp.Model, d: mjwarp.Data) -> wp.context.Graph:
    """
    Warms up and compiles the mjwarp.step function into a CUDA graph.
    """
    # Warmup step to ensure all kernels are compiled
    mjwarp.step(m, d)
    wp.synchronize()
    # Capture the step function into a CUDA graph for faster execution
    with wp.ScopedCapture() as capture:
        mjwarp.step(m, d)
    wp.synchronize()
    return capture.graph


def setup_model(
    dt_sim: float = 0.005,
    model_path: str = "./assets/inspire/scene_bimanual_wipe.xml",
) -> mujoco.MjModel:
    """Setup MuJoCo model with specified parameters."""
    m = mujoco.MjModel.from_xml_path(model_path)
    m.opt.timestep = dt_sim
    m.opt.iterations = 10
    m.opt.ls_iterations = 20
    m.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    return m


def load_data(
    data_path: str = "./data/inspire_bimanual_wipe.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory data from NPZ file."""
    raw_data = np.load(data_path)
    qpos_list = raw_data["qpos"]
    qvel_list = raw_data["qvel"]
    if "ctrl" in raw_data:
        ctrl_list = raw_data["ctrl"]
    else:
        # Fallback if 'ctrl' is not in the data file
        print("Warning: 'ctrl' data not found, using 'qpos' as a fallback for control.")
        ctrl_list = qpos_list[:, :-14]  # Assuming the model has 14 DoFs for objects
    return qpos_list, qvel_list, ctrl_list


def benchmark(
    batch_size: int = 1024,
    ls_parallel: bool = True,
    dt_sim: float = 0.005,
    model_path: str = "./assets/inspire/scene_bimanual_wipe.xml",
    data_path: str = "./data/inspire_bimanual_wipe.npz",
) -> None:
    # Load model and data
    model_cpu = setup_model(dt_sim, model_path)
    qpos_list, qvel_list, ctrl_list = load_data(data_path)
    nstep = ctrl_list.shape[0]
    data_cpu = mujoco.MjData(model_cpu)
    data_cpu.qpos[:] = qpos_list[0]
    data_cpu.qvel[:] = qvel_list[0]
    data_cpu.ctrl[:] = ctrl_list[0]
    mujoco.mj_forward(model_cpu, data_cpu)

    # Create control variables
    ctrls_np = data_cpu.ctrl[None] + 0.01 * np.random.normal(
        size=(batch_size, nstep, model_cpu.nu)
    )
    ctrls_wp = wp.array(ctrls_np.astype(np.float32))

    # Initialize batch of MJWarp models and data
    print(f"Initializing {batch_size} parallel simulations...")
    model_warp = mjwarp.put_model(model_cpu)
    model_warp.opt.ls_parallel = ls_parallel
    data_warp = mjwarp.put_data(
        model_cpu,
        data_cpu,
        nworld=batch_size,
        nconmax=18 * batch_size,
        njmax=45 * batch_size,
    )

    # Compile step function for each batch
    print("JIT-compiling the model physics step with CUDA graphs...")
    start_time = time.time()
    graph = _compile_step(model_warp, data_warp)
    elapsed = time.time() - start_time
    print(f"Compilation took {elapsed:.3f}s.")

    # Benchmark simulation
    print(f"Running benchmark for {nstep} steps with batch size {batch_size}...")
    start_time = time.time()
    for i in range(nstep):
        wp.copy(data_warp.ctrl, ctrls_wp[:, i])
        wp.capture_launch(graph)
    elapsed = time.time() - start_time
    print(f"Simulation took {elapsed:.3f}s.")


def visualize(
    dt_sim: float = 0.005,
    dt_ctrl: float = 0.005,
    model_path: str = "./assets/inspire/scene_bimanual_wipe.xml",
    data_path: str = "./data/inspire_bimanual_wipe.npz",
) -> None:
    """
    Main function to load data, set up the simulation, and run the viewer.

    Args:
        dt_sim: Simulation timestep.
        dt_ctrl: Control timestep.
        model_path: Path to the MuJoCo XML model file.
        data_path: Path to the NPZ file containing trajectory data (qpos, qvel, ctrl).
    """
    # --- 1. Load Trajectory Data ---
    print(f"Loading trajectory data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    raw_data = np.load(data_path)

    qpos_list = raw_data["qpos"]
    qvel_list = raw_data["qvel"]
    if "ctrl" in raw_data:
        ctrl_list = raw_data["ctrl"]
    else:
        # Fallback if 'ctrl' is not in the data file
        print("Warning: 'ctrl' data not found, using 'qpos' as a fallback for control.")
        ctrl_list = qpos_list[:, :-14]  # Assuming the model has 14 DoFs for object2

    # --- 2. Initialize MuJoCo (for viewer and initial state) ---
    print(f"Loading MuJoCo model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    mjm = mujoco.MjModel.from_xml_path(model_path)
    mjm.opt.timestep = dt_sim
    mjm.opt.iterations = 10
    mjm.opt.ls_iterations = 20
    mjm.opt.o_solimp = [0.9, 0.95, 0.001, 0.5, 2]
    mjd = mujoco.MjData(mjm)

    # Set initial state from the first frame of the trajectory
    mjd.qpos[:] = qpos_list[0]
    mjd.qvel[:] = qvel_list[0]
    mjd.ctrl[:] = ctrl_list[0]
    mujoco.mj_forward(mjm, mjd)

    # --- 3. Initialize MJWarp and Compile Step ---
    print("Putting model and data onto Warp device (GPU)...")
    m = mjwarp.put_model(mjm)
    d = mjwarp.put_data(mjm, mjd)

    print("JIT-compiling the model physics step with CUDA graphs...")
    start_time = time.time()
    graph = _compile_step(m, d)
    elapsed = time.time() - start_time
    print(f"Compilation took {elapsed:.3f}s.")

    # --- 4. Launch Viewer and Run Simulation Loop ---
    print("Launching passive viewer...")
    viewer = mujoco.viewer.launch_passive(mjm, mjd)
    with viewer:
        rate_limiter = RateLimiter(1.0 / dt_sim)

        while viewer.is_running():
            # Determine the control index based on simulation time, looping the trajectory
            ctrl_idx = int(mjd.time / dt_ctrl) % qpos_list.shape[0]
            mjd.ctrl[:] = ctrl_list[ctrl_idx]

            # --- MJWarp Simulation Step ---
            # a. Copy current state from MuJoCo (CPU) to MJWarp (GPU)
            # This is necessary to keep the Warp simulation state in sync with any
            # potential external changes (like control updates).
            wp.copy(d.ctrl, wp.array([mjd.ctrl.astype(np.float32)]))
            wp.copy(d.act, wp.array([mjd.act.astype(np.float32)]))
            wp.copy(d.xfrc_applied, wp.array([mjd.xfrc_applied.astype(np.float32)]))
            wp.copy(d.qpos, wp.array([mjd.qpos.astype(np.float32)]))
            wp.copy(d.qvel, wp.array([mjd.qvel.astype(np.float32)]))
            wp.copy(d.time, wp.array([mjd.time], dtype=wp.float32))

            # b. Execute the pre-compiled physics step on the GPU
            wp.capture_launch(graph)

            # c. Copy results from MJWarp (GPU) back to MuJoCo (CPU) for rendering
            mjwarp.get_data_into(mjd, mjm, d)

            # d. Sync the viewer to render the new state
            viewer.sync()

            # e. Maintain the simulation rate
            rate_limiter.sleep()


if __name__ == "__main__":
    # benchmark(batch_size=1024, ls_parallel=True)
    visualize()
