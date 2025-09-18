from llm_tale.utils.rotation_utils import (
    rotation_matrix_from_quat,
)
import numpy as np


def check_reachable(pose):
    """
    Heuristic reachability check using a simple 2-link (J2->J4->J6) geometry.

    pose : (7,) array-like [x, y, z, qx, qy, qz, qw]  (adjust if your quat order differs!)
    Returns:
        reachable : bool
        score     : float  (cos(theta) after clamping, or np.nan on degenerate cases)
    """
    # --- Robot-specific constants (meters) ---
    joint2_pos = np.array([-0.270, -0.006, 1.080], dtype=np.float64)  # base frame coords of J2
    d24 = 0.3264  # |J2-J4|
    d46 = 0.3922  # |J4-J6|
    ee_offset_along_z = 0.22  # EE z-axis back from TCP to J6
    ee_offset_along_x = 0.0808  # small x-offset to J6

    pose = np.asarray(pose, dtype=np.float64)
    p = pose[:3]
    q = pose[3:]

    # If your rotation function expects wxyz, reorder accordingly:
    # q_wxyz = np.array([q[3], q[0], q[1], q[2]])
    # R = rotation_matrix_from_quat(q_wxyz)

    # Assuming rotation_matrix_from_quat expects XYZW and returns a world-from-EE rotation.
    R = rotation_matrix_from_quat(q / (np.linalg.norm(q) + 1e-12))  # normalize for safety
    # Columns of R are EE basis vectors expressed in world if R is world_from_ee
    x_axis = R[:, 0]
    z_axis = R[:, 2]

    # Estimate J6 position from TCP pose + fixed offsets in EE frame
    j6 = p - ee_offset_along_z * z_axis + ee_offset_along_x * x_axis

    d26_vec = j6 - joint2_pos
    d26 = float(np.linalg.norm(d26_vec))

    # Degenerate / numeric guard
    if not np.isfinite(d26) or d26 < 1e-6:
        return False, np.nan

    # Triangle inequality check first: |d24 - d46| <= d26 <= d24 + d46
    if d26 > (d24 + d46 + 1e-6) or d26 < (abs(d24 - d46) - 1e-6):
        return False, np.nan

    # Law of cosines: cos(theta) = (d26^2 + d24^2 - d46^2) / (2 d26 d24)
    denom = 2.0 * d26 * d24
    if denom < 1e-9 or not np.isfinite(denom):
        return False, np.nan

    cos_theta = (d26**2 + d24**2 - d46**2) / denom

    # Reachable if the triangle inequality holds and cos is within [-1, 1]
    # (the clamp ensures this; we use the unclamped value to detect gross errors)
    reachable = np.isfinite(cos_theta) and (-1.0 <= cos_theta <= 1.0)

    return bool(reachable), cos_theta


def check_object_grasp(task):
    obj = task._scene.robot.gripper._grasped_objects
    return len(obj) > 0
