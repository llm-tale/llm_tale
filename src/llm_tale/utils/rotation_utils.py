import torch
import numpy as np


def quat2rotm(q, mode="wxyz", device="cpu"):
    if mode == "wxyz":
        w, x, y, z = q[0], q[1], q[2], q[3]
    elif mode == "xyzw":
        x, y, z, w = q[0], q[1], q[2], q[3]
    return torch.tensor(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        device=device,
    )


def quat2rotmwxyz(q, device="cpu"):
    return quat2rotm(q, mode="wxyz", device=device)


def quat2rotmxyzw(q, device="cpu"):
    return quat2rotm(q, mode="xyzw", device=device)


def rotm2quat(rot, mode="wxyz", device="cpu"):
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = 2.0 * torch.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * torch.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    if mode == "wxyz":
        return torch.tensor([w, x, y, z], device=device)
    elif mode == "xyzw":
        return torch.tensor([x, y, z, w], device=device)


def rotm2quatxyzw(rot, device="cpu"):
    return rotm2quat(rot, mode="xyzw", device=device)


def rotm2quatwxyz(rot, device="cpu"):
    return rotm2quat(rot, mode="wxyz", device=device)


def rotm2axisangle(rot, device="cpu"):
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    if angle == 0:
        axis = torch.tensor([0.0, 0.0, 0.0], device=device)
    else:
        axis = torch.tensor([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]], device=device) / (
            2 * torch.sin(angle)
        )
    return axis, angle


def axisangle2rotm(axis_angle, device="cpu"):
    angle = torch.norm(axis_angle)
    if angle == 0:
        return torch.eye(3, device=device)
    axis = axis_angle / angle
    c = torch.cos(angle)
    s = torch.sin(angle)
    skew_symmetric = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=device)
    rot = torch.eye(3, device=device) + s * skew_symmetric + (1 - c) * skew_symmetric @ skew_symmetric
    return rot


def quat_from_axis_angle(axis_angle, modes="xyzw"):
    if np.linalg.norm(axis_angle) == 0:
        if modes == "xyzw":
            return np.array([0, 0, 0, 1])
        elif modes == "wxyz":
            return np.array([1, 0, 0, 0])
    angle = np.linalg.norm(axis_angle)
    axis = axis_angle / angle
    w = np.cos(angle / 2)
    x, y, z = np.sin(angle / 2) * axis
    if modes == "xyzw":
        return np.array([x, y, z, w])
    elif modes == "wxyz":
        return np.array([w, x, y, z])


def rotation_matrix_from_quat(quat, mode="xyzw"):
    """
    Return a 3x3 rotation matrix from a quaternion.

    Args:
        quat: iterable of 4 numbers. If mode='xyzw', it's [x, y, z, w].
              If mode='wxyz', it's [w, x, y, z].
        mode: 'xyzw' or 'wxyz' (default: 'xyzw').

    Returns:
        R (3, 3) ndarray, orthonormal.

    Notes:
        - Columns of R are the rotated frame's axes expressed in the parent frame
          (i.e., if q encodes EE-in-world, R[:,0] and R[:,2] are EE x/z in world).
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    if mode == "xyzw":
        x, y, z, w = q
    elif mode == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError("mode must be 'xyzw' or 'wxyz'")

    # Normalize for numerical safety
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        print("Warning: Quaternion norm is too small, returning identity matrix.")
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n

    # Standard right-handed rotation matrix for unit quaternion (w scalar part)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def axisangle2quat(axis_angle, mode="xyzw", device="cpu"):
    angle = torch.norm(axis_angle)
    if angle == 0:
        return torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    axis = axis_angle / angle
    half_angle = angle / 2
    return torch.cat([torch.sin(half_angle) * axis, torch.cos(half_angle).unsqueeze(0)], dim=0)


def quat_dis(q1, q2):
    q1 = q1 / torch.norm(q1)
    q2 = q2 / torch.norm(q2)
    dot_product = torch.sum(q1 * q2)
    dot_product = torch.clamp(dot_product, -1, 1)
    angle = torch.acos(torch.abs(dot_product))
    return angle


def norm(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + eps)


def eul_from_R(R):
    # World-frame XYZ Euler from rotation matrix whose columns are [X,Y,Z] (local axes in world)
    sy = -R[2, 0]
    cy = float(np.sqrt(max(0.0, 1.0 - sy * sy)))
    if cy > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        ry = np.arcsin(sy)
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        ry = np.arcsin(sy)
        rz = 0.0
    return [float(rx), float(ry), float(rz)]
