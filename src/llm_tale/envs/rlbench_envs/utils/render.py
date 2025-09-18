import numpy as np
from llm_tale.utils.rotation_utils import norm, eul_from_R
from pyrep.objects.vision_sensor import VisionSensor


def aim_cam_Z_fwd_X_left_Y_up(cam: VisionSensor, target_pos, world_up=(0, 0, 1)):
    """
    Build an orthonormal frame with columns [X(left), Y(up), Z(forward)]
    so that Z points to target_pos, Y is as 'up' as possible, and X is left.
    """
    cam_pos = cam.get_position()  # Camera position

    # Z (forward, out of camera) toward target
    Z = norm(target_pos - cam_pos)

    # Y 'up' = world_up projected onto plane orthogonal to Z (handle degeneracy)
    up = norm(np.array(world_up, dtype=np.float32))
    Y = up - Z * float(np.dot(up, Z))
    if np.linalg.norm(Y) < 1e-6:
        alt = np.array([1, 0, 0], dtype=np.float32) if abs(Z[0]) < 0.9 else np.array([0, 1, 0], dtype=np.float32)
        Y = alt - Z * float(np.dot(alt, Z))
    Y = norm(Y)
    X = norm(np.cross(Y, Z))

    R = np.stack([X, Y, Z], axis=1)  # columns are local axes in world
    orientation = eul_from_R(R.transpose())
    cam.set_orientation(-np.asarray(orientation))
