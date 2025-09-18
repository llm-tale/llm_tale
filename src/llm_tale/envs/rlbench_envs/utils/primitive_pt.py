import numpy as np
import torch
from llm_tale.utils.rotation_utils import quat2rotmxyzw, rotm2quatxyzw, rotm2axisangle, quat_dis


class primitive:
    def __init__(self, target_object, device="cpu") -> None:
        # check error detection
        self.contact_region = [0.05, 0.05, 0.25, 0.25]
        self.finish_region = [0.05, 0.02, 0.1, 0.1]
        self.k = [1, 1]
        self.motion_const = [0.05, 0.15]
        self.base_limitation = [0.10, 0.3]
        self.initialized = False
        self.success = None
        self.device = device
        self.last_action = None
        self.no_position = False

    def initialize(self):
        # parse target position
        self.initialized = True
        self.object_in_hand = False  # check if object in hand
        self.in_contact = False  # check if in contact region
        self.finish = False  # check if there is in finish region(close to the target) used for open loop control
        self.success = None
        self.obj_init_pose = torch.zeros(7, device=self.device)

    def in_region(self, ee_pose, target_pose, region):
        target_quat = target_pose[3:]
        rot_axis = quat2rotmxyzw(target_quat)[:, 2]
        pos_dis = target_pose[:3] - ee_pose[:3]
        z_dis = pos_dis.dot(rot_axis)
        dis = np.linalg.norm(pos_dis)
        r_dis = torch.sin(quat_dis(ee_pose[3:7], target_quat))
        c_1 = dis < region[0]
        c_2 = z_dis < region[1]
        c_3 = r_dis < region[2]

        return c_1 and c_2 and c_3

    def update_state(self, gripper_pose, target_pose, objects):
        if self.name == "transport":
            contact_target_pose = self.move_tar(gripper_pose, target_pose)
        else:
            contact_target_pose = target_pose.clone()

        if not self.in_contact:
            self.in_contact = self.in_region(gripper_pose, contact_target_pose, self.contact_region)
        self.finish = self.in_region(gripper_pose, contact_target_pose, self.finish_region)
        if self.name == "pick":
            self.success = self.check_pickup(target_pose)

    def init_bias(self, gripper_pose, object_pose):
        # init pos and quat bias
        pos_bias = object_pose[:3] - gripper_pose[:3]
        quat_bias = rotm2quatxyzw(quat2rotmxyzw(object_pose[3:]) @ quat2rotmxyzw(gripper_pose[3:]).transpose(0, 1))
        self.obj_init_pose[:3] = pos_bias
        self.obj_init_pose[3:] = quat_bias

    def check_drop(self, gripper_pose, object_pose):
        pos_bias = object_pose[:3] - gripper_pose[:3]
        init_pos_bias = self.obj_init_pose[:3]
        pos_error = torch.norm(pos_bias - init_pos_bias)
        if pos_error > 0.03:
            # print("pos_error",pos_error)
            self.success = False

    def reward(self, gripper_pose, action, obs):
        target_pose = self.target_pose
        gripper_pose = gripper_pose
        pos_error = torch.norm(gripper_pose[:3] - target_pose[:3])
        error = pos_error * (1 - self.no_position)
        reward = -torch.tanh(5 * error)

        is_transport = self.name == "transport"
        if is_transport and self.object_in_hand:
            reward = 2 * reward + 2
        elif self.name == "pick":
            reward = reward
        q_vel = obs["q_vel"]
        static_reward = -0.25 * torch.tanh(
            torch.linalg.norm(
                q_vel,
            )
            / np.pi
        )
        reward += static_reward

        return reward

    def check_pickup(self, object_pose):
        # check if obejct been picked up
        pick_axis = self.pick_axis
        objmove = object_pose[:3] - self.obj_init_pose[:3]
        objmove = objmove @ pick_axis
        obj_up = objmove > 0.05
        return obj_up

    def move_tar(self, gripper_pose, target_pose):
        target_pose = target_pose.clone()
        if self.name == "pick":
            if self.object_in_hand:
                target_pose[:3] = (
                    target_pose[:3] + (0.1 - (target_pose[:3] - self.obj_init_pose[:3]) @ self.pick_axis) * self.pick_axis
                )

        elif self.name == "transport":
            if not self.in_contact:
                target_pose[:3] = target_pose[:3] - self.transport_axis
                # designed for insertion by default gripper z axis * 0.05
            return target_pose
        return target_pose

    def check_object_grasp(self, gripper_pose, object_pose):
        return False

    def prime_action(self, action):
        if self.name == "pick":
            if self.object_in_hand:
                action[-1] = -1.0
            elif not self.in_contact:
                action[-1] = 1.0
        elif self.name == "transport":
            action[-1] = -1.0
        return action

    def dummy_action(self, pos_only=False):
        if not pos_only:
            action = torch.zeros(7, device=self.device)
        else:
            action = torch.zeros(4, device=self.device)
        if self.object_in_hand:
            action[-1] = -1.0
        else:
            if self.finish:
                print("close gripper because finish")
                action[-1] = -1.0
            else:
                action[-1] = 1.0
        return action


def _move_to(k, gripper_pose, target_pose):
    k_vel, k_omega = k
    vel = (target_pose[:3] - gripper_pose[:3]) * k_vel
    gripper_rot = quat2rotmxyzw(gripper_pose[3:])
    rot_dis = gripper_rot.transpose(0, 1) @ quat2rotmxyzw(target_pose[3:])
    axis, angle = rotm2axisangle(rot_dis)
    axis = gripper_rot @ axis
    omega = axis * angle * k_omega
    return vel, omega


def residual_action(gripper_poses, target_poses, prim, action):
    target_poses = prim.move_tar(gripper_poses, target_poses)
    k = prim.k
    d_pos, d_angle = _move_to(k, gripper_poses, target_poses)
    lim_p, lim_r = prim.base_limitation[0], prim.base_limitation[1]
    cons_p, cons_r = prim.motion_const[0], prim.motion_const[1]
    action[:3] = torch.clamp(d_pos, -lim_p, lim_p) + cons_p * action[:3]
    action[3:6] = torch.clamp(d_angle, -lim_r, lim_r) + cons_r * action[3:6]
    return action


class pick(primitive):
    def __init__(self, target_object) -> None:
        super().__init__(target_object)
        self.target_object = target_object
        self.name = "pick"
        self.gripper_goal = torch.tensor([0, 1.0], device=self.device)
        self.pick_axis = torch.tensor([0, 0, 1.0], device=self.device)

    def set_pickaxis(self, pick_axis):
        self.pick_axis = pick_axis


class transport(primitive):
    def __init__(self, picked_object, target_object, transport_axis=None) -> None:
        super().__init__(target_object)
        self.contact_region = [0.08, 0.05, 0.25, 0.25]
        self.picked_object = picked_object
        self.target_object = target_object
        self.name = "transport"
        if transport_axis is not None:
            self._transport_axis = transport_axis
        else:
            self.transport_axis = torch.tensor(
                [0, 0, 0.0],
            )
        self.gripper_goal = torch.tensor([1, 0], device=self.device)

    def get_transport_axis(self, objects):
        object_pose = objects[self.target_object]
        if self._transport_axis == "vertical":
            self.transport_axis = object_pose[3:6]
        elif self._transport_axis == "x_axis":
            rot = quat2rotmxyzw(object_pose[3:])
            self.transport_axis = rot[:, 0]
        elif self._transport_axis == "y_axis":
            rot = quat2rotmxyzw(object_pose[3:])
            self.transport_axis = rot[:, 1]
        elif self._transport_axis == "z_axis":
            rot = quat2rotmxyzw(object_pose[3:])
            self.transport_axis = rot[:, 2]
