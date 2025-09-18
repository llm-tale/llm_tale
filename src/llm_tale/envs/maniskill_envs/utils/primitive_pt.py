import numpy as np
import torch
from llm_tale.utils.rotation_utils import quat2rotmwxyz, rotm2quatwxyz, rotm2axisangle, quat_dis


class primitive:
    def __init__(self, target_object, device="cpu") -> None:
        # check error detection
        self.contact_region = [0.05, 0.03, 0.25, 0.15]
        self.finish_region = [0.03, 0.01, 0.1, 0.1]
        self.k = [5, 5]
        self.max_vel = [1, 1]
        self.motion_const = [0.5, 0.5]
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
        # determine contact-rich region
        target_quat = target_pose[3:]
        r_dis = torch.sin(quat_dis(ee_pose[3:7], target_quat))
        rot_axis = quat2rotmwxyz(target_quat)[:, 2]
        pos_dis = target_pose[:3] - ee_pose[:3]
        z_dis = pos_dis.dot(rot_axis)
        dis = np.linalg.norm(pos_dis)
        c_1 = dis < region[0]
        c_2 = z_dis < region[1]
        c_3 = r_dis < region[2]
        return c_1 and c_2 and c_3

    def update_state(self, gripper_pose, target_pose, objects):
        contact_target_pose = self.move_tar(gripper_pose, target_pose)

        if not self.in_contact:
            self.in_contact = self.in_region(gripper_pose, contact_target_pose, self.contact_region)
        self.finish = self.in_region(gripper_pose, target_pose, self.finish_region)
        if self.name == "pick":
            self.success = self.check_pickup(target_pose) if self.check_pickup(target_pose) else None
        if self.name == "transport":
            pickedobj_pose = objects[self.picked_object]
            self.check_drop(gripper_pose, pickedobj_pose)

    def init_bias(self, gripper_pose, object_pose):
        # init pos and quat bias
        pos_bias = quat2rotmwxyz(gripper_pose[3:]).transpose(0, 1) @ (object_pose[:3] - gripper_pose[:3])
        quat_bias = rotm2quatwxyz(quat2rotmwxyz(gripper_pose[3:]).transpose(0, 1) @ quat2rotmwxyz(object_pose[3:]))
        self.obj_init_pose[:3] = pos_bias
        self.obj_init_pose[3:] = quat_bias

    def check_drop(self, gripper_pose, object_pose):
        pass

    def reward(self, gripper_pose, action, obs):
        target_pose = self.target_pose
        gripper_pose = gripper_pose
        pos_error = torch.norm(gripper_pose[:3] - target_pose[:3])
        quat_error = torch.sin(quat_dis(gripper_pose[3:7], target_pose[3:7]))
        error = pos_error * (1 - self.no_position) + quat_error / 10
        reward = -torch.tanh(10 * error)

        is_transport = self.name == "transport"
        if is_transport:
            reward = 2 * reward + 2
        elif self.name == "pick":
            reward = reward
        q_vel = obs["q_vel"]
        static_reward = -torch.tanh(
            3
            * torch.linalg.norm(
                q_vel,
            )
        )
        reward += static_reward

        return reward

    def check_pickup(self, object_pose):
        # check if object been picked up
        pick_axis = self.pick_axis
        objmove = object_pose[:3] - self.obj_init_pose[:3]
        objmove = objmove @ pick_axis
        obj_up = objmove > 0.04
        return obj_up

    def move_tar(self, gripper_pose, target_pose):
        target_pose = target_pose.clone()
        if self.name == "pick":
            if not self.in_contact:
                gripper_zaxis = quat2rotmwxyz(gripper_pose[3:])[:, 2]
                target_pose[:3] = target_pose[:3] - gripper_zaxis * 0.03 * 0
            if self.object_in_hand:
                target_pose[:3] = target_pose[:3] + self.pick_axis * 0.1
        elif self.name == "transport":
            if not self.in_contact:
                target_pose[:3] = target_pose[:3] - self.transport_axis  # transport_axis lenth and direction
        return target_pose

    def check_object_grasp(self, gripper_pose, object_pose):
        return False

    def prime_action(self, action):
        if self.name == "pick":
            if self.object_in_hand:
                action[6] = -1.0
            elif not self.in_contact:
                action[6] = 1.0
        elif self.name == "transport":
            action[6] = -1.0
        return action

    def dummy_action(self):
        action = torch.zeros(7, device=self.device)
        if self.object_in_hand:
            action[6] = -1.0
        else:
            if self.finish:
                print("close gripper because finish")
                action[6] = -1.0
            else:
                action[6] = 1.0
        return action


def _move_to(k, gripper_pose, target_pose):
    target_pose = target_pose.clone()
    k_vel, k_omega = k
    vel = (target_pose[:3] - gripper_pose[:3]) * k_vel
    gripper_rot = quat2rotmwxyz(gripper_pose[3:])
    rot_dis = gripper_rot.transpose(0, 1) @ quat2rotmwxyz(target_pose[3:])
    axis, angle = rotm2axisangle(rot_dis)
    axis = gripper_rot @ axis
    omega = -axis * angle * k_omega
    return vel, omega


def residual_action(gripper_poses, target_poses, prim, action):
    action = action.clone()
    target_poses = prim.move_tar(gripper_poses, target_poses)
    vel_limit, omega_limit = prim.max_vel
    cons_p, cons_r = prim.motion_const
    k = prim.k
    vel, omega = _move_to(k, gripper_poses, target_poses)
    action[:3] = torch.clamp(vel, -vel_limit, vel_limit) * (1 - prim.no_position) + cons_p * action[:3]
    action[3:6] = torch.clamp(omega, -omega_limit, omega_limit) + cons_r * action[3:6]
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
    def __init__(self, picked_object, target_object, transport_axis=None, no_position=False) -> None:
        super().__init__(target_object)
        self.contact_region = [0.05, 0.05, 0.1, 0.25]

        self.picked_object = picked_object
        self.target_object = target_object
        self.name = "transport"
        self.no_position = no_position
        if transport_axis is not None:
            self._transport_axis = transport_axis
        else:
            self.transport_axis = torch.tensor([0, 0, -0.00], device=self.device)
        self.gripper_goal = torch.tensor([1, 0], device=self.device)


class insert(transport):
    def __init__(self, picked_object, target_object, transport_axis=None) -> None:
        super().__init__(picked_object, target_object, transport_axis)
        self.contact_region = [0.00, 0.01, 0.03, 0.25]

    def update_insert_info(self, error):
        dis_yz = torch.norm(error[1:])
        pre_insert = dis_yz < 0.01
        if pre_insert:
            self.in_contact = True
        self.insert_error = error

    def reward(self, gripper_pose, action, obs):
        error = torch.norm(self.insert_error)
        reward = -torch.tanh(5 * error)
        reward = 2 + 2 * reward
        q_vel = obs["q_vel"]
        static_reward = -torch.tanh(
            3
            * torch.linalg.norm(
                q_vel,
            )
        )
        reward += static_reward
        return reward

    def update_state(self, gripper_pose, target_pose, objects):
        super().update_state(gripper_pose, target_pose, objects)
        if self.in_contact:
            self.max_vel = [1, 1]
            self.k = [5, 5]
            self.motion_const = [0.5, 0.5]


class place(transport):
    def __init__(self, picked_object, target_object, transport_axis=None) -> None:
        super().__init__(picked_object, target_object, transport_axis)

    def prime_action(self, action):
        if not self.in_contact:
            action[6] = -1.0
        if not self.object_in_hand:
            action[6] = 1.0
        return action

    def dummy_action(self):
        action = torch.zeros(7, device=self.device)
        if self.in_contact:
            action[6] = 1.0

        return action
