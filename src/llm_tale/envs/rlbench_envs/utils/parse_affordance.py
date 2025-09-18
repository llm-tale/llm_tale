import torch
import numpy as np
from llm_tale.envs.rlbench_envs.utils.primitive_pt import pick, transport
from llm_tale.utils.rotation_utils import quat2rotmxyzw, rotm2quatxyzw, axisangle2rotm


# quat wxyz
class Obj:
    def __init__(
        self,
        name,
        shape,  # l,w,h
        attribute=None,  # dictionary {text: torch.tensor(7)#SE3}
        major_axis=0,  # axis index
        minor_axis=1,
        plane_vertical=2,
        articulate_axis=2,  # axis of a moving articulate object
    ):
        self.name = name
        self.shape = shape
        self.attribute = attribute if attribute is not None else []
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.plane_vertical = plane_vertical
        self.articulate_axis = articulate_axis

    def update_pose(self, pose):
        self.pose = pose

    def get_attribute(self, text):
        if isinstance(text, str):
            attrs = text.split(".")
        elif isinstance(text, list):
            attrs = text
        if len(attrs) == 1:
            attrs.append("centroid")
        for idx, attr in enumerate(attrs[1:]):
            if attr == "centroid":
                return self.pose[:3]
            elif attr in self.attribute:
                if isinstance(self.attribute[attr], torch.Tensor):
                    pos = self.pose[:3]
                    pos = pos + quat2rotmxyzw(self.pose[3:]) @ self.attribute[attr][:3]
                if isinstance(self.attribute[attr], Obj):
                    pos = self.attribute[attr].get_attribute(attrs[1:])
                return pos
            else:
                rot = quat2rotmxyzw(self.pose[3:])
                if attr == "major_axis":
                    return rot[:, self.major_axis]
                elif attr == "minor_axis":
                    return rot[:, self.minor_axis]
                elif attr == "z_axis":
                    return rot[:, 2]
                elif attr == "plane_vertical":
                    sign = np.sign(self.plane_vertical)
                    self.plane_vertical = abs(self.plane_vertical)
                    return sign * rot[:, self.plane_vertical]
                elif attr == "articulate_axis":
                    sign = np.sign(self.articulate_axis)
                    self.articulate_axis = abs(self.articulate_axis)
                    return sign * rot[:, self.articulate_axis]
                else:
                    return None
        return self.attribute


def parse_object(text, task_objects, pick=True):
    if text == "ABSOLUTE_VERTICAL":
        return torch.tensor([0.0, 0.0, -1]) if pick else torch.tensor([0.0, 0.0, 1])
    attrs = text.split(".")
    obj: Obj = task_objects[attrs[0]]
    return obj.get_attribute(text)


def parse_pick_affordance(aff, cur_plan, gripper_pose, task_objects):
    position_text = aff["gripper position"]
    z_axis_text = aff["gripper ori z_axis"]
    x_axis_text = aff["gripper ori x_axis"]
    if x_axis_text == "DEFAULT" and z_axis_text == "ABSOLUTE_VERTICAL":
        z_axis = torch.tensor([0.0, 0.0, -1])
        x_axis = torch.tensor([1.0, 0.0, 0.0])
        x_robot = quat2rotmxyzw(gripper_pose[3:])[:, 0]
        theta = np.arccos(x_axis @ x_robot)
        if theta > np.pi / 2:
            x_axis = -x_axis
        y_axis = torch.cross(z_axis, x_axis)
        rot = torch.stack([x_axis, y_axis, z_axis], dim=1)
    elif x_axis_text == "ABSOLUTE_VERTICAL" and z_axis_text == "DEFAULT":
        rot = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
    else:
        z_axis = parse_object(z_axis_text, task_objects)
        x_axis = parse_object(x_axis_text, task_objects)
        x_robot = quat2rotmxyzw(gripper_pose[3:])[:, 0]
        theta = np.arccos(x_axis @ x_robot)
        if theta > np.pi / 2:
            x_axis = -x_axis
        y_axis = torch.cross(z_axis, x_axis)
        rot = torch.stack([x_axis, y_axis, z_axis], dim=1)
    pos = parse_object(position_text, task_objects)
    attrs = position_text.split(".")
    obj: Obj = task_objects[attrs[0]]
    obj_rot = quat2rotmxyzw(obj.pose[3:])

    re_pos = obj_rot.transpose(0, 1) @ (obj.pose[:3] - pos)
    re_rot = obj_rot.transpose(0, 1) @ rot

    picked_obj = cur_plan.target_object

    pick_axis = parse_object(f"{picked_obj}.articulate_axis", task_objects)
    if pick_axis is not None:
        cur_plan.pick_axis = pick_axis
    return re_pos, re_rot


def parse_transport_affordance(aff, cur_plan, gripper_pose, task_objects):
    # get object in hand
    # calculate se(3) from inhand to target
    # apply the se(3) to the gripper
    # get first key value of aff
    aff = {k: v for k, v in aff.items() if ("." in k) or ("axis" in k) or ("orientation" in k)}

    k1 = list(aff.keys())[0]
    v1 = aff[k1]
    in_hand_obj = k1.split(".")[0]
    in_hand_obj = task_objects[in_hand_obj]
    target = v1.split(".")[0]
    target = task_objects[target]
    pos_target = parse_object(v1, task_objects)
    pos_inhand = parse_object(k1, task_objects)

    # position
    dis = gripper_pose[:3] - pos_inhand[:3]
    # orientation
    if "orientation" in aff.keys():
        if aff["orientation"] == "DEFAULT":
            goal_rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            goal_quat = rotm2quatxyzw(goal_rot)
            rot = goal_rot @ quat2rotmxyzw(gripper_pose[3:]).transpose(0, 1)
        elif aff["orientation"] == "TRANSLATION":
            rot = torch.eye(3)
            goal_quat = gripper_pose[3:]
        else:
            raise NotImplementedError
    elif len(aff.keys()) == 3:
        raise NotImplementedError
    elif len(aff.keys()) == 4:
        obj_1_axis, obj_2_axis = list(aff.keys())[1:3]
        target_1_axis, target_2_axis = aff[obj_1_axis], aff[obj_2_axis]

        target_1_axis = parse_object(target_1_axis, task_objects)
        target_2_axis = parse_object(target_2_axis, task_objects)
        obj_1_axis = parse_object(obj_1_axis, task_objects, pick=False)
        obj_2_axis = parse_object(obj_2_axis, task_objects, pick=False)

        if target_1_axis @ obj_1_axis < 0:
            target_1_axis = -target_1_axis
        rot1 = axisangle2rotm(torch.cross(obj_1_axis, target_1_axis))
        if target_2_axis @ (rot1 @ obj_2_axis) < 0:
            target_2_axis = -target_2_axis
        rot2 = axisangle2rotm(torch.cross(rot1 @ obj_2_axis, target_2_axis))
        rot = rot2 @ rot1
        goal_quat = rotm2quatxyzw(rot @ quat2rotmxyzw(gripper_pose[3:]))

    # transport axis
    transport_axis = aff["transportation axis"]
    if transport_axis == "None":
        transport_axis = torch.tensor([0.0, 0.0, 0.0])
    else:
        transport_axis = parse_object(transport_axis, task_objects) * max(in_hand_obj.shape)
        cur_plan.transport_axis = transport_axis
    gripper_pos = pos_target + rot @ dis
    if target.pose[3:].shape[0] == 4:
        relative_rot = quat2rotmxyzw(target.pose[3:]).transpose(0, 1) @ quat2rotmxyzw(goal_quat)
        relative_pos = quat2rotmxyzw(target.pose[3:]).transpose(0, 1) @ (gripper_pos - target.pose[:3])
    else:
        relative_rot = quat2rotmxyzw(goal_quat)
        relative_pos = gripper_pos - target.pose[:3]
    return relative_pos, relative_rot


def parse_code(code):
    # robot.pick(CubeA)
    txt_prim, txt_objs = code[:-1].split(".")[1].split("(")
    if txt_prim == "pick":
        prim = pick
        obj = txt_objs[:]
        return [prim, obj]
    elif txt_prim == "transport":
        picked_obj, target, mode = txt_objs.split(", ")
        mode = mode[1:-1]
        print(picked_obj, target, mode)
        prim = transport
        return [prim, picked_obj, target]
