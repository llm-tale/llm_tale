import os
import torch
import gymnasium.spaces as spaces
import numpy as np
from pyrep.objects.vision_sensor import VisionSensor

from llm_tale.envs.rlbench_envs.utils.skrl_wrapper import _Wrapper
from llm_tale.envs.rlbench_envs.utils.parse_affordance import Obj
from llm_tale.envs.rlbench_envs.utils.robot import check_reachable
from llm_tale.envs.rlbench_envs.utils.render import aim_cam_Z_fwd_X_left_Y_up
from llm_tale.utils.rotation_utils import quat2rotmxyzw, quat_from_axis_angle
from llm_tale.envs.rlbench_envs.utils.arm_action_modes import SafeEndEffectorPoseViaIK as EndEffectorPoseViaIK

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig

from typing import List, Tuple
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.objects.object import Object
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.backend.task import Task, TASKS_PATH
from rlbench.backend.conditions import DetectedCondition, NothingGrasped
from rlbench.backend.spawn_boundary import SpawnBoundary


GROCERY_NAMES = [
    "chocolate jello",
]


class PutBox(Task):

    def __init__(self, *args, **kwargs) -> None:
        assert len(args) < 3, "Do not set the name of the task manually"
        assert "name" not in kwargs, "Do not set the name of the task manually"
        name = "put_groceries_in_cupboard"
        super().__init__(*args, name=name, **kwargs)

    def init_task(self) -> None:
        self.groceries = [Shape(name.replace(" ", "_")) for name in GROCERY_NAMES]
        self.grasp_points = [Dummy("%s_grasp_point" % name.replace(" ", "_")) for name in GROCERY_NAMES]
        self.waypoint1 = Dummy("waypoint1")
        self.register_graspable_objects(self.groceries)
        self.boundary = SpawnBoundary([Shape("workspace")])

    def init_episode(self, index: int) -> List[str]:
        self.boundary.clear()
        [self.boundary.sample(g, min_distance=0.1) for g in self.groceries]
        self.waypoint1.set_pose(self.grasp_points[index].get_pose())
        self.register_success_conditions(
            [DetectedCondition(self.groceries[index], ProximitySensor("success")), NothingGrasped(self.robot.gripper)]
        )
        return [
            "put the %s in the cupboard" % GROCERY_NAMES[index],
            "pick up the %s and place it in the cupboard" % GROCERY_NAMES[index],
            "move the %s to the bottom shelf" % GROCERY_NAMES[index],
            "put away the %s in the cupboard" % GROCERY_NAMES[index],
        ]

    def variation_count(self) -> int:
        return len(GROCERY_NAMES)

    def boundary_root(self) -> Object:
        return Shape("boundary_root")

    def base_rotation_bounds(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        return (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)

    def load(self) -> Object:
        if Object.exists(self.name):
            return Dummy(self.name)
        ttm_file = os.path.join(TASKS_PATH, "../task_ttms/%s.ttm" % self.name)
        if not os.path.isfile(ttm_file):
            raise FileNotFoundError("The following is not a valid task .ttm file: %s" % ttm_file)
        self._base_object = self.pyrep.import_model(ttm_file)
        TO_REMOVE = [
            "crackers",
            "strawberry_jello",
            "soup",
            "tuna",
            "spam",
            "sugar",
            "coffee",
            "mustard",
        ]
        for obj in self._base_object.get_objects_in_tree(exclude_base=True):
            if obj.get_name() in TO_REMOVE:
                obj.remove()
        return self._base_object


class PutBoxEnv(_Wrapper):
    def __init__(
        self,
        env=None,
        episode_length: int = 15,
        LLM_instuction=False,
        eval: bool = False,
        pos_only=False,
        image_input=False,
        expert_reward=False,
        null_space=None,
        reward_scale=1,
        on_policy=True,
        record=False,
    ) -> None:
        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        if record:
            obs_config.front_camera.rgb = True
            obs_config.front_camera.image_size = (1080, 1080)
        action_mode = MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=False), Discrete())
        env = Environment(action_mode, "", obs_config, headless=True)
        env.launch()
        self.task = env.get_task(PutBox)
        self.image_input = image_input
        self.LLM_instuction = LLM_instuction
        env.observation_space = self.define_observation_space()
        if pos_only:
            env.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)
        else:
            env.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=float)

        self.null_space = null_space
        self.record = record

        super().__init__(
            env=env,
            LLM_instuction=LLM_instuction,
            eval=eval,
            episode_length=episode_length,
            image_input=image_input,
            expert_reward=expert_reward,
            reward_scale=reward_scale,
            on_policy=on_policy,
            pos_only=pos_only,
        )

    def restart_simulation(self):
        self._env.shutdown()
        obs_config = ObservationConfig()
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False if not self.image_input else True)
        action_mode = MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=False), Discrete())
        env = Environment(action_mode, "", obs_config, headless=True)  # see (3)
        env.launch()
        self.task = env.get_task(PutBox)
        env.observation_space = self.define_observation_space()
        env.action_space = spaces.Box(low=-1, high=1, shape=(4 if self.pos_only else 7,), dtype=float)
        self._env = env

    def define_observation_space(self):
        if self.image_input:
            state_num = 7
            observation_space = spaces.Dict(
                {
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=float),
                    "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=int),
                }
            )
        else:
            state_num = 1 + 7 + 7 + 9 + 14
            if self.LLM_instuction:
                state_num += 7
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_num,), dtype=float)
        return observation_space

    def set_objects(self):
        _objects = self.task._task._initial_objs_in_scene

        object_dict = {}
        for i in _objects:
            object_dict[i[0].get_name()] = i[0]
        self.collide_objects = {
            "cupboard": object_dict["cupboard"],
        }

        self._objects = {
            "chocolate_box": object_dict["chocolate_jello"],
            "cupboard": object_dict["waypoint4"],
            "real_cupboard": object_dict["cupboard"],
        }

        pose = self._objects["chocolate_box"].get_pose()
        position = np.array([0.2, 0, pose[2]]) + 2 * np.clip(
            np.random.normal(0.05, 0.05, 3) * np.array([1, 1, 0]), np.array([-0.05, -0.05, 0]), np.array([0.05, 0.05, 0])
        )
        axis_angle = np.array([0, 0, 1]) * -np.abs(np.random.uniform(0, np.pi / 6))
        orientation = quat_from_axis_angle(axis_angle)
        pose = np.concatenate([position, orientation])
        self._objects["chocolate_box"].set_pose(pose)

        pose = self._objects["real_cupboard"].get_pose()
        pose[3:] = np.array([0, 0, 1, 0])
        pose[0] = np.clip(pose[0], 0.58, 0.62)
        pose[2] -= 0.1
        # print("pose",pose)
        self._objects["real_cupboard"].set_pose(pose)
        chocolate_box = Obj(
            name="chocolate_box",
            shape=self._objects["chocolate_box"].get_bounding_box(),
            major_axis=1,
        )
        cupboard = Obj(
            name="cupboard",
            shape=None,
            plane_vertical=-2,
        )
        self.task_objects = {
            "chocolate_box": chocolate_box,
            "cupboard": cupboard,
        }
        if self.llm_tale is not None:
            self.llm_tale.task_objects = self.task_objects

        # Set the camera to look at the cupboard
        if self.record:
            cam = VisionSensor("cam_front")
            cam.set_position([-1, 1.5, 1.8])  # Set camera position
            aim_cam_Z_fwd_X_left_Y_up(cam, [0.5, 0.0, 1.2], world_up=(0, 0, 1))

    def get_object_state(self, obs_dict):
        choco_pose = torch.tensor(self._objects["chocolate_box"].get_pose(), dtype=torch.float32)
        cupboard_pose = torch.tensor(self._objects["cupboard"].get_pose(), dtype=torch.float32)
        self.objects = {"chocolate_box": choco_pose, "cupboard": cupboard_pose}
        obj_states = torch.cat([choco_pose, cupboard_pose])
        for obj_name, obj in self.task_objects.items():
            obj.update_pose(self.objects[obj_name])
        return obj_states

    def get_task_relevent_info(self, info: dict) -> dict:
        # ---- plan state (Python scalars) ----
        cur_plan = self.llm_tale.plans[self.llm_tale.current_plannum] if self.llm_tale is not None else None
        pick_success = bool(cur_plan.success) if (cur_plan and cur_plan.name == "pick") else False

        reachable, s = (False, 1e9)
        if cur_plan is not None:
            reachable, s = check_reachable(cur_plan.target_pose)
        reachable = bool(reachable) or (abs(float(s)) < 1.1)

        # ---- orientation / "falling" from quaternion ----
        # Expecting quaternion in xyzw; normalize to avoid numerical drift.
        quat_box = self.objects["chocolate_box"][3:].detach().cpu().to(torch.float32)
        qn = quat_box / (torch.norm(quat_box) + 1e-12)

        R = quat2rotmxyzw(qn)  # ensure this returns a 3x3 on CPU
        if not isinstance(R, torch.Tensor):
            R = torch.as_tensor(R, dtype=torch.float32)
        zaxis = R[:, 2]
        zz = float(zaxis[2].item())  # cos(angle between box-z and world-z)
        object_falling = zz < 0.707  # ~ cos(45°)

        # ---- position/orientation progress (floats) ----
        pos_vec = (self.objects["chocolate_box"][:3] - self.objects["cupboard"][:3]).detach().cpu().to(torch.float32)
        weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)
        pos_error = float(torch.norm(weights * pos_vec).item())
        ori_error = float(1.0 - zz)
        pos_progress = 1 - (pos_error - 0.01) / 0.3
        ori_progress = 1 - (ori_error - 0.005) / 0.3
        progress = float(np.clip(pos_progress, 0, 1) * np.clip(ori_progress, 0, 1))

        # ---- collision (bool) ----
        collision = bool(self.task._scene.robot.arm.check_arm_collision(self.collide_objects["cupboard"]))

        # ---- success criterion (Python scalars only) ----
        q_vel = self.observation["q_vel"]
        if isinstance(q_vel, torch.Tensor):
            qmax = float(torch.max(torch.abs(q_vel)).item())
        else:
            qmax = float(np.max(np.abs(q_vel)))

        # ---- write Python types into info ----
        info["reachable"] = bool(reachable)
        info["pick_success"] = bool(pick_success)
        info["object_falling"] = bool(object_falling)
        info["collision"] = bool(collision)
        info["success"] = bool(progress > 0.85 and qmax < 0.3)
        return info

    def task_relevant_reward(self, info):
        reward = 0
        terminated = False
        if info["pick_success"]:
            reward += 5
            print("pick_success")
        if info["collision"]:
            reward -= 10
            terminated = True
            print("collision")
        if not info["reachable"]:
            reward -= 10
            terminated = True
            print("not reachable")
        if info["object_falling"]:
            reward -= 10
            terminated = True
            print("object_falling")
        elif info["success"]:
            terminated = True
            if self.on_policy:
                reward += 10 + 3 * (self.episode_length - self.step_num)
            else:
                reward += 100

        return reward, terminated


env = PutBoxEnv


episode_length = 20
task = "PutBox"
llm_plan_path = f"cache/{task.lower()}_code_gpt-4o.pkl"
Instruction = "Put the chocolate box in cupboard"
Objects = """
Objects:
    chocolate_box = Object(
        position, 
        orientation, # z-axis pointing up
        attributes=None)
    cupboard = Object(
        position, 
        orientation, 
        attributes=None) 
"""
env_kwargs = {
    "ppo_explo": {
        "LLM_instuction": True,
        "episode_length": episode_length,
    },
    "td3_explo": {
        "episode_length": episode_length,
        "LLM_instuction": True,
        "on_policy": False,
        "reward_scale": 0.01,
    },
}
llm_tale_kwargs = {
    "ppo_explo": {
        "env_type": "rlbench",
        "temperature": 5,
        "start_choice_ep": 500,
    },
    "td3_explo": {
        "env_type": "rlbench",
        "temperature": 15,
        "start_choice_ep": 500,
    },
}
kwargs = {
    "ppo_explo": {
        "train": {
            "start_std": -1.5,
            "device": "cuda",
            "learning_rate_scheduler": "exp",
            "tb_path": "exps_llm_tale/rlbench/{}/PPO".format(task),
        },
        "eval": {
            "start_std": -10,
            "device": "cuda",
            "tb_path": None,
        },
    },
    "td3_explo": {
        "train": {
            "discount": 0.97,
            "batch_size": 512,
            "learning_starts": 10_000,
            "actor_learning_rate": 5e-4,
            "smooth_regularization_noise": 0.05,
            "tb_path": "exps_llm_tale/rlbench/{}/TD3".format(task),
            "state_preprocessor": True,
        },
        "eval": {
            "exploration_timesteps": -1,
            "learning_starts": 0,
            "tb_path": None,
            "state_preprocessor": True,
        },
    },
}
