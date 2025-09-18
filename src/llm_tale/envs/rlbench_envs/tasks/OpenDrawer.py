from typing import Any, Tuple
import torch
import gymnasium.spaces as spaces
import numpy as np
from pyrep.objects.vision_sensor import VisionSensor

from llm_tale.llm_tale import LLM_TALE
from llm_tale.envs.rlbench_envs.utils.skrl_wrapper import _Wrapper
from llm_tale.envs.rlbench_envs.utils.parse_affordance import Obj
from llm_tale.envs.rlbench_envs.utils.render import aim_cam_Z_fwd_X_left_Y_up
from llm_tale.envs.rlbench_envs.utils.arm_action_modes import SafeEndEffectorPoseViaIK as EndEffectorPoseViaIK

from rlbench.tasks import OpenDrawer
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


class OpenDrawerEnv(_Wrapper):
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
        variation_number=1,
    ) -> None:
        obs_config = ObservationConfig(gripper_touch_forces=True)
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        if record:
            obs_config.front_camera.rgb = True
            obs_config.front_camera.image_size = (1080, 1080)
        action_mode = MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=False), Discrete())
        env = Environment(action_mode, "", obs_config, headless=True)
        env.launch()
        self.variation_number = variation_number
        self.task = env.get_task(OpenDrawer)
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

    def reset(self, fake_reset=False, seed=None, options=None) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        self.step_num = 0
        self.current_episode += 1
        descriptions, self._obs_dict = self.task.reset()
        self.last_obs_dict = self._obs_dict
        self.task.set_variation(self.variation_number)
        self.set_objects()

        if fake_reset:
            return self._obs_dict, {}
        if self.LLM_instuction:
            self.llm_tale.reset_high_level()
        obs = self._process_obs(self._obs_dict)
        obs = self.change_device(obs)
        info = {}
        if self.LLM_instuction:
            info = self.init_task_relevent_info(info)

        return obs, info

    def set_objects(self):
        _objects = self.task._task._initial_objs_in_scene
        options = ["bottom", "middle", "top"]

        object_dict = {}
        for i in _objects:
            object_dict[i[0].get_name()] = i[0]
        self._objects = {
            "drawer": object_dict["drawer_frame"],
            "drawer_handle": object_dict[f"waypoint_anchor_{options[self.variation_number]}"],
            "target_position": object_dict["waypoint2"],
        }
        drawer_handle = Obj(
            "drawer_handle",
            [0.1, 0.01, 0.01],
            plane_vertical=2,
            articulate_axis=-2,
        )
        self.task_objects = {
            "drawer_handle": drawer_handle,
            "drawer": Obj(
                "drawer",
                shape=None,
                attribute={
                    "drawer_handle": drawer_handle,
                },
                plane_vertical=2,
                articulate_axis=-2,
            ),
            "target_position": Obj(
                "target_position",
                [0.01, 0.01, 0.01],
                attribute={"centroid": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)},
            ),
        }
        if self.llm_tale is not None:
            self.llm_tale.task_objects = self.task_objects

        # Set the camera to look at the drawer
        if self.record:
            cam = VisionSensor("cam_front")
            cam.set_position([1.0, 1.0, 1.8])  # Set camera position
            aim_cam_Z_fwd_X_left_Y_up(cam, [0.25, 0.0, 1.1], world_up=(0, 0, 1))

    def set_llm_tale(
        self,
        llm_tale: LLM_TALE = None,
        learned_basepolicy=None,
    ):
        self.llm_tale = llm_tale
        self.rl_policy = self.llm_tale.rl_policy
        self.learned_basepolicy = self.llm_tale.learned_basepolicy
        self.llm_tale.set_env(
            task_objects=self.task_objects,
            _post_process_obs=self._post_process_obs,
            change_device=self.change_device,
            get_picked_obj=None,
            is_grasping=self.check_grasp,
            on_policy=self.on_policy,
        )

    def reset_objects(self):
        pass

    def get_object_state(self, obs_dict):
        handle_pose = torch.tensor(self._objects["drawer_handle"].get_pose(), dtype=torch.float32)
        target_pose = torch.tensor(self._objects["target_position"].get_pose(), dtype=torch.float32)
        self.objects = {
            "drawer": handle_pose,
            "drawer_handle": handle_pose,
            "target_position": target_pose,
        }
        obj_states = torch.cat([handle_pose, target_pose])
        for obj_name, obj in self.task_objects.items():
            obj.update_pose(self.objects[obj_name])
        return obj_states

    def check_grasp(self):
        gripper_touch_forces = self._obs_dict.gripper_touch_forces
        norm1 = np.linalg.norm(gripper_touch_forces[0:3])
        norm2 = np.linalg.norm(gripper_touch_forces[3:6])
        return (norm1 > 0.02) and (norm2 > 0.02)

    def init_task_relevent_info(self, info):
        return info

    def get_task_relevent_info(self, info):
        info["object_drop"] = False
        info["collision"] = False

        cur_plan = self.llm_tale.plans[self.llm_tale.current_plannum] if self.llm_tale is not None else None
        if cur_plan and cur_plan.name == "pick":
            info["collision"] = bool(self.task._scene.robot.arm.check_arm_collision(self._objects["drawer"]))
        if cur_plan and cur_plan.name == "transport" and not cur_plan.object_in_hand:
            info["object_drop"] = True

        return info

    def task_relevant_reward(self, info):
        reward = 0.0
        terminated = False
        if info["object_drop"] or info["collision"]:
            terminated = True
            reward = -10.0
        if info["success"]:
            terminated = True
            if self.on_policy:
                reward += 10 + 3 * (self.episode_length - self.step_num)
            else:
                reward += 100
        return reward, torch.tensor(terminated)


env = OpenDrawerEnv

task = "OpenDrawer"
llm_plan_path = f"cache/{task.lower()}_code_gpt-4o.pkl"
Instruction = "Open the drawer"
episode_length = 15
Objects = """
Objects:
    drawer = Object(position,orientation,
    attributes=[drawer_handle]
    )
    target_position = Object(position,orientation=None)
"""
env_kwargs = {
    "ppo_explo": {
        "LLM_instuction": True,
        "episode_length": episode_length,
    },
    "td3_explo": {
        "LLM_instuction": True,
        "episode_length": episode_length,
        "on_policy": False,
        "reward_scale": 0.01,
    },
}
llm_tale_kwargs = {
    "ppo_explo": {
        "env_type": "rlbench",
    },
    "td3_explo": {
        "env_type": "rlbench",
    },
}
kwargs = {
    "ppo_explo": {
        "train": {
            "device": "cuda",
            "start_std": -2.5,
            "learning_rate": 5e-4,
            "device": "cuda",
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
            "state_preprocessor": True,
            "smooth_regularization_noise": 0.1,
            "tb_path": "exps_llm_tale/rlbench/{}/TD3".format(task),
        },
        "eval": {
            "state_preprocessor": True,
            "exploration_timesteps": -1,
            "learning_starts": 0,
            "device": "cuda",
            "tb_path": None,
        },
    },
}
