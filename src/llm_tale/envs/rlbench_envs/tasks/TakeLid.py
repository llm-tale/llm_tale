import torch
import gymnasium.spaces as spaces
import numpy as np
from pyrep.objects.vision_sensor import VisionSensor

from llm_tale.llm_tale import LLM_TALE
from llm_tale.envs.rlbench_envs.utils.skrl_wrapper import _Wrapper
from llm_tale.envs.rlbench_envs.utils.parse_affordance import Obj
from llm_tale.envs.rlbench_envs.utils.render import aim_cam_Z_fwd_X_left_Y_up
from llm_tale.envs.rlbench_envs.utils.arm_action_modes import SafeEndEffectorPoseViaIK as EndEffectorPoseViaIK

from rlbench.tasks import TakeLidOffSaucepan
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig


class TakeLidOffSaucepanEnv(_Wrapper):
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
        obs_config = ObservationConfig(gripper_touch_forces=True)
        obs_config.set_all_low_dim(True)
        obs_config.set_all_high_dim(False)
        if record:
            obs_config.front_camera.rgb = True
            obs_config.front_camera.image_size = (1080, 1080)
        action_mode = MoveArmThenGripper(EndEffectorPoseViaIK(absolute_mode=False), Discrete())
        env = Environment(action_mode, "", obs_config, headless=True)
        env.launch()
        self.task = env.get_task(TakeLidOffSaucepan)
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
            state_num = 1 + 7 + 7 + 9 + 7
            if self.LLM_instuction:
                state_num += 7
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_num,), dtype=float)
        return observation_space

    def set_objects(self):
        _objects = self.task._task._initial_objs_in_scene

        object_dict = {}
        for i in _objects:
            object_dict[i[0].get_name()] = i[0]
        self._objects = {
            "Lid": object_dict["saucepan_lid_grasp_point"],
        }
        self.task_objects = {
            "Lid": Obj(
                name="Lid",
                shape=[0.05, 0.05, 0.05],
                attribute={"lid_handle": torch.tensor([0.0, 0.0, 0.0])},
            )
        }
        if self.llm_tale is not None:
            self.llm_tale.task_objects = self.task_objects

        # Set the camera to look at the pan
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

    def get_object_state(self, obs_dict):
        handle_pose = torch.tensor(self._objects["Lid"].get_pose(), dtype=torch.float32)
        self.objects = {
            "Lid": handle_pose,
        }
        obj_states = torch.cat([handle_pose])
        for obj_name, obj in self.task_objects.items():
            obj.update_pose(self.objects[obj_name])
        return obj_states

    def task_relevant_reward(self, info):
        reward = 0
        terminated = False

        if info["success"]:
            terminated = True
            if self.on_policy:
                reward += 10 + 3 * (self.episode_length - self.step_num)
            else:
                reward += 100
        return reward, terminated


env = TakeLidOffSaucepanEnv

task = "TakeLid"
llm_plan_path = f"cache/{task.lower()}_code_gpt-4o.pkl"
Instruction = "Pick the lid"
Objects = """
Objects:
    Lid = Object(position,orientation=None,attributes=[lid_handle])
"""
episode_length = 15
env_kwargs = {
    "ppo_explo": {
        "LLM_instuction": True,
        "episode_length": episode_length,
    },
    "td3_explo": {
        "LLM_instuction": True,
        "on_policy": False,
        "reward_scale": 0.01,
        "episode_length": episode_length,
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
            "start_std": -1.5,
            "device": "cuda",
            "learning_rate_scheduler": "exp",
            "tb_path": "exps_llm_tale/rlbench/{}/PPO".format(task),
        },
        "eval": {
            "start_std": -10.0,
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
            "tb_path": None,
        },
    },
}
