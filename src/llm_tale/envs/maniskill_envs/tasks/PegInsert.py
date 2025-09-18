from typing import Any, Tuple
import torch
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
from llm_tale.envs.maniskill_envs.utils.skrl_wrapper import Wrapper
from llm_tale.envs.maniskill_envs.utils.parse_affordance import Obj

from mani_skill.envs.tasks.tabletop.peg_insertion_side import PegInsertionSideEnv, _build_box_with_hole
from mani_skill.envs.utils import randomization
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Actor, Pose
from mani_skill.sensors.camera import CameraConfig
from llm_tale.envs.maniskill_envs.utils.primitive_pt import insert
import sapien
from llm_tale.llm_tale import LLM_TALE


@register_env("PegInsertionSide-v2", max_episode_steps=100)
class PegInsertV2(PegInsertionSideEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = self._episode_rng.uniform(0.1, 0.1, size=(self.num_envs,))
            radii = self._episode_rng.uniform(0.02, 0.02, size=(self.num_envs,))
            centers = 0.5 * (lengths - radii)[:, None] * self._episode_rng.uniform(-1, 1, size=(self.num_envs, 2))

            # save some useful values for use later
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            # in each parallel env we build a different box with a hole and peg (the task is meant to be quite difficult)
            pegs = []
            boxes = []

            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                # peg head
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                # peg tail
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5,
                    specular=0.5,
                )
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius],
                    material=mat,
                )
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")

                # box with hole

                inner_radius, outer_radius, depth = (
                    radius + self._clearance,
                    length,
                    length,
                )
                builder = _build_box_with_hole(self.scene, inner_radius, outer_radius, depth, center=centers[i])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")

                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # initialize the box and peg
            xy = randomization.uniform(low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2))
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 2]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 6, np.pi / 2 + np.pi / 6),
            )
            self.peg.set_pose(Pose.create_from_pq(pos, quat))

            xy = randomization.uniform(
                low=torch.tensor([-0.05, 0.2]),
                high=torch.tensor([0.05, 0.4]),
                size=(b, 2),
            )
            pos = torch.zeros((b, 3))
            pos[:, :2] = xy
            pos[:, 2] = self.peg_half_sizes[env_idx, 0]
            quat = randomization.random_quaternions(
                b,
                self.device,
                lock_x=True,
                lock_y=True,
                bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8),
            )
            self.box.set_pose(Pose.create_from_pq(pos, quat))

            # Initialize the robot
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    -np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))


class PegInsertEnv(Wrapper):
    def __init__(
        self,
        LLM_instuction=False,
        eval: bool = False,
        episode_length: int = 150,
        image_input=False,
        expert_reward=False,
        reward_scale=1,
        on_policy=True,
        record=False,
    ) -> None:
        reward_mode = "dense" if expert_reward else "sparse"
        obs_mode = "state_dict" if not image_input else "rgb"
        if record:
            pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
            human_render_camera_configs = CameraConfig("render_camera", pose, 1080, 1080, 1, 0.01, 100).__dict__
        else:
            human_render_camera_configs = None
        env = gym.make(
            "PegInsertionSide-v2",
            reward_mode=reward_mode,
            obs_mode=obs_mode,
            control_mode="pd_ee_delta_pose",
            num_envs=1,
            render_mode="rgb_array",
            human_render_camera_configs=human_render_camera_configs,
        )

        super().__init__(
            env=env,
            LLM_instuction=LLM_instuction,
            eval=eval,
            episode_length=episode_length,
            image_input=image_input,
            expert_reward=expert_reward,
            reward_scale=reward_scale,
            on_policy=on_policy,
        )
        self.set_objects()

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
            get_picked_obj=self.get_picked_obj,
            is_grasping=self.is_grasping,
            on_policy=self.on_policy,
        )

    def is_grasping(self, obj):
        object = self._env.peg
        return self._env.agent.is_grasping(object)

    def define_observation_space(self):
        if self.image_input:
            state_num = 2 + 7 + 7 + 9
            if self.LLM_instuction:
                state_num += 7
            observation_space = spaces.Dict(
                {
                    "state": spaces.Box(low=-np.inf, high=np.inf, shape=(state_num,), dtype=float),
                    "image": spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=int),
                }
            )
        else:
            state_num = 2 + 7 + 7 + 14
            if self.LLM_instuction:
                state_num += 7 + 9
            observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_num,), dtype=float)
        return observation_space

    def set_objects(self):
        self.task_objects = {
            "Peg": Obj(
                "Peg",
                shape=[0.2, 0.04, 0.04],
                attribute={"head": torch.tensor([0.06, 0.0, 0.0]), "end": torch.tensor([-0.06, 0.0, 0.0])},
            ),
            "Hole": Obj("Hole", shape=None),
        }

    def get_object_state(self, obs_dict):
        if self.image_input:
            peg_pose = self._env.peg.pose.raw_pose[0]
            box_hole_pose = self._env.box_hole_pose.raw_pose[0]
        else:
            peg_pose = obs_dict["extra"]["peg_pose"][0]
            box_hole_pose = obs_dict["extra"]["box_hole_pose"][0]
        self.objects = {"Peg": peg_pose, "Hole": box_hole_pose}
        for obj_name, obj in self.task_objects.items():
            obj.update_pose(self.objects[obj_name])
        obj_states = torch.cat([peg_pose, box_hole_pose])
        return obj_states

    def get_picked_obj(self):
        return self._env.peg

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        obs, reward, terminated, truncated, info = super().step(actions)
        return obs, reward, terminated, truncated, info

    def get_task_relevent_info(self, info):
        cur_plan = self.llm_tale.plans[self.llm_tale.current_plannum]
        if cur_plan.name == "transport":
            cur_plan: insert
            cur_plan.update_insert_info(info["peg_head_pos_at_hole"][0])
        return info

    def task_relevant_reward(self, info):
        reward = 0
        terminated = False

        if info["success"]:
            if self.on_policy:
                reward += 10 + 4 * (self.episode_length - self.step_num)
            else:
                reward += 100
            terminated = True

        return reward, torch.tensor(terminated)


task = "PegInsert"
llm_plan_path = f"cache/{task.lower()}_code_gpt-4o.pkl"

env = PegInsertEnv
Instruction = "insert the peg into the hole"
Objects = """
Objects:
    Peg = Object( # a peg is laying on the ground
        position, 
        orientation, # x-axis as center axis
        attributes=[head, end])
    Hole = Object(
        position, 
        orientation, # x-axis as center axis
        attributes=None) 
"""

episode_length = 75
env_kwargs = {
    "ppo_explo": {
        "episode_length": episode_length,
        "LLM_instuction": True,
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
        "env_type": "maniskill",
        "temperature": 5,
        "start_choice_ep": 500,
    },
    "td3_explo": {
        "env_type": "maniskill",
        "temperature": 10,
        "start_choice_ep": 500,
    },
}
kwargs = {
    "ppo_explo": {
        "train": {
            "n_steps": 3072,
            "learning_rate": 2e-4,
            "start_std": -1.5,  # -2
            "device": "cuda",
            "discount": 0.99,
            "value_clip": 0.5,
            "ratio_clip": 0.1,
            "tb_path": "exps_llm_tale/maniskill/{}/PPO".format(task),
        },
        "eval": {
            "start_std": -10,
            "device": "cuda",
            "tb_path": None,
        },
    },
    "td3_explo": {
        "train": {
            "exploration_noise": 0.4,
            "discount": 0.93,
            "gradient_steps": 4,
            "batch_size": 256,
            "buffer_size": 500_000,
            "learning_starts": 40_000,
            "critic_learning_rate": 3e-4,
            "tb_path": "exps_llm_tale/maniskill/{}/TD3".format(task),
        },
        "eval": {
            "exploration_timesteps": -1,
            "learning_starts": 0,
            "tb_path": None,
        },
    },
}
