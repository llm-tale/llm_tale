from typing import Any, Tuple

import torch
from skrl.envs.wrappers.torch.base import Wrapper
from llm_tale.llm_tale import LLM_TALE


class Wrapper(Wrapper):
    def __init__(
        self,
        env: Any,
        LLM_instuction=False,
        eval: bool = False,
        episode_length: int = 150,
        image_input=False,
        expert_reward=False,
        reward_scale=1,
        on_policy=True,
    ) -> None:
        super().__init__(env)
        self._vectorized = False
        self._obs_dict = None
        self.eval = eval
        self.LLM_instuction = LLM_instuction
        self.device = torch.device("cuda")
        self.image_input = image_input
        self.episode_length = episode_length
        self.expert_reward = expert_reward
        self.reward_scale = reward_scale
        self.on_policy = on_policy
        self.current_episode = 0
        self.pick_success_memory = torch.zeros(1000000, dtype=torch.bool)
        self.success_memory = torch.zeros(1000000, dtype=torch.float32)
        self.rl_policy = None
        self._observation_space = self.define_observation_space()

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
            is_grasping=self._env.agent.is_grasping,
            on_policy=self.on_policy,
        )

    def set_llm_plan(self, llm_plan_path):
        if self.LLM_instuction:
            self.llm_tale.set_llm_plan(llm_plan_path)

    def define_observation_space(self):
        raise NotImplementedError

    def _process_obs(self, obs_dict):
        gripper_pose = obs_dict["extra"]["tcp_pose"][0]
        gripper_open = obs_dict["agent"]["qpos"][0, 7:]
        q_vel = obs_dict["agent"]["qvel"][0, :7]
        goal_space = torch.zeros(
            9,
        )
        obs = {
            "gripper_open": gripper_open,
            "gripper_pose": gripper_pose,
            "q_vel": q_vel,
            "goal_space": goal_space,
        }
        object_states = self.get_object_state(obs_dict)
        if not self.image_input:
            obs["obj_states"] = object_states
        else:
            obs["image"] = obs_dict["sensor_data"]["base_camera"]["rgb"][0].float()
            obs["image"] = obs["image"].permute(2, 0, 1)  # Change the order of dimensions to (3, 128, 128)

            obs["image"] = torch.nn.functional.interpolate(
                obs["image"].unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False
            )
            obs["image"] = obs["image"].squeeze(0)  # Remove the batch dimension
        if self.LLM_instuction:
            start_choose = self.current_episode > self.llm_tale.start_choice_ep
            goal_space = self.llm_tale.step_high_level(obs, self.objects, start_choose, self.eval)
            obs = self.llm_tale.reform_obs_with_action(obs, goal_space)
        elif self.learned_basepolicy is not None:
            _obs = torch.cat([obs["gripper_pose"], obs["gripper_open"], obs["obj_states"]])
            action = self.learned_basepolicy(_obs)
            obs["base_action"] = action
        observation = self._post_process_obs(obs, goal_space, replace_goal_space=True)
        return observation

    def get_object_state(self, obs_dict):
        raise NotImplementedError

    def _post_process_obs(self, obs, goal_space, replace_goal_space=False):
        if replace_goal_space:
            obs["goal_space"] = goal_space
            self.observation = obs
        state = torch.cat([obs["gripper_pose"], obs["gripper_open"], obs["q_vel"]])
        if self.LLM_instuction:
            base_action = obs["base_action"] if self.learned_basepolicy is None else obs["base_action"].to(torch.device("cpu"))
            state = torch.cat([state, goal_space, base_action])
        if not self.image_input:
            state = torch.cat([state, obs["obj_states"]])
        else:
            observation = {"state": state, "image": obs["image"]}
            observation = self.change_device(observation)
            state = torch.cat([(observation["image"] / 255 - 0.5).flatten(), observation["state"]])
        observation = torch.unsqueeze(state, 0)
        return observation

    def _process_action(self, action):
        if action.device != "cpu":
            action = action.cpu()
        if action.dim() == 2:
            action = action[0]
        if self.LLM_instuction:
            target_pose = self.observation["goal_space"]
            gripper_pose = self.observation["gripper_pose"]
            if self.learned_basepolicy is None:
                action = self.llm_tale._prim["residual_action"](
                    gripper_poses=gripper_pose,
                    target_poses=target_pose,
                    prim=self.llm_tale.plans[self.llm_tale.current_plannum],
                    action=action,
                )
            else:
                action = action + self.observation["base_action"].cpu()

            action = self.llm_tale.plans[self.llm_tale.current_plannum].prime_action(action)
        elif self.learned_basepolicy is not None:
            _obs = torch.cat(
                [self.observation["gripper_pose"], self.observation["gripper_open"], self.observation["obj_states"]]
            )
            _action = self.learned_basepolicy(_obs)
            action[:-1] = action[:-1] * 0.5
            action = _action + action

        return action

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor
        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """

        with torch.no_grad():
            raw_actions = actions.clone()
            self.step_num += 1
            actions = self._process_action(actions)
            self._obs_dict, reward, terminated, truncated, info = self._env.step(actions)

            obs = self._process_obs(self._obs_dict)
            obs = self.change_device(obs)
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
            if self.LLM_instuction:
                gripper_pose = self._obs_dict["extra"]["tcp_pose"][0]
                info = self.get_task_relevent_info(info)
                reward = self.llm_tale.get_llm_tale_reward(gripper_pose, raw_actions, self.observation)
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

            terminated = torch.tensor(terminated, device=self.device, dtype=torch.bool)
            truncated = torch.tensor([self.step_num >= self.episode_length], device=self.device, dtype=torch.bool)
            if self.LLM_instuction:
                # sparse reward
                ts_reward, ts_terminated = self.task_relevant_reward(info)  # sparse reward and terminate condition
                reward = reward + ts_reward
                terminated = torch.logical_or(terminated, ts_terminated.to(self.device))

            if terminated or truncated:
                terminated = torch.logical_or(terminated, info["success"].to(self.device))
                self.success_memory[self.current_episode] = info["success"]
                if self.rl_policy is not None:
                    self.rl_policy.track_data("Episode /Success Rate", info["success"])
                    if self.LLM_instuction:
                        pick_success = 1 if self.llm_tale.plans[0].success else 0
                        self.rl_policy.track_data("Episode /Pick Success Rate", pick_success)

        return obs, reward * self.reward_scale, terminated, truncated, info

    def change_device(self, obs):
        if isinstance(obs, dict):
            for key in obs.keys():
                obs[key] = obs[key].to(self.device)
        else:
            obs = obs.to(self.device)
        return obs

    def reset(self, fake_reset=False, seed=None, options=None) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        if fake_reset:
            return self._obs_dict, {}
        self.step_num = 0
        self.current_episode += 1
        self._obs_dict, info = self._env.reset()
        if self.LLM_instuction:
            self.llm_tale.reset_high_level()
        obs = self._process_obs(self._obs_dict)
        obs = self.change_device(obs)
        if self.LLM_instuction:
            info = self.init_task_relevent_info(info)

        return obs, info

    def render(self, *args, **kwargs) -> None:
        """Render the environment"""
        frame = self._env.render(*args, **kwargs)
        if type(frame) is torch.Tensor:
            frame = frame.cpu().numpy()[0]
        return frame

    def close(self) -> None:
        """Close the environment"""
        self._env.close()

    def init_task_relevent_info(self, info):
        self.max_progress = 0
        info["progress"] = 0
        return info

    def get_task_relevent_info(self, info):
        info["progress"] = 0
        return info

    def task_relevant_reward(self, info):
        if info["success"]:
            return 100, torch.tensor([True])
        else:
            return 0, torch.tensor([False])
