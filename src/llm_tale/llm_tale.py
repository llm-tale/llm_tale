import torch
import pickle
import numpy as np
from llm_tale.utils.rotation_utils import quat2rotm, rotm2quat


class LLM_TALE:
    def __init__(
        self,
        rl_policy,
        env_type="maniskill",  # maniskill or rlbench
        learned_basepolicy=None,
        start_choice_ep=100,
        temperature=1.25,
        conf_coeff=0.0002,
        min_conf=0.2,
    ):
        self.rl_policy = rl_policy
        self.env_type = env_type
        self.start_choice_ep = start_choice_ep
        self.conf_coeff = conf_coeff
        self.min_conf = min_conf
        self.temperature = temperature
        self.learned_basepolicy = learned_basepolicy
        self._plan = []
        self.plan_var = []
        self._affordance = []

    def set_llm_plan(self, llm_plan_path):
        with open(llm_plan_path, "rb") as f:
            res = pickle.load(f)
        if self.env_type == "maniskill":
            from llm_tale.envs.maniskill_envs.utils.primitive_pt import (
                pick,
                transport,
                place,
                insert,
                residual_action,
                primitive,
            )
            from llm_tale.envs.maniskill_envs.utils.parse_affordance import (
                parse_code,
                parse_pick_affordance,
                parse_transport_affordance,
            )

            self._prim = {
                "pick": pick,
                "transport": transport,
                "place": place,
                "insert": insert,
                "residual_action": residual_action,
                "primtive": primitive,
                "parse_pick_affordance": parse_pick_affordance,
                "parse_transport_affordance": parse_transport_affordance,
            }
        elif self.env_type == "rlbench":
            from llm_tale.envs.rlbench_envs.utils.primitive_pt import pick, transport, residual_action, primitive
            from llm_tale.envs.rlbench_envs.utils.parse_affordance import (
                parse_code,
                parse_pick_affordance,
                parse_transport_affordance,
            )

            self._prim = {
                "pick": pick,
                "transport": transport,
                "residual_action": residual_action,
                "primtive": primitive,
                "parse_pick_affordance": parse_pick_affordance,
                "parse_transport_affordance": parse_transport_affordance,
            }
        else:
            raise NotImplementedError
        for i in range(len(res["code"])):
            code_item = parse_code(res["code"][str(i + 1)])
            self._plan.append(code_item)
        for i in range(len(res["aff_space"])):
            _aff = []
            key = "pick" if i == 0 else "place"
            for j in range(len(res["aff_space"][key])):
                _step_mode = res["aff_space"][key][str(j + 1)]
                _aff.append(res["affordance"][_step_mode])
            self.plan_var.append(len(_aff))
            self._affordance.append(_aff)
        self.uncertainty = [torch.ones(_) for _ in self.plan_var]
        print("setting LLM plan########################")
        print(self._plan)
        print(self._affordance)

    def get_llm_tale_reward(self, gripper_pose, raw_actions, observation):
        reward = self.plans[self.current_plannum].reward(gripper_pose, raw_actions, observation)
        return reward

    def set_env(
        self,
        task_objects,
        _post_process_obs,
        change_device,
        get_picked_obj=None,
        is_grasping=None,
        on_policy=False,
    ):
        # bridge the function from env to llm_tale
        self.task_objects = task_objects
        self._post_process_obs = _post_process_obs
        self.change_device = change_device
        self.on_policy = on_policy
        self.get_picked_obj = get_picked_obj
        self.is_grasping = is_grasping

    def reset_high_level(self):
        self.plans = []  # a local plan holder for each episode
        for i in range(len(self._plan)):
            func = self._plan[i][0]
            if func == self._prim["pick"]:
                obj = self._plan[i][1]
                self.plans.append(func(obj))
            elif func == self._prim["transport"] or func == self._prim["place"] or func == self._prim["insert"]:
                obj1 = self._plan[i][1]
                obj2 = self._plan[i][2]
                self.plans.append(func(obj1, obj2))
        self.plan_choice = [0, 0]
        self.current_plannum = 0

    def step_high_level(self, obs, objects, start_choose, _eval):
        # start_time = time.time()
        gripper_pose = obs["gripper_pose"]

        if self.plans[self.current_plannum].success:
            print("plan {} success".format(self.current_plannum))
            if self.current_plannum < len(self.plans) - 1:
                self.current_plannum += 1

        cur_plan = self.plans[self.current_plannum]
        if not cur_plan.initialized:
            cur_plan.initialize()
            if cur_plan.name == "pick":
                cur_plan.obj_init_pose = objects[cur_plan.target_object]
            if cur_plan.name == "transport":
                pickedobj_pose = objects[cur_plan.picked_object]
                cur_plan.init_bias(gripper_pose, pickedobj_pose)

            self.choose_value(obs, objects, start_choose, _eval)
            aff = self._affordance[self.current_plannum][self.plan_choice[self.current_plannum]]
            cur_plan.relative_pos, cur_plan.relative_rot = self.cal_relative_se3(cur_plan, gripper_pose, aff)

        target_pose = self.calculate_target_pose(cur_plan.relative_pos, cur_plan.relative_rot, objects)
        if self.env_type == "maniskill":
            obj = self.get_picked_obj()
            object_in_hand = self.is_grasping(obj)
        elif self.env_type == "rlbench":
            object_in_hand = self.is_grasping()

        cur_plan.object_in_hand = object_in_hand
        cur_plan.update_state(gripper_pose, target_pose, objects)
        cur_plan.target_pose = target_pose

        goal_pose = target_pose
        goal_space = torch.cat([goal_pose, self.plans[self.current_plannum].gripper_goal])
        return goal_space

    def reform_obs_with_action(
        self,
        obs,
        goal_space,
        input_action=torch.zeros(
            7,
        ),
    ):
        # given the goal space, reform the obs with the action
        # if input action is zeros, wchich means only for the llm_bc action in observation space
        if self.learned_basepolicy is not None:
            _obs = torch.cat([obs["gripper_pose"], obs["gripper_open"], obs["obj_states"], goal_space]).to(
                torch.device("cuda")
            )
            action = self.learned_basepolicy(_obs)
        else:
            action = self._prim["residual_action"](
                gripper_poses=obs["gripper_pose"],
                target_poses=goal_space,
                prim=self.plans[self.current_plannum],
                action=input_action,
            )
        action = self.plans[self.current_plannum].prime_action(action)
        obs["base_action"] = action
        return obs

    def cal_relative_se3(self, cur_plan, gripper_pose, aff):
        # calculate relative pose based on affordance

        aff_func = self._prim["parse_pick_affordance"] if cur_plan.name == "pick" else self._prim["parse_transport_affordance"]
        relative_pos, relative_rot = aff_func(aff, cur_plan, gripper_pose, self.task_objects)
        return relative_pos, relative_rot

    def get_value(self, obs, goal_space):
        states = self._post_process_obs(obs, goal_space)
        states = self.change_device(states)
        with torch.no_grad():
            states = self.rl_policy._state_preprocessor(states)
            if self.on_policy:
                self.rl_policy.value._shared_output = None
                value = self.rl_policy.value.act({"states": states}, role="value")
            else:
                sampled_actions = self.rl_policy.policy.act({"states": states}, role="policy")[0]
                value = self.rl_policy.critic_1.act(
                    {"states": states, "taken_actions": sampled_actions},
                    role="critic_1",
                )
            return value[0]

    def choose_value(self, obs, objects, start_choose, _eval=False):
        # choosing affordance plan based on value function (Q or V)
        affs = self._affordance[self.current_plannum]
        if len(affs) > 1 and self.rl_policy is not None:
            affs = self._affordance[self.current_plannum]
            cur_plan = self.plans[self.current_plannum]
            t = self.temperature
            values = []
            scores = []
            for i in range(len(affs)):
                aff = self._affordance[self.current_plannum][i]
                relative_pos, relative_rot = self.cal_relative_se3(cur_plan, obs["gripper_pose"], aff)
                goal_pose = self.calculate_target_pose(relative_pos, relative_rot, objects)
                obs = self.reform_obs_with_action(obs, torch.cat([goal_pose, cur_plan.gripper_goal]))
                value = self.get_value(obs, torch.cat([goal_pose, cur_plan.gripper_goal])).item()
                values.append(value)
            values_mean = np.mean(values).item()
            for i in range(len(affs)):
                if not start_choose:
                    score = self.uncertainty[self.current_plannum][i].item()
                else:
                    score = (
                        np.exp(np.clip(values[i] - values_mean, -1.0, 1.0) * t).item()
                        * self.uncertainty[self.current_plannum][i].item()
                    )
                scores.append(score)
            for i in range(len(affs)):
                self.rl_policy.track_data(
                    "EXPLO value/{}_{}".format(self.current_plannum, i),
                    values[i],
                )
                self.rl_policy.track_data(
                    "EXPLO uncertainty/{}_{}".format(self.current_plannum, i),
                    self.uncertainty[self.current_plannum][i].item(),
                )

            if not _eval:
                scores = np.array(scores) / sum(scores)
                choice = np.random.choice(len(affs), 1, p=scores)[0]
            else:
                print("eval mode")
                values = np.array([v for v in values])
                choice = np.argmax(values)
            print("##########value##########", values)
            print("#####choice#######", choice)
            self.plan_choice[self.current_plannum] = choice
            self.uncertainty[self.current_plannum][choice] -= self.conf_coeff * self.uncertainty[self.current_plannum][choice]
            self.uncertainty[self.current_plannum][choice] = max(self.uncertainty[self.current_plannum][choice], self.min_conf)
        elif len(affs) > 1:
            # for those do not have policy
            choice = np.random.choice(len(affs), 1)[0]
            self.plan_choice[self.current_plannum] = choice
            print("random choice", self.plan_choice[self.current_plannum])
        else:
            # for those that only have one affordance
            self.plan_choice[self.current_plannum] = 0

    def calculate_target_pose(self, relative_pos, relative_rot, objects):
        obj_name = self.plans[self.current_plannum].target_object
        mode = "wxyz" if self.env_type == "maniskill" else "xyzw"
        if obj_name is not None:
            targetobj_pose = objects[obj_name]
            if targetobj_pose.shape[0] < 7:
                # no rotation specified
                target_pos = targetobj_pose[:3] + relative_pos
                target_rot = relative_rot
            else:
                target_pos = targetobj_pose[:3] + quat2rotm(targetobj_pose[3:], mode=mode) @ relative_pos
                target_rot = quat2rotm(targetobj_pose[3:], mode=mode) @ relative_rot
            target_quat = rotm2quat(target_rot, mode=mode)
            target_pose = torch.cat([target_pos, target_quat])
            return target_pose
        else:
            # if not target object specify, relative means absolute
            target_quat = rotm2quat(relative_rot, mode=mode)
            target_pose = torch.cat([relative_pos, target_quat])
            return target_pose

    def print_value(self, obs):
        if self.rl_policy is not None:
            if not self.on_policy:
                states = self.rl_policy._state_preprocessor(obs)
                sampled_actions = self.rl_policy.policy.act({"states": states}, role="policy")[0]
                value = self.rl_policy.critic_1.act({"states": states, "taken_actions": sampled_actions}, role="critic_1")
                print("value: ", value)
