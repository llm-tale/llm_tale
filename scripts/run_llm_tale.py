import torch
import argparse
import glob
import json
import numpy as np
import os
import imageio
from skrl.utils import set_seed
from skrl.trainers.torch import SequentialTrainer, StepTrainer

from llm_tale.llm_tale import LLM_TALE
from llm_tale.agents.td3_skrl import setup_td3
from llm_tale.agents.ppo_skrl import setup_ppo
from llm_tale.llm_bc import BC


def run(env_id, seed, train, algo, image_input, llm_bc, timesteps, wandb, record=False):
    if env_id == "PegInsert":
        from llm_tale.envs.maniskill_envs.tasks.PegInsert import (
            env,
            env_kwargs,
            kwargs,
            llm_plan_path,
            llm_tale_kwargs,
        )
    elif env_id == "PickCube":
        from llm_tale.envs.maniskill_envs.tasks.PickCube import env, env_kwargs, kwargs, llm_plan_path, llm_tale_kwargs
    elif env_id == "StackCube":
        from llm_tale.envs.maniskill_envs.tasks.StackCube import (
            env,
            env_kwargs,
            kwargs,
            llm_plan_path,
            llm_tale_kwargs,
        )
    elif env_id == "PutBox":
        from llm_tale.envs.rlbench_envs.tasks.PutBox import env, llm_tale_kwargs, env_kwargs, kwargs, llm_plan_path
    elif env_id == "OpenDrawer":
        from llm_tale.envs.rlbench_envs.tasks.OpenDrawer import env, llm_tale_kwargs, env_kwargs, kwargs, llm_plan_path
    elif env_id == "TakeLid":
        from llm_tale.envs.rlbench_envs.tasks.TakeLid import env, llm_tale_kwargs, env_kwargs, kwargs, llm_plan_path
    else:
        print("Invalid environment")
        return

    algos = {
        "ppo_explo": setup_ppo,
        "td3_explo": setup_td3,
    }
    mode = "train" if train else "eval"
    kwargs_copy = kwargs.copy()
    kwargs = kwargs[algo][mode]
    kwargs["wandb"] = wandb
    llm_tale_kwargs = llm_tale_kwargs[algo]

    if not train:
        llm_tale_kwargs["start_choice_ep"] = 0
        env_kwargs[algo]["eval"] = True
        if record:
            env_kwargs[algo]["record"] = True
        if algo == "td3_explo":
            kwargs["buffer_size"] = 1

    # Image input and llm_bc learning flags
    if image_input:
        assert env_id in ["PickCube", "StackCube"], "Image input is only supported for PickCube and StackCube"
        assert not llm_bc, "llm_bc learning is not supported with image input"
        exp_name = "image_env_{}_algo_{}_seed_{}".format(env_id, algo, seed)
        kwargs["image_input"] = True
        env_kwargs[algo]["image_input"] = True
        kwargs_copy[algo]["image_input"] = True
        if algo == "td3_explo":
            kwargs["batch_size"] = 512
            kwargs["actor_learning_rate"] = 0.75e-3
            kwargs["critic_learning_rate"] = 0.75e-3

    elif llm_bc:
        assert env_id in [
            "PickCube",
            "StackCube",
            "PegInsert",
        ], "llm_bc learning is supported only for PickCube, StackCube, and PegInsert"
        exp_name = "residual_env_{}_algo_{}_seed_{}".format(env_id, algo, seed)
        model = BC(19 + 9, 7) if env_id == "PickCube" else BC(23 + 9, 7)
        demo_seed = seed - 1
        model.load_state_dict(torch.load(f"models/{env_id}_llmbc_seed{demo_seed}.pth"))
        model.eval()
        model.to(torch.device("cuda"))
        llm_tale_kwargs["learned_basepolicy"] = model
    else:
        exp_name = "state_env_{}_algo_{}_seed_{}".format(env_id, algo, seed)

    env_instance = env(**env_kwargs[algo])
    kwargs["env"] = env_instance
    kwargs["exp_name"] = exp_name

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not train and algo == "td3_explo":
        kwargs["buffer_size"] = 1
    agent = algos[algo](**kwargs)
    llm_tale_kwargs["rl_policy"] = agent
    llm_tale_agent = LLM_TALE(**llm_tale_kwargs)

    env_instance.reset(fake_reset=True)
    env_instance.set_llm_tale(llm_tale_agent)
    env_instance.set_llm_plan(llm_plan_path)

    print(env_instance.observation_space)
    print(env_instance.action_space)
    print(train)

    if train:
        cfg_trainer = {"timesteps": timesteps, "headless": True}
        trainer = SequentialTrainer(cfg=cfg_trainer, env=env_instance, agents=agent)
        agent.write_checkpoint(0, 0)
        trainer.train()
    else:
        eval_seed = seed + 1000
        set_seed(eval_seed)
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)

        tb_path = kwargs_copy[algo]["train"]["tb_path"] + f"/{exp_name}/checkpoints"
        files = glob.glob(tb_path + "/*.pt")
        cfg = {"timesteps": env_kwargs[algo]["episode_length"] * 10, "headless": True}
        trainer = StepTrainer(env=env_instance, agents=agent, cfg=cfg)

        results = {}
        results_file = os.path.join(
            tb_path,
            "results.json",
        )
        # load existing results if they exist
        if os.path.exists(results_file):
            # check if the file is empty
            if os.path.getsize(results_file) == 0:
                results = {}
                assert not record, "Results file is empty, but record is set to True. Please evaluate first."
            else:
                with open(results_file, "r") as f:
                    results = json.load(f)
                if record:
                    best = -np.inf
                    ckpt = "agent_0.pt"
                    for key, value in results.items():
                        if isinstance(value, list):
                            value = np.array(value)
                        if isinstance(value, np.ndarray):
                            value = value.mean().item()
                        if value >= best and key != "best_agent.pt":
                            if int(key.split("_")[-1][:-3]) >= int(ckpt.split("_")[-1][:-3]):
                                best = value
                                ckpt = key
                    files = [os.path.join(tb_path, ckpt)]
                    print(f"Recording best checkpoint: {ckpt}")
        else:
            assert not record, "Results file does not exist, but record is set to True. Please evaluate first."

        env_instance.episode_trigger = lambda x: x % 1 == 0
        for file in files:
            ckpt = file.split("/")[-1]
            # check if results already contains the file
            if ckpt in results and not record:
                print(f"Already evaluated {ckpt}")
                continue
            agent.load(file)
            agent.set_running_mode("eval")
            env_instance.current_episode == 0
            frames = []
            successes = 0
            for i in range(10):
                obs, info = env_instance.reset()
                if record:
                    frames.append(env_instance.render())
                done = False
                while not done:
                    with torch.no_grad():
                        actions = agent.act(obs, 1, 1)
                    obs, reward, terminated, truncated, info = env_instance.step(actions[0])
                    if record:
                        frames.append(env_instance.render())
                    done = terminated or truncated
                if info["success"]:
                    successes += 1
                if record:
                    # Add last frame 5 times to show the final state
                    for _ in range(5):
                        frames.append(frames[-1])
            if record:
                video_file = os.path.join(
                    tb_path,
                    "best_agent.mp4",
                )
                imageio.mimsave(video_file, frames, fps=15)
            else:
                results[ckpt] = successes / 10
                with open(results_file, "w") as f:
                    json.dump(results, f)
    env_instance.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument("--env", type=str, help="Environment to train on")
    parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Random seed(s) for reproducibility",
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "eval", "record"], default="train", help="Mode: train, eval, or record"
    )
    parser.add_argument("--algo", type=str, default="td3_explo", help="Algorithm to use")
    parser.add_argument(
        "--image-input",
        action=argparse.BooleanOptionalAction,
        help="Flag to use image input (default: False)",
    )
    parser.add_argument(
        "--llm_bc",
        action=argparse.BooleanOptionalAction,
        help="Flag to use llm_bc learning (default: False)",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        help="Flag to log results to WandB (default: False)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Now you can use the arguments anywhere in your program
    print(f"Environment: {args.env}")
    print(f"Seed: {args.seed}")
    print(f"Mode: {args.mode}")

    if args.env == "PegInsert":
        timesteps = 2000000
    elif args.env in ["PickCube", "StackCube"]:
        timesteps = 1000000
    elif args.env in ["PutBox", "OpenDrawer"]:
        timesteps = 300000
    elif args.env == "TakeLid":
        timesteps = 100000
    else:
        raise ValueError("Invalid environment")

    seeds = args.seed
    if len(seeds) > 1:
        import torch.multiprocessing as mp

        processes = []
        for seed in seeds:
            p = mp.Process(
                target=run,
                args=(
                    args.env,
                    seed,
                    args.mode == "train",
                    args.algo,
                    args.image_input,
                    args.llm_bc,
                    timesteps,
                    args.wandb,
                    args.mode == "record",
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        run(
            env_id=args.env,
            seed=seeds[0],
            train=args.mode == "train",
            algo=args.algo,
            image_input=args.image_input,
            llm_bc=args.llm_bc,
            timesteps=timesteps,
            wandb=args.wandb,
            record=args.mode == "record",
        )
