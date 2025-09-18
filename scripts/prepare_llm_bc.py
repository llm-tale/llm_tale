import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skrl.utils import set_seed

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from llm_tale.llm_tale import LLM_TALE
from llm_tale.llm_bc import BC, BCDataset, train_epoch, validate_epoch


def main(env_id, mode, seed, plot):
    env_name = env_id
    if env_id == "PegInsert":
        from llm_tale.envs.maniskill_envs.tasks.PegInsert import (
            env,
            env_kwargs,
            llm_plan_path,
            llm_tale_kwargs,
        )
    elif env_id == "PickCube":
        from llm_tale.envs.maniskill_envs.tasks.PickCube import env, env_kwargs, llm_plan_path, llm_tale_kwargs
    elif env_id == "StackCube":
        from llm_tale.envs.maniskill_envs.tasks.StackCube import (
            env,
            env_kwargs,
            llm_plan_path,
            llm_tale_kwargs,
        )
    else:
        raise ValueError("Unsupported environment")
    env = env(**env_kwargs["ppo_explo"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    set_seed(seed)
    llm_tale_kwargs = llm_tale_kwargs["td3_explo"]
    llm_tale_kwargs["rl_policy"] = None
    llm_tale_agent = LLM_TALE(**llm_tale_kwargs)
    env.reset(fake_reset=True)
    env.set_llm_tale(llm_tale_agent)
    env.set_llm_plan(llm_plan_path)
    if mode == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        file_name = "demos/{}_data_seed{}.h5".format(env_name, seed)
        with h5py.File(file_name, "r") as hf:
            observations = {key: torch.tensor(hf[key][:]) for key in hf.keys()}
            actions = torch.tensor(hf["actions"][:])

        action_data = observations["base_action"]
        action_data[:, -1] = actions[:, -1]
        obs_data = torch.cat(
            [
                observations["gripper_pose"],
                observations["gripper_open"],
                observations["obj_states"],
                observations["goal_space"],
            ],
            dim=-1,
        )

        # Create dataset and dataloaders
        dataset = BCDataset(obs_data, action_data)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

        # Define the model
        input_dim = obs_data.shape[1]
        output_dim = action_data.shape[1]
        model = BC(input_dim, output_dim).to(device)

        # Setup the optimizer and loss function with L2 regularization
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=2e-5)
        criterion = nn.MSELoss()

        # Training and Validation
        if env_name == "PickCube":
            epochs = 200
        elif env_name == "StackCube":
            epochs = 400
        elif env_name == "PegInsert":
            epochs = 1000
        train_losses, val_losses = [], []
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss = validate_epoch(model, val_loader, criterion, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # save the model
        torch.save(model.state_dict(), f"models/{env_name}_llmbc_seed{seed}.pth")
        if plot:
            # Plot the training and validation loss
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Training Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.title("Training and Validation Losses")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()
    elif mode == "eval":
        if env_name == "PickCube":
            model = BC(19 + 9, 7)
        elif env_name == "StackCube":
            model = BC(23 + 9, 7)
        elif env_name == "PegInsert":
            model = BC(23 + 9, 7)
        model.load_state_dict(torch.load(f"models/{env_name}_llmbc_seed{seed}.pth"))
        model.eval()
        env.learned_basepolicy = model
        env.episode_length = 50
        obs, instruction = env.reset()
        success = []
        episide_num = 0
        max_episode = 50
        while episide_num < max_episode:
            action = torch.zeros(7)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated | truncated
            env.render()
            if done:
                if info["success"]:
                    success.append(True)
                else:
                    success.append(False)

                obs, instruction = env.reset()
                episide_num += 1
        env.close()
        print(np.mean(success))
    elif mode == "collect":
        obs, instruction = env.reset()
        success = []
        episide_num = 0
        max_episode = 200
        observations = []
        actions = []
        epi_obs = []
        epi_act = []
        while episide_num < max_episode:
            action = llm_tale_agent.plans[llm_tale_agent.current_plannum].dummy_action()
            obs, reward, terminated, truncated, info = env.step(action)
            epi_obs.append(env.observation)
            epi_act.append(action)
            done = terminated | truncated
            if done:
                if llm_tale_agent.plans[0].success is not None:
                    print("saving {} th data".format(episide_num))
                    observations += epi_obs
                    actions += epi_act
                    episide_num += 1
                epi_obs = []
                epi_act = []
                if info["success"]:
                    success.append(True)
                else:
                    success.append(False)

                obs, instruction = env.reset()
        print(np.mean(success))
        print("saving data")
        print("observation:", len(observations))
        print("actions:", len(actions))

        file_name = "demos/{}_data_seed{}.h5".format(env_name, seed)
        with h5py.File(file_name, "w") as hf:
            if isinstance(observations[0], dict):
                for key in observations[0].keys():
                    hf.create_dataset(key, data=np.stack([obs[key].cpu().numpy() for obs in observations]))
            hf.create_dataset("actions", data=np.stack([action.cpu().numpy() for action in actions]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument(
        "--env", type=str, help="Environment to train on", default="PickCube", choices=["PickCube", "StackCube", "PegInsert"]
    )
    parser.add_argument("--mode", type=str, default="eval", help="Mode to run in", choices=["train", "eval", "collect"])
    parser.add_argument("--seed", type=int, default=0, help="Random seed, use 0, 2, 4 to reproduce results in the paper")
    parser.add_argument(
        "--plot",
        action=argparse.BooleanOptionalAction,
        help="Flag to plot the training and validation loss (default: False)",
    )

    args = parser.parse_args()

    main(args.env, args.mode, args.seed, args.plot)
