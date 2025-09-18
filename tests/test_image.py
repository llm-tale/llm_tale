"""Integration tests for tasks."""

from absl.testing import absltest
from absl.testing import parameterized


class TaskTest(parameterized.TestCase):
    @parameterized.parameters(
        ("PickCube", "td3_explo", True),
        ("PickCube", "td3_explo", False),
        ("StackCube", "td3_explo", True),
        ("StackCube", "td3_explo", False),
        ("PickCube", "ppo_explo", True),
        ("PickCube", "ppo_explo", False),
        ("StackCube", "ppo_explo", True),
        ("StackCube", "ppo_explo", False),
    )
    def test_tasks(self, env_id, algo, train):
        from scripts.run_llm_tale import run

        args = {
            "env_id": env_id,
            "algo": algo,
            "train": train,
            "seed": 0,
            "timesteps": 75,
            "image_input": True,
            "llm_bc": False,
            "wandb": False,
        }
        print(f"Running test for task: {env_id}, algo: {algo}, train: {train}")
        run(**args)


if __name__ == "__main__":
    absltest.main()
