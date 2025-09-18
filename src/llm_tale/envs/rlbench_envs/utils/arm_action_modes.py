import time
import numpy as np
from pyrep.errors import IKError
from rlbench.backend.exceptions import InvalidActionError
from rlbench.action_modes.arm_action_modes import (
    ArmActionMode,
    assert_action_shape,
    assert_unit_quaternion,
    calculate_delta_pose,
    RelativeFrame,
)


class SafeEndEffectorPoseViaIK(ArmActionMode):
    """High-level action where target pose is given and reached via IK.

    Given a target pose, IK via inverse Jacobian is performed. This requires
    the target pose to be close to the current pose, otherwise the action
    will fail. It is up to the user to constrain the action to
    meaningful values.

    The decision to apply collision checking is a crucial trade off!
    With collision checking enabled, you are guaranteed collision free paths,
    but this may not be applicable for task that do require some collision.
    E.g. using this mode on pushing object will mean that the generated
    path will actively avoid not pushing the object.
    """

    def __init__(
        self, absolute_mode: bool = True, frame: RelativeFrame = RelativeFrame.WORLD, collision_checking: bool = False
    ):
        """
        Args:
            absolute_mode: If we should opperate in 'absolute', or 'delta' mode.
            frame: Either WORLD or EE.
            collision_checking: IF collision checking is enabled.
        """
        self._absolute_mode = absolute_mode
        self._frame = frame
        self._collision_checking = collision_checking
        self._max_steps = 50
        self._time_limit_s = 5
        self._reached_atol = 1e-2
        self._stagnant_atol = 1e-3
        self._stagnant_patience = 1

    def action(self, scene, action: np.ndarray):
        # --- Input checks and frame handling (same intent as original) ---
        assert_action_shape(action, (7,))
        assert_unit_quaternion(action[3:])

        if not self._absolute_mode and self._frame != RelativeFrame.EE:
            action = calculate_delta_pose(scene.robot, action)

        relative_to = None if self._frame == RelativeFrame.WORLD else scene.robot.arm.get_tip()

        # --- Solve IK and set target joint positions ---
        try:
            joint_positions = scene.robot.arm.solve_ik_via_jacobian(action[:3], quaternion=action[3:], relative_to=relative_to)
            scene.robot.arm.set_joint_target_positions(joint_positions)
        except IKError as e:
            raise InvalidActionError(
                "Could not perform IK via Jacobian; most likely due to current "
                "end-effector pose being too far from the given target pose. "
                "Try limiting/bounding your action space."
            ) from e

        # --- Convergence watchdogs ---
        remaining_steps = self._max_steps
        t0 = time.time()
        stagnant_steps = 0
        prev_positions = None

        while True:
            scene.step()

            cur_positions = scene.robot.arm.get_joint_positions()
            reached = np.allclose(cur_positions, joint_positions, atol=self._reached_atol)

            if prev_positions is not None:
                if np.allclose(cur_positions, prev_positions, atol=self._stagnant_atol):
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0
            prev_positions = cur_positions

            not_moving = stagnant_steps >= self._stagnant_patience
            if reached or not_moving:
                break

            # Hard stops
            if remaining_steps is not None:
                remaining_steps -= 1
                if remaining_steps <= 0:
                    # Exiting gracefully; if you'd rather treat as error, raise instead.
                    break

            if self._time_limit_s is not None and (time.time() - t0) > self._time_limit_s:
                break

    def action_shape(self, scene) -> tuple:
        return (7,)
