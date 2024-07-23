import time

import gym
import numpy as np
from pyquaternion import Quaternion
from widowx_envs.widowx_env_service import WidowXClient
import cv2


def state_to_eep(xyz_coor, zangle: float):
    """
    Implement the state to eep function.
    Refered to `bridge_data_robot`'s `widowx_controller/widowx_controller.py`
    return a 4x4 matrix
    """
    assert len(xyz_coor) == 3
    DEFAULT_ROTATION = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
    new_pose = np.eye(4)
    new_pose[:3, -1] = xyz_coor
    new_quat = Quaternion(axis=np.array([0.0, 0.0, 1.0]), angle=zangle) * Quaternion(
        matrix=DEFAULT_ROTATION
    )
    new_pose[:3, :3] = new_quat.rotation_matrix
    # yaw, pitch, roll = quat.yaw_pitch_roll
    return new_pose

def load_frames_from_video(video_path):
    """
    Loads frames from a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List of frames extracted from the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def wait_for_obs_from_input(frame_list, index, im_size):
    """
    Fetches the next frame from the frame list.

    Args:
        frame_list (list): List of frames.
        index (int): Current index in the frame list.

    Returns:
        obs: The next observation frame.
        new_index: The updated index.
    """
    if index >= len(frame_list):
        raise IndexError("Index out of range. No more frames available.")
    
    obs = frame_list[index]
    return obs, index + 1

def wait_for_obs(widowx_client):
    obs = widowx_client.get_observation()
    while obs is None:
        print("Waiting for observations...")
        obs = widowx_client.get_observation()
        time.sleep(1)
    return obs

def convert_obs(obs, im_size):
    print(obs)
    print(f"Original shape of obs['image']: {obs['image'].shape}")
    obs['image'] = (obs["image"].reshape(3, 256, 256).transpose(1, 2, 0) * 255).astype(np.uint8)
    # print(obs["image"].shape)

    obs['image'] = cv2.resize(np.array(obs['image']), (256, 256))
    image_obs = obs["image"]

    # print(obs["image"].shape)

    # image_obs = (
    #     obs["image"].reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
    # ).astype(np.uint8)
    # add padding to proprio to match training

    # TODO: sumedh look at this man
    # proprio = np.concatenate([obs["state"][:6], [0], obs["state"][-1:]])

    proprio = np.concatenate([obs["state"][:6],[0], obs["state"][-1:]])

    # NOTE: assume image_1 is not available
    return {
        "image_primary": image_obs,
    }


def null_obs(img_size):
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
    }


class WidowXGym(gym.Env):
    """
    A Gym environment for the WidowX controller provided by:
    https://github.com/rail-berkeley/bridge_data_robot
    Needed to use Gym wrappers.
    """

    def __init__(
        self,
        widowx_client: WidowXClient,
        im_size: int = 256,
        blocking: bool = True,
        sticky_gripper_num_steps: int = 1,
    ):
        self.widowx_client = widowx_client
        self.im_size = im_size
        self.blocking = blocking
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float64
                ),
                # "proprio": gym.spaces.Box(
                #     low=np.ones((8,)) * -1, high=np.ones((8,)), dtype=np.float64
                # ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64
        )
        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

    def step(self, action):
        # sticky gripper logic
        if (action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.is_gripper_closed else 1.0
        print(action)

        self.widowx_client.step_action(action, blocking=self.blocking)

        raw_obs = self.widowx_client.get_observation()

        truncated = False
        if raw_obs is None:
            # this indicates a loss of connection with the server
            # due to an exception in the last step so end the trajectory
            truncated = True
            obs = null_obs(self.im_size)  # obs with all zeros
        else:
            obs = convert_obs(raw_obs, self.im_size)

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.widowx_client.reset()

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        raw_obs = wait_for_obs(self.widowx_client)
        obs = convert_obs(raw_obs, self.im_size)

        return obs, {}
