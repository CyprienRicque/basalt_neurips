import numpy as np
import torch as th
import cv2
from gym3.types import DictType
from gym import spaces

from src.openai_vpt.lib.action_mapping import CameraHierarchicalMapping
from src.openai_vpt.lib.actions import ActionTransformer
from src.openai_vpt.lib.policy import MinecraftAgentPolicy
from src.openai_vpt.lib.torch_util import default_device_type, set_default_torch_device

from src.openai_vpt.agent import AGENT_RESOLUTION, ACTION_TRANSFORMER_KWARGS, ENV_KWARGS, TARGET_ACTION_SPACE, resize_image


class MineRLAgentPP:
    def __init__(self, env, device=None):
        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)

        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    def _env_obs_to_agent(self, minerl_obs):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        agent_input = resize_image(minerl_obs["pov"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(self.device)}
        return agent_input

    def _agent_action_to_env(self, agent_action):
        """
        Turn output from policy into action for MineRL
        This is quite important step (for some reason).
        For the sake of your sanity, remember to do this step (manual conversion to numpy)
        before proceeding. Otherwise, your agent might be a little derp.
        """
        action = agent_action
        if isinstance(action["buttons"], th.Tensor):
            action = {
                "buttons": agent_action["buttons"].cpu().numpy(),
                "camera": agent_action["camera"].cpu().numpy()
            }
        minerl_action = self.action_mapper.to_factored(action)
        minerl_action_transformed = self.action_transformer.policy2env(minerl_action)
        return minerl_action_transformed

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(self.device) for k, v in action.items()}
        return action
