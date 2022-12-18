import logging
from argparse import ArgumentParser
import pickle

import aicrowd_gym

from src.openai_vpt.agent import MineRLAgent


def main(model, weights, env, n_episodes=3, max_steps=int(1e9), show=False):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    logging.debug("env = aicrowd_gym.make(env)")
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    logging.debug(f"MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)")
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    logging.debug(f"agent.load_weights(weights)")
    agent.load_weights(weights)

    logging.debug(f"Running {n_episodes} with {max_steps=}")

    for _ in range(n_episodes):
        logging.debug("obs = env.reset()")
        obs = env.reset()
        for s in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            if done:
                logging.debug(f"done.")
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)
