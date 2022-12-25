from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS


def main():
    run_agent_main(
        model="data/VPT-models/2x.model",
        weights="train/foundation-model-2x-MineRLBasaltMakeWaterfall.weights",
        env="MineRLBasaltMakeWaterfall-v0",
        n_episodes=EVAL_EPISODES,
        max_steps=1000,
        # max_steps=EVAL_MAX_STEPS,
        show=True
    )


if __name__ == "__main__":
    main()
