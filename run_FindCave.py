from run_agent import main as run_agent_main
from config import EVAL_EPISODES, EVAL_MAX_STEPS
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    run_agent_main(
        model="data/VPT-models/1x.model",
        weights="train/MineRLBasaltFindCave.weights",
        env="MineRLBasaltFindCave-v0",
        n_episodes=EVAL_EPISODES,
        # max_steps=EVAL_MAX_STEPS,
        max_steps=1000,
        show=True
    )


if __name__ == "__main__":
    main()


