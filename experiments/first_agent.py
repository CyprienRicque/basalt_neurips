import logging
import gym
import minerl

import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

"""
Now we can choose any one of the many environments included in the minerl package. 
To learn more about the environments checkout the environment documentation.
"""

"""
For this tutorial we’ll choose the MineRLBasaltFindCave-v0 environment. 
In this task, the agent is placed to a new world and its (subjective) goal is to find a cave, and end the episode.

To create the environment, simply invoke gym.make
"""

env = gym.make('MineRLBasaltFindCave-v0')


"""
Now we can reset this environment to its first position and get our first observation 
from the agent by resetting the environment.
"""
# Note that this command will launch the MineRL environment, which takes time.
# Be patient!
obs = env.reset()

"""
The obs variable will be a dictionary containing the following observations returned by the environment. 
In the case of the MineRLBasaltFindCave-v0 environment, 
only one observation is returned: pov, an RGB image of the agent’s first person perspective.
"""

"""
{
    'pov': array([[[ 63,  63,  68],
        [ 63,  63,  68],
        [ 63,  63,  68],
        ...,
        [ 92,  92, 100],
        [ 92,  92, 100],
        [ 92,  92, 100]],,

        ...,


        [[ 95, 118, 176],
        [ 95, 119, 177],
        [ 96, 119, 178],
        ...,
        [ 93, 116, 172],
        [ 93, 115, 171],
        [ 92, 115, 170]]], dtype=uint8)
}
"""

# if '__main__' == __name__:

done = False

print(env.action_space)
print(env.observation_space)

while not done:  # There is time limit + the agent can die
    # Take a random action
    action = env.action_space.sample()
    # In BASALT environments, sending ESC action will end the episode
    # Lets not do that
    action["ESC"] = 0
    obs, reward, done, _ = env.step(action)
    env.render()





