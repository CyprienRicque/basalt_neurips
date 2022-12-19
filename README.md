# NeurIPS 2022: MineRL BASALT Behavioural Cloning Baseline

[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/BT9uegr)

This repository contains a behavioural cloning baseline solution for the MineRL BASALT 2022 Competition ("basalt" track)! This solution fine-tunes the "width-x1" models of OpenAI VPT for more sample-efficient training.

You can find the "intro" track baseline solution [here](https://github.com/minerllabs/basalt-2022-intro-track-baseline).

MineRL BASALT is a competition on solving human-judged tasks. The tasks in this competition do not have a pre-defined reward function: the goal is to produce trajectories that are judged by real humans to be effective at solving a given task.

See [the AICrowd competition page](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition) for further details on the competition.


## Downloading the BASALT dataset

You can find the index files containing all the download URLs for the full BASALT dataset in the [OpenAI VPT repository at the bottom](https://github.com/openai/Video-Pre-Training#basalt-2022-dataset).

We have included a download utility (`utils/download_dataset.py`) to help you download the dataset. You can use it with the index files from the OpenAI VPT repository. For example, if you download the FindCave dataset index file, named `find-cave-Jul-28.json`, you can download first 100 demonstrations to `MineRLBasaltFindCave-v0` directory with:

```
python download_dataset.py --json-file find-cave-Jul-28.json --output-dir MineRLBasaltFindCave-v0 --num-demos 100
```

Basic dataset statistics (note: one trajectory/demonstration may be split up into multiple videos):
```
Size  #Videos  Name
--------------------------------------------------
146G  1399     MineRLBasaltBuildVillageHouse-v0
165G  2833     MineRLBasaltCreateVillageAnimalPen-v0
165G  5466     MineRLBasaltFindCave-v0
175G  4230     MineRLBasaltMakeWaterfall-v0
```



## Setting up

Install [MineRL v1.0.0](https://github.com/minerllabs/minerl) (or newer) and the requirements for [OpenAI VPT](https://github.com/openai/Video-Pre-Training).

Download the dataset following above instructions. Also download the 1x width foundational model [.weights](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.weights) and [.model](https://openaipublic.blob.core.windows.net/minecraft-rl/models/foundation-model-1x.model) files for the OpenAI VPT model.

Place these data files under `data` to match the following structure:

```
├── data
│   ├── MineRLBasaltBuildVillageHouse-v0
│   │   ├── Player70-f153ac423f61-20220707-111912.jsonl
│   │   ├── Player70-f153ac423f61-20220707-111912.mp4
│   │   └── ... rest of the files
│   ├── MineRLBasaltCreateVillageAnimalPen-v0
│   │   └── ... files as above
│   ├── MineRLBasaltFindCave-v0
│   │   └── ... files as above
│   ├── MineRLBasaltMakeWaterfall-v0
│   │   └── ... files as above
│   └── VPT-models
│       ├── foundation-model-1x.model
│       └── foundation-model-1x.weights
```


## Training models

Running following code will save a fine-tuned network for each task under `train` directory. This has been tested to fit into a 8GB GPU.

```
python train.py
```

## Visualizing/enjoying/evaluating models

To run the trained model for `MineRLBasaltFindCave-v0`, run the following:

```
python run_agent.py --model data/VPT-models/foundation-model-1x.model --weights train/MineRLBasaltFindCave.weights --env MineRLBasaltFindCave-v0 --show
```

Change `FindCave` to other tasks to run for different tasks.

## How to Submit a Model on AICrowd.

**Note:** This repository is *not* submittable as-is. You first need to train the models, add them to the git repository and then submit to AICrowd.

To submit this baseline agent follow the [submission instructions](https://github.com/minerllabs/basalt_2022_competition_submission_template/), but use this repo instead of the starter kit repo.


# TMP

This code defines a PyTorch neural network model that appears to be designed for use in a reinforcement learning (RL) task. The model, called MinecraftPolicy, extends PyTorch's nn.Module class and contains several layers and submodules.

The MinecraftPolicy class takes several parameters that determine the structure and behavior of the model, including the recurrence_type which specifies whether and how the model will use recurrence (e.g. LSTM), the n_recurrence_layers which specifies the number of consecutive LSTM layers to use if recurrence_type is set to "multi_layer_lstm" or "multi_masked_lstm", and the dense_init_norm_kwargs and init_norm_kwargs which specify the arguments for initializing and normalizing the model's layers.

The model contains two submodules: ImgObsProcess and ImpalaCNN. The ImgObsProcess submodule takes an input image and applies the ImpalaCNN submodule to it, followed by a linear layer. The ImpalaCNN submodule appears to be a convolutional neural network (CNN) with a particular architecture and normalization methods.

The MinecraftPolicy class also contains several layers and functions that implement various RL-related tasks such as an action head (make_action_head), a loss function (ScaledMSEHead), and a residual recurrent blocks layer (ResidualRecurrentBlocks).

Overall, this code defines a PyTorch neural network model that appears to be designed for use in an RL task, specifically in the context of the game Minecraft.


The MinecraftPolicy model is a PyTorch neural network that extends the nn.Module class. The model takes as input an image and produces as output a predicted value, policy, and action. The model consists of several layers and submodules, including an ImgObsProcess submodule, a recurrent layer, a value_head submodule, and a pi_head and action_head submodule pair.

The ImgObsProcess submodule applies the ImpalaCNN submodule to the input image, followed by a linear layer. The ImpalaCNN submodule is a CNN with a particular architecture and normalization methods. The output of the ImgObsProcess submodule is then passed to the recurrent layer, which applies recurrence (e.g. LSTM) to the sequence of hidden states computed from the output of the ImgObsProcess submodule.

The output of the recurrent layer is then passed to the value_head submodule, which is a linear layer with a single output. The output of the value_head is used in the calculation of the loss function. The output of the recurrent layer is also passed to the make_action_head function, which creates the pi_head and action_head submodule pair. The pi_head is a linear layer with one output for each possible action followed by a softmax function, while the action_head is a linear layer with one output for each possible action. The output of the pi_head is used to select the next action and update the model's parameters during training.

Overall, the MinecraftPolicy model is a complex neural network with multiple layers and submodules that processes input images and produces predictions for the value, policy, and action in an RL task.


The value_head refers to a specific part of the MinecraftPolicy model that outputs a value prediction. In RL, the value of a state is a measure of how good it is for the agent to be in that state, and the value head is a neural network layer or submodule that predicts the value of a given state. In this specific implementation, the value_head is a submodule of the MinecraftPolicy class that contains a linear layer with a single output. This layer takes as input the output of the recurrent layer, which is a sequence of hidden states computed by applying recurrence (e.g. LSTM) to the output of the ImgObsProcess submodule. The output of the value_head is then used in the calculation of the loss function, which is used to update the model's parameters during training.

The action_head and pi_head refer to specific parts of the MinecraftPolicy model that output predictions for the next action and policy, respectively. In RL, the action is the choice that the agent makes at each timestep, and the policy is the function that determines the probability of choosing each possible action given a state. The action_head and pi_head are neural network layers or submodules that predict the next action and policy, respectively, given a state.

In this specific implementation, the action_head and pi_head are submodules of the MinecraftPolicy class that are created by the make_action_head function. This function takes as input the output of the recurrent layer and returns two submodules: the action_head and pi_head. The action_head is a linear layer with one output for each possible action, and the pi_head is a linear layer with one output for each possible action followed by a softmax function, which outputs a probability distribution over the actions.

The action_head and pi_head are different from the value_head in that they predict different quantities. The value_head predicts the value of a state, whereas the action_head and pi_head predict the next action and policy, respectively. These predictions are used in different ways during training: the value prediction is used in the calculation of the loss function, while the action and policy predictions are used to select the next action and update the model's parameters.

Yes, you are correct. In this specific implementation, the pi_head is created using the make_action_head function, which means that the pi_head is actually the same as the action_head with an additional softmax function applied to its output. This means that the pi_head and action_head predict the same thing, but the pi_head outputs a probability distribution over the actions while the action_head outputs the raw predicted action values.