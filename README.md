# moving-firefighter-env

<div align="center">
![moving-firefighter example](https://raw.githubusercontent.com/JoseLuisC99/moving-firefighter-env/docs/images/graph_example.gif)
</div>

## Description

This environment simulates the [Moving Firefighter Problem](https://www.mdpi.com/2227-7390/11/1/179), a model where a firefighter races against the spread of a fire across a network.

### The Dynamics

* **Firefighter**: The firefighter strategically moves across the network, protecting nodes from the fire. Protected nodes cannot be burned in the future.
* **Fire**: The fire starts at specific nodes and spreads to connected neighbors each time step. The fire's network (graph) can be different from the one the firefighter navigates.

The firefighter can't teleport; movement between nodes takes time. The firefighter receives information about their location, the networks, and the fire's progress (protected vs. burned nodes). They can then choose to move to different valid nodes (see [Reward Structure](https://github.com/JoseLuisC99/moving-firefighter-env?tab=readme-ov-file#reward-structure)).

The goal is to test the agent's ability to strategically move the firefighter to minimize the total number of nodes burned by the fire.

## Installation

```shell
pip install gymnasium
git clone https://github.com/JoseLuisC99/moving-firefighter-env.git
cd moving-firefighter-env
pip install -e .
```

## Usage

```python
import gymnasium as gym
import numpy as np

env = gym.make("mfp/MovingFirefighter-v0")
observation, info = env.reset()

for _ in range(1000):
  # Replace with your agent's action selection logic
  action = np.random.choice(env.unwrapped.valid_actions())
  observation, reward, terminated, truncated, info = env.step(action)

  if terminated or truncated:
     observation, info = env.reset()

env.close()
```

### Observation Space

The observation is a dictionary with the following keys:
* `"graph_burnt"`: the graph showing how the fire spreads, with edges representing the time it takes the fire to move between nodes.
* `"graph_fighter"`: the complete graph where the firefighter navigates, defining travel times between nodes.
* `"fighter_pos"`: the firefighter's current location.
* `"burnt_nodes"`: nodes already burned by the fire.
* `"defended_nodes"`: nodes protected by the firefighter and safe from the fire.

### Action Space

The action space is `gym.spaces.Discrete(n)`, representing the action of moving to a node to protect it, and `n` is the number of nodes.

### Reward Structure

* The agent receives a large negative reward if it moves to an invalid node (defended, burnt, or unreachable).
* Otherwise, the agent receives a negative reward equivalent to the number of burnt nodes up to that time. 

## Contributing

Feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is licensed under the [MIT License](https://raw.githubusercontent.com/JoseLuisC99/moving-firefighter-env/main/LICENSE).

## Contact

José Luis Castro García (<jlcastrog99@gmail.com>)
