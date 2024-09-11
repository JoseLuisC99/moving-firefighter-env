from typing import Any, SupportsFloat
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from gymnasium.core import ActType, ObsType, RenderFrame
import networkx as nx
from itertools import combinations


class MovingFirefighter(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 n: int = 20,
                 p: float = 0.05,
                 num_fires: int = 3,
                 is_tree: bool = False,
                 time_slots: float = 1.0,
                 render_mode: str | None = None,
                 seed: int | None = None) -> None:
        assert num_fires >= 1, "there must be at least one burned node"
        assert n > num_fires, "number of nodes must be greater that number of burned nodes"
        assert 0 < p < 1, "probability must be greater than zero and less than 1"
        assert time_slots > 0, "time_slots must be greater than zero"
        assert render_mode is None or render_mode in self.metadata["render_modes"], ("render_mode must be one of the "
                                                                                     "available render modes")

        self.observation_space = spaces.Dict({
            "fire_graph": spaces.Graph(
                node_space=spaces.Box(low=-1.0, high=1.0, shape=(2,)),
                edge_space=spaces.Discrete(1)
            ),
            "fighter_graph": spaces.Graph(
                node_space=spaces.Box(low=-1.0, high=1.0, shape=(2,)),
                edge_space=spaces.Box(low=0, high=np.sqrt(8), shape=(1,))
            ),
            "fighter_pos": spaces.Discrete(n + 1),
            "burned_nodes": spaces.Sequence(spaces.Discrete(n + 1)),
            "defended_nodes": spaces.Sequence(spaces.Discrete(n + 1)),
        })
        self.action_space = spaces.Discrete(n)

        np.random.seed(seed)
        fire_graph = nx.random_tree(n, seed=seed)
        if not is_tree:
            edges_to_add = []
            for i, j in combinations(range(n), 2):
                if not fire_graph.has_edge(i, j) and np.random.random() < p:
                    edges_to_add.append((i, j))
            fire_graph.add_edges_from(edges_to_add)
        node_pos = nx.spring_layout(fire_graph, seed=seed)
        node_pos[n] = np.random.uniform(-1, 1, (2,))
        fighter_graph = nx.complete_graph(n + 1)

        nodes = []
        for i in range(n + 1):
            nodes.append(node_pos[i])

        distances = dict()
        for i, j in combinations(range(n + 1), 2):
            x1, y1 = node_pos[i]
            x2, y2 = node_pos[j]
            distances[(i, j)] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        fire_edges = []
        fire_edges_values = []
        for i, j in fire_graph.edges:
            fire_edges.append([i, j])
            fire_edges_values.append(0)

        fighter_edges = []
        fighter_edges_values = []
        for i, j in combinations(range(n + 1), 2):
            fighter_edges.append([i, j])
            fighter_edges_values.append(distances[(i, j)])

        nx.set_edge_attributes(fighter_graph, distances, "distance")

        self.nodes = np.array(nodes, dtype=np.float32)

        self.fire_edges = np.array(fire_edges, dtype=np.int32)
        self.fire_edges_values = np.array(fire_edges_values, dtype=np.int64)

        self.fighter_edges = np.array(fighter_edges, dtype=np.int32)
        self.fighter_edges_values = np.array(fighter_edges_values, dtype=np.float32).reshape(-1, 1)

        self.is_tree = is_tree
        self.n = n
        self.num_fires = num_fires
        self.time_slot = time_slots
        self.fire_graph = fire_graph
        self.fighter_graph = fighter_graph
        self.node_pos = node_pos

        self.fighter_pos = None
        self.burned_nodes = None
        self.defended_nodes = None
        self.fighter_time = 0
        self.fire_time = 0

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 720

    def _get_obs(self) -> dict[str, Any]:
        return {
            "fire_graph": spaces.GraphInstance(self.nodes[:-1], self.fire_edges_values, self.fire_edges),
            "fighter_graph": spaces.GraphInstance(self.nodes, self.fighter_edges_values, self.fighter_edges),
            "fighter_pos": self.fighter_pos,
            "burned_nodes": tuple(self.burned_nodes),
            "defended_nodes": tuple(self.defended_nodes),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "burned_nodes": len(self.burned_nodes),
            "fighter_time": self.fighter_time,
            "fire_time": self.fire_time,
            "sequence": self.defended_nodes,
        }

    def _propagate(self) -> int:
        self.fire_time += self.time_slot
        new_burnt = []
        for node in self.burned_nodes:
            for neighbor in self.fire_graph.neighbors(node):
                if neighbor not in self.defended_nodes and neighbor not in self.burned_nodes:
                    new_burnt.append(neighbor)
        self.burned_nodes.extend(new_burnt)
        return len(new_burnt)

    def valid_actions(self) -> np.ndarray:
        valid_actions = [self.fighter_pos]
        for neighbor in self.fighter_graph.neighbors(self.fighter_pos):
            # Can we move to a defended node? Maybe yes, to be closer to another interesting node
            # if neighbor in self.burned_nodes or neighbor in self.defended_nodes:
            if neighbor in self.burned_nodes:
                continue
            t = self.fighter_graph.edges[self.fighter_pos, neighbor]["distance"]
            if self.fighter_time + t <= self.fire_time + self.time_slot:
                valid_actions.append(neighbor)
        return np.array(valid_actions)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.fighter_pos = self.n
        self.burned_nodes = list(self.np_random.choice(range(self.n), self.num_fires, replace=False))
        self.defended_nodes = [self.n]
        self.fighter_time = 0
        self.fire_time = 0

        return self._get_obs(), self._get_info()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        terminated = False
        truncated = False
        t = 0 if action == self.fighter_pos else self.fighter_graph.edges[self.fighter_pos, action]["distance"]

        # The agent decides to let time flow
        if action == self.fighter_pos:
            new_burned_nodes = self._propagate()
            terminated = new_burned_nodes == 0
            reward = new_burned_nodes
        # the agent moves to a burned node
        elif action in self.burned_nodes:
            truncated = True
            reward = float("-inf")
        # the agent performs an invalid action
        elif self.fighter_time + t > self.fire_time + self.time_slot:
            truncated = True
            reward = float("-inf")
        else:
            self.fighter_time += t
            if action not in self.defended_nodes:
                self.defended_nodes.append(action)
            self.fighter_pos = action
            reward = 0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        node_colors = ((51, 51, 51), (193, 193, 193))
        burnt_colors = ((222, 49, 99), (245, 193, 208))
        anchor_colors = ((101, 148, 237), (186, 199, 222))
        defended_colors = ((65, 224, 208), (198, 246, 241))

        def get_position(position):
            return ((position + 1) * ((self.window_size - 40) / 2)) + 20

        def draw_circle(position, circle_colors, text):
            pygame.draw.circle(
                canvas,
                circle_colors[0],
                position,
                15,
            )
            pygame.draw.circle(
                canvas,
                circle_colors[1],
                position,
                14,
            )
            text_surface = font.render(text, True, circle_colors[0])
            text_rect = text_surface.get_rect(center=position)

            canvas.blit(text_surface, text_rect)

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        font = pygame.font.Font(None, 18)

        for i, j in self.fire_graph.edges:
            pygame.draw.line(
                canvas,
                node_colors[0],
                get_position(self.node_pos[i]),
                get_position(self.node_pos[j]),
                width=1,
            )

        for i in range(self.n):
            if i in self.burned_nodes:
                colors = burnt_colors
            elif i in self.defended_nodes:
                colors = defended_colors
            else:
                colors = node_colors
            draw_circle(get_position(self.node_pos[i]), colors, f"{i}")
        draw_circle(get_position(self.node_pos[self.n]), anchor_colors, "A")

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
